"""
Batch Inference Script for Preference-Guided Language Model

Supports:
- Multi-GPU distributed inference using Accelerate
- Resume capability from checkpoints (Fixed Distributed Logic)
- Incremental CSV saving
- Two-stage inference: preference prediction + guided generation
"""
import sys, os
sys.path.insert(0, os.path.abspath('../'))
import argparse
import csv
import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

from sga_utils import load_model_withhead
from load_datasets import build_hh_rlhf_dataset, parse_hh_rlhf_dialogue
from preference_injection import create_injector


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch Inference for Preference-Guided Model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name or path")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, 
                       default="path/to/no_1-hh-rlhf",
                       help="Path to HH-RLHF test dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save inference results")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--save_interval", type=int, default=20,
                       help="Save results every N steps")
    
    # Preference arguments
    parser.add_argument("--num_preferences", type=int, default=2,
                       help="Number of preference dimensions")
    parser.add_argument("--preference_names", nargs="+", 
                       default=["harmless", "helpful"],
                       help="Names of preference dimensions")
    
    # Value head configuration
    parser.add_argument("--layer_type", type=str, default="mlp",
                       help="Value head layer type")
    parser.add_argument("--num_layers", type=int, default=3,
                       help="Number of layers in value head")
    parser.add_argument("--num_neurons", type=int, default=1024,
                       help="Number of neurons in value head")
    
    # Generation arguments
    parser.add_argument("--format_style", type=str, default="plain",
                       choices=["angle_brackets", "angle_brackets_outer", "plain"],
                       help="Format style for preference text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                       help="Repetition penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                       help="N-gram size for no repeat")
    
    # Resume and checkpoint
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if exists")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    return parser.parse_args()


class InferenceCheckpoint:
    """Manages checkpoint for resume capability"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.processed_indices = set()
        self.last_saved_step = 0
        
    def load(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                self.processed_indices = set(data.get('processed_indices', []))
                self.last_saved_step = data.get('last_saved_step', 0)
                return True
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                return False
        return False
    
    def save(self, processed_indices: set, current_step: int):
        """Save checkpoint"""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            data = {
                'processed_indices': list(processed_indices),
                'last_saved_step': current_step
            }
            with open(self.checkpoint_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def is_processed(self, idx: int) -> bool:
        """Check if an index has been processed"""
        return idx in self.processed_indices
    
    def mark_processed(self, indices: List[int]):
        """Mark indices as processed"""
        self.processed_indices.update(indices)


class CSVResultWriter:
    """Incremental CSV writer for results"""
    
    def __init__(self, csv_path: str, max_buffer_size: int = 100):
        self.csv_path = csv_path
        self.fieldnames = [
            'index', 'prompt', 'history_messages', 
            'predicted_harmless', 'predicted_helpful',
            'generated_response',
            'ground_truth_response', 'ground_truth_harmless', 'ground_truth_helpful'
        ]
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        
        # Create directory if needed
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Check if file exists to determine if we need to write header
        self.file_exists = os.path.exists(csv_path)
    
    def add_result(self, result: Dict):
        """Add a result to buffer"""
        self.buffer.append(result)
        # Auto-flush when buffer reaches max size
        if len(self.buffer) >= self.max_buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered results to CSV"""
        if not self.buffer:
            return
        
        mode = 'a' if self.file_exists else 'w'
        try:
            with open(self.csv_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not self.file_exists:
                    writer.writeheader()
                    self.file_exists = True
                writer.writerows(self.buffer)
            self.buffer.clear()
        except Exception as e:
            print(f"Error writing to CSV: {e}")


class IndexWrapperDataset(Dataset):
    """Wraps a dataset to inject the original global index into the sample item."""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Inject the original index so it survives Subset and DataLoader batching
        # Use copy to avoid modifying the original dataset in memory unexpectedly
        if isinstance(item, dict):
            item = item.copy()
            item['original_index'] = idx
        return item


def load_model(args, device, rank=0):
    """Load model with value head"""
    print(f"Loading model from {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        if 'Llama' in args.base_model:
            # Better to use unk_token or eos_token than adding new one to avoid embedding resize issues
            if tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    if not hasattr(tokenizer, 'max_length'):
        tokenizer.max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048
    
    device_str = str(device)
    if 'cuda' in device_str:
        device_for_load = rank
    else:
        device_for_load = 'cpu'
    
    model = load_model_withhead(
        model_name=args.base_model,
        peft_name=args.model_path,
        tokenizer=tokenizer,
        device=device_for_load,
        layer_type=args.layer_type,
        num_neurons=args.num_neurons,
        num_layers=args.num_layers,
        num_output=args.num_preferences
    )
    
    model.eval()
    return model, tokenizer


def predict_preferences_batch(
    model, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        _, _, preferences = model(input_ids=input_ids, attention_mask=attention_mask)
    
    if len(preferences.shape) == 1:
        preferences = preferences.unsqueeze(0)
    
    return preferences


def generate_response_batch(
    model,
    tokenizer,
    history_messages_batch: List[List[Dict]],
    last_user_messages_batch: List[str],
    preferences_batch: torch.Tensor,
    injector,
    args,
    device
) -> List[str]:
    """
    Generates responses one by one (pseudo-batch) as per user requirements.
    """
    batch_size = len(history_messages_batch)
    generated_texts = []
    
    for i in range(batch_size):
        try:
            history_messages = history_messages_batch[i]
            last_user_message = last_user_messages_batch[i]
            preferences = preferences_batch[i]
            
            pref_text = injector.format_preferences(preferences)
            last_user_with_pref = f"{last_user_message}  {pref_text}"
            
            inference_messages = history_messages.copy()
            inference_messages.append({"role": "user", "content": last_user_with_pref})
            
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt_formatted = tokenizer.apply_chat_template(
                    inference_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt_formatted = last_user_with_pref
            
            inputs = tokenizer(
                prompt_formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model.pretrained_model.generate(
                    **inputs,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size
                )
            
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            del inputs, outputs, generated_ids
            
        except Exception as e:
            print(f"Error generating response for sample {i}: {e}")
            generated_texts.append(f"[ERROR: {str(e)}]")
    
    return generated_texts


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return batch


def get_gpu_memory_info(device):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    return None


def main():
    args = parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    
    model_path_parts = Path(args.model_path).parts
    
    if model_path_parts[-1].startswith('checkpoint-'):
        checkpoint_name = model_path_parts[-1]
        model_name = model_path_parts[-2]
    else:
        model_name = model_path_parts[-1]
        checkpoint_name = "xx"
    
    output_dir_name = f"{model_name}_{checkpoint_name}"
    final_output_dir = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print("="*70)
        print("Batch Inference for Preference-Guided Model (Fixed Resume Logic)")
        print("="*70)
        print(f"Model path: {args.model_path}")
        print(f"Final output dir: {final_output_dir}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print("="*70)
    
    model, tokenizer = load_model(args, device, rank)
    
    if accelerator.is_main_process:
        print(f"\nLoading test dataset from {args.data_path}...")
    
    from datasets import load_from_disk
    ds_temp = load_from_disk(args.data_path)
    
    target_split = 'test'
    if hasattr(ds_temp, 'keys'):
        if target_split in ds_temp:
            ds_temp = ds_temp[target_split]
        else:
            ds_temp = ds_temp[list(ds_temp.keys())[0]]
    
    harmless_scores = [float(item.get('harmless', 0.0)) for item in ds_temp]
    helpful_scores = [float(item.get('helpful', 0.0)) for item in ds_temp]
    harmless_min, harmless_max = min(harmless_scores), max(harmless_scores)
    helpful_min, helpful_max = min(helpful_scores), max(helpful_scores)
    
    # 1. Load the raw dataset
    raw_test_dataset = build_hh_rlhf_dataset(
        args.data_path,
        tokenizer,
        split='test',
        size=args.max_samples,
        harmless_min=harmless_min,
        harmless_max=harmless_max,
        helpful_min=helpful_min,
        helpful_max=helpful_max
    )
    
    # Wrap dataset to inject permanent global indices
    test_dataset = IndexWrapperDataset(raw_test_dataset)
    if accelerator.is_main_process:
        print(f"Loaded and indexed {len(test_dataset)} samples")
    
    checkpoint_path = os.path.join(final_output_dir, f"inference_checkpoint_rank{rank}.json")
    checkpoint = InferenceCheckpoint(checkpoint_path)
    
    if args.resume:
        loaded = checkpoint.load()
        if loaded and accelerator.is_main_process:
            print(f"Resumed from checkpoint: {len(checkpoint.processed_indices)} samples already processed by rank {rank}")
    
    # ==========================================================================
    # CRITICAL FIX: Do not subset the dataset before Accelerate!
    # We use the FULL dataset here, allowing Accelerate to shard it consistently.
    # ==========================================================================
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model, dataloader = accelerator.prepare(model, dataloader)
    
    csv_path = os.path.join(final_output_dir, f"results_rank{rank}.csv")
    csv_writer = CSVResultWriter(csv_path)

    if 'outer' in args.model_path:
        format_style1 = 'angle_brackets_outer'
    elif 'plain' in args.model_path:
        format_style1 = 'plain'
    else:
        format_style1 = 'angle_brackets'
    
    injector = create_injector(
        injection_type='text',
        tokenizer=tokenizer,
        preference_names=args.preference_names,
        format_style=format_style1
    )
    
    step = 0
    total_processed = 0
    processed_indices_in_run = []
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process)):
            # ==================================================================
            # CRITICAL FIX: Filter batch elements manually based on checkpoint
            # ==================================================================
            
            # 'batch' is a List[Dict] because of the collate_fn
            unprocessed_batch = []
            for item in batch:
                if not checkpoint.is_processed(item['original_index']):
                    unprocessed_batch.append(item)
            
            # If entire batch is already processed, skip it
            if not unprocessed_batch:
                continue
                
            # Use the filtered batch for inference
            batch = unprocessed_batch
            batch_size = len(batch)
            
            # Now construct input tensors only for the unprocessed items
            input_ids_list = [item['input_ids_prompt'] for item in batch]
            attention_mask_list = [item['attention_mask_prompt'] for item in batch]
            
            max_len = max(len(ids) for ids in input_ids_list)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_len - len(ids)
                padded_ids = torch.cat([ids, torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device)])
                padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype, device=mask.device)])
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            input_ids_prompt = torch.stack(padded_input_ids).to(device)
            attention_mask_prompt = torch.stack(padded_attention_masks).to(device)
            
            history_messages_batch = []
            for item in batch:
                history_msg = item['history_messages']
                if isinstance(history_msg, str):
                    history_msg = json.loads(history_msg)
                history_messages_batch.append(history_msg)
            
            last_user_messages_batch = [item['last_user_message'] for item in batch]
            response_texts_batch = [item['response_text'] for item in batch]
            preference_labels_batch = torch.stack([item['preference_labels'] for item in batch])
            
            predicted_preferences = predict_preferences_batch(
                model, input_ids_prompt, attention_mask_prompt
            )
            
            # Keeping your original pseudo-batch logic
            generated_responses = generate_response_batch(
                model,
                tokenizer,
                history_messages_batch,
                last_user_messages_batch,
                predicted_preferences,
                injector,
                args,
                device
            )
            
            for i in range(batch_size):
                original_idx = batch[i]['original_index']
                
                full_dialogue = ""
                for msg in history_messages_batch[i]:
                    role_name = "Human" if msg['role'] == 'user' else "Assistant"
                    full_dialogue += f"\n\n{role_name}: {msg['content']}"
                full_dialogue += f"\n\nHuman: {last_user_messages_batch[i]}\n\nAssistant:"
                
                result = {
                    'index': original_idx,
                    'prompt': full_dialogue,
                    'history_messages': json.dumps(history_messages_batch[i]),
                    'predicted_harmless': predicted_preferences[i][0].item(),
                    'predicted_helpful': predicted_preferences[i][1].item(),
                    'generated_response': generated_responses[i],
                    'ground_truth_response': response_texts_batch[i],
                    'ground_truth_harmless': preference_labels_batch[i][0].item(),
                    'ground_truth_helpful': preference_labels_batch[i][1].item()
                }
                
                csv_writer.add_result(result)
                processed_indices_in_run.append(original_idx)
            
            # Clean up
            del input_ids_prompt, attention_mask_prompt, predicted_preferences, preference_labels_batch
            del padded_input_ids, padded_attention_masks
            
            step += 1
            total_processed += batch_size
            
            if step % args.save_interval == 0:
                csv_writer.flush()
                checkpoint.mark_processed(processed_indices_in_run)
                checkpoint.save(checkpoint.processed_indices, step)
                processed_indices_in_run.clear()
                
                # Removed empty_cache() inside loop for speed
                
                if accelerator.is_main_process:
                    mem_info = get_gpu_memory_info(device)
                    if mem_info:
                        # Note: total_processed might be approximate in resumed runs for logging
                        print(f"Checkpoint saved at step {step} ({total_processed} new samples processed)")
        
        csv_writer.flush()
        checkpoint.mark_processed(processed_indices_in_run)
        checkpoint.save(checkpoint.processed_indices, step)
        
        if accelerator.is_main_process:
            print(f"\nInference completed! Processed {total_processed} new samples")
            print(f"Results saved to: {csv_path}")
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nInterrupted! Saving checkpoint...")
        csv_writer.flush()
        checkpoint.mark_processed(processed_indices_in_run)
        checkpoint.save(checkpoint.processed_indices, step)
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\nError during inference: {e}")
            import traceback
            traceback.print_exc()
        csv_writer.flush()
        checkpoint.mark_processed(processed_indices_in_run)
        checkpoint.save(checkpoint.processed_indices, step)
        raise
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\nMerging results from all GPUs...")
        merge_results(final_output_dir, accelerator.num_processes)
        print("Done!")


def merge_results(output_dir: str, num_processes: int):
    all_results = []
    
    for rank in range(num_processes):
        csv_path = os.path.join(output_dir, f"results_rank{rank}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                all_results.append(df)
                print(f"Loaded {len(df)} results from rank {rank}")
            except Exception as e:
                print(f"Warning: Failed to load results from rank {rank}: {e}")
    
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        # Sort by index to ensure final order is correct
        merged_df = merged_df.sort_values('index').drop_duplicates(subset=['index'], keep='last').reset_index(drop=True)
        merged_path = os.path.join(output_dir, "final_results.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"Merged {len(merged_df)} total results to: {merged_path}")
    else:
        print("Warning: No results found to merge")


if __name__ == "__main__":
    main()