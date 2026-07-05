"""
Stage-1 Inference Script: Preference Score Prediction

Loads HH-RLHF, HelpSteer2, or UltraFeedback dataset, predicts preference scores, and outputs formatted JSON files for Stage-2 (vllm_gam.py).
"""

# ========== HH-RLHF Dataset Inference Example ==========
# python infer/infer_preference.py \
#     --model_path "path/to/checkpoint-1501" \
#     --device "cuda:0" \
#     --dataset_type "hh_rlhf" \
#     --dataset_path "path/to/no_1-hh-rlhf_test" \
#     --dataset_split "test" \
#     --max_samples 100 \
#     --batch_size 4 \
#     --output_dir "./inference_results/hh_rlhf_run"

# ========== HelpSteer2 Dataset Inference Example ==========
# python infer/infer_preference.py \
#     --model_path "/path/to/helpsteer2_model/checkpoint" \
#     --device "cuda:0" \
#     --dataset_type "helpsteer2" \
#     --dataset_path "path/to/HelpSteer2_processed" \
#     --dataset_split "validation" \
#     --max_samples 100 \
#     --batch_size 4 \
#     --output_dir "./inference_results/helpsteer2_run"

# ========== UltraFeedback Dataset Inference Example ==========
# python infer/infer_preference.py \
#     --model_path "/path/to/ultrafeedback_model/checkpoint" \
#     --device "cuda:0" \
#     --dataset_type "ultrafeedback" \
#     --dataset_path "path/to/ultrafeedback-binarized-preferences-cleaned-kto/processed" \
#     --dataset_split "test" \
#     --max_samples 100 \
#     --batch_size 4 \
#     --output_dir "./inference_results/ultrafeedback_run"

import sys
import os

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk

from sga_utils import load_model_withhead, model_withhead_forward
from load_datasets import build_hh_rlhf_dataset, build_helpsteer2_dataset, build_ultrafeedback_dataset, build_alpaca_eval_dataset, build_arena_hard_dataset, parse_hh_rlhf_dialogue
from preference_injection import create_injector
from normalization import create_normalizer, PreferenceNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 Preference Score Prediction")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--base_model", type=str, default=None, help="Base model for LoRA")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    # Value head config
    parser.add_argument("--num_preferences", type=int, default=None,
                       help="Number of preferences (auto-detected from model config or dataset type)")
    parser.add_argument("--preference_names", nargs="+", default=None,
                       help="Preference names (auto-detected from model config or dataset type)")
    parser.add_argument("--layer_type", type=str, default='mlp')
    parser.add_argument("--num_neurons", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_type", type=str, default="hh_rlhf",
                       choices=["hh_rlhf", "helpsteer2", "ultrafeedback", "alpaca_eval", "arena_hard"],
                       help="Dataset type: hh_rlhf (2 prefs), helpsteer2 (5 prefs), ultrafeedback (4 prefs), alpaca_eval (benchmark), or arena_hard (benchmark)")
    parser.add_argument("--alpaca_preference_source", type=str, default="hh_rlhf",
                       choices=["hh_rlhf", "helpsteer2", "ultrafeedback"],
                       help="For alpaca_eval/arena_hard only: which preference set to use (hh_rlhf/helpsteer2/ultrafeedback)")
    parser.add_argument("--reference_answer_path", type=str, default=None,
                       help="For arena_hard only (optional): reference model_answer jsonl path (e.g., gpt-4-0613.jsonl)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for AlpacaEval output (generator field)")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # Normalization arguments
    parser.add_argument("--normalizer_load_path", type=str, default=None)
    parser.add_argument("--normalization_method", type=str, default='sigmoid',
                       choices=['minmax', 'robust', 'sigmoid', 'quantile'])
    parser.add_argument("--normalization_dataset_path", type=str, default=None)
    
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    # Set default preference names based on dataset_type
    default_preference_names = {
        'hh_rlhf': ['harmless', 'helpful'],
        'helpsteer2': ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
        'ultrafeedback': ['helpfulness', 'honesty', 'instruction_following', 'truthfulness'],
        'alpaca_eval': ['harmless', 'helpful'],
        'arena_hard': ['harmless', 'helpful'],
    }

    if args.dataset_type in ['alpaca_eval', 'arena_hard']:
        default_preference_names[args.dataset_type] = default_preference_names.get(
            args.alpaca_preference_source,
            default_preference_names['hh_rlhf'],
        )
    
    # Try to read configuration from config.json
    config_path = os.path.join(args.model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Read parameters from config file
        layer_type = config.get('vhead_layer_type', args.layer_type)
        num_neurons = config.get('vhead_num_neurons', args.num_neurons)
        num_layers = config.get('vhead_num_layers', args.num_layers)
        num_preferences = config.get('num_preferences', args.num_preferences)
        preference_names = config.get('preference_names', args.preference_names)
        preference_activation = config.get('preference_activation', 'sigmoid')  # Read activation function config
        
        # If no preference_names in config, set default value based on dataset_type
        if preference_names is None:
            preference_names = default_preference_names.get(args.dataset_type, ['harmless', 'helpful'])
            num_preferences = len(preference_names)
        
        print(f"Parameters loaded from config file:")
        print(f"  Value Head Type: {layer_type}")
        print(f"  Preference Dimensions: {num_preferences}")
        print(f"  Preference Names: {preference_names}")
        print(f"  Preference Activation: {preference_activation}")
        
    else:
        layer_type = args.layer_type
        num_neurons = args.num_neurons
        num_layers = args.num_layers
        # If not specified, set default value based on dataset_type
        preference_names = args.preference_names if args.preference_names else default_preference_names.get(args.dataset_type, ['harmless', 'helpful'])
        num_preferences = args.num_preferences if args.num_preferences else len(preference_names)

    if num_preferences is None:
        num_preferences = len(preference_names)
    if int(num_preferences) != len(preference_names):
        raise ValueError(
            f"num_preferences ({num_preferences}) does not match the length of preference_names ({len(preference_names)}): {preference_names}"
        )
    
    # Check if this is a LoRA model
    is_lora = os.path.exists(os.path.join(args.model_path, 'adapter_config.json'))
    
    # For LoRA models, base_model path must be determined first to load tokenizer from it
    # Because the tokenizer config in the LoRA checkpoint may be incomplete or use an abstract base class
    if is_lora:
        if args.base_model is None:
            # Try to read from adapter_config.json
            adapter_config_path = os.path.join(args.model_path, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get('base_model_name_or_path')
            if base_model is None:
                raise ValueError("LoRA model requires --base_model argument")
        else:
            base_model = args.base_model
        tokenizer_path = base_model
    else:
        base_model = None
        tokenizer_path = args.model_path
    
    # Load tokenizer - Do not specify use_fast, let AutoTokenizer automatically choose the most appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_lora:
        
        print(f"\nLoading LoRA model:")
        print(f"  Base Model: {base_model}")
        print(f"  LoRA Adapter: {args.model_path}")
        
        model = load_model_withhead(
            model_name=base_model,
            peft_name=args.model_path,
            tokenizer=tokenizer,
            device=args.device,
            layer_type=layer_type,
            num_neurons=num_neurons,
            num_layers=num_layers,
            num_output=num_preferences,
            load_in_8bit=False
        )
    else:
        print(f"\nLoading full model: {args.model_path}")
        
        model = load_model_withhead(
            model_name=args.model_path,
            peft_name='',
            tokenizer=tokenizer,
            device=args.device,
            layer_type=layer_type,
            num_neurons=num_neurons,
            num_layers=num_layers,
            num_output=num_preferences,
            load_in_8bit=False
        )
    
    model.eval()
    print(f"\nModel loaded successfully!")
    
    return model, tokenizer, preference_names


def predict(model, tokenizer, question, answer="", args=None):
    """Predict preference weights for a single query-response pair."""
    if args is None:
        raise ValueError("args parameter is required")
    
    # Build input
    if hasattr(tokenizer, 'apply_chat_template'):
        if answer:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        else:
            messages = [{"role": "user", "content": question}]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not bool(answer)
        )
    else:
        input_text = f"{question}\n{answer}" if answer else question
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_length
    )
    
    input_ids = inputs['input_ids'].to(args.device)
    attention_mask = inputs['attention_mask'].to(args.device)
    
    # Forward pass
    with torch.no_grad():
        if args.use_amp:
            with torch.cuda.amp.autocast():
                preference_scores = model_withhead_forward(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=args.device,
                    forward_type='reward'
                )
        else:
            preference_scores = model_withhead_forward(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=args.device,
                forward_type='reward'
            )
    
    if preference_scores.dim() == 2:
        preference_scores = preference_scores.squeeze(0)
    
    return preference_scores.cpu()


def batch_predict_dataset(model, tokenizer, dataset, preference_names, injector, normalizer, args):
    """
    Batch predict preference scores for dataset and output formatted JSON for Stage-2.
    
    Supports HH-RLHF (2-dim preference) and HelpSteer2 (5-dim preference) datasets.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        dataset: Processed dataset
        preference_names: List of preference dimension names
        injector: Preference formatter
        normalizer: Normalizer (None for HelpSteer2)
        args: Command line arguments
    """
    model.eval()
    tokenizer.padding_side = 'left'  # Use left padding for batch inference
    
    formatted_results = []
    all_predictions = []
    all_labels = []
    
    num_prefs = len(preference_names)
    # HelpSteer2, UltraFeedback and AlpacaEval datasets are pre-normalized or have no ground truth labels, no denormalization needed
    skip_denormalize = (args.dataset_type in ['helpsteer2', 'ultrafeedback', 'alpaca_eval'])
    
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    print(f"\nStarting batch inference (batch_size={args.batch_size}, total_batches={total_batches})...")
    print(f"Dataset type: {args.dataset_type}, Preference dimensions: {num_prefs}")
    
    # Process by batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(dataset), args.batch_size), desc="Inference Progress"):
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_size = end_idx - start_idx
            
            # Extract batch data from dataset
            input_ids_list = [dataset[i]['input_ids_prompt'] for i in range(start_idx, end_idx)]
            attention_mask_list = [dataset[i]['attention_mask_prompt'] for i in range(start_idx, end_idx)]
            
            # Pad to same length
            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_len - len(ids)
                padded_ids = torch.cat([
                    torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device),
                    ids
                ])
                padded_mask = torch.cat([
                    torch.zeros(padding_length, dtype=mask.dtype, device=mask.device),
                    mask
                ])
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            input_ids = torch.stack(padded_input_ids).to(args.device)
            attention_mask = torch.stack(padded_attention_masks).to(args.device)
            
            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    preference_scores = model_withhead_forward(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=args.device,
                        forward_type='reward'
                    )
            else:
                preference_scores = model_withhead_forward(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=args.device,
                    forward_type='reward'
                )
            
            # Convert to numpy
            predictions = preference_scores.cpu().numpy()
            all_predictions.append(predictions)
            
            # Process each sample
            for i in range(batch_size):
                idx = start_idx + i
                sample = dataset[idx]
                
                # Extract ground truth labels (for evaluation comparison)
                true_labels = sample['preference_labels'].cpu().numpy()
                all_labels.append(true_labels)
                
                # Model predicted scores (Dynamic dimension)
                pred_scores = [float(predictions[i, j]) for j in range(num_prefs)]
                
                # Parse history messages
                history_messages = json.loads(sample['history_messages'])
                last_user_message = sample['last_user_message']
                response_text = sample['response_text']
                
                # Build preference text (using model predicted scores)
                pred_scores_tensor = torch.tensor(pred_scores)
                pref_text = injector.format_preferences(pred_scores_tensor)
                
                # Add preference constraint to last human message
                last_user_with_pref = f"{last_user_message}\n{pref_text}"
                
                # Build complete conversation list (compatible with vllm_gam.py input format)
                conversation = []
                for msg in history_messages:
                    if msg['role'] == 'user':
                        conversation.append({"human": msg['content'], "assistant": ""})
                    elif msg['role'] == 'assistant':
                        if conversation:
                            conversation[-1]["assistant"] = msg['content']
                
                # Add final conversation turn (with preference constraint)
                conversation.append({
                    "human": last_user_with_pref,
                    "assistant": response_text
                })
                
                # Build output format (compatible with test_angle_brackets.json format)
                # Build dynamic metadata
                metadata = {"index": idx}
                metadata["num_preferences"] = num_prefs
                metadata["preference_names"] = preference_names
                for j, pref_name in enumerate(preference_names):
                    metadata[f"predicted_{pref_name}"] = pred_scores[j]
                    metadata[f"true_{pref_name}"] = float(true_labels[j])
                    # HH-RLHF requires denormalization, HelpSteer2/UltraFeedback/AlpacaEval do not
                    if normalizer is not None and not skip_denormalize:
                        metadata[f"true_{pref_name}_orig"] = normalizer.denormalize(float(true_labels[j]), pref_name)
                
                # AlpacaEval specific fields: for final output format
                if args.dataset_type == 'alpaca_eval':
                    metadata["alpaca_dataset"] = sample.get('alpaca_dataset', '')
                    metadata["alpaca_instruction"] = sample.get('alpaca_instruction', last_user_message)
                    metadata["model_name"] = args.model_name if args.model_name else "GAM"
                    metadata["alpaca_preference_source"] = args.alpaca_preference_source

                if args.dataset_type == 'arena_hard':
                    metadata["arena_uid"] = sample.get('arena_uid', '')
                    metadata["arena_category"] = sample.get('arena_category', '')
                    metadata["arena_cluster"] = sample.get('arena_cluster', '')
                    metadata["arena_prompt"] = sample.get('arena_prompt', last_user_message)
                    metadata["reference_model"] = sample.get('reference_model', '')
                    metadata["model_name"] = args.model_name if args.model_name else "GAM"
                    metadata["arena_preference_source"] = args.alpaca_preference_source
                
                formatted_result = {
                    "system": "",
                    "conversation": conversation,
                    "_metadata": metadata
                }
                
                formatted_results.append(formatted_result)
    
    # Merge all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.array(all_labels)
    
    return formatted_results


def main():
    args = parse_args()
    
    print("="*80)
    print("Stage-1: Preference Score Prediction")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Output Dir: {args.output_dir}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, preference_names = load_model(args)
    
    print("\n" + "="*80)
    print("Batch Inference Mode")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"AMP: {'Enabled' if args.use_amp else 'Disabled'}")
    
    # ========== Step 1: Load or create normalizer (only for HH-RLHF) ==========
    print("\n" + "="*80)
    print("Step 1: Load Normalizer")
    print("="*80)
    
    normalizer = None
    
    # HelpSteer2, UltraFeedback and AlpacaEval/Arena-Hard benchmark datasets are already normalized during preprocessing or have no ground truth labels, no renorm needed
    if args.dataset_type in ['helpsteer2', 'ultrafeedback', 'alpaca_eval', 'arena_hard']:
        print(f"{args.dataset_type.upper()} dataset: Pre-normalized or benchmark, no normalizer needed.")
        print("Skipping normalizer loading...")
    else:
        # HH-RLHF needs normalizer
        # Try to load normalizer saved during training
        import os
        normalizer_paths_to_try = []
        
        print("Searching for normalizer saved during training...")
        
        # Priority 1: User-specified path
        if args.normalizer_load_path:
            normalizer_paths_to_try.append(('User-specified', args.normalizer_load_path))
            print(f"  [1] User-specified path: {args.normalizer_load_path}")
        
        # Priority 2: Auto-detect model directory
        # If model_path is a file, use its directory; if it's a directory, use it directly
        if os.path.isfile(args.model_path):
            model_dir = os.path.dirname(args.model_path)
        else:
            model_dir = args.model_path
        
        # Checkpoint directory itself
        checkpoint_normalizer = os.path.join(model_dir, 'normalizer.json')
        normalizer_paths_to_try.append(('Checkpoint dir', checkpoint_normalizer))
        print(f"  [2] Checkpoint directory: {checkpoint_normalizer}")
        
        # Parent directory (where training saves normalizer)
        parent_dir = os.path.dirname(model_dir)
        parent_normalizer = os.path.join(parent_dir, 'normalizer.json')
        normalizer_paths_to_try.append(('Parent dir', parent_normalizer))
        print(f"  [3] Parent directory: {parent_normalizer}")
        
        # Grandparent directory (in case model path structure differs)
        grandparent_dir = os.path.dirname(parent_dir)
        grandparent_normalizer = os.path.join(grandparent_dir, 'normalizer.json')
        normalizer_paths_to_try.append(('Grandparent dir', grandparent_normalizer))
        print(f"  [4] Grandparent directory: {grandparent_normalizer}")
        
        print()
        
        # Try loading in order
        for idx, (location, path) in enumerate(normalizer_paths_to_try, 1):
            print(f"Trying [{idx}] {location}: ", end='')
            if os.path.exists(path):
                print("File exists, loading...", end='')
                try:
                    normalizer = PreferenceNormalizer.load(path)
                    print(" Success!")
                    print(f"\n{'='*80}")
                    print(f" Successfully loaded normalizer from training")
                    print(f"{'='*80}")
                    print(f"Location: {path}")
                    print(f"Method: {normalizer.method}")
                    if normalizer.method == 'robust':
                        print(f"Parameters: percentile_range={normalizer.percentile_range}")
                    elif normalizer.method == 'sigmoid':
                        print(f"Parameters: sigmoid_scale={getattr(normalizer, 'sigmoid_scale', 'N/A')}")
                    
                    print(f"\nNormalizer Statistics:")
                    stats = normalizer.get_stats_summary()
                    for pref_name, pref_stats in stats['stats'].items():
                        print(f"  {pref_name}:")
                        print(f"    Mean: {pref_stats['mean']:.4f}, Std: {pref_stats['std']:.4f}")
                        if 'percentile_lower' in pref_stats:
                            print(f"    Normalization Range: [{pref_stats['percentile_lower']:.4f}, {pref_stats['percentile_upper']:.4f}] (percentile)")
                        else:
                            print(f"    Normalization Range: [{pref_stats['min']:.4f}, {pref_stats['max']:.4f}]")
                    print(f"{'='*80}\n")
                    break
                except Exception as e:
                    print(f" Failed: {e}")
            else:
                print("File not found")
        
        # If not found, refit from training data (warn user)
        if normalizer is None:
            print(f"\n Warning: Normalizer saved during training not found!")
            print(f" Will refit normalizer using training dataset...")
            print(f" Please ensure the following configs match training exactly:")
            print(f"    - Normalization method: {args.normalization_method}")
            print(f"    - Dataset path: {args.normalization_dataset_path}")
            print()
            
            if args.normalization_dataset_path is None:
                if args.dataset_type == 'hh_rlhf':
                    raise ValueError("Normalizer not found and --normalization_dataset_path not specified")
                print("Skipping normalizer refit (not required for this dataset_type); proceeding without normalizer...\n")
                normalizer = None
            else:
                # Fit from training data
                ds_temp = load_from_disk(args.normalization_dataset_path)
            
                # Use training split
                if hasattr(ds_temp, 'keys'):
                    if 'train' in ds_temp:
                        ds_temp = ds_temp['train']
                        print(f"Fitting normalizer using training split")
                    else:
                        available_splits = list(ds_temp.keys())
                        ds_temp = ds_temp[available_splits[0]]
                        print(f"Fitting normalizer using split '{available_splits[0]}'")
                
                # Extract scores
                print(f"Extracting scores... (num_samples: {len(ds_temp)})")
                harmless_scores = [float(item.get('harmless', 0.0)) for item in ds_temp]
                helpful_scores = [float(item.get('helpful', 0.0)) for item in ds_temp]
                
                # Create normalizer
                normalizer = create_normalizer(
                    method=args.normalization_method,
                    scores_dict={
                        'harmless': harmless_scores,
                        'helpful': helpful_scores
                    }
                )
                
                print(f" Normalizer created successfully")
    
    # ========== Step 2: Determine preference text format ==========
    print("\n" + "="*80)
    print("Step 2: Determine Preference Text Format")
    print("="*80)
    
    # Auto-select format based on model path
    if 'outer' in args.model_path:
        format_style = 'angle_brackets_outer'
    elif 'plain' in args.model_path:
        format_style = 'plain'
    else:
        format_style = 'angle_brackets'
    
    print(f"Detected format style: {format_style}")
    
    # Create preference injector
    injector = create_injector(
        injection_type='text',
        tokenizer=tokenizer,
        preference_names=preference_names,
        format_style=format_style
    )
    
    # ========== Step 3: Load and process dataset ==========
    print("\n" + "="*80)
    print(f"Step 3: Load and Process Dataset ({args.dataset_type.upper()})")
    print("="*80)
    
    if args.dataset_type == 'helpsteer2':
        # HelpSteer2 dataset loading
        # Note: HelpSteer2's dataset_path should be the parent directory containing train/validation subdirectories
        # Example: path/to/HelpSteer2_processed
        # dataset_split should be 'train' or 'validation'
        dataset = build_helpsteer2_dataset(
            data_path=args.dataset_path,
            tokenizer=tokenizer,
            split=args.dataset_split,
            size=args.max_samples,
            preference_names=preference_names,
            format_style=format_style,
            for_inference=True  # Inference mode: preserve original text fields
        )
    elif args.dataset_type == 'ultrafeedback':
        # UltraFeedback dataset loading
        # Note: UltraFeedback's dataset_path should be the parent directory containing train/test subdirectories
        # Example: path/to/ultrafeedback-binarized-preferences-cleaned-kto/processed
        # dataset_split should be 'train' or 'test'
        # UltraFeedback is single-turn QA data with 4 preference dimensions
        dataset = build_ultrafeedback_dataset(
            data_path=args.dataset_path,
            tokenizer=tokenizer,
            split=args.dataset_split,
            size=args.max_samples,
            preference_names=preference_names,
            format_style=format_style,
            for_inference=True  # Inference mode: preserve original text fields
        )
    elif args.dataset_type == 'alpaca_eval':
        # AlpacaEval 2.0 benchmark dataset loading
        # AlpacaEval is a JSON file, not a HuggingFace Dataset format
        dataset = build_alpaca_eval_dataset(
            data_path=args.dataset_path,
            tokenizer=tokenizer,
            size=args.max_samples,
            preference_names=preference_names,
            format_style=format_style,
            for_inference=True  # Inference mode: preserve original text fields
        )
    elif args.dataset_type == 'arena_hard':
        dataset = build_arena_hard_dataset(
            question_path=args.dataset_path,
            reference_answer_path=args.reference_answer_path,
            tokenizer=tokenizer,
            size=args.max_samples,
            preference_names=preference_names,
            format_style=format_style,
            for_inference=True,
        )
    else:
        # HH-RLHF dataset loading
        dataset = build_hh_rlhf_dataset(
            data_path=args.dataset_path,
            tokenizer=tokenizer,
            split=args.dataset_split,
            size=args.max_samples,
            normalizer=normalizer,  # Use normalizer
            preference_names=preference_names,
            format_style=format_style,
            for_inference=True  # Inference mode: preserve original text fields
        )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {list(dataset.features.keys())}")
    print("Sample data:")
    if len(dataset) > 0:
        sample = dataset[0]
        print(sample)

    
    # ========== Step 4: Batch predict preference scores ==========
    print("\n" + "="*80)
    print("Step 4: Batch Predict Preference Scores")
    print("="*80)
    
    results = batch_predict_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        preference_names=preference_names,
        injector=injector,
        normalizer=normalizer,  # Pass normalizer for denormalization
        args=args
    )
    
    # Build output directory following batch_inference_preference_guided.py naming
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    output_json_path = output_dir / "stage1_preference_scores.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"  - Prediction results (JSON format, for Stage-2 use): {output_json_path}")
    
    print("Stage-1 Inference Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
