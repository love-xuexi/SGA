import os
import argparse
import json
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.lora.utils import parse_fine_tuned_lora_name


# CUDA_VISIBLE_DEVICES=0 python infer/vllm_gam.py \
#     --dataset_path "path/to/stage1_preference_scores.json" \
#     --base_model "path/to/Llama-3.2-1B-Instruct" \
#     --lora_path "path/to/checkpoint-1501" \
#     --output_dir "./inference_results/hh_rlhf_run" \
#     --dataset_type "hh_rlhf" \
#     --max_samples 10 \
#     --tensor_parallel_size 1 \
#     --gpu_memory_utilization 0.5 \
#     --temperature 0.6 \
#     --top_p 0.9 \
#     --max_tokens 1024 \
#     --seed 3407 \
#     --max_lora_rank 128


def _append_suppress_score_instruction(dataset):
    instruction = "Do not mention or repeat any preference scores(e.g. harmless_score, helpful_score) in your response."
    for entry in dataset:
        conv = entry.get("conversation", []) if isinstance(entry, dict) else []
        if not conv:
            continue
        last_turn = conv[-1]
        if not isinstance(last_turn, dict):
            continue
        human = str(last_turn.get("human", "") or "")
        last_turn["human"] = f"{human}\n{instruction}"
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Stage-2 GAM Inference")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to stage-1 JSON dataset")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for debugging")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_weight", type=int, default=1)
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--dataset_type", type=str, default="hh_rlhf",
                       choices=["hh_rlhf", "helpsteer2", "ultrafeedback", "alpaca_eval", "arena_hard"],
                       help="Dataset type: hh_rlhf, helpsteer2, ultrafeedback, alpaca_eval (benchmark) or arena_hard (benchmark)")
    parser.add_argument("--alpaca_preference_source", type=str, default="hh_rlhf",
                       choices=["hh_rlhf", "helpsteer2", "ultrafeedback"],
                       help="For alpaca_eval only: which preference set was used in stage-1 (hh_rlhf/helpsteer2/ultrafeedback)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for AlpacaEval output (generator field)")
    parser.add_argument("--suppress_score_echo", action="store_true",
                        help="Append an instruction to the last user message to avoid echoing preference scores.")
    return parser.parse_args()


def build_output_dir(output_root: str, lora_path: Optional[str], base_model: str) -> Path:
    """Build output directory following batch_inference_preference_guided.py naming convention"""
    if lora_path:
        parts = Path(lora_path).parts
        if parts[-1].startswith("checkpoint-"):
            ckpt = parts[-1]
            model_name = parts[-2]
        else:
            model_name = parts[-1]
            ckpt = "xx"
    else:
        model_name = Path(base_model).name
        ckpt = "base"
    
    dir_name = f"{model_name}_{ckpt}"
    final_dir = Path(output_root).expanduser().resolve() / dir_name
    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir



Message = Dict[str, str]
ConversationTurn = Dict[str, str]
DatasetEntry = Dict[str, object]


def load_dataset(dataset_path: Path) -> List[DatasetEntry]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        with dataset_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of conversation objects.")
    return dataset


def discretize_score(score: float) -> float:
    """
    Discretize continuous scores to the nearest HelpSteer2 discrete values (0, 0.25, 0.5, 0.75, 1).
    
    Args:
        score: Continuous score value
    
    Returns:
        The nearest discrete value
    """
    discrete_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Find the nearest discrete value
    closest = min(discrete_values, key=lambda x: abs(x - score))
    return closest


def discretize_helpsteer2_scores(dataset: List[DatasetEntry]) -> List[DatasetEntry]:
    """
    Discretize preference scores for the HelpSteer2 dataset.
    
    Example of HelpSteer2 score format (at the end of the human field):
    "Your response must satisfy the following scores: helpfulness_score: 0.696 and correctness_score: 0.693 ..."
    
    Round continuous scores to the nearest discrete values (0, 0.25, 0.5, 0.75, 1).
    
    Args:
        dataset: Original dataset
    
    Returns:
        Discretized dataset
    """
    # Define regex patterns for preference scores.
    # Support two formats:
    # 1) helpfulness_score: 0.696
    # 2) <helpfulness_score> 0.696  (angle_brackets_outer)
    score_pattern = re.compile(r'(\w+_score):\s*([\d.]+)')
    score_pattern_outer = re.compile(r'<(\w+_score)>\s*([\d.]+)')
    
    processed_dataset = []
    discretization_stats = {"total_scores": 0, "modified_scores": 0}
    
    for entry in dataset:
        new_entry = entry.copy()
        conversation = entry.get("conversation", [])
        new_conversation = []
        
        for turn in conversation:
            new_turn = turn.copy()
            human_text = turn.get("human", "")
            
            # Find and replace all scores
            def replace_score(match):
                score_name = match.group(1)
                original_score = float(match.group(2))
                discretized_score = discretize_score(original_score)
                
                discretization_stats["total_scores"] += 1
                if abs(original_score - discretized_score) > 0.001:
                    discretization_stats["modified_scores"] += 1
                
                return f"{score_name}: {discretized_score:.2f}"

            def replace_score_outer(match):
                score_name = match.group(1)
                original_score = float(match.group(2))
                discretized_score = discretize_score(original_score)

                discretization_stats["total_scores"] += 1
                if abs(original_score - discretized_score) > 0.001:
                    discretization_stats["modified_scores"] += 1

                return f"<{score_name}> {discretized_score:.2f}"
            
            new_human_text = score_pattern.sub(replace_score, human_text)
            new_human_text = score_pattern_outer.sub(replace_score_outer, new_human_text)
            new_turn["human"] = new_human_text
            new_conversation.append(new_turn)
        
        new_entry["conversation"] = new_conversation
        processed_dataset.append(new_entry)
    
    print(f"✓ Discretization complete: {discretization_stats['modified_scores']}/{discretization_stats['total_scores']} scores modified")
    
    return processed_dataset


def build_entry_components(entry: DatasetEntry) -> Tuple[List[Message], List[Message], str, str, str]:
    system_prompt = str(entry.get("system", "") or "")
    conversation: List[ConversationTurn] = entry.get("conversation", [])
    if not conversation:
        raise ValueError("Each dataset entry must contain a 'conversation' list.")

    messages: List[Message] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    history_messages: List[Message] = []
    prompt_segments: List[str] = []

    for idx, turn in enumerate(conversation):
        human_text = str(turn.get("human", "") or "")
        assistant_text = str(turn.get("assistant", "") or "")

        messages.append({"role": "user", "content": human_text})

        if idx < len(conversation) - 1:
            messages.append({"role": "assistant", "content": assistant_text})
            history_messages.append({"role": "user", "content": human_text})
            history_messages.append({"role": "assistant", "content": assistant_text})
            prompt_segments.append(f"Human: {human_text}\n\nAssistant: {assistant_text}")
        else:
            prompt_segments.append(f"Human: {human_text}\n\nAssistant:")

    prompt_text = "\n" + "\n\n".join(prompt_segments)
    last_user = str(conversation[-1].get("human", "") or "")
    target_response = str(conversation[-1].get("assistant", "") or "")
    return messages, history_messages, prompt_text, last_user, target_response


def prepare_messages_and_metadata(dataset: List[DatasetEntry]) -> Tuple[List[List[Message]], List[Dict[str, object]]]:
    message_batches: List[List[Message]] = []
    metadata: List[Dict[str, object]] = []
    for idx, entry in enumerate(dataset):
        messages, history_messages, prompt_text, last_user, target = build_entry_components(entry)
        message_batches.append(messages)
        
        # Extract stage-1 metadata (contains alpaca_eval specific fields)
        stage1_metadata = entry.get("_metadata", {})
        
        metadata.append(
            {
                "index": idx,
                "system": entry.get("system", ""),
                "history_messages": history_messages,
                "prompt": prompt_text,
                "last_user": last_user,
                "reference": target,
                # AlpacaEval specific fields (from Stage-1)
                "alpaca_dataset": stage1_metadata.get("alpaca_dataset", ""),
                "alpaca_instruction": stage1_metadata.get("alpaca_instruction", ""),
                "model_name": stage1_metadata.get("model_name", ""),
                # Arena-Hard specific fields (from Stage-1)
                "arena_uid": stage1_metadata.get("arena_uid", ""),
                "arena_category": stage1_metadata.get("arena_category", ""),
                "arena_cluster": stage1_metadata.get("arena_cluster", ""),
                "arena_prompt": stage1_metadata.get("arena_prompt", ""),
                "reference_model": stage1_metadata.get("reference_model", ""),
            }
        )
    return message_batches, metadata


# Llama 2 chat template (for base models without built-in template)
LLAMA2_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""


def ensure_chat_template(tokenizer, base_model: str):
    """
    Ensure tokenizer has a chat template. For Llama-2 base models,
    set the standard Llama 2 chat template if not present.
    """
    if tokenizer.chat_template is not None:
        return tokenizer
    
    # Check if it's a Llama-2 base model
    model_name_lower = base_model.lower()
    if "llama-2" in model_name_lower or "llama2" in model_name_lower:
        print(f"⚠ Model {base_model} has no chat_template, setting Llama 2 chat template...")
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
        print(f"✓ Llama 2 chat template set successfully")
    else:
        raise ValueError(
            f"Tokenizer for {base_model} has no chat_template. "
            "Please set a chat template manually or use a model with built-in chat template."
        )
    
    return tokenizer


def apply_chat_template_batch(
    message_batches: List[List[Message]],
    tokenizer
) -> List[str]:
    prompts: List[str] = []
    for messages in message_batches:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception as e:
                # Fallback: try without enable_thinking (some models don't support it)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            raise ValueError("Tokenizer does not support apply_chat_template")
        prompts.append(prompt)
    return prompts


def run_generation(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest] = None,
) -> List[str]:
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )

    generations: List[str] = []
    for output in outputs:
        if not output.outputs:
            generations.append("")
            continue
        generations.append(output.outputs[0].text)
    return generations


def ensure_parent_directory(output_path: Path) -> None:
    output_dir = output_path.expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_vllm_compatible_lora(source_dir: Path) -> Tuple[Path, List[str]]:
    source_dir = source_dir.expanduser().resolve()
    
    # 1. Determine source weight file path.
    # Prioritize adapter_model.safetensors, then model.safetensors.
    if (source_dir / "adapter_model.safetensors").exists():
        adapter_path = source_dir / "adapter_model.safetensors"
    elif (source_dir / "model.safetensors").exists():
        adapter_path = source_dir / "model.safetensors"
    else:
        if (source_dir / "adapter_model.bin").exists() or (source_dir / "pytorch_model.bin").exists():
             raise ValueError("Found .bin weights but vLLM prefers .safetensors. Please convert format first.")
        raise FileNotFoundError(f"No suitable LoRA weights (adapter_model.safetensors or model.safetensors) found in {source_dir}")

    # 2. Prepare target directory.
    target_dir = source_dir.parent / f"{source_dir.name}_vllm_fixed"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # vLLM strictly requires the filename to be adapter_model.safetensors
    target_adapter_path = target_dir / "adapter_model.safetensors"

    # 3. Check if regeneration is needed (based on modification time).
    regenerate = True
    if target_adapter_path.exists():
        if target_adapter_path.stat().st_mtime > adapter_path.stat().st_mtime:
            regenerate = False

    skipped_keys = []
    
    if regenerate:
        print(f"Processing LoRA weights from: {adapter_path.name}")
        print(f"Target: {target_adapter_path}")
        
        weights = load_file(str(adapter_path))
        new_weights = {}
        
        for key, tensor in weights.items():
            # Filter logic
            if "v_head" in key:
                skipped_keys.append(key)
                continue
            
            # Key correction logic
            new_key = key.replace(".default", "")
            
            # Target keys must map "model.layers..." to "base_model.model.model.layers..."
            # Explanation:
            #   The first base_model.model is the PEFT wrapper.
            #   The second model is the Llama model's attribute.
            
            if new_key.startswith("model."):
                new_key = "base_model.model." + new_key
            elif new_key.startswith("lm_head."):
                new_key = "base_model.model." + new_key
            else:
                # Fallback: if the key does not start with model. or lm_head.,
                # add the base_model prefix if it's missing.
                if not new_key.startswith("base_model.model."):
                    new_key = "base_model.model." + new_key
            
            new_weights[new_key] = tensor

        if not new_weights:
            raise ValueError("No compatible LoRA weights found after filtering!")

        save_file(new_weights, str(target_adapter_path))
        print(f"✓ Successfully converted {len(new_weights)} weights.")
        print(f"  (Ignored {len(skipped_keys)} keys related to v_head)")

    # 4. Copy configuration files.
    # Only copy adapter and Tokenizer configs needed by vLLM.
    # Do not copy config.json or generation_config.json, which belong to the Base Model.
    files_to_copy = [
        "adapter_config.json", 
        "tokenizer.json", 
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    for filename in files_to_copy:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)
            
    # 5. Additional check for adapter_config.json
    # Ensure modules_to_save does not contain v_head, otherwise vLLM fails to find weights.
    config_path = target_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        changed = False
        if "modules_to_save" in config and config["modules_to_save"]:
            original_len = len(config["modules_to_save"])
            # Filter out v_head
            config["modules_to_save"] = [m for m in config["modules_to_save"] if "v_head" not in m]
            if len(config["modules_to_save"]) != original_len:
                changed = True
                print("✓ Cleaned 'modules_to_save' in adapter_config.json")
        
        if changed:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    return target_dir, skipped_keys


def main():
    args = parse_args()
    
    # Set GPU
    
    print("="*80)
    print("Stage-2: vLLM GAM Inference")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Output Dir: {args.output_dir}")
    print("="*80)
    
    # Build output directory
    final_output_dir = build_output_dir(args.output_dir, args.lora_path, args.base_model)
    print(f"Final output directory: {final_output_dir}")
    
    # Initialize LLM
    enable_lora = args.lora_path is not None
    start_time = time.time()
    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank,
        max_model_len=4096,
    )
    print(f"✓ Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Load dataset
    dataset = load_dataset(Path(args.dataset_path))
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]
        print(f"✓ Loaded {len(dataset)} samples (debug mode)")
    else:
        print(f"✓ Loaded {len(dataset)} samples")
    
    inferred_num_preferences = None
    if dataset:
        stage1_meta = dataset[0].get("_metadata", {}) if isinstance(dataset[0], dict) else {}
        if isinstance(stage1_meta, dict):
            inferred_num_preferences = stage1_meta.get("num_preferences", None)
            if inferred_num_preferences is None:
                predicted_keys = [k for k in stage1_meta.keys() if isinstance(k, str) and k.startswith("predicted_")]
                if predicted_keys:
                    inferred_num_preferences = len(predicted_keys)

    if inferred_num_preferences is None:
        default_num_preferences = {
            "hh_rlhf": 2,
            "helpsteer2": 5,
            "ultrafeedback": 4,
        }
        if args.dataset_type == "alpaca_eval":
            inferred_num_preferences = default_num_preferences.get(args.alpaca_preference_source, 2)
        else:
            inferred_num_preferences = default_num_preferences.get(args.dataset_type, None)

    should_discretize = inferred_num_preferences is not None and int(inferred_num_preferences) != 2

    if should_discretize:
        print("\n" + "="*80)
        print("Score Discretization")
        print("="*80)
        print("Discretizing predicted scores to nearest values in {0, 0.25, 0.5, 0.75, 1}...")
        dataset = discretize_helpsteer2_scores(dataset)
    
    if args.suppress_score_echo:
        dataset = _append_suppress_score_instruction(dataset)
    
    # Prepare messages
    message_batches, metadata = prepare_messages_and_metadata(dataset)
    print(f"✓ Prepared {len(message_batches)} message sequences")
    
    # Apply chat template
    tokenizer = llm.get_tokenizer()
    tokenizer = ensure_chat_template(tokenizer, args.base_model)
    prompts = apply_chat_template_batch(message_batches, tokenizer)
    print(f"✓ Applied chat template to {len(prompts)} prompts")
    
    # Preview first prompt
    print("\n" + "="*80)
    print("Sample Prompt:")
    print("="*80)
    for prompt in prompts[0:1]:
        print(prompt)
        print("-"*100)
    
    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    
    # Prepare history strings
    history_strings = [json.dumps(meta["history_messages"], ensure_ascii=False) for meta in metadata]
    
    # Run inference
    print("\n" + "="*80)
    print("Running Inference")
    print("="*80)
    
    if enable_lora:
        # Only run LoRA inference
        lora_path = Path(args.lora_path).expanduser().resolve()
        sanitized_lora_path, skipped_keys = ensure_vllm_compatible_lora(lora_path)
        
        if skipped_keys:
            preview = ", ".join(skipped_keys[:3])
            if len(skipped_keys) > 3:
                preview += ", ..."
            print(f"⚠ Detected {len(skipped_keys)} non-LoRA weights (ignored): {preview}")
        
        lora_name = lora_path.name
        print(f"Loading LoRA adapter '{lora_name}'...")
        lora_request = LoRARequest(
            lora_name,
            1,
            str(sanitized_lora_path),
        )
        
        start_lora = time.time()
        lora_generations = run_generation(llm, prompts, sampling_params, lora_request=lora_request)
        print(f"✓ LoRA inference completed in {time.time() - start_lora:.2f} seconds")
        
        # Build results dataframe
        lora_rows = []
        for meta, lora_text, history_json in zip(metadata, lora_generations, history_strings):
            lora_rows.append({
                "index": meta["index"],
                "prompt": meta["prompt"],
                "history_messages": history_json,
                "generated_response": lora_text,
                "ground_truth_response": meta["reference"],
                "model_variant": f"lora:{lora_name}",
            })
        results_df = pd.DataFrame(lora_rows)
    else:
        # Run base model inference
        print("Running base model inference...")
        start_infer = time.time()
        base_generations = run_generation(llm, prompts, sampling_params)
        print(f"✓ Base model inference completed in {time.time() - start_infer:.2f} seconds")
        
        # Build results dataframe
        base_rows = []
        for meta, base_text, history_json in zip(metadata, base_generations, history_strings):
            base_rows.append({
                "index": meta["index"],
                "prompt": meta["prompt"],
                "history_messages": history_json,
                "generated_response": base_text,
                "ground_truth_response": meta["reference"],
                "model_variant": "base",
            })
        results_df = pd.DataFrame(base_rows)
    
    print(f"\n✓ Prepared {len(results_df)} results")
    
    # Preview results
    print("\n" + "="*80)
    print("Sample Results:")
    print("="*80)
    print(json.dumps(results_df.head(1).to_dict(orient="records"), ensure_ascii=False, indent=2))
    
    # Save results
    if args.dataset_type == "alpaca_eval":
        # AlpacaEval: Output JSON format compatible with AlpacaEval 2.0
        # Format: [{"dataset": "...", "instruction": "...", "output": "...", "generator": "..."}]
        
        # Determine model name for generator field
        if args.model_name:
            generator_name = args.model_name
        elif args.lora_path:
            # Use LoRA path as model name
            generator_name = Path(args.lora_path).name
        else:
            generator_name = Path(args.base_model).name
        
        # Build AlpacaEval format output
        alpaca_results = []
        generations = lora_generations if enable_lora else base_generations
        for meta, gen_text in zip(metadata, generations):
            alpaca_results.append({
                "dataset": meta.get("alpaca_dataset", ""),
                "instruction": meta.get("alpaca_instruction", meta.get("last_user", "")),
                "output": gen_text,
                "generator": generator_name
            })
        
        # Save as JSON
        output_json_path = final_output_dir / "alpaca_eval_results.json"
        ensure_parent_directory(output_json_path)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ AlpacaEval results saved to: {output_json_path}")
        
        # Preview AlpacaEval output format
        print("\n" + "="*80)
        print("AlpacaEval Output Sample:")
        print("="*80)
        print(json.dumps(alpaca_results[:1], ensure_ascii=False, indent=2))
    elif args.dataset_type == "arena_hard":
        if args.model_name:
            generator_name = args.model_name
        elif args.lora_path:
            generator_name = Path(args.lora_path).name
        else:
            generator_name = Path(args.base_model).name

        generations = lora_generations if enable_lora else base_generations
        now_ts = time.time()
        output_jsonl_path = final_output_dir / "model_answer.jsonl"
        ensure_parent_directory(output_jsonl_path)
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for meta, gen_text in zip(metadata, generations):
                uid = meta.get("arena_uid", "") or meta.get("index")
                record = {
                    "uid": uid,
                    "ans_id": f"{generator_name}:{uid}",
                    "model": generator_name,
                    "messages": [
                        {"role": "user", "content": meta.get("arena_prompt", meta.get("last_user", ""))},
                        {"role": "assistant", "content": {"answer": gen_text}},
                    ],
                    "tstamp": now_ts,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n✓ Arena-Hard model answers saved to: {output_jsonl_path}")

        print("\n" + "="*80)
        print("Arena-Hard Output Sample:")
        print("="*80)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        # Other datasets: Output CSV format
        output_csv_path = final_output_dir / "vllm_gam_results.csv"
        ensure_parent_directory(output_csv_path)
        results_df.to_csv(output_csv_path, index=False)
        print(f"\n✓ Results saved to: {output_csv_path}")
    
    print("\n" + "="*80)
    print("Stage-2 Inference Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

