import torch
import pandas as pd
import re
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import argparse

# ================= Configuration =================
# Default configurations (can be overridden by command line arguments)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MODEL_PATH = "path/to/ArmoRM-Llama3-8B-v0_1"
DEFAULT_INPUT_FILES = [
    "path/to/vllm_gam_results.csv"
]
DEFAULT_OUTPUT_DIR = "path/to/scored_results" 
# ===========================================

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        device_map=device, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer

def clean_noise(text: str) -> str:
    if not isinstance(text, str) or not text: 
        return ""
    
    gam_pattern = r"Your response must satisfy the following scores:\s*(?:\w+_score:\s*[\d.]+(?:\s+and\s+)?)+\s*"
    ric_pattern = r"(?:<\w+_score>\s*[\d.]+\s*)+"
    cpo_pattern = r"(?:<\s*\w+:\s*[\d.]+\s*>\s*)+"
    
    text = re.sub(gam_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(ric_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(cpo_pattern, "", text, flags=re.IGNORECASE)
    
    return text.strip()

def parse_multi_turn_prompt(raw_prompt: str) -> list:
    # global l 
    if not isinstance(raw_prompt, str) or not raw_prompt:
        return []

    clean_text = clean_noise(raw_prompt)

        # === Modification starts ===
    # Check if dialogue markers exist (ignore case)
    has_markers = re.search(r"(?i)(Human:|Assistant:)", clean_text)

    if not has_markers:
        # l += 1 
        # print(l)
        # If no markers, return directly as a single User message
        return [{"role": "user", "content": clean_text}]

    parts = re.split(r"(?i)\n?(Human:|Assistant:)", clean_text)
    
    messages = []
    current_role = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if re.match(r"(?i)^Human:$", part):
            current_role = "user"
        elif re.match(r"(?i)^Assistant:$", part):
            current_role = "assistant"
        else:
            if current_role:
                messages.append({"role": current_role, "content": part})
            else:
                messages.append({"role": "user", "content": part})
                current_role = "user"
                
    return messages

def get_output_path(input_path, output_root):
    """
    Calculate the output path based on the input path.
    Logic: find 'inference_results', keep all parts after it, and append to output_root.
    """
    # Normalize path separators
    norm_path = os.path.normpath(input_path)
    parts = norm_path.split(os.sep)
    
    keyword = "inference_results"
    
    if keyword in parts:
        # Find the index of the keyword
        idx = parts.index(keyword)
        # Extract all parts after the keyword (e.g., GAM/llama3.../file.csv)
        relative_path = os.path.join(*parts[idx+1:])
        final_path = os.path.join(output_root, relative_path)
    else:
        # If keyword is not found, save directly under root with original filename
        print(f"[Warning] '{keyword}' not found in path, will save with filename directly.")
        final_path = os.path.join(output_root, os.path.basename(norm_path))
        
    return final_path

def process_single_file(model=None, tokenizer=None, input_path=None, output_root=None, device="cuda:0"):
    # 1. Calculate output path
    output_path = get_output_path(input_path, output_root)
    # print(output_path)
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nProcessing file: {input_path}")
    print(f"Results will be saved to: {output_path}")

    # 2. Read CSV
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully read CSV, {len(df)} rows of data in total.")
    except FileNotFoundError:
        print(f"Error: File {input_path} not found, skipping this file.")
        return
    except Exception as e:
        print(f"Error reading file: {e}, skipping.")
        return

    results = []
    obj_transform = model.reward_transform_matrix.data.cpu().float()

    # 3. Iterate over each row
    # Use filename as progress bar description
    desc_name = os.path.basename(input_path)
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {desc_name}"):
        result_row = row.to_dict()
        
        raw_prompt = row.get('prompt', "")
        generated_response = row.get('generated_response', "")
        
        messages = parse_multi_turn_prompt(raw_prompt)
        messages.append({"role": "assistant", "content": generated_response})
        
        if index == 455 or index == 0:
            print(f"[DEBUG First Message]: {messages}")

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            ).to(device)
        except Exception as e:
            print(f"Skipping index {index} due to template error: {e}")
            results.append(result_row)
            continue

        with torch.no_grad():
            output = model(input_ids)
            
            multi_obj_rewards = output.rewards.cpu().float()
            gating_output = output.gating_output.cpu().float()
            preference_score = output.score.cpu().float().item()

            hs_scores = multi_obj_rewards[0, :5] * 5 - 0.5
            uf_scores = multi_obj_rewards[0, 5:10] * 5 + 0.5

        # Write scores
        result_row['armo_preference_score'] = preference_score

        result_row['hs_helpfulness'] = hs_scores[0].item()
        result_row['hs_correctness'] = hs_scores[1].item()
        result_row['hs_coherence'] = hs_scores[2].item()
        result_row['hs_complexity'] = hs_scores[3].item()
        result_row['hs_verbosity'] = hs_scores[4].item()
        
        # result_row['uf_overall'] = uf_scores[0].item()
        result_row['uf_instruction_following'] = uf_scores[1].item()
        result_row['uf_truthfulness'] = uf_scores[2].item()
        result_row['uf_honesty'] = uf_scores[3].item()
        result_row['uf_helpfulness'] = uf_scores[4].item()
        
        multi_obj_coeffs = gating_output @ obj_transform.T
        top_obj_dim = torch.argmax(torch.abs(multi_obj_coeffs), dim=1).item()
        attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
           'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
           'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
           'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
           'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
           'code-style','code-explanation','code-instruction-following','code-readability']
        # result_row['primary_attribute'] = attributes[top_obj_dim]

        results.append(result_row)

    # 4. Save
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"File processing complete: {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Score responses using ArmoRM")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the ArmoRM model")
    parser.add_argument("--input_files", nargs='+', default=DEFAULT_INPUT_FILES, help="List of CSV files to process")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root directory for outputs")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use (e.g., cuda:0)")
    
    args = parser.parse_args()

    # # 1. Load model
    model, tokenizer = load_model(args.model_path, args.device)
    
    # 2. Loop processing
    for input_file in args.input_files:
        process_single_file(model, tokenizer, input_file, args.output_dir, args.device)
        # process_single_file(None, None, input_file, args.output_dir, args.device)
    
    print("\nAll tasks completed!")