"""
Brief description: This script is used to read CSV files containing prompts and responses, load specified reward models
(harmless, helpful, etc.) to batch score the responses, optionally clean the scoring prompts in the prompt, and
append the scores of each reward model back to the CSV before outputting to the specified directory. hhrlf
"""
import argparse
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from multi_reward_models import RewardModels

# python score.py \
#     --csv_path path/to/vllm_gam_results.csv \
#     --gpu_id 0 \
#     --batch_size 16 \
#     --prompt_column prompt \
#     --response_column generated_response \
#     --reward_names harmless,helpful \
#     --clean_prompt

# Score the responses for gam and ric
def parse_args():
    parser = argparse.ArgumentParser(description='Score CSV file with reward models')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--reward_names', type=str, default='harmless,helpful',
                        help='Comma-separated reward model names (harmless,helpful,deberta,summary,faithful,humor)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--prompt_column', type=str, default='prompt',
                        help='Column name for prompt')
    parser.add_argument('--response_column', type=str, default='generated_response',
                        help='Column name for response to score')
    parser.add_argument('--clean_prompt', action='store_true',
                        help='Remove score requirement text from prompts (e.g., "Your response must satisfy..." and "<rm1_score>...")')
    return parser.parse_args()



def clean_prompt_text(text):
    """
    Remove score requirement text from prompts.
    
    Removes:
    1. "Your response must satisfy the following scores: harmless_score: X.XXX and helpful_score: X.XXX"
    2. "<rm1_score> X.XXX <rm2_score> X.XXX" and similar patterns
    3. The newlines following these patterns
    """
    # Pattern 1: "Your response must satisfy the following scores: ..."
    # Matches the entire line and the following newline(s)
    pattern1 = r'Your response must satisfy the following scores:[^\n]*\n+'
    text = re.sub(pattern1, '', text)
    
    # Pattern 2: "<rm{N}_score> X.XXX" patterns
    # Matches: "<rm1_score> 0.361 <rm2_score> 0.571" and the following newline(s)
    pattern2 = r'(?:<rm\d+_score>\s+[-+]?\d+\.?\d*\s*)+\n+'
    text = re.sub(pattern2, '', text)
    
    return text


def main():
    args = parse_args()
    
    # Auto-generate output path
    output_dir = 'path/to/inference_results/modpo/hh'
    os.makedirs(output_dir, exist_ok=True)
    # Get the name of the directory containing the input file (i.e., checkpoint directory name)
    # input_dir =os.path.basename(args.csv_path)
    # input_dir =  os.path.splitext(input_dir)[0]
    input_dir = os.path.dirname(args.csv_path)
    if args.response_column == 'ground_truth_response':
        folder_name = os.path.basename(input_dir)+ '_truth' + '.csv'
    else:
        folder_name = os.path.basename(input_dir) + '.csv'

    output_path = os.path.join(output_dir, folder_name)
    print("output_path:",output_path)
    # Reward model paths mapping
        'harmless': 'path/to/gpt2-large-harmless-reward_model',
        'helpful': 'path/to/gpt2-large-helpful-reward_model',
        'deberta': 'path/to/reward-model-deberta-v3-large-v2',
        'summary': 'path/to/gpt2_reward_summarization',
        'faithful': 'path/to/bart-faithful-summary-detector',
        'humor': 'path/to/humor-no-humor',
    }
    
    # Parse reward names
    reward_names = [x.strip() for x in args.reward_names.split(',')]
    print(f"Using reward models: {reward_names}")
    
    # Validate reward names
    for name in reward_names:
        if name not in reward_path_dict:
            raise ValueError(f"Unknown reward model: {name}. Available: {list(reward_path_dict.keys())}")
    
    # Get reward model paths
    reward_model_paths = [reward_path_dict[name] for name in reward_names]
    rm_tokenizer_paths = [reward_path_dict[name] for name in reward_names]
    
    print("Loading reward models...")
    reward_models = RewardModels(reward_model_paths, rm_tokenizer_paths, args.gpu_id)
    
    # Load CSV file
    print(f"Loading CSV file: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Total rows: {len(df)}")
    
    # Check if required columns exist
    if args.prompt_column not in df.columns:
        raise ValueError(f"Column '{args.prompt_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    if args.response_column not in df.columns:
        raise ValueError(f"Column '{args.response_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Clean prompts if requested
    if args.clean_prompt:
        print("Cleaning prompts (removing score requirement text)...")
        df[args.prompt_column] = df[args.prompt_column].apply(
            lambda x: clean_prompt_text(str(x)) if pd.notna(x) else x
        )
    
    # Prepare query-response pairs
    print("Preparing query-response pairs...")
    queries_responses = []
    for idx, row in df.iterrows():
        prompt = str(row[args.prompt_column]) if pd.notna(row[args.prompt_column]) else ""
        response = str(row[args.response_column]) if pd.notna(row[args.response_column]) else ""
        queries_responses.append((prompt, response))
    
    # Get reward scores
    print("Computing reward scores...")
    rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
    # Add scores to dataframe
    for i, reward_name in enumerate(reward_names):
        column_name = f'{reward_name}_score'
        df[column_name] = rewards_list[i]
        print(f"{column_name}: mean={np.mean(rewards_list[i]):.4f}, std={np.std(rewards_list[i]):.4f}")
    
    # Save results
    print(f"Saving results to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
