import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import pandas as pd
import re


# === Utility: Mask Thinking Block ===
def mask_thinking_labels(labels: torch.Tensor, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Scan input_ids to locate <think>...</think> blocks and set their corresponding labels to -100."""
    think_start_token = "<think>"
    think_end_token = "</think>"
    
    # Retrieve token IDs (Qwen series usually uses them as special tokens or regular vocabulary).
    try:
        think_start_ids = tokenizer.encode(think_start_token, add_special_tokens=False)
        think_end_ids = tokenizer.encode(think_end_token, add_special_tokens=False)
    except:
        return labels
    
    if len(think_start_ids) == 0 or len(think_end_ids) == 0:
        return labels
    
    input_ids_list = input_ids.tolist()
    
    i = 0
    len_input = len(input_ids_list)
    len_start = len(think_start_ids)
    len_end = len(think_end_ids)

    while i < len(input_ids_list):
        if input_ids_list[i:i+len(think_start_ids)] == think_start_ids:
            start_pos = i
            j = i + len(think_start_ids)
            while j <= len(input_ids_list) - len(think_end_ids):
                if input_ids_list[j:j+len(think_end_ids)] == think_end_ids:
                    end_pos = j + len(think_end_ids)
                    labels[start_pos:end_pos] = -100
                    i = end_pos
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1
    
    return labels


# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'SGA' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'SGA' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected,
                # "margin": margin, # SGA does not need this
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin, 
            }
        

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col and 'target_scores' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'SGA' in model_name:
            # add label mask for sft and dpo
            prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")
    return ds

def build_dataset_HS2(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    """
    Process HelpSteer2 dataset (full version):
    1. Calculate 5-dimensional average scores and pair data (Chosen, Rejected).
    2. Parse the <extra_id_1> token in the prompt to restore multi-turn conversation history.
    3. Append the 5 scores of the Chosen sample to the content of the last User turn.
    4. In SGA mode: Mask the Prompt section and the <think>...</think> reasoning process.
    """

    try:
        ds = load_dataset(data_path, split=split)
    except:
        ds = load_dataset("path/to/HelpSteer2", split=split)

    df = ds.to_pandas()
    
    score_cols = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    valid_cols = [c for c in score_cols if c in df.columns]
    df['avg_score'] = df[valid_cols].mean(axis=1)

    paired_rows = []
    df = df[df.duplicated(subset=['prompt'], keep=False)]
    grouped = df.groupby('prompt')
    
    for prompt, group in grouped:
        if len(group) < 2: continue
        
        sorted_group = group.sort_values('avg_score', ascending=False)
        chosen_row = sorted_group.iloc[0]
        rejected_row = sorted_group.iloc[-1]
        
        margin = chosen_row['avg_score'] - rejected_row['avg_score']
        
        if margin == 0: continue

        chosen_scores = {col: chosen_row[col] for col in valid_cols}

        paired_rows.append({
            'prompt': prompt, 
            'chosen_content': chosen_row['response'],
            'rejected_content': rejected_row['response'],
            'margin': margin,
            'chosen_scores': chosen_scores
        })

    ds = Dataset.from_list(paired_rows)
    print(f"Number of rows after constructing paired data: {len(ds)}")

    if size is not None and size < len(ds):
        ds = ds.select(range(0, size))

    def format_preferences(scores_dict):
        texts = []
        for name in score_cols:
            if name in scores_dict:
                # Assume score range is 0-4, normalize by dividing by 4.
                val = scores_dict[name] / 4
                texts.append(f"<{name}_score:{val}>") 
        pref_scores = " and ".join(texts)
        return f"\nYour response must satisfy the following scores: {pref_scores}"

    def parse_multiturn_prompt(raw_prompt):
        messages = []
        parts = raw_prompt.split('<extra_id_1>')
        if parts[0].strip():
            messages.append({"role": "user", "content": parts[0].strip()})
        for part in parts[1:]:
            part = part.strip()
            if part.startswith('Assistant'):
                content = part[len('Assistant'):].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif part.startswith('User'):
                content = part[len('User'):].strip()
                if content:
                    messages.append({"role": "user", "content": content})
        return messages

    # === 4. Formatting Function (Tokenization & Masking) ===
    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": 4096, "return_tensors": "pt"}
        
        harmless_val = example['chosen_scores'].get('helpfulness', 0.0) / 4
        correctness_val = example['chosen_scores'].get('correctness', 0.0) / 4
        coherence_val = example['chosen_scores'].get('coherence', 0.0) / 4
        complexity_val = example['chosen_scores'].get('complexity', 0.0) / 4
        verbosity_val = example['chosen_scores'].get('verbosity', 0.0) / 4
        target_scores = torch.tensor([harmless_val, correctness_val, coherence_val, complexity_val, verbosity_val], dtype=torch.float)
        target_scores = torch.round(target_scores, decimals=2)
        # A. Construct Prompt (with scores)
        history_messages = parse_multiturn_prompt(example['prompt'])

        # Use history_messages directly to generate template for reference.
        raw_prompt_str = tokenizer.apply_chat_template(history_messages, tokenize=False, add_generation_prompt=False)
        tokens_raw_prompt = tokenizer.encode_plus(raw_prompt_str, **kwargs)

        score_suffix = format_preferences(example['chosen_scores'])
        
        modified_history = [dict(msg) for msg in history_messages]
        if modified_history and modified_history[-1]['role'] == 'user':
            modified_history[-1]['content'] = modified_history[-1]['content'] + score_suffix
        else:
            modified_history.append({"role": "user", "content": score_suffix.strip()})

        # B. Construct Chosen / Rejected
        chosen_messages = modified_history + [{"role": "assistant", "content": example['chosen_content']}]
        rejected_messages = modified_history + [{"role": "assistant", "content": example['rejected_content']}]
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        
        margin = example['margin']

        if 'SGA' in model_name:
            # Calculate Mask length based on prompt with scores.
            # Note: prompt_template_with_scores must correspond to modified_history because chosen/rejected contain scores.
            prompt_template_with_scores = tokenizer.apply_chat_template(modified_history, tokenize=False, add_generation_prompt=True)
            tokens_prompt_with_scores = tokenizer.encode_plus(prompt_template_with_scores, **kwargs)['input_ids'][0]
            len_prompt_mask = len(tokens_prompt_with_scores)
            
            # --- Chosen Processing ---
            input_ids_chosen = tokens_chosen["input_ids"][0]
            label_chosen = input_ids_chosen.clone()
            
            mask_len_c = min(len_prompt_mask, len(label_chosen))
            label_chosen[:mask_len_c] = -100
            
            label_chosen = mask_thinking_labels(label_chosen, input_ids_chosen, tokenizer)
            
            # --- Rejected Processing ---
            input_ids_rejected = tokens_rejected["input_ids"][0]
            label_rejected = input_ids_rejected.clone()
            
            mask_len_r = min(len_prompt_mask, len(label_rejected))
            label_rejected[:mask_len_r] = -100

            label_rejected = mask_thinking_labels(label_rejected, input_ids_rejected, tokenizer)
            
            return {
                "input_ids_chosen": input_ids_chosen, "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": input_ids_rejected, "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen, 'label_rejected': label_rejected,
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0],
                "target_scores": target_scores
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin,
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0]
            }
        
    print(f"Processing HelpSteer2 dataset (Multi-turn + Scores + Thinking Mask)...")
    
    ds = ds.map(formatting_func, batched=False, num_proc=32)
    
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col and 'target_scores' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    """
    Process UltraFeedback dataset:
    1. Parse annotations to extract scores for 4 dimensions and calculate avg_score.
    2. Pair responses based on avg_score (Chosen=Highest, Rejected=Lowest).
    3. Append the scores of the Chosen sample to the Prompt.
    4. In SGA mode: Mask the Prompt section and the <think>...</think> reasoning process.
    """

    # UltraFeedback commonly features one response per line if using the standard version.
    try:
        ds = load_dataset(data_path, split=split)
    except:
        ds = load_dataset("path/to/ultrafeedback-binarized-preferences-cleaned-kto", split=split)

    df = ds.to_pandas()
    
    # Four main metrics for UltraFeedback
    uf_score_cols = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    
    def extract_score(row, col_name):
        """Extract score from annotations (e.g., {'helpfulness': {'Rating': '4', ...}, ...})."""
        try:
            if row.get('annotations') and isinstance(row['annotations'], dict):
                if col_name in row['annotations']:
                    return float(row['annotations'][col_name]['Rating'])
        except Exception as e:
            raise ValueError("annotations error")

    for col in uf_score_cols:
        df[col] = df.apply(lambda row: extract_score(row, col), axis=1)

    df['avg_score'] = df[uf_score_cols].mean(axis=1)

    paired_rows = []
    
    # Filter duplicate prompts to ensure comparison within the same prompt.
    # Note: If the prompt in the dataset is a list or unhashable type, convert to string first.
    # Assuming prompt is string here.
    grouped = df.groupby('prompt')
    
    for prompt, group in grouped:
        if len(group) < 2: continue
        
        sorted_group = group.sort_values('avg_score', ascending=False)
        
        chosen_row = sorted_group.iloc[0]
        rejected_row = sorted_group.iloc[-1]
        
        margin = chosen_row['avg_score'] - rejected_row['avg_score']
        
        if margin == 0: continue

        chosen_scores = {col: chosen_row[col] for col in uf_score_cols}

        paired_rows.append({
            'prompt': prompt, 
            'chosen_content': chosen_row['completion'] if 'completion' in chosen_row else chosen_row['response'],
            'rejected_content': rejected_row['completion'] if 'completion' in rejected_row else rejected_row['response'],
            'margin': margin,
            'chosen_scores': chosen_scores
        })

    ds = Dataset.from_list(paired_rows)

    if size is not None and size < len(ds):
        ds = ds.select(range(0, size))

    def format_preferences(scores_dict):
        texts = []
        for name in uf_score_cols:
            if name in scores_dict:
                val = (scores_dict[name] - 1) / 4
                texts.append(f"<{name}_score:{val}>") 
        pref_scores = " and ".join(texts)
        return f"\nYour response must satisfy the following scores: {pref_scores}"

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": 4096, "return_tensors": "pt"}

        helpfulness_val = (example['chosen_scores'].get('helpfulness', 0.0) - 1) / 4
        honesty_val = (example['chosen_scores'].get('honesty', 0.0) - 1) / 4
        instruction_following_val = (example['chosen_scores'].get('instruction_following', 0.0) - 1) / 4
        truthfulness_val = (example['chosen_scores'].get('truthfulness', 0.0) - 1) / 4
        target_scores = torch.tensor([helpfulness_val, honesty_val, instruction_following_val, truthfulness_val], dtype=torch.float)
        target_scores = torch.round(target_scores, decimals=2)
        raw_user_content = example['prompt']
        
        messages_raw = [{"role": "user", "content": raw_user_content}]
        raw_prompt_str = tokenizer.apply_chat_template(messages_raw, tokenize=False, add_generation_prompt=False)
        tokens_raw_prompt = tokenizer.encode_plus(raw_prompt_str, **kwargs)

        score_suffix = format_preferences(example['chosen_scores'])
        messages_with_score = [{"role": "user", "content": raw_user_content + score_suffix}]

        chosen_messages = messages_with_score + [{"role": "assistant", "content": example['chosen_content']}]
        rejected_messages = messages_with_score + [{"role": "assistant", "content": example['rejected_content']}]
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        
        margin = example['margin']

        if 'SGA' in model_name:
            prompt_template_with_scores = tokenizer.apply_chat_template(messages_with_score, tokenize=False, add_generation_prompt=True)
            tokens_prompt_with_scores = tokenizer.encode_plus(prompt_template_with_scores, **kwargs)['input_ids'][0]
            len_prompt_mask = len(tokens_prompt_with_scores)
            
            input_ids_chosen = tokens_chosen["input_ids"][0]
            label_chosen = input_ids_chosen.clone()
            
            mask_len_c = min(len_prompt_mask, len(label_chosen))
            label_chosen[:mask_len_c] = -100
            
            label_chosen = mask_thinking_labels(label_chosen, input_ids_chosen, tokenizer)
            
            input_ids_rejected = tokens_rejected["input_ids"][0]
            label_rejected = input_ids_rejected.clone()
            
            mask_len_r = min(len_prompt_mask, len(label_rejected))
            label_rejected[:mask_len_r] = -100

            label_rejected = mask_thinking_labels(label_rejected, input_ids_rejected, tokenizer)
            
            return {
                "input_ids_chosen": input_ids_chosen, "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": input_ids_rejected, "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen, 'label_rejected': label_rejected,
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0],
                "target_scores": target_scores
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin,
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0]
            }
        
    print(f"Processing UltraFeedback dataset (Score Formatting + Thinking Mask)...")

    ds = ds.map(formatting_func, batched=False, num_proc=32)
    
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col and 'target_scores' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds
    

def build_dataset_HH(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    """
    Process HH-RLHF dataset (with Sigmoid normalization):
    1. Group by Prompt, only process groups containing 2 Responses.
    2. Determine Chosen/Rejected based on the label column.
    3. Apply Sigmoid normalization to the Chosen Response's score (mapped to 0-1).
    4. In SGA mode: Mask Prompt + Mask Thinking.
    """

    try:
        ds = load_from_disk(data_path)
    except :
        ds = load_from_disk('path/to/train_harmhelp_llama3')

    df = ds.to_pandas()
    
    # Ensure scores and labels are numeric types
    df['score1'] = pd.to_numeric(df['score1'], errors='coerce').fillna(0)
    df['score2'] = pd.to_numeric(df['score2'], errors='coerce').fillna(0)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0)

    def sigmoid(x):
        # Limit the range of x to prevent overflow
        x = np.clip(x, -100, 100)
        return 1.0 / (1.0 + np.exp(-x))

    paired_rows = []
    
    # Filter out prompts with only one response
    df = df[df.duplicated(subset=['prompt'], keep=False)]
    
    # Process grouped by Prompt
    grouped = df.groupby('prompt')
    
    print("Start building Preference Pairs (Chosen filtering + Sigmoid normalization)...")
    
    for prompt, group in grouped:
        if len(group) != 2: 
            continue
        
        row_a = group.iloc[0]
        row_b = group.iloc[1]
        
        # Logic: label 1 is Chosen
        if int(row_a['label']) == 1 and int(row_b['label']) == 0:
            chosen_row = row_a
            rejected_row = row_b
        elif int(row_a['label']) == 0 and int(row_b['label']) == 1:
            chosen_row = row_b
            rejected_row = row_a
        else:
            continue # Skip conflicting labels

        # Calculate margin (Margin usually keeps the original score difference for Loss calculation, no need to normalize, or decide based on your Loss requirement)
        # Calculate margin using original scores difference
        avg_score_chosen = (chosen_row['score1'] + chosen_row['score2']) / 2
        avg_score_rejected = (rejected_row['score1'] + rejected_row['score2']) / 2
        margin = avg_score_chosen - avg_score_rejected

        # Only normalize scores used for Prompt Conditioning
        chosen_scores = {
            'harmless': sigmoid(chosen_row['score1']),
            'helpfulness': sigmoid(chosen_row['score2'])
        }

        paired_rows.append({
            'prompt': prompt, 
            'chosen_content': chosen_row['response'],
            'rejected_content': rejected_row['response'],
            'margin': margin,
            'chosen_scores': chosen_scores
        })

    ds = Dataset.from_list(paired_rows)

    if size is not None and size < len(ds):
        ds = ds.select(range(0, size))

    def format_preferences(scores_dict):
        """Convert scores to Prompt suffix string"""
        texts = []
        keys = ['harmless', 'helpfulness']
        for name in keys:
            if name in scores_dict:
                val = scores_dict[name] 
                texts.append(f"<{name}_score:{val:.2f}>") 
        
        if not texts:
            return ""
            
        pref_scores = " and ".join(texts)
        return f"\nYour response must satisfy the following scores: {pref_scores}"

    def parse_hh_multiturn_prompt(raw_prompt):
        messages = []
        segments = re.split(r'(\n\nHuman:|\n\nAssistant:)', raw_prompt)
        current_role = None
        for segment in segments:
            if segment == '\n\nHuman:':
                current_role = 'user'
            elif segment == '\n\nAssistant:':
                current_role = 'assistant'
            else:
                content = segment.strip()
                if current_role and content:
                    messages.append({"role": current_role, "content": content})
        return messages

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": 4096, "return_tensors": "pt"}

        harmless_val = example['chosen_scores'].get('harmless', 0.0)
        helpful_val = example['chosen_scores'].get('helpfulness', 0.0)
        target_scores = torch.tensor([harmless_val, helpful_val], dtype=torch.float)
        target_scores = torch.round(target_scores, decimals=2)

        history_messages = parse_hh_multiturn_prompt(example['prompt'])
        
        raw_prompt_str = tokenizer.apply_chat_template(history_messages, tokenize=False, add_generation_prompt=False)
        tokens_raw_prompt = tokenizer.encode_plus(raw_prompt_str, **kwargs)

        score_suffix = format_preferences(example['chosen_scores'])
        
        modified_history = [dict(msg) for msg in history_messages]
        if modified_history and modified_history[-1]['role'] == 'user':
            modified_history[-1]['content'] = modified_history[-1]['content'] + score_suffix
        else:
            modified_history.append({"role": "user", "content": score_suffix.strip()})

        chosen_messages = modified_history + [{"role": "assistant", "content": example['chosen_content']}]
        rejected_messages = modified_history + [{"role": "assistant", "content": example['rejected_content']}]
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        
        if 'SGA' in model_name:
            prompt_template_with_scores = tokenizer.apply_chat_template(modified_history, tokenize=False, add_generation_prompt=True)
            tokens_prompt_with_scores = tokenizer.encode_plus(prompt_template_with_scores, **kwargs)['input_ids'][0]
            len_prompt_mask = len(tokens_prompt_with_scores)
            
            input_ids_chosen = tokens_chosen["input_ids"][0]
            label_chosen = input_ids_chosen.clone()
            mask_len_c = min(len_prompt_mask, len(label_chosen))
            label_chosen[:mask_len_c] = -100
            label_chosen = mask_thinking_labels(label_chosen, input_ids_chosen, tokenizer)
            
            input_ids_rejected = tokens_rejected["input_ids"][0]
            label_rejected = input_ids_rejected.clone()
            mask_len_r = min(len_prompt_mask, len(label_rejected))
            label_rejected[:mask_len_r] = -100
            label_rejected = mask_thinking_labels(label_rejected, input_ids_rejected, tokenizer)
            
            return {
                "input_ids_chosen": input_ids_chosen, "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": input_ids_rejected, "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen, 'label_rejected': label_rejected,
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0],
                "target_scores": target_scores
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": example['margin'],
                "input_ids_prompt": tokens_raw_prompt["input_ids"][0],
                "attention_mask_prompt": tokens_raw_prompt["attention_mask"][0]
            }
        
    print(f"Processing HH-RLHF dataset (Pairwise + Label Check + Sigmoid Scores)...")
    
    ds = ds.map(formatting_func, batched=False, num_proc=32)
    
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col and 'target_scores' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds

def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name=''):
    """
    Load train and eval datasets based on data_path.
    
    Supported datasets:
    - HelpSteer2: 5-dimensional scores (helpfulness, correctness, coherence, complexity, verbosity)
    - UltraFeedback: 4-dimensional scores (helpfulness, honesty, instruction_following, truthfulness)
    - HH-RLHF: 2-dimensional scores (harmless, helpfulness)
    - Unified-Feedback: Original pairwise preference data
    - Skywork: Original pairwise preference data
    - Others: Generic pairwise preference data
    """
    data_path_lower = data_path.lower()
    
    # HelpSteer2 dataset (5 dimensions)
    if 'helpsteer2' in data_path_lower or 'help_steer2' in data_path_lower or 'helpsteer-2' in data_path_lower:
        print(f"Loading HelpSteer2 dataset from {data_path}...")
        dataset = build_dataset_HS2(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    
    # UltraFeedback dataset (4 dimensions) - new version with scores
    elif 'ultrafeedback' in data_path_lower or 'ultra_feedback' in data_path_lower:
        print(f"Loading UltraFeedback dataset from {data_path}...")
        dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    
    # HH-RLHF dataset (2 dimensions: harmless, helpfulness)
    elif 'hh-rlhf' in data_path_lower or 'hh_rlhf' in data_path_lower or 'harmhelp' in data_path_lower:
        print(f"Loading HH-RLHF dataset from {data_path}...")
        dataset = build_dataset_HH(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    
    # Unified-Feedback (original pairwise, no multi-dim scores)
    elif 'unified' in data_path_lower:
        print(f"Loading Unified-Feedback dataset from {data_path}...")
        # Use the old build_dataset_UF for original unified feedback
        train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name) 
        eval_dataset = build_dataset_UF(data_path, tokenizer, split='val', model_name=model_name)
    
    # Skywork dataset (original pairwise)
    elif 'skywork' in data_path_lower:
        print(f"Loading Skywork dataset from {data_path}...")
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    
    # Generic pairwise preference data
    else:
        print(f"Loading generic preference dataset from {data_path}...")
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name) 
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    
    return train_dataset, eval_dataset


def get_num_output_for_dataset(data_path):
    """
    Get the number of output dimensions for the regression head based on dataset.
    
    Returns:
        int: Number of output dimensions
            - HelpSteer2: 5
            - UltraFeedback: 4
            - HH-RLHF: 2
            - Others: 1 (default single-dim reward)
    """
    data_path_lower = data_path.lower()
    
    if 'helpsteer2' in data_path_lower or 'help_steer2' in data_path_lower or 'helpsteer-2' in data_path_lower:
        return 5
    elif 'ultrafeedback' in data_path_lower or 'ultra_feedback' in data_path_lower:
        return 4
    elif 'hh-rlhf' in data_path_lower or 'hh_rlhf' in data_path_lower or 'harmhelp' in data_path_lower:
        return 2
    else:
        return 1  # Default single-dim reward for original datasets