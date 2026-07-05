from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
from sga_trainer import SGADataCollatorWithPadding, SGATrainer
from load_datasets import load_train_eval_dataset, get_num_output_for_dataset
from utils import print_trainable_parameters, sga_compute_metrics
from sga_utils import AutoModelForCausalLMWithValueHead
from datetime import datetime


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=2) 
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "The number of training epochs for the reward model."})
    optim: Optional[str] = field(default="adamw_torch",  metadata={"help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # data
    dataset: Optional[str] = field(default='llm-blender/Unified-Feedback')
    dataset_mode: Optional[str] = field(default='', metadata={"help": "use from '', '40k', and '400k' for the paper's experiments"},)
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=100)
    # model and loss
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    # log
    report_to: Optional[str] = field(default='none', metadata={'help': "use 'none', 'wandb'. "})
    output_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=1000)
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})
    # SGA original parameters
    weight_ratio: Optional[float] = field(default=0.01, metadata={'help': 'Weight ratio for DPO in dpo_bt mode'})
    beta: Optional[float] = field(default=0.1, metadata={'help': 'beta for DPO'})
    layer_type: Optional[str] = field(default='mlp', metadata={'help': 'mlp or linear for value head'})
    num_layers: Optional[int] = field(default=3, metadata={'help': 'Number of layers in value head MLP'})
    num_neurons: Optional[int] = field(default=1024, metadata={'help': 'Number of neurons in value head MLP'})
    reference_free: Optional[bool] = field(default=False)
    sft_only: Optional[bool] = field(default=False)
    no_logsigmoid_sft: Optional[bool] = field(default=False)
    # New DPO+MSE mode parameters
    training_mode: Optional[str] = field(
        default='dpo_mse', 
        metadata={'help': "Training mode: 'dpo_bt' (original DPO+BT) or 'dpo_mse' (new DPO+MSE)"}
    )
    num_output: Optional[int] = field(
        default=None, 
        metadata={'help': 'Number of output dimensions for regression head. If None, auto-detect from dataset.'}
    )
    dpo_weight: Optional[float] = field(
        default=0.1, 
        metadata={'help': 'Weight for DPO loss in dpo_mse mode. Default 0.1'}
    )
    mse_weight: Optional[float] = field(
        default=0.9, 
        metadata={'help': 'Weight for MSE loss in dpo_mse mode. Default 0.9'}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]

# Auto-detect num_output based on dataset if not specified
if script_args.num_output is None:
    script_args.num_output = get_num_output_for_dataset(script_args.dataset)
    print(f"Auto-detected num_output={script_args.num_output} for dataset {script_args.dataset}")

# Determine training mode based on num_output if not explicitly set to dpo_mse
if script_args.num_output > 1 and script_args.training_mode == 'dpo_bt':
    print(f"Warning: num_output={script_args.num_output} > 1, but training_mode='dpo_bt'. "
          f"Consider using training_mode='dpo_mse' for multi-dimensional regression.")

current_time = datetime.now().strftime("%m_%d_%H_%M")

if script_args.use_lora:
    output_name = (
        f"{script_args.output_dir}/{model_name_split}_dpo_"
        f"len{script_args.max_length}_lora{script_args.lora_r}_"
        f"{script_args.learning_rate}_{current_time}"
    )
else:
    output_name = (
        f"{script_args.output_dir}/{model_name_split}_dpo_"
        f"len{script_args.max_length}_fulltrain_"
        f"{script_args.learning_rate}_{current_time}"
    )
device = Accelerator().local_process_index 

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    eval_strategy=script_args.eval_strategy,
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing, 
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    # warmup_ratio=0.03,
    optim=script_args.optim,
    adam_beta1=0.95,
    adam_beta2=0.95,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True, 
    dataloader_num_workers=2,  
    dataloader_prefetch_factor=4,  
    dataloader_persistent_workers=True,  
    logging_nan_inf_filter=True,
    auto_find_batch_size=False,  
)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast = False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(script_args.dataset, tokenizer, mode=script_args.dataset_mode, model_name='SGA', size=1000 if script_args.debug else None)
print('Training dataset size: {}, validation dataset size: {}'.format(len(train_dataset), len(eval_dataset)))

# ==================== PRINT ALL TRAINING PARAMETERS ====================
print("\n" + "="*80)
print("ALL TRAINING PARAMETERS")
print("="*80)

# Print all script arguments
print("\n--- Model & Dataset Parameters ---")
print(f"Base Model:                    {script_args.base_model}")
print(f"Dataset:                       {script_args.dataset}")
print(f"Dataset Mode:                  {script_args.dataset_mode}")

print("\n--- Training Configuration ---")
print(f"Training Mode:                 {script_args.training_mode}")
print(f"Per Device Train Batch Size:   {script_args.per_device_train_batch_size}")
print(f"Gradient Accumulation Steps:   {script_args.gradient_accumulation_steps}")
print(f"Learning Rate:                 {script_args.learning_rate}")
print(f"Number of Training Epochs:     {script_args.num_train_epochs}")
print(f"Optimizer:                     {script_args.optim}")
print(f"LR Scheduler Type:             {script_args.lr_scheduler_type}")
print(f"Max Sequence Length:           {script_args.max_length}")
print(f"Gradient Checkpointing:        {script_args.gradient_checkpointing}")
print(f"BF16 Training:                 {script_args.bf16}")

print("\n--- Loss & Weight Parameters ---")
print(f"Number of Output Dimensions:   {script_args.num_output}")
if script_args.training_mode == 'dpo_mse':
    print(f"DPO Weight:                    {script_args.dpo_weight}")
    print(f"MSE Weight:                    {script_args.mse_weight}")
else:
    print(f"Weight Ratio (DPO):            {script_args.weight_ratio}")
print(f"Beta (DPO parameter):          {script_args.beta}")

print("\n--- LoRA Configuration ---")
print(f"Use LoRA:                      {script_args.use_lora}")
if script_args.use_lora:
    print(f"LoRA Rank (r):                 {script_args.lora_r}")
    print(f"LoRA Alpha:                    {script_args.lora_alpha}")
    print(f"LoRA Dropout:                  {script_args.lora_dropout}")
    print(f"LoRA Target Modules:           {script_args.lora_target_modules}")

print("\n--- Evaluation & Saving ---")
print(f"Evaluation Strategy:           {script_args.eval_strategy}")
print(f"Evaluation Steps:              {script_args.eval_steps}")
print(f"Save Strategy:                 {script_args.save_strategy}")
print(f"Save Steps:                    {script_args.save_steps}")
print(f"Per Device Eval Batch Size:    {script_args.per_device_eval_batch_size}")

print("\n--- Value Head Configuration ---")
print(f"Value Head Layer Type:         {script_args.layer_type}")
print(f"Value Head Num Layers:         {script_args.num_layers}")
print(f"Value Head Num Neurons:        {script_args.num_neurons}")

print("\n--- Model & Loss Flags ---")
print(f"Reference Free:                {script_args.reference_free}")
print(f"SFT Only:                      {script_args.sft_only}")
print(f"No LogSigmoid SFT:             {script_args.no_logsigmoid_sft}")
print(f"Attention Implementation:      {script_args.attn_implementation}")

print("\n--- Logging & Output ---")
print(f"Report To:                     {script_args.report_to}")
print(f"Output Directory:              {script_args.output_dir}")
print(f"Wandb Name:                    {script_args.wandb_name}")
print(f"Debug Mode:                    {script_args.debug}")

print("\n--- TrainingArguments (Derived) ---")
print(f"Final Output Dir:              {output_name}")
print(f"Warmup Ratio:                  0.03")
print(f"Max Grad Norm:                 5.0")
print(f"Logging Strategy:              steps")
print(f"Logging Steps:                 10")
print(f"DDP Find Unused Parameters:    False")
print(f"Remove Unused Columns:         False")

print("="*80 + "\n")

model_params = {
    'vhead_layer_type': script_args.layer_type,
    'vhead_num_neurons': script_args.num_neurons,
    'vhead_num_layers': script_args.num_layers,
    'vhead_num_output': script_args.num_output,  # Multi-dimensional output support
}
if len(script_args.attn_implementation):
    model_params["attn_implementation"] = script_args.attn_implementation


### load model
if not script_args.reference_free:
    reference_model = AutoModelForCausalLM.from_pretrained(script_args.base_model, 
    device_map=device, 
    torch_dtype=torch.bfloat16, 
    use_cache=False if 'qwen' in script_args.base_model.lower() else True, 
    attn_implementation="flash_attention_2")
    
    reference_model.resize_token_embeddings(len(tokenizer))
    reference_model.config.pad_token_id = tokenizer.pad_token_id


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.base_model, device_map=device, 
    torch_dtype=torch.bfloat16,
    use_cache=False if 'qwen' in script_args.base_model.lower() else True,
    **model_params,
)

model.pretrained_model.resize_token_embeddings(len(tokenizer))
print_trainable_parameters(model)
model.config.pad_token_id = tokenizer.pad_token_id

if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=script_args.lora_target_modules,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)

## let value head trainable
# After LoRA wrapping, v_head may be at different locations
def set_vhead_trainable(m):
    """Set v_head parameters trainable, handling different model structures"""
    vhead = None
    if hasattr(m, 'v_head'):
        vhead = m.v_head
    elif hasattr(m, 'base_model') and hasattr(m.base_model, 'model') and hasattr(m.base_model.model, 'v_head'):
        # PeftModel structure: model.base_model.model.v_head
        vhead = m.base_model.model.v_head
    
    if vhead is not None:
        for parameter in vhead.parameters():
            parameter.requires_grad = True
        print(f"Set v_head trainable: {sum(p.numel() for p in vhead.parameters())} parameters")
    else:
        print("Warning: v_head not found in model structure")

set_vhead_trainable(model)
print_trainable_parameters(model)


# Define the trainer parameters
trainer_params = {
    "model": model,
    "reference_model": reference_model if not script_args.reference_free else None,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "compute_metrics": sga_compute_metrics,
    "data_collator": SGADataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    # Original DPO+BT parameters
    'weight_ratio': script_args.weight_ratio,
    'reference_free': script_args.reference_free,
    'sft_only': script_args.sft_only,
    'no_logsigmoid_sft': script_args.no_logsigmoid_sft,
    'beta': script_args.beta,
    'use_lora': script_args.use_lora,
    # New DPO+MSE parameters
    'training_mode': script_args.training_mode,
    'dpo_weight': script_args.dpo_weight,
    'mse_weight': script_args.mse_weight,
    # Model config to save
    'info_to_save': {
        'base_model': script_args.base_model,
        'layer_type': script_args.layer_type,
        'num_neurons': script_args.num_neurons,
        'num_layers': script_args.num_layers,
        'num_output': script_args.num_output,  # Save multi-dim output config
    }
}


# Train the model, woohoo.
trainer = SGATrainer(**trainer_params)
print(f'Training start with mode: {script_args.training_mode}')
trainer.train()
