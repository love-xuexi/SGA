from accelerate import Accelerator
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score  # Accuracy evaluation function


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return {"accuracy": accuracy_score(labels, predictions)}


def sga_compute_metrics(eval_pred):
    rewards = eval_pred.label_ids
    predictions = eval_pred.predictions
    
    # Handle multi-dimensional rewards: shape (batch, 2, num_dims) or (batch, 2)
    if rewards.ndim == 3:
        # Multi-dimensional case: average across dimensions for comparison
        rewards_chosen = rewards[:, 0, :].mean(axis=-1)  # (batch,)
        rewards_rejected = rewards[:, 1, :].mean(axis=-1)  # (batch,)
        reward_accuracy = (rewards_chosen > rewards_rejected).mean()
        
        # Also compute per-dimension accuracy
        per_dim_acc = {}
        for dim in range(rewards.shape[-1]):
            dim_acc = (rewards[:, 0, dim] > rewards[:, 1, dim]).mean()
            per_dim_acc[f'reward_accuracy_dim{dim}'] = dim_acc
    else:
        # Original 1D case: shape (batch, 2)
        reward_accuracy = (rewards[:, 0] > rewards[:, 1]).mean()
        per_dim_acc = {}
    
    # DPO accuracy from log probabilities
    accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    
    result = {
        'dpo_accuracy': accuracy,
        'reward_accuracy': reward_accuracy
    }
    result.update(per_dim_acc)
    return result


def print_trainable_parameters(model, print_trainable_name=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_trainable_name:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


