from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
from utils import get_trainable_weights


@dataclass
class SGADataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 1024  # Default max length is 1024
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Check if we have the new SGA-style fields (input_ids_prompt and target_scores)
        has_prompt_fields = "input_ids_prompt" in features[0] and "target_scores" in features[0]
        
        # === Filter over-length samples (discard strategy) ===
        max_len = self.max_length if self.max_length is not None else 1024
        valid_features = []
        for feature in features:
            len_chosen = len(feature["input_ids_chosen"])
            len_rejected = len(feature["input_ids_rejected"])
            # Only keep samples where both chosen and rejected sequences do not exceed max_length
            if len_chosen <= max_len and len_rejected <= max_len:
                valid_features.append(feature)
        
        # === Empty batch fallback: if all discarded, keep the shortest sample and truncate ===
        if len(valid_features) == 0:
            # Find the sample with the shortest max(chosen, rejected)
            shortest_feature = min(
                features,
                key=lambda f: max(len(f["input_ids_chosen"]), len(f["input_ids_rejected"]))
            )
            # Truncate to max_length
            truncated_feature = {
                "input_ids_chosen": shortest_feature["input_ids_chosen"][:max_len],
                "attention_mask_chosen": shortest_feature["attention_mask_chosen"][:max_len],
                "input_ids_rejected": shortest_feature["input_ids_rejected"][:max_len],
                "attention_mask_rejected": shortest_feature["attention_mask_rejected"][:max_len],
                "label_chosen": shortest_feature["label_chosen"][:max_len],
                "label_rejected": shortest_feature["label_rejected"][:max_len],
            }
            # Keep other fields (like input_ids_prompt, target_scores)
            if has_prompt_fields:
                truncated_feature["input_ids_prompt"] = shortest_feature["input_ids_prompt"][:max_len]
                truncated_feature["attention_mask_prompt"] = shortest_feature["attention_mask_prompt"][:max_len]
                truncated_feature["target_scores"] = shortest_feature["target_scores"]
            valid_features = [truncated_feature]
        
        features = valid_features
        
        # Merge chosen and rejected sequences
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        # Dynamic padding: pad to the length of the longest sequence in the current batch (not exceeding max_length)
        batch = self.tokenizer.pad(
            merged_features,
            padding=True,  # Dynamic padding
            max_length=max_len,  # Ensure it does not exceed max_length
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Pad labels
        paded_length = batch["input_ids"].shape[1]
        label_paded = []
        for feature in features:
            label_chosen_paded = torch.tensor(feature["label_chosen"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_chosen"])) , dtype=torch.int64)
            label_rejected_paded = torch.tensor(feature["label_rejected"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_rejected"])) , dtype=torch.int64)
            label_paded.extend([label_chosen_paded.view(1, -1), label_rejected_paded.view(1, -1)])
        label_paded = torch.concatenate(label_paded, dim=0)
        
        result_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "label": label_paded,  
        }
        
        # Handle new SGA-style fields for DPO+MSE training
        if has_prompt_fields:
            # Pad prompt sequences (original prompt without score info, for MSE calculation)
            prompt_features = []
            for feature in features:
                prompt_features.append(
                    {
                        "input_ids": feature["input_ids_prompt"],
                        "attention_mask": feature["attention_mask_prompt"],
                    }
                )
            prompt_batch = self.tokenizer.pad(
                prompt_features,
                padding=True,  # Dynamic padding
                max_length=max_len,  # Ensure it does not exceed max_length
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            result_batch["input_ids_prompt"] = prompt_batch["input_ids"]
            result_batch["attention_mask_prompt"] = prompt_batch["attention_mask"]
            
            # Stack target scores (already fixed dimension, no padding needed)
            target_scores_list = [feature["target_scores"] for feature in features]
            result_batch["target_scores"] = torch.stack(target_scores_list, dim=0)
        
        return result_batch



class SGATrainer(RewardTrainer):    
    def __init__(self, **kwargs):
        self.reference_free = kwargs.pop('reference_free', True)
        self.reference_model = kwargs.pop('reference_model', None)
        self.sft_only = kwargs.pop('sft_only', True)
        self.no_logsigmoid_sft = kwargs.pop('no_logsigmoid_sft', False)
        # Legacy weight_ratio for backward compatibility
        self.weight_ratio = kwargs.pop('weight_ratio', 0.01)
        # New separate weights for DPO and MSE (DPO+MSE mode)
        self.dpo_weight = kwargs.pop('dpo_weight', 0.1)
        self.mse_weight = kwargs.pop('mse_weight', 0.9)
        self.beta = kwargs.pop('beta', 0.1)
        self.label_pad_token_id = -100
        self.use_lora = kwargs.pop('use_lora', True)
        self.info_to_save = kwargs.pop('info_to_save', {})
        # Training mode: 'dpo_bt' (original) or 'dpo_mse' (new SGA-style)
        self.training_mode = kwargs.pop('training_mode', 'dpo_mse')
        super(SGATrainer, self).__init__(**kwargs)


    def get_batch_logps(
        self, 
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss based on training mode:
        - 'dpo_bt': Original DPO + BT reward loss
        - 'dpo_mse': New DPO + MSE loss (SGA-style with two forward passes)
        """
        if self.training_mode == 'dpo_mse':
            return self.compute_loss_dpo_mse(model, inputs, return_outputs, num_items_in_batch)
        else:
            return self.compute_loss_dpo_bt(model, inputs, return_outputs, num_items_in_batch)

    def compute_loss_dpo_bt(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Original DPO + BT reward loss computation."""
        logits, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if not self.reference_free:
            with torch.no_grad():
                ref_logits = self.reference_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2) # chosen_ids
        kidx = jidx + 1                # rejected_ids  
        reward_loss = -nn.functional.logsigmoid(rewards[jidx] - rewards[kidx]).mean()

        ## text-generation regularization
        if self.weight_ratio > 0:
            logps = self.get_batch_logps(logits, inputs['label'])
            if self.sft_only:
                pi_logratios = logps[jidx] # positive

                if self.no_logsigmoid_sft:
                    dpo_loss = - pi_logratios.mean()
                else:
                    dpo_loss = -F.logsigmoid(self.beta * (pi_logratios)).mean()
            else:
                pi_logratios = logps[jidx] - logps[kidx]  
                if self.reference_free or self.sft_only:
                    ref_logratios = torch.tensor(0.0)
                else:
                    ref_logps = self.get_batch_logps(ref_logits, inputs['label'])
                    ref_logratios = ref_logps[jidx] - ref_logps[kidx]

                pi_logratios = pi_logratios.to(rewards[0].device)
                ref_logratios = ref_logratios.to(rewards[0].device)
                dpo_loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()

            loss = self.weight_ratio * dpo_loss + (1 - self.weight_ratio) * reward_loss
        else:
            loss = reward_loss

        if return_outputs:
            return loss, {}
        return loss

    def _get_underlying_model(self, model):
        """
        Helper to get the underlying model, unwrapping DDP/FSDP and PeftModel if needed.
        - DDP wraps the model in model.module
        - PeftModel wraps in model.base_model.model
        Custom methods like forward_regression_only need to be accessed through the unwrapped model.
        """
        # Unwrap DDP/FSDP first
        if hasattr(model, 'module'):
            model = model.module
        # Unwrap PeftModel (LoRA wrapper)
        # PeftModel has peft_config and base_model.model structure
        if hasattr(model, 'peft_config') and hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model'):
                return model.base_model.model
        return model

    def compute_loss_dpo_mse(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        New DPO + MSE loss computation (SGA-style).
        
        Two forward passes:
        1. First forward: Use original prompt (without score info) to compute MSE loss via regression head
        2. Second forward: Use enhanced prompt (with score info) + response to compute DPO loss
        
        Total loss = dpo_weight * DPO_loss + mse_weight * MSE_loss
        """
        # Validate required fields for DPO+MSE mode
        required_fields = ["input_ids_prompt", "attention_mask_prompt", "target_scores"]
        for field in required_fields:
            if field not in inputs:
                raise ValueError(
                    f"'{field}' not found in inputs. DPO+MSE mode requires dataset to provide: {required_fields}. "
                    f"Make sure you are using a compatible dataset (HelpSteer2, UltraFeedback, or HH-RLHF) with model_name='SGA'."
                )
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Get underlying model for custom method access (unwrap DDP and PeftModel if needed)
        underlying_model = self._get_underlying_model(model)
        
        # ========== First Forward Pass: MSE Loss ==========
        # Use original prompt (input_ids_prompt) to predict scores
        # Only use chosen samples since target_scores corresponds to chosen
        predicted_scores = underlying_model.forward_regression_only(
            input_ids=inputs["input_ids_prompt"].to(device),
            attention_mask=inputs["attention_mask_prompt"].to(device),
            apply_sigmoid=True  # Normalize to [0, 1]
        )  # shape: (batch_size, num_output)
        
        # Get target scores and compute MSE loss
        target_scores = inputs["target_scores"].to(device).float()
        mse_loss = F.mse_loss(predicted_scores, target_scores)
        
        # ========== Second Forward Pass: DPO Loss ==========
        # Use enhanced prompt (with score info) + response for DPO
        logits, _, rewards = model(
            input_ids=inputs["input_ids"].to(device), 
            attention_mask=inputs["attention_mask"].to(device)
        )
        
        # Get reference model logits if needed
        if not self.reference_free:
            with torch.no_grad():
                ref_logits = self.reference_model(
                    input_ids=inputs["input_ids"].to(device), 
                    attention_mask=inputs["attention_mask"].to(device)
                )[0]
        
        bsz = logits.size(0)
        jidx = torch.arange(0, bsz, 2, device=device)  # chosen_ids
        kidx = jidx + 1                                 # rejected_ids
        
        # Compute DPO loss
        logps = self.get_batch_logps(logits, inputs['label'].to(device))
        
        if self.sft_only:
            # SFT-only mode: only use chosen samples
            pi_logratios = logps[jidx]
            if self.no_logsigmoid_sft:
                dpo_loss = -pi_logratios.mean()
            else:
                dpo_loss = -F.logsigmoid(self.beta * pi_logratios).mean()
        else:
            # Full DPO: use both chosen and rejected
            pi_logratios = logps[jidx] - logps[kidx]
            
            if self.reference_free:
                ref_logratios = torch.tensor(0.0, device=device)
            else:
                ref_logps = self.get_batch_logps(ref_logits, inputs['label'].to(device))
                ref_logratios = ref_logps[jidx] - ref_logps[kidx]
            
            dpo_loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()
        
        # ========== Combine Losses ==========
        total_loss = self.dpo_weight * dpo_loss + self.mse_weight * mse_loss
        
        if return_outputs:
            return total_loss, {
                "dpo_loss": dpo_loss.detach(),
                "mse_loss": mse_loss.detach(),
                "predicted_scores": predicted_scores.detach(),
            }
        return total_loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if self.training_mode == 'dpo_mse':
                # For DPO+MSE mode, also evaluate MSE
                device = next(model.parameters()).device
                
                # Get underlying model for custom method access (unwrap DDP if needed)
                underlying_model = self._get_underlying_model(model)
                
                # Predict scores using original prompt
                if "input_ids_prompt" in inputs:
                    predicted_scores = underlying_model.forward_regression_only(
                        input_ids=inputs["input_ids_prompt"].to(device),
                        attention_mask=inputs["attention_mask_prompt"].to(device),
                        apply_sigmoid=True
                    )
                else:
                    predicted_scores = None
                
                # DPO evaluation
                logits, _, rewards = model(
                    input_ids=inputs["input_ids"].to(device), 
                    attention_mask=inputs["attention_mask"].to(device)
                )
                logps = self.get_batch_logps(logits, inputs['label'].to(device))
                
                if self.reference_free:
                    dpo_logp_diff = logps
                else:
                    ref_logits = self.reference_model(
                        input_ids=inputs["input_ids"].to(device), 
                        attention_mask=inputs["attention_mask"].to(device)
                    )[0]
                    ref_logps = self.get_batch_logps(ref_logits, inputs['label'].to(device))
                    dpo_logp_diff = logps - ref_logps
                
                # Return both DPO and MSE related outputs
                # Handle multi-dimensional rewards
                if rewards.dim() == 2:
                    # Multi-dimensional output, reshape accordingly
                    return (None, dpo_logp_diff.reshape(-1, 2), rewards.reshape(-1, 2, rewards.size(-1)))
                else:
                    return (None, dpo_logp_diff.reshape(-1, 2), rewards.reshape(-1, 2))
            else:
                # Original DPO+BT mode
                logits, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                logps = self.get_batch_logps(logits, inputs['label'])
                if self.reference_free:
                    dpo_logp_diff = logps
                else:
                    ref_logits = self.reference_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
                    ref_logps = self.get_batch_logps(ref_logits, inputs['label'])
                    dpo_logp_diff = logps - ref_logps

                return (None, dpo_logp_diff.reshape(-1, 2), rewards.reshape(-1, 2))


    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            ## add config
            model.config.vhead_layer_type = self.info_to_save['layer_type']
            model.config.vhead_num_neurons = self.info_to_save['num_neurons']
            model.config.vhead_num_layers = self.info_to_save['num_layers']
            # Save num_output for multi-dimensional regression head
            if 'num_output' in self.info_to_save:
                model.config.vhead_num_output = self.info_to_save['num_output']
            if self.use_lora:
                state_dict = get_trainable_weights(model.base_model.model)
                model.base_model.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
                model.peft_config['default'].base_model_name_or_path = self.info_to_save['base_model']
                model.peft_config['default'].save_pretrained(output_dir)
            else: # for full training and deepspeed
                state_dict = get_trainable_weights(model)
                model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(output_dir)
