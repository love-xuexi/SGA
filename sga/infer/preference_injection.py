"""
Preference Injection Module for Multi-Preference Language Model

Implements three methods to inject preference weights into the language model:
1. Text-based injection: Format weights as text tokens
2. Embedding-based injection: Map weights to embedding space
3. Condition-based injection: Use weights as conditional input
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer


class TextBasedInjection:
    """
    Format preference weights as text and append to the input prompt.
    Requires two forward passes: one to get weights, another to generate.
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        preference_names: List[str],
        format_style: str = 'angle_brackets'
    ):
        """
        Args:
            tokenizer: The tokenizer to use
            preference_names: Names of preference dimensions (e.g., ['helpful', 'harmless', 'honest'])
            format_style: 'angle_brackets' -> "<helpful:0.879>" or 'plain' -> "helpful: 0.879"
        """
        self.tokenizer = tokenizer
        self.preference_names = preference_names
        self.format_style = format_style
    
    def format_preferences(self, preference_scores: torch.Tensor) -> str:
        """
        Format preference weights as text string.
        
        Args:
            preference_scores: Tensor of shape (num_preferences,)
        
        Returns:
            Formatted string, e.g., "<helpful:0.879> <harmless:-0.53> <honest:0.234>"
        """
        texts = []
        for name, weight in zip(self.preference_names, preference_scores):
            if self.format_style == 'angle_brackets':
                texts.append(f"<{name}_score:{weight:.3f}>")
            elif self.format_style == 'angle_brackets_outer':
                texts.append(f"<{name}_score> {weight:.3f}")
            elif self.format_style == 'plain':
                texts.append(f"{name}_score: {weight:.3f}")
            else:
                raise ValueError(f"Unknown format_style: {self.format_style}")
        pref_scores = " and ".join(texts)
        return f"Your response must satisfy the following scores: {pref_scores}"
        # return " ".join(texts)
    
    def inject_to_prompt(
        self, 
        input_text: str, 
        preference_scores: torch.Tensor,
        position: str = 'end'
    ) -> str:
        """
        Inject preference weights into the input text.
        
        Args:
            input_text: Original input text
            preference_scores: Tensor of shape (num_preferences,)
            position: 'end' or 'start' - where to add preferences
        
        Returns:
            Modified input text with preferences
        """
        pref_text = self.format_preferences(preference_scores)
        
        if position == 'end':
            return f"{input_text} {pref_text}"
        elif position == 'start':
            return f"{pref_text} {input_text}"
        else:
            raise ValueError(f"Unknown position: {position}")
    
    def tokenize_with_preferences(
        self,
        input_text: str,
        preference_scores: torch.Tensor,
        max_length: int = 512,
        position: str = 'end'
    ) -> dict:
        """
        Tokenize input text with injected preferences.
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        modified_text = self.inject_to_prompt(input_text, preference_scores, position)
        return self.tokenizer(
            modified_text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )


class EmbeddingBasedInjection(nn.Module):
    """
    Map preference weights to embedding space and combine with input embeddings.
    Requires single forward pass with modified embeddings.
    """
    
    def __init__(
        self, 
        num_preferences: int,
        embedding_dim: int,
        projection_type: str = 'linear',
        combination_mode: str = 'concat'
    ):
        """
        Args:
            num_preferences: Number of preference dimensions
            embedding_dim: Dimension of token embeddings
            projection_type: 'linear' or 'mlp' - how to project preferences
            combination_mode: 'concat' or 'add' - how to combine with input embeddings
        """
        super().__init__()
        self.num_preferences = num_preferences
        self.embedding_dim = embedding_dim
        self.projection_type = projection_type
        self.combination_mode = combination_mode
        
        # Create projection layer
        if projection_type == 'linear':
            self.projection = nn.Linear(num_preferences, embedding_dim)
        elif projection_type == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(num_preferences, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")
    
    def forward(
        self, 
        input_embeds: torch.Tensor,
        preference_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inject preference weights into input embeddings.
        
        Args:
            input_embeds: Tensor of shape (batch_size, seq_len, embedding_dim)
            preference_scores: Tensor of shape (batch_size, num_preferences)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
        
        Returns:
            Tuple of (modified_embeds, modified_attention_mask)
        """
        batch_size = input_embeds.size(0)
        
        # Project preference weights to embedding space
        pref_embeds = self.projection(preference_scores)  # (batch_size, embedding_dim)
        pref_embeds = pref_embeds.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        if self.combination_mode == 'concat':
            # Concatenate preference embeddings to the input
            modified_embeds = torch.cat([input_embeds, pref_embeds], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                pref_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
                modified_mask = torch.cat([attention_mask, pref_mask], dim=1)
                return modified_embeds, modified_mask
            else:
                return modified_embeds, None
                
        elif self.combination_mode == 'add':
            # Add preference embedding to the last token embedding
            modified_embeds = input_embeds.clone()
            if attention_mask is not None:
                # Find the last valid position for each sequence
                last_positions = attention_mask.sum(dim=1) - 1
                for i in range(batch_size):
                    modified_embeds[i, last_positions[i]] += pref_embeds[i, 0]
            else:
                modified_embeds[:, -1] += pref_embeds[:, 0]
            
            return modified_embeds, attention_mask
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")


class ConditionBasedInjection(nn.Module):
    """
    Use preference weights as conditional input for generation.
    Most efficient - single forward pass with condition vector.
    """
    
    def __init__(
        self,
        num_preferences: int,
        hidden_size: int,
        num_layers: Optional[int] = None,
        conditioning_method: str = 'prepend'
    ):
        """
        Args:
            num_preferences: Number of preference dimensions
            hidden_size: Hidden size of the model
            num_layers: Number of transformer layers (for per-layer conditioning)
            conditioning_method: 'prepend', 'cross_attention', or 'film'
        """
        super().__init__()
        self.num_preferences = num_preferences
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conditioning_method = conditioning_method
        
        if conditioning_method == 'prepend':
            # Simple projection to hidden dimension
            self.projection = nn.Linear(num_preferences, hidden_size)
            
        elif conditioning_method == 'film':
            # FiLM (Feature-wise Linear Modulation)
            # Generate scale and shift parameters for each layer
            self.film_generator = nn.ModuleList([
                nn.Linear(num_preferences, hidden_size * 2)
                for _ in range(num_layers or 1)
            ])
        else:
            raise ValueError(f"Conditioning method {conditioning_method} not implemented")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        preference_scores: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply conditioning based on preference weights.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            preference_scores: Tensor of shape (batch_size, num_preferences)
            layer_idx: Layer index (for per-layer conditioning)
        
        Returns:
            Conditioned hidden states
        """
        if self.conditioning_method == 'prepend':
            # Project preferences and prepend to hidden states
            pref_hidden = self.projection(preference_scores)  # (batch_size, hidden_size)
            pref_hidden = pref_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
            return torch.cat([pref_hidden, hidden_states], dim=1)
            
        elif self.conditioning_method == 'film':
            # Apply FiLM conditioning
            if layer_idx is None:
                layer_idx = 0
            
            film_params = self.film_generator[layer_idx](preference_scores)
            scale, shift = torch.chunk(film_params, 2, dim=-1)
            scale = scale.unsqueeze(1)  # (batch_size, 1, hidden_size)
            shift = shift.unsqueeze(1)  # (batch_size, 1, hidden_size)
            
            return hidden_states * (1 + scale) + shift
        
        return hidden_states


def create_injector(
    injection_type: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    preference_names: Optional[List[str]] = None,
    num_preferences: int = 3,
    embedding_dim: int = 768,
    hidden_size: int = 768,
    **kwargs
):
    """
    Factory function to create the appropriate injector.
    
    Args:
        injection_type: 'text', 'embedding', or 'condition'
        tokenizer: Required for text-based injection
        preference_names: Required for text-based injection
        num_preferences: Number of preference dimensions
        embedding_dim: Embedding dimension (for embedding-based)
        hidden_size: Hidden size (for condition-based)
        **kwargs: Additional arguments for specific injectors
    
    Returns:
        Appropriate injector instance
    """
    if injection_type == 'text':
        if tokenizer is None or preference_names is None:
            raise ValueError("Text-based injection requires tokenizer and preference_names")
        return TextBasedInjection(tokenizer, preference_names, **kwargs)
    
    elif injection_type == 'embedding':
        return EmbeddingBasedInjection(num_preferences, embedding_dim, **kwargs)
    
    elif injection_type == 'condition':
        return ConditionBasedInjection(num_preferences, hidden_size, **kwargs)
    
    else:
        raise ValueError(f"Unknown injection_type: {injection_type}")

