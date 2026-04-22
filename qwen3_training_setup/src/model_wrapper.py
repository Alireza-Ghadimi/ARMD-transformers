"""
Model wrapper utilities for Qwen3 with optional layer surgery.

This module provides utilities to modify the official Transformers Qwen3 model
while maintaining maximum compatibility with pretrained weights.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
)


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    """
    Example custom decoder layer that extends the official Qwen3DecoderLayer.
    
    You can modify this class to implement your own architectural changes:
    - Different attention mechanisms
    - Modified MLP architectures
    - Additional residual connections
    - Custom normalization schemes
    
    This example keeps the same structure but adds a marker for identification.
    """
    
    def __init__(self, config: Qwen3Config, layer_idx: int, custom_tag: str = "modified"):
        super().__init__(config, layer_idx)
        self.custom_tag = custom_tag
        
        # Example: You could replace components here
        # self.mlp = YourCustomMLP(config)
        # self.self_attn = YourCustomAttention(config, layer_idx)
    
    def forward(self, *args, **kwargs):
        # Call parent forward - modify if needed
        return super().forward(*args, **kwargs)


def replace_decoder_layers(
    model: Qwen3ForCausalLM,
    layer_indices: List[int],
    replacement_layer_class: Optional[type] = None,
    custom_kwargs: Optional[Dict[str, Any]] = None,
) -> Qwen3ForCausalLM:
    """
    Replace specific decoder layers in a Qwen3 model.
    
    Args:
        model: The Qwen3ForCausalLM model to modify
        layer_indices: List of layer indices to replace (0-indexed)
        replacement_layer_class: Custom layer class (defaults to CustomQwen3DecoderLayer)
        custom_kwargs: Additional kwargs to pass to the replacement layer constructor
    
    Returns:
        Modified model with replaced layers
    
    Example:
        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> # Replace middle layers 12, 13, 14
        >>> model = replace_decoder_layers(model, [12, 13, 14])
    """
    if replacement_layer_class is None:
        replacement_layer_class = CustomQwen3DecoderLayer
    
    if custom_kwargs is None:
        custom_kwargs = {}
    
    config = model.config
    num_layers = config.num_hidden_layers
    
    # Validate indices
    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers-1}]")
    
    print(f"Replacing layers at indices: {layer_indices}")
    print(f"Using replacement class: {replacement_layer_class.__name__}")
    
    # Replace specified layers
    for idx in layer_indices:
        old_layer = model.model.layers[idx]
        
        # Create new layer with same config
        new_layer = replacement_layer_class(
            config=config,
            layer_idx=idx,
            **custom_kwargs
        )
        
        # Optionally copy weights from old layer to new layer
        # This only works if the architectures are compatible
        try:
            new_layer.load_state_dict(old_layer.state_dict(), strict=False)
            print(f"  ✓ Layer {idx}: Copied compatible weights from pretrained layer")
        except Exception as e:
            print(f"  ⚠ Layer {idx}: Could not copy weights (expected for custom architectures): {e}")
            print(f"      New layer will use random initialization")
        
        # Replace in model
        model.model.layers[idx] = new_layer
    
    return model


def selective_load_pretrained(
    model: Qwen3ForCausalLM,
    pretrained_model_name: str,
    modified_layer_indices: List[int],
) -> Qwen3ForCausalLM:
    """
    Load pretrained weights selectively, skipping modified layers.
    
    This is useful when you've replaced some layers and want to load
    pretrained weights only for the unchanged layers.
    
    Args:
        model: Model with some modified layers
        pretrained_model_name: Name of pretrained model to load from
        modified_layer_indices: Indices of layers that were modified
    
    Returns:
        Model with selectively loaded pretrained weights
    """
    print(f"\nLoading pretrained weights from {pretrained_model_name}")
    print(f"Skipping layers: {modified_layer_indices}")
    
    # Load pretrained state dict
    pretrained_model = Qwen3ForCausalLM.from_pretrained(pretrained_model_name)
    pretrained_state = pretrained_model.state_dict()
    
    # Current model state
    current_state = model.state_dict()
    
    # Create filtered state dict (exclude modified layers)
    filtered_state = {}
    skipped_keys = []
    
    for key, value in pretrained_state.items():
        # Check if this key belongs to a modified layer
        skip = False
        for idx in modified_layer_indices:
            if f"model.layers.{idx}." in key:
                skip = True
                break
        
        if skip:
            skipped_keys.append(key)
        else:
            filtered_state[key] = value
    
    # Load filtered weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    
    print(f"\n✓ Loaded {len(filtered_state)} pretrained parameters")
    print(f"  Skipped {len(skipped_keys)} parameters from modified layers")
    if missing_keys:
        print(f"  Missing keys (random init): {len(missing_keys)}")
    
    return model


def get_layer_info(model: Qwen3ForCausalLM) -> None:
    """
    Print information about model layers for inspection.
    
    Useful for understanding the model structure and identifying
    which layers to modify.
    """
    config = model.config
    print(f"\n{'='*60}")
    print(f"Qwen3 Model Structure: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'custom'}")
    print(f"{'='*60}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of key-value heads: {config.num_key_value_heads}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    print(f"\nLayer types: {config.layer_types}")
    print(f"\nLayers:")
    
    for idx, layer in enumerate(model.model.layers):
        layer_type = type(layer).__name__
        is_custom = "Custom" in layer_type
        marker = " [MODIFIED]" if is_custom else ""
        print(f"  Layer {idx:2d}: {layer_type}{marker}")
    
    print(f"{'='*60}\n")


def count_parameters(model: Qwen3ForCausalLM, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters by component.
    
    Args:
        model: The model to analyze
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Dictionary with parameter counts by component
    """
    def count_params(module):
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())
    
    counts = {
        "total": count_params(model),
        "embeddings": count_params(model.model.embed_tokens),
        "layers": count_params(model.model.layers),
        "norm": count_params(model.model.norm),
        "lm_head": count_params(model.lm_head),
    }
    
    # Per-layer counts
    for idx, layer in enumerate(model.model.layers):
        counts[f"layer_{idx}"] = count_params(layer)
    
    return counts


def freeze_layers(model: Qwen3ForCausalLM, frozen_layer_indices: List[int]) -> Qwen3ForCausalLM:
    """
    Freeze specific layers (no gradient updates during training).
    
    Useful for training only modified layers or doing progressive unfreezing.
    
    Args:
        model: The model to modify
        frozen_layer_indices: List of layer indices to freeze
    
    Returns:
        Model with frozen layers
    """
    print(f"Freezing layers: {frozen_layer_indices}")
    
    for idx in frozen_layer_indices:
        for param in model.model.layers[idx].parameters():
            param.requires_grad = False
    
    # Report trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return model


def unfreeze_all_layers(model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
    """Unfreeze all layers in the model."""
    for param in model.parameters():
        param.requires_grad = True
    return model


# Example usage demonstration
if __name__ == "__main__":
    print("This module provides utilities for Qwen3 model surgery.")
    print("\nExample usage:")
    print("""
    from transformers import Qwen3ForCausalLM
    from model_wrapper import replace_decoder_layers, get_layer_info
    
    # Load model
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Inspect structure
    get_layer_info(model)
    
    # Replace middle layers
    model = replace_decoder_layers(model, [12, 13, 14])
    
    # Verify changes
    get_layer_info(model)
    """)
