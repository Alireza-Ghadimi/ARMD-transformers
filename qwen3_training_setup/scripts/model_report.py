import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
)

from data_utils import load_text_dataset, DataCollatorForCausalLM, create_sample_dataset
from training_utils import (
    Qwen3TrainingConfig,
    setup_model_for_training,
    print_training_info,
    save_training_summary,
    PerformanceCallback,
)
from model_wrapper import (
    replace_decoder_layers,
    CustomQwen3DecoderLayer,
    get_layer_info,
    freeze_layers,
    count_parameters,
)


if __name__ == "__main__":
    print("="*80)
    print("QWEN3 MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    print()
    
    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading model: {model_name}")
    
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    
    print(f"✓ Model loaded successfully\n")
    
    # Print overall model structure
    print("="*80)
    print("MODEL STRUCTURE OVERVIEW")
    print("="*80)
    print(f"\nModel class: {model.__class__.__name__}")
    print(f"Config: {model.config.model_type}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Number of KV heads: {model.config.num_key_value_heads}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Max position embeddings: {model.config.max_position_embeddings}")
    
    # Print parameter counts
    print("\n" + "="*80)
    print("PARAMETER COUNTS")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Print layer-by-layer architecture
    print("\n" + "="*80)
    print("LAYER-BY-LAYER ARCHITECTURE")
    print("="*80)
    
    # Embedding layer
    print("\n[EMBEDDING LAYER]")
    print(f"  embed_tokens: {model.model.embed_tokens}")
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    print(f"  Parameters: {embed_params:,}")
    
    # Decoder layers
    print("\n[DECODER LAYERS]")
    for i, layer in enumerate(model.model.layers):
        print(f"\n  Layer {i}:")
        print(f"    Type: {layer.__class__.__name__}")
        
        # Self attention
        if hasattr(layer, 'self_attn'):
            print(f"    Self Attention:")
            print(f"      q_proj: {layer.self_attn.q_proj}")
            print(f"      k_proj: {layer.self_attn.k_proj}")
            print(f"      v_proj: {layer.self_attn.v_proj}")
            print(f"      o_proj: {layer.self_attn.o_proj}")
            attn_params = sum(p.numel() for p in layer.self_attn.parameters())
            print(f"      Attention parameters: {attn_params:,}")
        
        # MLP
        if hasattr(layer, 'mlp'):
            print(f"    MLP:")
            print(f"      gate_proj: {layer.mlp.gate_proj}")
            print(f"      up_proj: {layer.mlp.up_proj}")
            print(f"      down_proj: {layer.mlp.down_proj}")
            mlp_params = sum(p.numel() for p in layer.mlp.parameters())
            print(f"      MLP parameters: {mlp_params:,}")
        
        # Layer norms
        if hasattr(layer, 'input_layernorm'):
            print(f"    input_layernorm: {layer.input_layernorm}")
        if hasattr(layer, 'post_attention_layernorm'):
            print(f"    post_attention_layernorm: {layer.post_attention_layernorm}")
        
        # Total layer parameters
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"    Total layer parameters: {layer_params:,}")
    
    # Final layer norm
    print("\n[FINAL LAYER NORM]")
    print(f"  norm: {model.model.norm}")
    norm_params = sum(p.numel() for p in model.model.norm.parameters())
    print(f"  Parameters: {norm_params:,}")
    
    # LM head
    print("\n[LANGUAGE MODEL HEAD]")
    print(f"  lm_head: {model.lm_head}")
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  Parameters: {lm_head_params:,}")
    
    # Summary by component
    print("\n" + "="*80)
    print("PARAMETER SUMMARY BY COMPONENT")
    print("="*80)
    
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    decoder_params = sum(p.numel() for p in model.model.layers.parameters())
    norm_params = sum(p.numel() for p in model.model.norm.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    print(f"\nEmbedding layer: {embed_params:,} ({100*embed_params/total_params:.2f}%)")
    print(f"Decoder layers: {decoder_params:,} ({100*decoder_params/total_params:.2f}%)")
    print(f"Final layer norm: {norm_params:,} ({100*norm_params/total_params:.2f}%)")
    print(f"LM head: {lm_head_params:,} ({100*lm_head_params/total_params:.2f}%)")
    print(f"\nTotal: {total_params:,}")
    
    # Print memory footprint
    print("\n" + "="*80)
    print("MEMORY FOOTPRINT")
    print("="*80)
    
    param_size_bytes = total_params * 2  # bf16 = 2 bytes per parameter
    param_size_mb = param_size_bytes / (1024 ** 2)
    param_size_gb = param_size_bytes / (1024 ** 3)
    
    print(f"\nModel size (bf16): {param_size_mb:.2f} MB ({param_size_gb:.4f} GB)")
    print(f"Model size (fp32): {param_size_mb * 2:.2f} MB ({param_size_gb * 2:.4f} GB)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
