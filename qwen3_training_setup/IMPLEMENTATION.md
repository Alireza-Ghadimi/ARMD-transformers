# Qwen3-0.6B Training Setup - Implementation Summary

## Overview

This is a clean, research-friendly training setup based on the **official Hugging Face Transformers Qwen3 implementation**. It provides full fine-tuning capabilities with optional layer surgery for architecture experiments.

## Key Design Principles

1. **Use Official Implementation**: No custom Qwen3 reimplementation - builds directly on `transformers.models.qwen3.modeling_qwen3`
2. **Pretrained Weight Loading**: Maintains compatibility with `Qwen/Qwen3-0.6B` pretrained weights
3. **Research-Friendly**: Easy to inspect, modify, and extend
4. **Minimal Dependencies**: Only essential packages
5. **Clear Separation**: Model, data, training utilities in separate modules

## Architecture Details

### Official Qwen3 Classes Used

From `transformers.models.qwen3.modeling_qwen3`:

```python
Qwen3ForCausalLM           # Main model class
├── model: Qwen3Model      # Base transformer
│   ├── embed_tokens       # Token embeddings
│   ├── layers: ModuleList # 28 x Qwen3DecoderLayer
│   │   └── Qwen3DecoderLayer
│   │       ├── self_attn: Qwen3Attention
│   │       ├── mlp: Qwen3MLP
│   │       ├── input_layernorm: Qwen3RMSNorm
│   │       └── post_attention_layernorm: Qwen3RMSNorm
│   ├── norm: Qwen3RMSNorm # Final layer norm
│   └── rotary_emb         # Rotary position embeddings
└── lm_head: Linear        # Output projection
```

### Qwen3-0.6B Configuration

- **Layers**: 28 decoder layers
- **Hidden size**: 1024
- **Intermediate size**: 2816
- **Attention heads**: 16
- **KV heads**: 16 (no GQA)
- **Vocab size**: 151,936
- **Max context**: 8,192 tokens
- **Parameters**: ~609M

## Component Breakdown

### 1. Model Wrapper (`src/model_wrapper.py`)

**Purpose**: Provides utilities for optional layer surgery

**Key Functions**:
- `replace_decoder_layers()`: Replace specific layers with custom implementations
- `CustomQwen3DecoderLayer`: Example custom layer extending official implementation
- `selective_load_pretrained()`: Load weights for unchanged layers only
- `get_layer_info()`: Inspect model structure
- `count_parameters()`: Analyze parameter distribution
- `freeze_layers()`: Freeze specific layers for selective training

**Hook Points for Modifications**:
```python
# Example: Replace attention mechanism
class MyCustomLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Replace attention
        self.self_attn = MyCustomAttention(config, layer_idx)
    
# Example: Modify MLP
class MyCustomLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Replace MLP
        self.mlp = MyCustomMLP(config)
```

### 2. Data Utilities (`src/data_utils.py`)

**Purpose**: Handle dataset loading and preprocessing

**Key Components**:
- `TextDataset`: PyTorch Dataset for causal language modeling
- `load_text_dataset()`: Unified loader for .txt, .json, .jsonl, and HF datasets
- `create_sample_dataset()`: Generate sample data for testing
- `DataCollatorForCausalLM`: Handles batching and padding

**Supported Data Formats**:
1. Plain text (`.txt`): Splits on double newlines
2. JSON (`.json`): Expects `{"text": "..."}` format
3. JSONL (`.jsonl`): One JSON object per line
4. Hugging Face datasets: Any dataset name from Hub

### 3. Training Utilities (`src/training_utils.py`)

**Purpose**: Training configuration and helpers

**Key Components**:
- `Qwen3TrainingConfig`: Dataclass with sensible defaults
- `setup_model_for_training()`: Prepare model (gradient checkpointing, cache, etc.)
- `print_training_info()`: Display training configuration
- `save_training_summary()`: Record training details
- `PerformanceCallback`: Track training performance
- `estimate_memory_usage()`: Predict GPU memory requirements

### 4. Scripts

#### a) `scripts/inference_test.py`
Smoke test that verifies:
1. Tokenizer loads
2. Model loads with pretrained weights
3. Forward pass works
4. Text generation works

Run this FIRST to verify setup.

#### b) `scripts/train.py`
Standard full fine-tuning script.

**Usage**:
```bash
python scripts/train.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path data/train.txt \
    --output_dir ./outputs/qwen3_finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5
```

**Features**:
- Automatic tokenizer handling
- Sample data creation
- Progress tracking
- Checkpoint saving
- TensorBoard/WandB logging

#### c) `scripts/train_with_surgery.py`
Training with optional layer modifications.

**Usage**:
```bash
python scripts/train_with_surgery.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path data/train.txt \
    --replace_layers 12,13,14 \
    --freeze_unchanged_layers \
    --output_dir ./outputs/qwen3_modified
```

**Features**:
- Selective layer replacement
- Optional freezing of unchanged layers
- Compatible weight loading
- Modified structure inspection

## Where to Modify for Experiments

### 1. Change Attention Mechanism

Edit `src/model_wrapper.py`:

```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

class MyAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Your modifications here
        
    def forward(self, *args, **kwargs):
        # Your custom attention logic
        pass

class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = MyAttention(config, layer_idx)
```

### 2. Change MLP Architecture

```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

class MyMLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        # Your modifications here
        
    def forward(self, x):
        # Your custom MLP logic
        pass

class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = MyMLP(config)
```

### 3. Modify Full Decoder Layer

```python
class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Add extra components
        self.extra_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, hidden_states, *args, **kwargs):
        # Custom forward logic
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Your modifications here
        hidden_states = self.extra_layer(hidden_states)
        
        # Standard attention
        hidden_states, _ = self.self_attn(hidden_states, *args, **kwargs)
        hidden_states = residual + hidden_states
        
        # Standard MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

### 4. Add Hooks for Analysis

```python
# In your training script
def attention_hook(module, input, output):
    # Analyze attention patterns
    print(f"Attention output shape: {output[0].shape}")

# Attach hook to specific layer
model.model.layers[14].self_attn.register_forward_hook(attention_hook)
```

## Weight Loading Strategy

### Standard Training
- All pretrained weights loaded
- Full model training
- Compatible with official checkpoints

### Layer Surgery
1. **Load base model**: `Qwen3ForCausalLM.from_pretrained()`
2. **Replace layers**: Use `replace_decoder_layers()`
3. **Weight handling**:
   - Unchanged layers: Keep pretrained weights
   - Modified layers: 
     - Try to copy compatible weights
     - Fall back to random initialization
4. **Optional**: Freeze unchanged layers for focused training

## Memory Optimization

### Gradient Checkpointing
Enable with `--gradient_checkpointing`
- Saves ~30-40% memory
- Increases training time ~20%
- Essential for limited VRAM

### Batch Size Tuning
Effective batch size = `per_device_batch_size × gradient_accumulation_steps × num_gpus`

**Recommendations**:
```python
# 8GB GPU
per_device_train_batch_size=1
gradient_accumulation_steps=4
gradient_checkpointing=True

# 16GB GPU
per_device_train_batch_size=4
gradient_accumulation_steps=1

# 24GB+ GPU
per_device_train_batch_size=8
gradient_accumulation_steps=1
```

## Training Configurations

### Quick Test (Verify Setup)
```bash
python scripts/train.py \
    --max_steps 10 \
    --create_sample_data
```

### Development (Small Dataset)
```bash
python scripts/train.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --logging_steps 10
```

### Production (Large Dataset)
```bash
python scripts/train.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_steps 1000 \
    --report_to wandb
```

### Layer Probing
```bash
python scripts/train_with_surgery.py \
    --replace_layers 14 \
    --freeze_unchanged_layers \
    --num_train_epochs 5
```

## Extending the Setup

### Add Custom Loss Function

Edit `scripts/train.py`:

```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Your custom loss
        loss = my_custom_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Use CustomTrainer instead of Trainer
trainer = CustomTrainer(...)
```

### Add Evaluation

```python
# In train.py, add eval_dataset
eval_dataset = load_text_dataset("data/eval.txt", tokenizer, max_length)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add this
    data_collator=data_collator,
)
```

### Add Custom Metrics

```python
from datasets import load_metric

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Your metrics
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    ...,
    compute_metrics=compute_metrics,
)
```

## Comparison with Alternatives

### vs LoRA (PEFT)
- **This setup**: Full fine-tuning, all parameters updated
- **LoRA**: Only adapters trained, base model frozen
- **When to use this**: 
  - You have sufficient compute
  - Want maximum performance
  - Need to modify architecture

### vs Custom Implementation
- **This setup**: Uses official Transformers code
- **Custom**: Reimplemented from scratch
- **Advantages**:
  - Pretrained weight compatibility
  - Ecosystem integration (vLLM, TGI, etc.)
  - Maintained and tested

### vs Training Frameworks (Axolotl, etc.)
- **This setup**: Direct control, minimal abstraction
- **Frameworks**: High-level, many features
- **When to use this**:
  - Research experiments
  - Architecture modifications
  - Understanding internals

## Best Practices

1. **Always run smoke test first**: `python scripts/inference_test.py`
2. **Start with small data**: Verify pipeline before scaling
3. **Monitor GPU memory**: Use `nvidia-smi` during training
4. **Save often**: Set reasonable `save_steps`
5. **Log to TensorBoard/WandB**: Track experiments
6. **Document changes**: Update training_summary.txt
7. **Version control**: Commit after successful runs
8. **Test layer surgery on one layer first**: Before replacing many

## Troubleshooting Guide

See [CHECKLIST.md](CHECKLIST.md) for detailed troubleshooting steps.

Common issues and fixes:
- **CUDA OOM**: Reduce batch size, enable gradient checkpointing
- **Slow training**: Check dataloader workers, use mixed precision
- **Poor convergence**: Adjust learning rate, check data quality
- **Import errors**: Reinstall with `uv pip install -e .`

## References

- **Qwen3 Model Card**: https://huggingface.co/Qwen/Qwen3-0.6B
- **Qwen3 Docs**: https://huggingface.co/docs/transformers/model_doc/qwen3
- **Source Code**: `src/transformers/models/qwen3/modeling_qwen3.py`
- **Transformers Docs**: https://huggingface.co/docs/transformers/

## Summary

This setup provides:
✅ Official Transformers Qwen3 implementation
✅ Full fine-tuning support
✅ Optional layer surgery for experiments
✅ Pretrained weight compatibility
✅ Research-friendly code structure
✅ Comprehensive documentation

**You're ready to start training! Follow [SETUP.md](SETUP.md) for step-by-step instructions.**
