# Qwen3-0.6B Training Setup

Clean, research-friendly training setup based on the official Hugging Face Transformers Qwen3 implementation.

## Features

✅ Uses official `transformers` Qwen3 implementation (no custom reimplementation)
✅ Loads pretrained `Qwen/Qwen3-0.6B` weights
✅ Full fine-tuning support (not LoRA)
✅ **Default dataset**: HuggingFace's `fineweb-edu` (high-quality educational text)
✅ **Default logging**: Weights & Biases (wandb) with real-time tracking
✅ Optional layer modification mechanism for architecture experiments
✅ Simple, maintainable codebase

## Project Structure

```
qwen3_training_setup/
├── pyproject.toml           # Dependencies (uv-compatible)
├── README.md                # This file
├── src/
│   ├── model_wrapper.py     # Optional layer surgery utilities
│   ├── data_utils.py        # Dataset loading and preprocessing
│   └── training_utils.py    # Training helpers
├── scripts/
│   ├── inference_test.py    # Smoke test: load model + run inference
│   ├── train.py             # Full fine-tuning script
│   └── train_with_surgery.py # Training with optional layer modifications
├── configs/
│   ├── training_config.yaml # Default training hyperparameters
│   └── surgery_config.yaml  # Layer modification settings
└── data/
    └── (your datasets go here)
```

## Setup Instructions

### 1. Environment Setup (You need to run this on your server)

```bash
cd qwen3_training_setup

# Create virtual environment with uv
uv venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies (includes wandb)
uv sync

# Login to Weights & Biases (required for default logging)
wandb login
# Get your API key from: https://wandb.ai/authorize

# Optional: Install dev dependencies
uv pip install -e ".[dev]"
```

> **Note**: For detailed wandb setup instructions, see [WANDB_SETUP.md](WANDB_SETUP.md)

### 2. GPU Allocation (If using SLURM)

```bash
# Allocate GPU resources
salloc --account=aip-aghodsib --time=01:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1

# After allocation, activate environment
source .venv/bin/activate
```

### 3. Run Smoke Test

```bash
# Test that model loads and runs inference
python scripts/inference_test.py
```

Expected output:
- ✓ Tokenizer loads successfully
- ✓ Model loads with pretrained weights
- ✓ Forward pass works
- ✓ Text generation works

### 4. Prepare Your Dataset (Optional)

**Default**: The training script uses HuggingFace's `fineweb-edu` dataset by default - no preparation needed!

If you want to use your own data, place it in the `data/` directory. Supported formats:
- A text file (`.txt`)
- A JSON/JSONL file with a `"text"` field
- Any HuggingFace dataset name

Example:
```bash
echo "This is example training text." > data/train.txt
```

### 5. Run Training

#### Quick Start (Uses fineweb-edu by default)

```bash
# Train on 10K samples from fineweb-edu with wandb logging
python scripts/train.py --max_samples 10000

# Or use the example commands script
./example_train_commands.sh  # See file for more examples
```

#### Option A: Vanilla Full Fine-tuning

**Using fineweb-edu (default)**:
```bash
# Simple: train on 10K samples
python scripts/train.py --max_samples 10000

# Full training with custom hyperparameters
python scripts/train.py \
    --max_samples 100000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/qwen3_fineweb
```

**Using your own dataset**:
```bash
python scripts/train.py \
    --dataset_path data/train.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/qwen3_custom
```

**Using a different HuggingFace dataset**:
```bash
python scripts/train.py \
    --dataset_path "wikitext" \
    --dataset_split "train" \
    --text_field "text" \
    --max_samples 5000 \
    --output_dir ./outputs/qwen3_wikitext
```

**Disable wandb (use tensorboard instead)**:
```bash
python scripts/train.py \
    --report_to tensorboard \
    --max_samples 10000
```

> **More examples**: See [example_train_commands.sh](example_train_commands.sh) for additional use cases

#### Option B: Training with Layer Surgery

To modify specific layers (e.g., replace middle decoder blocks):

```bash
python scripts/train_with_surgery.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path data/train.txt \
    --output_dir ./outputs/qwen3_modified \
    --replace_layers 12,13,14 \
    --num_train_epochs 3
```

## Key Implementation Details

### Official Transformers Classes Used

From `transformers.models.qwen3.modeling_qwen3`:

- **`Qwen3ForCausalLM`**: Main model class with LM head
- **`Qwen3Model`**: Base transformer model
- **`Qwen3DecoderLayer`**: Individual transformer block
  - `self_attn`: `Qwen3Attention` 
  - `mlp`: `Qwen3MLP`
  - `input_layernorm`: `Qwen3RMSNorm`
  - `post_attention_layernorm`: `Qwen3RMSNorm`

### Where to Modify Layers

The model structure is:
```python
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens
│   ├── layers: ModuleList[Qwen3DecoderLayer]  # ← Modify these
│   ├── norm
│   └── rotary_emb
└── lm_head
```

To modify specific layers, see `src/model_wrapper.py` which provides:
- `replace_decoder_layers()`: Replace specific layer indices
- `CustomQwen3DecoderLayer`: Example custom layer implementation
- Selective weight loading from pretrained checkpoints

### Layer Modification Hook Points

You can modify:
1. **Full decoder layer**: Replace `Qwen3DecoderLayer` at index `i`
2. **Attention only**: Replace `layer.self_attn`
3. **MLP only**: Replace `layer.mlp`
4. **Add hooks**: Use PyTorch hooks on any submodule

See `scripts/train_with_surgery.py` for examples.

## Configuration Files

### Training Config (`configs/training_config.yaml`)

Controls training hyperparameters:
- Learning rate, batch size, epochs
- Gradient accumulation, mixed precision
- Optimizer settings
- Logging and checkpointing

### Surgery Config (`configs/surgery_config.yaml`)

Controls layer modifications:
- Which layers to replace
- Custom layer parameters
- Weight initialization strategy

## Testing Your Setup

```bash
# 1. Test inference
python scripts/inference_test.py

# 2. Test training (1 step)
python scripts/train.py --max_steps 1 --output_dir ./test_output

# 3. Test layer surgery
python scripts/train_with_surgery.py --max_steps 1 --replace_layers 14
```

## Important Notes

⚠️ **Pretrained Weight Loading**: When you modify layers, only compatible weights are loaded. Modified layers are randomly initialized.

⚠️ **Memory Usage**: Qwen3-0.6B requires ~2-3GB GPU memory for inference, ~8-12GB for training with batch size 4.

⚠️ **Gradient Checkpointing**: Enable with `--gradient_checkpointing` flag to reduce memory usage.

## Troubleshooting

**Issue**: CUDA out of memory
- Reduce batch size: `--per_device_train_batch_size 1`
- Enable gradient checkpointing: `--gradient_checkpointing`
- Use gradient accumulation: `--gradient_accumulation_steps 4`

**Issue**: Model not loading pretrained weights
- Check model name: `Qwen/Qwen3-0.6B` (case-sensitive)
- Ensure internet connection for downloading
- Check Hugging Face Hub access

**Issue**: Layer surgery breaks training
- Verify layer indices are valid (0 to num_layers-1)
- Check custom layer implementation matches expected interface
- Review weight loading logs for mismatches

## Next Steps

1. **Monitor training**: View real-time metrics at https://wandb.ai (or use tensorboard)
2. **Run experiments**: Try different layer modifications
3. **Add custom layers**: Implement your own `Qwen3DecoderLayer` variants
4. **Try different datasets**: Use `--dataset_path` to try wikitext, openwebtext, etc.
5. **Scale up**: Use multi-GPU with `accelerate launch`

## Additional Documentation

- **[WANDB_SETUP.md](WANDB_SETUP.md)**: Complete wandb configuration guide
- **[example_train_commands.sh](example_train_commands.sh)**: Ready-to-use training examples
- **[SETUP.md](SETUP.md)**: Detailed setup instructions
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: Technical implementation details

## References

- Qwen3 Model: https://huggingface.co/Qwen/Qwen3-0.6B
- Transformers Docs: https://huggingface.co/docs/transformers/model_doc/qwen3
- Qwen3 Implementation: `src/transformers/models/qwen3/modeling_qwen3.py`
