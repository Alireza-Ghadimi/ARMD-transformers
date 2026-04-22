# Qwen3-0.6B Training Setup - Quick Start Guide

## Step-by-Step Setup (Run these commands yourself)

### 1. Navigate to Project Directory

```bash
cd /home/ghadimi/projects/aip-aghodsib/ghadimi/ARMD-transformers/qwen3_training_setup
```

### 2. Create Virtual Environment with UV

```bash
# Create virtual environment
uv venv .venv

# Activate environment
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all dependencies (including PyTorch, Transformers, etc.)
uv pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 4. Allocate GPU Resources (SLURM)

If you're using SLURM, allocate a GPU:

```bash
# Navigate to parent directory for salloc script
cd /home/ghadimi/projects/aip-aghodsib/ghadimi/ARMD-transformers

# Allocate GPU
salloc --account=aip-aghodsib --time=01:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1

# After allocation, return to project and activate environment
cd qwen3_training_setup
source .venv/bin/activate
```

### 5. Run Smoke Test

```bash
# Test that everything works
python scripts/inference_test.py
```

Expected output:
- ✓ Tokenizer loads
- ✓ Model loads with pretrained weights
- ✓ Forward pass works
- ✓ Text generation works

### 6. Create Sample Training Data (Optional)

```bash
# Create sample data for testing
python -c "
from src.data_utils import create_sample_dataset
create_sample_dataset('data/sample_train.jsonl', num_samples=100)
"
```

Or create your own training data in `data/` directory:
- Text file: `data/train.txt`
- JSON file: `data/train.json` (with "text" field)
- JSONL file: `data/train.jsonl` (one JSON per line)

### 7. Run Training

#### Option A: Standard Full Fine-tuning

```bash
python scripts/train.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path data/sample_train.jsonl \
    --output_dir ./outputs/qwen3_finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --create_sample_data
```

#### Option B: Training with Layer Surgery

```bash
python scripts/train_with_surgery.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path data/sample_train.jsonl \
    --output_dir ./outputs/qwen3_modified \
    --replace_layers 12,13,14 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --create_sample_data
```

### 8. Monitor Training

While training runs, in another terminal:

```bash
# View logs with TensorBoard
cd /home/ghadimi/projects/aip-aghodsib/ghadimi/ARMD-transformers/qwen3_training_setup
source .venv/bin/activate
tensorboard --logdir outputs/qwen3_finetuned/runs
```

Then open the URL shown in your browser.

## Troubleshooting

### Issue: CUDA out of memory

**Solutions:**
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Enable gradient checkpointing
--gradient_checkpointing

# Reduce sequence length
--max_length 256

# Combine all three
python scripts/train.py \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --max_length 256 \
    --gradient_accumulation_steps 4
```

### Issue: Tokenizer/Model download fails

**Solutions:**
- Check internet connection
- Ensure Hugging Face Hub is accessible
- Try: `export HF_HOME=/path/to/cache` to change cache location

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
source .venv/bin/activate
uv pip install -e . --force-reinstall
```

## Memory Requirements

- **Inference (bf16)**: ~2-3 GB GPU memory
- **Training (batch size 1)**: ~6-8 GB GPU memory
- **Training (batch size 4)**: ~12-16 GB GPU memory
- **Training (batch size 8)**: ~20-24 GB GPU memory

Enable gradient checkpointing to reduce memory by ~30-40%.

## Next Steps After Training

### 1. Load Your Trained Model

```python
from transformers import Qwen3ForCausalLM, AutoTokenizer

model = Qwen3ForCausalLM.from_pretrained("./outputs/qwen3_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./outputs/qwen3_finetuned")

# Generate text
inputs = tokenizer("Hello,", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 2. Experiment with Layer Modifications

Edit `src/model_wrapper.py` to implement custom layers:
- Custom attention mechanisms
- Modified MLP architectures
- Additional residual connections
- Different normalization schemes

### 3. Scale to Multi-GPU

```bash
# Use accelerate for multi-GPU training
accelerate config  # Configure once
accelerate launch scripts/train.py [args...]
```

## File Structure Reference

```
qwen3_training_setup/
├── pyproject.toml              # Dependencies (uv-compatible)
├── README.md                   # Main documentation
├── SETUP.md                    # This file
├── src/
│   ├── __init__.py
│   ├── model_wrapper.py        # Layer surgery utilities
│   ├── data_utils.py           # Dataset loading
│   └── training_utils.py       # Training helpers
├── scripts/
│   ├── inference_test.py       # Smoke test
│   ├── train.py                # Full fine-tuning
│   └── train_with_surgery.py   # Training with layer mods
├── configs/
│   ├── training_config.yaml    # Training hyperparameters
│   └── surgery_config.yaml     # Layer modification settings
├── data/                       # Your datasets
├── outputs/                    # Training outputs
└── .venv/                      # Virtual environment (after setup)
```

## Important Notes

⚠️ **This setup uses the official Transformers Qwen3 implementation** - no custom reimplementation.

⚠️ **When you modify layers**, pretrained weights are loaded for unchanged layers only. Modified layers use random initialization unless architecturally compatible.

⚠️ **Always run the smoke test first** to verify your setup before training.

## Getting Help

- Check error messages carefully - they usually indicate the issue
- Review the smoke test output to identify setup problems
- Ensure GPU is allocated before running GPU operations
- Verify all paths are correct (datasets, output directories)

## Quick Command Reference

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# Test
python scripts/inference_test.py

# Train (standard)
python scripts/train.py --create_sample_data --max_steps 10

# Train (with surgery)
python scripts/train_with_surgery.py --replace_layers 14 --max_steps 10

# View logs
tensorboard --logdir outputs/qwen3_finetuned/runs

# Check GPU
nvidia-smi
```
