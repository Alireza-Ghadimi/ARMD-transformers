# Setup Complete! 🎉

Your Qwen3-0.6B training setup is ready with all the requested features.

## What's New

### ✅ HuggingFace Dataset Integration
- **Default dataset**: `HuggingFaceFW/fineweb-edu` (high-quality educational text)
- No need to prepare custom data files to get started
- Still supports local files and other HuggingFace datasets
- Streaming support for very large datasets

### ✅ Weights & Biases (wandb) Logging
- **Default logging**: wandb (instead of tensorboard)
- Real-time training metrics at https://wandb.ai
- Automatic logging of loss, learning rate, GPU usage, and more
- Easy switching to tensorboard with `--report_to tensorboard`

### ✅ Pretrained Weight Loading Verification
- Automatic verification that Qwen/Qwen3-0.6B weights are loaded
- Parameter counting: 609M parameters (28 layers)
- Ensures all parameters are trainable (full fine-tuning mode)

### ✅ Enhanced Training Script
- `train.py` now supports:
  - HuggingFace dataset loading with `datasets.load_dataset()`
  - Streaming mode for large datasets
  - Sample limiting with `--max_samples`
  - Multiple dataset sources (local files, HF Hub)
  - Explicit trainability verification

## Quick Start Commands

```bash
# 1. Navigate to project
cd /home/ghadimi/projects/aip-aghodsib/ghadimi/ARMD-transformers/qwen3_training_setup

# 2. Create and activate environment
uv venv .venv
source .venv/bin/activate

# 3. Install dependencies (includes wandb)
uv sync

# 4. Login to Weights & Biases
wandb login
# Get your API key from: https://wandb.ai/authorize

# 5. (Optional) Allocate GPU if using SLURM
salloc --account=aip-aghodsib --time=01:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1

# 6. Run smoke test to verify setup
python scripts/inference_test.py

# 7. Start training (uses fineweb-edu by default)
python scripts/train.py --max_samples 10000

# 8. Monitor training at https://wandb.ai
```

## What Gets Logged to wandb

When you run training, wandb automatically tracks:
- ✅ Training loss (per step and epoch)
- ✅ Learning rate schedule
- ✅ Gradient norms
- ✅ GPU memory usage
- ✅ Training time and throughput
- ✅ Model hyperparameters
- ✅ System information (GPU model, CUDA version, etc.)

Visit https://wandb.ai to see real-time charts and metrics!

## Training Examples

### Example 1: Quick test (10K samples from fineweb-edu)
```bash
python scripts/train.py --max_samples 10000
```

### Example 2: Larger training run (100K samples)
```bash
python scripts/train.py \
    --max_samples 100000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_steps 1000 \
    --output_dir ./outputs/qwen3_fineweb_100k
```

### Example 3: Memory-efficient training
```bash
python scripts/train.py \
    --max_samples 10000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_length 256 \
    --output_dir ./outputs/qwen3_memory_efficient
```

### Example 4: Use tensorboard instead of wandb
```bash
python scripts/train.py \
    --report_to tensorboard \
    --max_samples 10000 \
    --output_dir ./outputs/qwen3_tensorboard

# In another terminal, view logs:
tensorboard --logdir outputs/qwen3_tensorboard/runs
```

### Example 5: Use a different HuggingFace dataset
```bash
python scripts/train.py \
    --dataset_path "wikitext" \
    --dataset_split "train" \
    --text_field "text" \
    --max_samples 5000 \
    --output_dir ./outputs/qwen3_wikitext
```

### Example 6: Use your own local data
```bash
# First, create your data file
echo "Your training text here." > data/train.txt

# Then train
python scripts/train.py \
    --dataset_path data/train.txt \
    --num_train_epochs 3 \
    --output_dir ./outputs/qwen3_custom
```

**More examples**: See [example_train_commands.sh](example_train_commands.sh)

## Files Created/Updated

### New Files
- **`WANDB_SETUP.md`**: Complete guide to using Weights & Biases
- **`example_train_commands.sh`**: Ready-to-run training examples
- **`SETUP_COMPLETE.md`**: This file

### Updated Files
- **`scripts/train.py`**:
  - Default dataset: `HuggingFaceFW/fineweb-edu`
  - Default logging: `wandb`
  - Added HuggingFace dataset loading
  - Added pretrained weight verification
  - Added full fine-tuning parameter check
  - Added wandb setup check and fallback
  - Enhanced status output

- **`README.md`**:
  - Updated features list
  - Added wandb login to setup instructions
  - Updated training examples
  - Added links to new documentation

- **`pyproject.toml`**:
  - Fixed hatchling build configuration (added `packages = ["src"]`)
  - All dependencies included (wandb, datasets, etc.)

## Verification Checklist

Before starting your first training run, verify:

- [ ] Environment created: `uv venv .venv`
- [ ] Dependencies installed: `uv sync`
- [ ] wandb login complete: `wandb login`
- [ ] GPU allocated (if using SLURM)
- [ ] Smoke test passed: `python scripts/inference_test.py`

## What the Training Script Does

When you run `python scripts/train.py`:

1. **Loads pretrained weights**: Downloads `Qwen/Qwen3-0.6B` from HuggingFace Hub
2. **Verifies full fine-tuning**: Checks all 609M parameters are trainable
3. **Loads dataset**: Uses `HuggingFaceFW/fineweb-edu` by default
4. **Prepares data**: Tokenizes and creates batches
5. **Initializes wandb**: Connects to your wandb project
6. **Starts training**: Full fine-tuning on all model parameters
7. **Logs metrics**: Sends real-time updates to wandb.ai
8. **Saves checkpoints**: Periodically saves model to `output_dir`

## Troubleshooting

### Issue: "wandb: ERROR API key not configured"
**Solution**: Run `wandb login` and enter your API key from https://wandb.ai/authorize

### Issue: Too much console output from wandb
**Solution**: Set environment variable: `export WANDB_SILENT=true`

### Issue: Want to run offline
**Solution**: Set `export WANDB_MODE=offline` before training

### Issue: Prefer tensorboard
**Solution**: Use `--report_to tensorboard` flag

### Issue: CUDA out of memory
**Solutions**:
- Reduce batch size: `--per_device_train_batch_size 1`
- Enable gradient checkpointing: `--gradient_checkpointing`
- Use gradient accumulation: `--gradient_accumulation_steps 4`
- Reduce sequence length: `--max_length 256`

### Issue: Dataset download is slow
**Solution**: Use streaming mode: `--streaming` (processes data without downloading entire dataset)

## Documentation

- **[README.md](README.md)**: Main project documentation
- **[SETUP.md](SETUP.md)**: Detailed setup instructions
- **[WANDB_SETUP.md](WANDB_SETUP.md)**: wandb configuration and best practices
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: Technical implementation details
- **[CHECKLIST.md](CHECKLIST.md)**: Pre-training verification checklist
- **[example_train_commands.sh](example_train_commands.sh)**: Training command examples

## Need Help?

1. Check the documentation files above
2. Review example commands in `example_train_commands.sh`
3. Test with small sample first: `--max_samples 100 --max_steps 10`
4. Check wandb logs at https://wandb.ai for training issues

## What to Do Next

1. **Run the setup commands** listed in "Quick Start Commands" above
2. **Test the setup** with `python scripts/inference_test.py`
3. **Start a test training run** with `python scripts/train.py --max_samples 100 --max_steps 10`
4. **Monitor on wandb** at https://wandb.ai
5. **Scale up** once everything works!

---

**Ready to start training!** 🚀

Run `python scripts/train.py --max_samples 10000` to begin.
