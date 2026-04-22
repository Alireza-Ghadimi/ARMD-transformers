# Quick Reference Card

## Setup (One-Time)
```bash
cd qwen3_training_setup
uv venv .venv && source .venv/bin/activate
uv sync
wandb login
```

## Common Commands

### Run Training (Default: fineweb-edu + wandb)
```bash
python scripts/train.py --max_samples 10000
```

### Test Setup
```bash
python scripts/inference_test.py
```

### Allocate GPU (SLURM)
```bash
salloc --account=aip-aghodsib --time=01:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1
```

## Useful Training Flags

```bash
# Limit dataset size
--max_samples 10000

# Use local data
--dataset_path data/train.txt

# Use different HF dataset
--dataset_path "wikitext" --dataset_split "train"

# Change logging
--report_to tensorboard  # or "wandb" (default) or "none"

# Memory optimization
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
--gradient_checkpointing

# Training control
--num_train_epochs 3
--learning_rate 2e-5
--max_steps 1000

# Output
--output_dir ./outputs/my_run

# Streaming (for large datasets)
--streaming
```

## Quick Training Templates

### Test Run (Fast)
```bash
python scripts/train.py --max_samples 100 --max_steps 10
```

### Small Experiment
```bash
python scripts/train.py --max_samples 10000
```

### Full Training
```bash
python scripts/train.py --max_samples 100000 --num_train_epochs 1
```

### Memory-Efficient
```bash
python scripts/train.py \
    --max_samples 10000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing
```

### Use Tensorboard
```bash
python scripts/train.py --report_to tensorboard --max_samples 10000
# Then: tensorboard --logdir outputs/qwen3_finetuned/runs
```

## wandb Quick Tips

```bash
# Silent mode (less output)
export WANDB_SILENT=true

# Offline mode
export WANDB_MODE=offline

# Custom project name
export WANDB_PROJECT=my_qwen3_project

# Custom run name
export WANDB_NAME=experiment_01

# Disable wandb entirely
export WANDB_DISABLED=true
```

## Model Info

- **Model**: Qwen/Qwen3-0.6B
- **Parameters**: 609M (all trainable)
- **Layers**: 28 decoder layers
- **Context**: 32768 tokens max
- **Training**: Full fine-tuning (not LoRA)

## Default Settings

- **Dataset**: HuggingFaceFW/fineweb-edu
- **Logging**: wandb
- **Batch size**: 4
- **Learning rate**: 2e-5
- **Sequence length**: 512
- **Optimizer**: AdamW

## Files to Check

- `README.md` - Main documentation
- `SETUP_COMPLETE.md` - Getting started guide
- `WANDB_SETUP.md` - wandb configuration
- `example_train_commands.sh` - Example commands
- `CHECKLIST.md` - Pre-training checklist

## Monitoring

- **wandb**: https://wandb.ai
- **Tensorboard**: `tensorboard --logdir outputs/*/runs`
- **Logs**: Check `output_dir/logs/`

## Emergency: Out of Memory

```bash
# Try in order:
--per_device_train_batch_size 1
--gradient_checkpointing
--gradient_accumulation_steps 4
--max_length 256
```

## Help

- See `example_train_commands.sh` for more examples
- Check WANDB_SETUP.md for wandb issues
- Run with `--max_steps 10` to test quickly
