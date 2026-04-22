#!/bin/bash
# Example training commands for Qwen3-0.6B

# ============================================================================
# IMPORTANT: Run these commands after:
#   1. uv sync (to install dependencies including wandb)
#   2. wandb login (to authenticate with Weights & Biases)
#   3. Allocating GPU resources (if using SLURM)
# ============================================================================

# ----------------------------------------------------------------------------
# Example 1: Full fine-tuning on fineweb-edu (default settings)
# ----------------------------------------------------------------------------
# This uses the HuggingFace fineweb-edu dataset by default
# Logs to wandb automatically
python scripts/train.py \
    --max_samples 10000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/qwen3_fineweb

# ----------------------------------------------------------------------------
# Example 2: Quick test run (10 steps only)
# ----------------------------------------------------------------------------
python scripts/train.py \
    --max_steps 10 \
    --max_samples 100 \
    --output_dir ./outputs/test_run

# ----------------------------------------------------------------------------
# Example 3: Full training on fineweb-edu with streaming
# ----------------------------------------------------------------------------
# For very large datasets, use streaming mode
python scripts/train.py \
    --streaming \
    --max_samples 100000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --save_steps 1000 \
    --output_dir ./outputs/qwen3_fineweb_large

# ----------------------------------------------------------------------------
# Example 4: Memory-efficient training (for GPUs with limited VRAM)
# ----------------------------------------------------------------------------
python scripts/train.py \
    --max_samples 10000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_length 256 \
    --output_dir ./outputs/qwen3_memory_efficient

# ----------------------------------------------------------------------------
# Example 5: Use a different HuggingFace dataset
# ----------------------------------------------------------------------------
python scripts/train.py \
    --dataset_path "wikitext" \
    --dataset_split "train" \
    --text_field "text" \
    --max_samples 5000 \
    --num_train_epochs 3 \
    --output_dir ./outputs/qwen3_wikitext

# ----------------------------------------------------------------------------
# Example 6: Use local dataset file
# ----------------------------------------------------------------------------
python scripts/train.py \
    --dataset_path "data/train.txt" \
    --num_train_epochs 3 \
    --output_dir ./outputs/qwen3_custom

# ----------------------------------------------------------------------------
# Example 7: Disable wandb and use tensorboard instead
# ----------------------------------------------------------------------------
python scripts/train.py \
    --report_to tensorboard \
    --max_samples 10000 \
    --output_dir ./outputs/qwen3_tensorboard

# ----------------------------------------------------------------------------
# Monitoring Training
# ----------------------------------------------------------------------------
# For wandb: Visit https://wandb.ai to see live training metrics
# For tensorboard: Run in another terminal:
#   tensorboard --logdir outputs/qwen3_tensorboard/runs

# ----------------------------------------------------------------------------
# Notes:
# ----------------------------------------------------------------------------
# - The model always loads Qwen/Qwen3-0.6B pretrained weights
# - All 609M parameters are trained (full fine-tuning)
# - Default dataset is HuggingFaceFW/fineweb-edu
# - Default logging is to wandb (run 'wandb login' first)
# - Adjust batch size and gradient accumulation based on GPU memory
# - Use --max_samples to limit dataset size for testing
# - Use --streaming for very large datasets
