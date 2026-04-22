#!/bin/bash
# GPU allocation script for Qwen3-0.6B deployment
# Adjust --time as needed for your workflow

salloc --account=aip-aghodsib --time=02:30:00 --cpus-per-task=8 --mem=32G --gres=gpu:1
source ~/projects/aip-aghodsib/ghadimi/ARMD-transformers/qwen3_training_setup/.venv/bin/activate
module load cuda12.8/toolkit/12.8.1
module load StdEnv/2023  

# Set HuggingFace cache to scratch directory (unlimited space)
export HF_HOME=/scratch/ghadimi/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/scratch/ghadimi/huggingface_cache/hub
export TRANSFORMERS_CACHE=/scratch/ghadimi/huggingface_cache/transformers

# Create cache directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $TRANSFORMERS_CACHE

echo "HuggingFace cache set to: $HF_HOME"

# After allocation, activate environment:
# source ~/projects/aip-aghodsib/ghadimi/QWen3/qwen3_0p6b/.venv/bin/activate