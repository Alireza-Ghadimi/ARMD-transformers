#!/bin/bash
# ============================================================================
# DeepSpeed ZeRO-3 Training Script
# ============================================================================
# This script uses DeepSpeed ZeRO-3 to balance memory across all 8 GPUs
# 
# ZeRO-3 distributes:
#   - Model parameters
#   - Gradients  
#   - Optimizer states
# Across all GPUs, achieving near-perfect memory balance!
#
# Usage:
#   bash train_deepspeed.sh
# ============================================================================

# Activate environment (if not already activated)
if [ -z "$VIRTUAL_ENV" ]; then
    cd /home1/aghadimi/ARMD-transformers
    source activate_env.sh
    cd qwen3_training_setup
fi

echo "============================================================================"
echo "Starting DeepSpeed ZeRO-3 Training on 8 GPUs"
echo "============================================================================"
echo ""

# Run with DeepSpeed (automatically detects all available GPUs)
deepspeed scripts/train.py \
    --max_samples 10000 \
    --streaming \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --deepspeed ds_config_zero3.json \
    --wandb_entity LHR_attention \
    --wandb_project qwen3-fineweb-training \
    --bf16

# Note: DeepSpeed automatically uses all visible GPUs
# To use specific GPUs, set: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# or use: deepspeed --num_gpus=8 ...

echo ""
echo "============================================================================"
echo "Training completed!"
echo "============================================================================"
