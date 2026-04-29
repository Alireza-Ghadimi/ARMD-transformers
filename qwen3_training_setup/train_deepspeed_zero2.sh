#!/bin/bash
# ============================================================================
# DeepSpeed ZeRO-2 Training Script (Alternative - Faster but less memory efficient)
# ============================================================================
# ZeRO-2 only distributes gradients and optimizer states (not model parameters)
# Use this if ZeRO-3 is too slow, but you still want better memory balance than DDP.
# ============================================================================

# Activate environment (if not already activated)
if [ -z "$VIRTUAL_ENV" ]; then
    cd /home1/aghadimi/ARMD-transformers
    source activate_env.sh
    cd qwen3_training_setup
fi

echo "============================================================================"
echo "Starting DeepSpeed ZeRO-2 Training on 8 GPUs"
echo "============================================================================"
echo ""

# Run with DeepSpeed ZeRO-2
deepspeed scripts/train.py \
    --max_samples 10000 \
    --streaming \
    --per_device_train_batch_size 50 \
    --gradient_accumulation_steps 1 \
    --deepspeed ds_config_zero2.json \
    --wandb_entity LHR_attention \
    --wandb_project qwen3-fineweb-training \
    --bf16

echo ""
echo "============================================================================"
echo "Training completed!"
echo "============================================================================"
