# Original DDP training (unbalanced memory - GPU 0 uses more)
# python scripts/train.py \
#     --max_samples 10000 \
#     --streaming \
#     --per_device_train_batch_size 50 \
#     --wandb_entity LHR_attention \
#     --wandb_project qwen3-fineweb-training

# RECOMMENDED: Use DeepSpeed ZeRO-3 for balanced memory across all 8 GPUs
deepspeed scripts/train.py \
    --max_samples 10000 \
    --streaming \
    --per_device_train_batch_size 50 \
    --deepspeed ds_config_zero3.json \
    --wandb_entity LHR_attention \
    --wandb_project qwen3-fineweb-training