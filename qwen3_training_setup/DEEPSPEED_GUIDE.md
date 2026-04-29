# DeepSpeed Training Guide for Qwen3

## GPU Memory Imbalance Problem

When running standard DDP training across 8 GPUs, you experienced:
- **GPU 0 (rank 0)**: High memory usage (~18-22GB)
- **GPU 1-7**: Lower memory usage (~12-15GB)

This happens because GPU 0 handles extra responsibilities: gradient aggregation, optimizer states, model checkpointing, and logging.

## Solution: DeepSpeed ZeRO

DeepSpeed ZeRO (Zero Redundancy Optimizer) distributes model components across all GPUs to balance memory usage perfectly.

### Stage Comparison

| Stage | What's Distributed | Memory Balance | Speed | Best For |
|-------|-------------------|----------------|-------|----------|
| **DDP** (current) | Nothing - full model on each GPU | ❌ Unbalanced | ⚡⚡⚡ Fast | Small models |
| **ZeRO-1** | Optimizer states | 🟡 Better | ⚡⚡⚡ Fast | Medium models |
| **ZeRO-2** | Optimizer + Gradients | 🟢 Good | ⚡⚡ Good | Medium-large models |
| **ZeRO-3** | Optimizer + Gradients + Parameters | ✅ Perfect | ⚡ Slower | Large models, memory-constrained |

## Quick Start

### Option 1: DeepSpeed ZeRO-3 (Recommended - Balanced Memory)

```bash
cd /home1/aghadimi/ARMD-transformers/qwen3_training_setup
bash train_deepspeed.sh
```

**Expected Memory Usage (8 GPUs):**
- All GPUs: ~13-15GB each (balanced!)
- Allows batch size 50+ per GPU

**Config file:** `ds_config_zero3.json`

### Option 2: DeepSpeed ZeRO-2 (Faster Alternative)

```bash
bash train_deepspeed_zero2.sh
```

**Expected Memory Usage:**
- GPU 0: ~16-18GB
- GPU 1-7: ~13-15GB each (better than DDP)

**Config file:** `ds_config_zero2.json`

### Option 3: Original DDP Training

```bash
# Comment out deepspeed line in train.sh and uncomment python line
python scripts/train.py \
    --max_samples 10000 \
    --streaming \
    --per_device_train_batch_size 32 \
    --gradient_checkpointing \
    --wandb_entity LHR_attention \
    --wandb_project qwen3-fineweb-training
```

## DeepSpeed Configuration Files

### ds_config_zero3.json
```json
{
  "zero_optimization": {
    "stage": 3,  // Full parameter sharding
    "overlap_comm": true,  // Overlap communication with computation
    "contiguous_gradients": true,  // Reduce memory fragmentation
    "stage3_max_live_parameters": 1e9,  // Parameters to keep in GPU memory
    "stage3_gather_16bit_weights_on_model_save": true  // Save in full precision
  }
}
```

### ds_config_zero2.json
```json
{
  "zero_optimization": {
    "stage": 2,  // Gradient + optimizer sharding only
    "overlap_comm": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8
  }
}
```

## Command Line Usage

### Basic DeepSpeed Command
```bash
deepspeed scripts/train.py \
    --deepspeed ds_config_zero3.json \
    --per_device_train_batch_size 50 \
    [other arguments...]
```

### Specify GPU Count
```bash
deepspeed --num_gpus=8 scripts/train.py \
    --deepspeed ds_config_zero3.json \
    [other arguments...]
```

### Use Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed scripts/train.py \
    --deepspeed ds_config_zero3.json \
    [other arguments...]
```

## Monitoring GPU Memory

### During Training
```bash
# Terminal 1: Run training
bash train_deepspeed.sh

# Terminal 2: Monitor GPUs (update every 1 second)
watch -n 1 rocm-smi

# Or get memory details
rocm-smi --showmeminfo vram
```

### Check Process Usage
```bash
# See which process is using which GPU
rocm-smi --showpids

# See detailed memory usage
rocm-smi --showmeminfo all
```

## Troubleshooting

### Issue: "No module named 'deepspeed'"
```bash
source /home1/aghadimi/ARMD-transformers/activate_env.sh
pip install deepspeed
```

### Issue: Still getting OOM on GPU 0
**Solutions:**
1. Reduce batch size: `--per_device_train_batch_size 32`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Increase gradient accumulation: `--gradient_accumulation_steps 2`

### Issue: Training slower with ZeRO-3
This is expected! ZeRO-3 trades speed for memory efficiency.
- **ZeRO-3**: ~10-20% slower, perfect memory balance
- **ZeRO-2**: ~5-10% slower, good memory balance
- **DDP**: Fastest, unbalanced memory

For Qwen3-0.6B on 8 GPUs, **ZeRO-2 is often the sweet spot**.

### Issue: DeepSpeed launcher not found
Make sure you're using the deepspeed launcher, not python directly:
```bash
# ✅ Correct
deepspeed scripts/train.py --deepspeed ds_config_zero3.json

# ❌ Wrong
python scripts/train.py --deepspeed ds_config_zero3.json
```

## Performance Comparison

### Effective Batch Sizes

| Config | Per-GPU BS | Grad Accum | GPUs | **Total BS** | Speed |
|--------|-----------|------------|------|-------------|-------|
| DDP | 50 | 1 | 8 | **400** | 100% |
| ZeRO-2 | 50 | 1 | 8 | **400** | ~95% |
| ZeRO-3 | 50 | 1 | 8 | **400** | ~85% |
| ZeRO-3 + Grad Ckpt | 64 | 1 | 8 | **512** | ~80% |

### Memory Usage (Qwen3-0.6B)

| Config | GPU 0 | GPU 1-7 | Imbalance |
|--------|-------|---------|-----------|
| DDP | 20GB | 14GB | ❌ 6GB |
| ZeRO-2 | 16GB | 14GB | 🟡 2GB |
| ZeRO-3 | 14GB | 14GB | ✅ 0GB |

## Recommended Settings

### For Your Current Setup (8 GPUs, Qwen3-0.6B)

**Best Performance:**
```bash
deepspeed scripts/train.py \
    --deepspeed ds_config_zero2.json \
    --per_device_train_batch_size 50 \
    --gradient_accumulation_steps 1 \
    --bf16
```

**Best Memory Balance:**
```bash
deepspeed scripts/train.py \
    --deepspeed ds_config_zero3.json \
    --per_device_train_batch_size 64 \
    --gradient_checkpointing \
    --bf16
```

**Conservative (Safe):**
```bash
deepspeed scripts/train.py \
    --deepspeed ds_config_zero2.json \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --bf16
```

## Advanced Configuration

### Enable CPU Offloading (for extremely large models)
Edit `ds_config_zero3.json`:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**Warning:** This is slower but allows much larger models. Not needed for Qwen3-0.6B.

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [Transformers + DeepSpeed](https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed)

---

**Quick Test Command:**
```bash
# SSH to your allocated server
ssh k002-001

# Activate environment
cd /home1/aghadimi/ARMD-transformers
source activate_env.sh

# Go to training directory
cd qwen3_training_setup

# Run with DeepSpeed ZeRO-3
bash train_deepspeed.sh
```

Monitor with: `watch -n 1 rocm-smi`
