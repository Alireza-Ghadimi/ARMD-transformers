# ARMD Transformers Training Setup Guide

This guide explains how to set up and run the Hugging Face Transformers training environment on your HPC cluster using **Lmod modules** (without Docker, conda, or uv).

## System Requirements

- **Python**: 3.10+ (Python 3.12 available at `/usr/bin/python3.12`)
- **PyTorch**: 2.4+ (PyTorch 2.10.0 available via Lmod module)
- **GPU**: AMD GPUs with ROCm support (ROCm 7.1.0)
- **Modules**: Lmod environment modules system

## Quick Start

### 1. Initial Setup (One-time only)

Run the setup script to create the training environment:

```bash
cd /home1/aghadimi/ARMD-transformers
bash setup_environment.sh
```

This script will:
- Load required Lmod modules (PyTorch 2.10.0, GCC 12.2.0, OpenMPI 4.1.8)
- Create a Python 3.12 virtual environment
- Install PyTorch with ROCm support
- Install Transformers in editable mode
- Install training dependencies (datasets, evaluate, wandb, etc.)

**Estimated time**: 10-15 minutes

### 2. Activate Environment (For each new session)

```bash
cd /home1/aghadimi/ARMD-transformers
source activate_env.sh
```

This will load all required modules and activate the virtual environment.

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

## Training on SLURM

### Interactive Training (Development/Testing)

1. Request an interactive session:
```bash
srun --nodes=1 --gres=gpu:1 --time=2:00:00 --pty bash
```

2. Activate environment:
```bash
cd /home1/aghadimi/ARMD-transformers
source activate_env.sh
```

3. Run your training script:
```bash
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --output_dir ./outputs/test_run \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

### Batch Training (Production)

1. Edit the SLURM job script:
```bash
nano train_slurm.sh
```

Customize:
- `#SBATCH --gres=gpu:N` - Number of GPUs
- `#SBATCH --time=HH:MM:SS` - Job duration
- `MODEL_NAME` - Your model
- `DATASET_NAME` - Your dataset
- Training hyperparameters

2. Create logs directory:
```bash
mkdir -p logs
```

3. Submit the job:
```bash
sbatch train_slurm.sh
```

4. Monitor the job:
```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/train_<JOB_ID>.out

# View errors
tail -f logs/train_<JOB_ID>.err
```

## Multi-GPU Training

### Data Parallel Training (Multiple GPUs, Single Node)

Use `torchrun` for distributed training:

```bash
torchrun --nproc_per_node=4 \
    examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name wikitext \
    --output_dir ./outputs/multi_gpu_run \
    --do_train \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

### DeepSpeed Training (Advanced)

For larger models with ZeRO optimization:

```bash
deepspeed --num_gpus=4 \
    examples/pytorch/language-modeling/run_clm.py \
    --deepspeed examples/pytorch/language-modeling/ds_config.json \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name wikitext \
    --output_dir ./outputs/deepspeed_run \
    --do_train \
    --per_device_train_batch_size 4
```

## Environment Details

### Loaded Modules
- `pytorch/2.10.0` - PyTorch with ROCm support
- `gnu12/12.2.0` - GCC compiler
- `openmpi4/4.1.8` - MPI library for distributed training
- `cmake/3.25.2` - Build tools

### Environment Variables
- `TRANSFORMERS_ROOT` - Repository root directory
- `PYTHONPATH` - Includes `src/` for editable install
- `ROCM_HOME` - ROCm installation path
- `HF_HOME` - Hugging Face cache directory
- `TRANSFORMERS_CACHE` - Model cache directory

### Virtual Environment
- Location: `ARMD-transformers/venv/`
- Python: 3.12
- Packages: PyTorch, Transformers, datasets, evaluate, wandb, peft, etc.

## Troubleshooting

### Issue: "Module not found: transformers"
**Solution**: Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/home1/aghadimi/ARMD-transformers/src:$PYTHONPATH
```

### Issue: "CUDA not available"
**Solution**: Check GPU allocation:
```bash
rocm-smi
echo $HIP_VISIBLE_DEVICES
```

### Issue: "Out of memory"
**Solution**: Reduce batch size or enable gradient checkpointing:
```python
--per_device_train_batch_size 2
--gradient_accumulation_steps 8
--gradient_checkpointing
```

### Issue: "Module load failed"
**Solution**: Check available modules:
```bash
module avail pytorch
module spider pytorch/2.10.0
```

## Useful Commands

### Check cluster resources
```bash
sinfo                    # Show partition info
squeue -u $USER          # Show your jobs
scancel <JOB_ID>        # Cancel a job
scontrol show job <ID>  # Job details
```

### Monitor GPUs
```bash
rocm-smi                # AMD GPU status
watch -n 1 rocm-smi     # Monitor in real-time
```

### Check logs
```bash
# Training progress
tail -f outputs/run_*/logs/events.out.tfevents.*

# Tensorboard
tensorboard --logdir outputs/run_*/logs --bind_all
```

## Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Training Examples](./examples/pytorch/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Support

For issues specific to this setup, check:
1. SLURM logs in `logs/` directory
2. Training outputs in `outputs/` directory
3. Cache directories in `.cache/huggingface/`

---

**Last Updated**: 2026-04-24
**Repository**: https://github.com/huggingface/transformers
