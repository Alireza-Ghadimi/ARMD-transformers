#!/bin/bash
#SBATCH --job-name=transformers_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================================
# SLURM Training Script for Hugging Face Transformers
# ============================================================================
# This is a template script for running transformer model training on SLURM
# with AMD GPUs (ROCm support).
#
# Usage:
#   sbatch train_slurm.sh
#
# Customize the SBATCH directives above based on your needs:
#   --nodes: Number of compute nodes
#   --gres=gpu:N: Number of GPUs per node
#   --time: Maximum job runtime
#   --partition: Queue/partition name (check with: sinfo)
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# Load Environment
# ============================================================================
echo "Loading modules and activating environment..."

# Load required modules
module load pytorch/2.10.0
module load gnu12/12.2.0
module load openmpi4/4.1.8

# Set environment variables
export TRANSFORMERS_ROOT="/home1/aghadimi/ARMD-transformers"
export VENV_PATH="${TRANSFORMERS_ROOT}/venv"
export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${PYTHONPATH}"

# ROCm settings
export ROCM_HOME="/opt/rocm-7.1.0"
export HIP_PLATFORM="amd"

# Hugging Face settings
export HF_HOME="${TRANSFORMERS_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# ============================================================================
# Training Configuration
# ============================================================================

# Model and data configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"  # Change to your model
DATASET_NAME="wikitext"               # Change to your dataset
OUTPUT_DIR="${TRANSFORMERS_ROOT}/outputs/run_${SLURM_JOB_ID}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TRANSFORMERS_ROOT}/logs"

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUM_STEPS=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_SEQ_LENGTH=512

# ============================================================================
# Display GPU Information
# ============================================================================
echo "============================================================================"
echo "GPU Information"
echo "============================================================================"
rocm-smi
echo "============================================================================"

# ============================================================================
# Run Training
# ============================================================================
echo "Starting training..."
echo "Output directory: ${OUTPUT_DIR}"

# Example training command - customize based on your needs
# This uses the Trainer API from transformers

python "${TRANSFORMERS_ROOT}/examples/pytorch/language-modeling/run_clm.py" \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUM_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --fp16 \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --load_best_model_at_end \
    --report_to "tensorboard" \
    --warmup_steps 500 \
    --weight_decay 0.01

# For multi-GPU training, use torchrun:
# torchrun --nproc_per_node=4 \
#     "${TRANSFORMERS_ROOT}/examples/pytorch/language-modeling/run_clm.py" \
#     [... same arguments as above ...]

# For DeepSpeed training:
# deepspeed --num_gpus=4 \
#     "${TRANSFORMERS_ROOT}/examples/pytorch/language-modeling/run_clm.py" \
#     --deepspeed "${TRANSFORMERS_ROOT}/examples/pytorch/language-modeling/ds_config.json" \
#     [... same arguments as above ...]

# ============================================================================
# Job Completion
# ============================================================================
echo "============================================================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "============================================================================"
