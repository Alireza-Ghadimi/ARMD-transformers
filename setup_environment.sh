#!/bin/bash
# ============================================================================
# Environment Setup Script for Transformers Training on HPC with Lmod
# ============================================================================
# This script sets up the training environment for Hugging Face Transformers
# on an HPC cluster using Lmod modules instead of conda/docker/uv.
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Setting up Transformers Training Environment with Lmod"
echo "============================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: Load Required Modules
# ============================================================================
echo -e "${GREEN}Step 1: Loading required Lmod modules...${NC}"

# Load PyTorch module (includes ROCm for AMD GPUs)
module load pytorch/2.10.0

# Load additional required modules
module load gnu12/12.2.0
module load openmpi4/4.1.8
module load cmake/3.25.2

echo "Loaded modules:"
module list

# ============================================================================
# Step 2: Set Environment Variables
# ============================================================================
echo -e "${GREEN}Step 2: Setting environment variables...${NC}"

export TRANSFORMERS_ROOT="/home1/aghadimi/ARMD-transformers"
export VENV_PATH="${TRANSFORMERS_ROOT}/venv"
export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${PYTHONPATH}"

# ROCm specific settings for AMD GPUs
export ROCM_HOME="/opt/rocm-7.1.0"
export HIP_PLATFORM="amd"

# Hugging Face cache directories
export HF_HOME="${TRANSFORMERS_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Create cache directories
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "Environment variables set:"
echo "  TRANSFORMERS_ROOT: ${TRANSFORMERS_ROOT}"
echo "  VENV_PATH: ${VENV_PATH}"
echo "  ROCM_HOME: ${ROCM_HOME}"
echo "  HF_HOME: ${HF_HOME}"

# ============================================================================
# Step 3: Create Virtual Environment with Python 3.12
# ============================================================================
echo -e "${GREEN}Step 3: Creating Python virtual environment...${NC}"

# Use Python 3.12 (required for transformers >= 5.0)
PYTHON_BIN="/usr/bin/python3.12"

if [ ! -f "${PYTHON_BIN}" ]; then
    echo -e "${RED}Error: Python 3.12 not found at ${PYTHON_BIN}${NC}"
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} --version)"

# Remove old venv if it exists
if [ -d "${VENV_PATH}" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf "${VENV_PATH}"
fi

# Create new virtual environment
${PYTHON_BIN} -m venv "${VENV_PATH}"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

echo "Virtual environment created and activated at: ${VENV_PATH}"
echo "Python in venv: $(which python)"
echo "Python version: $(python --version)"

# ============================================================================
# Step 4: Upgrade pip and Install Build Tools
# ============================================================================
echo -e "${GREEN}Step 4: Upgrading pip and installing build tools...${NC}"

pip install --upgrade pip setuptools wheel

# ============================================================================
# Step 5: Install PyTorch (matching the module version)
# ============================================================================
echo -e "${GREEN}Step 5: Installing PyTorch with ROCm support...${NC}"

# Install PyTorch 2.10+ with ROCm support
# Note: The pytorch module provides the libraries, but we need the Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# ============================================================================
# Step 6: Install Transformers in Editable Mode
# ============================================================================
echo -e "${GREEN}Step 6: Installing Transformers in editable mode...${NC}"

cd "${TRANSFORMERS_ROOT}"

# Install core dependencies first
pip install packaging numpy regex requests tqdm filelock

# Install transformers with essential extras for training
pip install -e ".[torch,accelerate,deepspeed,sentencepiece,tokenizers,sklearn]"

# ============================================================================
# Step 7: Install Additional Training Dependencies
# ============================================================================
echo -e "${GREEN}Step 7: Installing additional training dependencies...${NC}"

# Install common training tools
pip install \
    datasets \
    evaluate \
    wandb \
    tensorboard \
    scikit-learn \
    scipy \
    safetensors

# Install PEFT for parameter-efficient fine-tuning
pip install peft

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}============================================================================${NC}"

# ============================================================================
# Step 8: Verify Installation
# ============================================================================
echo -e "${GREEN}Step 8: Verifying installation...${NC}"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Setup Instructions:${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "To activate this environment in future sessions, run:"
echo ""
echo -e "${YELLOW}  source ${TRANSFORMERS_ROOT}/activate_env.sh${NC}"
echo ""
echo "Or manually:"
echo ""
echo -e "${YELLOW}  module load pytorch/2.10.0 gnu12/12.2.0 openmpi4/4.1.8${NC}"
echo -e "${YELLOW}  source ${VENV_PATH}/bin/activate${NC}"
echo -e "${YELLOW}  export PYTHONPATH=${TRANSFORMERS_ROOT}/src:\$PYTHONPATH${NC}"
echo ""
echo -e "${GREEN}============================================================================${NC}"
