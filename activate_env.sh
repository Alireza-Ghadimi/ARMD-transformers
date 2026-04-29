#!/bin/bash
# ============================================================================
# Quick Activation Script for Transformers Training Environment
# ============================================================================
# Source this script to activate the training environment:
#   source activate_env.sh
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Activating Transformers training environment...${NC}"

# Load required modules
module load pytorch/2.10.0
module load gnu12/12.2.0
module load openmpi4/4.1.8
module load cmake/3.25.2

# Set environment variables
export TRANSFORMERS_ROOT="/home1/aghadimi/ARMD-transformers"
export VENV_PATH="${TRANSFORMERS_ROOT}/venv"
export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${PYTHONPATH}"

# ROCm settings
export ROCM_HOME="/opt/rocm-7.1.0"
export HIP_PLATFORM="amd"

# Hugging Face cache
export HF_HOME="${TRANSFORMERS_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Activate virtual environment
if [ -d "${VENV_PATH}" ]; then
    source "${VENV_PATH}/bin/activate"
    echo -e "${GREEN}Environment activated!${NC}"
    echo ""
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
    echo ""
    echo "Loaded modules:"
    module list 2>&1 | grep -E "(pytorch|gnu|openmpi|cmake)"
else
    echo -e "${YELLOW}Warning: Virtual environment not found at ${VENV_PATH}${NC}"
    echo "Please run: bash setup_environment.sh"
fi
