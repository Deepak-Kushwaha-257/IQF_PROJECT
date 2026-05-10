#!/bin/bash
# ============================================================
# setup.sh — One-click environment setup for IQF project
# ============================================================

set -e

echo "============================================"
echo " IQF Project Environment Setup"
echo "============================================"

# 1. Create conda environment
echo "[1/5] Creating conda environment 'iqf_env' with Python 3.10..."
conda create -n iqf_env python=3.10 -y || echo "Env already exists, skipping."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iqf_env

# 2. Install PyTorch (auto-detects CUDA)
echo "[2/5] Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected — installing CUDA version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116
else
    echo "  No GPU detected — installing CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 3. Install TensorFlow
echo "[3/5] Installing TensorFlow..."
if command -v nvidia-smi &> /dev/null; then
    pip install tensorflow==2.10.0
else
    pip install tensorflow-cpu==2.10.0
fi

# 4. Install remaining requirements
echo "[4/5] Installing Python packages..."
pip install -r requirements.txt

# 5. Try to install signatory (path signature library for SIG model)
echo "[5/5] Attempting to install signatory (SIG model)..."
pip install signatory --no-binary signatory 2>/dev/null || \
    echo "  WARNING: signatory install failed. SIG model will be skipped."
    echo "  Manual fix: pip install signatory --no-binary signatory"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate with: conda activate iqf_env"
echo " Then:  python scripts/download_fred_data.py"
echo "        python run_experiment.py --models phs garch_t --dataset usdyc2"
echo "============================================"
