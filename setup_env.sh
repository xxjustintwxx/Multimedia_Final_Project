#!/bin/bash
# Create the 'multimedia' conda environment with all required packages.

set -e

CONDA_BIN="/home/xxjustin77xx/miniconda3/bin/conda"

echo "=== Creating multimedia env ==="
$CONDA_BIN create -n multimedia python=3.11 -y

# Activate
source /home/xxjustin77xx/miniconda3/etc/profile.d/conda.sh
conda activate multimedia

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing MediaPipe and data tools ==="
pip install \
    mediapipe \
    numpy \
    pillow \
    pandas \
    pyarrow \
    huggingface_hub \
    datasets \
    tqdm \
    opencv-python-headless

echo "=== Verifying installs ==="
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import datasets; print('datasets', datasets.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

echo "=== multimedia env ready ==="
