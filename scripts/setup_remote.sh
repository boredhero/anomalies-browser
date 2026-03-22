#!/bin/bash
# Setup script for 192.168.1.111 (Arch Linux + RX 6900 XT)
# Run via: ssh noah@192.168.1.111 "bash -s" < scripts/setup_remote.sh

set -euo pipefail
echo "=== Magic Eyes Remote Setup ==="
echo "Target: $(hostname) — $(uname -r)"

# --- ROCm Installation ---
echo ""
echo "=== Installing ROCm ==="
sudo pacman -S --needed --noconfirm \
    rocm-hip-runtime \
    rocm-opencl-runtime \
    rocm-smi-lib \
    hip-runtime-amd \
    rocminfo

# Add user to video and render groups
sudo usermod -aG video,render noah

# Verify ROCm
echo ""
echo "=== ROCm Verification ==="
rocminfo 2>/dev/null | grep -E "gfx|Name" | head -5 || echo "WARNING: rocminfo failed"

# --- Python / uv ---
echo ""
echo "=== Installing uv + Python 3.12 ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

# --- Clone/sync repo ---
echo ""
echo "=== Setting up project ==="
if [ ! -d "$HOME/magic-eyes" ]; then
    git clone git@github.com:boredhero/anomalies-browser.git "$HOME/magic-eyes"
fi
cd "$HOME/magic-eyes"
git pull

# --- Install Python deps with GPU extras ---
echo ""
echo "=== Installing Python dependencies (with GPU) ==="
# Set ROCm environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0

uv sync --extra dev
# GPU deps from ROCm PyTorch index
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
uv pip install ultralytics segmentation-models-pytorch

# --- Verify PyTorch + ROCm ---
echo ""
echo "=== PyTorch ROCm Verification ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- Data directory ---
echo ""
echo "=== Creating data directories ==="
sudo mkdir -p /data/magic-eyes/{raw,processed,models,ground_truth}
sudo chown -R noah:noah /data/magic-eyes

# --- Environment file ---
if [ ! -f "$HOME/magic-eyes/.env" ]; then
    cp "$HOME/magic-eyes/.env.example" "$HOME/magic-eyes/.env"
    echo "Created .env from example — edit passwords!"
fi

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Edit ~/.magic-eyes/.env with real passwords"
echo "  2. Start services: cd ~/magic-eyes-docker && docker compose up -d"
echo "  3. Run migrations: cd ~/magic-eyes && uv run alembic upgrade head"
echo "  4. Start workers: HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run celery -A magic_eyes.workers.celery_app worker -Q default,ingest,process,detect,gpu"
