#!/usr/bin/env bash
# Create .venv and install all dependencies with GPU (CUDA 12.4) support.
# Requires: Python 3.13, nvidia driver >=525, ffmpeg

set -e
cd "$(dirname "$0")"

echo "=== TTS Story Pipeline — Setup ==="

# 1. Create venv
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/3] .venv already exists, skipping creation."
fi

PIP=".venv/bin/pip"

# 2. Install GPU torch (CUDA 12.4 — works with driver >=525 / CUDA compat >=12.4)
echo "[2/3] Installing PyTorch (CUDA 12.4)..."
"$PIP" install --upgrade pip --quiet
"$PIP" install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# For CPU-only (no GPU), replace above with:
#   "$PIP" install torch torchvision torchaudio \
#       --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining packages
#    transformers must stay <5.0 — coqui-tts uses GPT2PreTrainedModel removed in 5.x
echo "[3/3] Installing coqui-tts, gradio, pydub, transformers..."
"$PIP" install \
    "coqui-tts>=0.27.0" \
    "gradio>=6.0.0" \
    "pydub>=0.25.1" \
    "transformers>=4.40.0,<5.0.0"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate and run:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo "  → http://localhost:7860"
echo ""
echo "NOTE: First launch downloads XTTS-v2 (~1.8 GB) to ~/.local/share/tts/."
