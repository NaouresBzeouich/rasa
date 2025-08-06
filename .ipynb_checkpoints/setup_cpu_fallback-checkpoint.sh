#!/bin/bash
# CPU-only fallback setup (if bitsandbytes fails)
echo "Setting up CPU-only version without quantization..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

echo "CPU setup complete. Use use_quantization: false in your config."
