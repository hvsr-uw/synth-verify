#!/bin/bash
# Create the LoRA training env on the login node (CPU only, no srun).
# Run once: bash setup_lora_env.sh
set -x
python3 -m venv /data/hvardhan2/lora-env
/data/hvardhan2/lora-env/bin/pip install --upgrade pip
/data/hvardhan2/lora-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
/data/hvardhan2/lora-env/bin/pip install peft transformers datasets accelerate
/data/hvardhan2/lora-env/bin/python -c "import torch, peft, transformers; print('env OK, torch', torch.__version__)"
echo "=== LORA ENV DONE ==="
