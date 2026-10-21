#!/bin/bash
# LoRA training v2: fails loudly if lora-env is missing.
set -x
export HF_HOME=/data/hvardhan2/hf-cache
cd /data/hvardhan2/jobs/synth
if [ ! -f /data/hvardhan2/lora-env/bin/activate ]; then
  echo "lora-env MISSING - run setup_lora_env.sh first"; exit 1
fi
source /data/hvardhan2/lora-env/bin/activate
python3 -c "import torch, peft; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
python3 synth_verify/train_lora.py --chunks synth_verify/logs/lora_chunk0.jsonl synth_verify/logs/lora_chunk1.jsonl synth_verify/logs/lora_chunk2.jsonl synth_verify/logs/lora_chunk3.jsonl --out logs/lora_adapter > logs/lora_train.out 2>&1
echo "TRAIN rc=$?"
ls -d logs/lora_adapter 2>/dev/null && echo "ADAPTER OK"
echo "=== TRAIN JOB DONE ==="
