#!/bin/bash
# Full run v2 (hardened): serve Qwen3-8B, cond1 (n=100), cond2 (n=100).
# Fixes from the 404 timeout: kill stale vllm, curl can never hang,
# abort loudly if vllm is unhealthy, unbuffered vllm log, 8h limit.
set -x
export HF_HOME=/data/hvardhan2/hf-cache
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /data/hvardhan2/jobs/synth
pkill -f "vllm serve" 2>/dev/null; sleep 3
fuser -k 8001/tcp 2>/dev/null; sleep 2
/data/hvardhan2/vllm-venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --max-model-len 8192 > vllm_full.log 2>&1 &
VPID=$!
echo "vllm pid $VPID, waiting for health"
HEALTHY=0
for i in $(seq 1 120); do
  if curl -sf --max-time 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo "vllm healthy after $i tries"; HEALTHY=1; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then
    echo "vllm process died, tail of vllm_full.log:"; tail -20 vllm_full.log; break
  fi
  sleep 10
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "VLLM NEVER BECAME HEALTHY - aborting"
  tail -30 vllm_full.log
  kill $VPID 2>/dev/null
  exit 1
fi
python3 synth_verify/run_cond.py --condition 1 --n 100 --seed 0 --log logs/cond1_trace.jsonl > logs/cond1_no_memory.out 2>&1
echo "COND1 rc=$?"
python3 synth_verify/run_cond.py --condition 2 --n 100 --seed 0 --log logs/cond2_trace.jsonl --playbook-out logs/cond2_final_playbook.txt > logs/cond2_playbook.out 2>&1
echo "COND2 rc=$?"
kill $VPID 2>/dev/null
echo "=== FULL JOB DONE ==="
