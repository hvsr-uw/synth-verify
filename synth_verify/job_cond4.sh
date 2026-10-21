#!/bin/bash
# Condition 4: EBM-style belief memory (n=100). Serve base model, run EBM.
set -x
export HF_HOME=/data/hvardhan2/hf-cache
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /data/hvardhan2/jobs/synth
pkill -f "vllm serve" 2>/dev/null; sleep 3
fuser -k 8001/tcp 2>/dev/null; sleep 2
/data/hvardhan2/vllm-venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --max-model-len 8192 > vllm_cond4.log 2>&1 &
VPID=$!
echo "vllm pid $VPID, waiting for health"
HEALTHY=0
for i in $(seq 1 120); do
  if curl -sf --max-time 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo "vllm healthy after $i tries"; HEALTHY=1; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then
    echo "vllm died, tail:"; tail -20 vllm_cond4.log; break
  fi
  sleep 10
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "VLLM NEVER BECAME HEALTHY - aborting"; tail -30 vllm_cond4.log
  kill $VPID 2>/dev/null; exit 1
fi
python3 synth_verify/run_cond.py --condition 4 --n 100 --seed 0 --log logs/cond4_trace.jsonl --belief-out logs/cond4_final_beliefs.txt > logs/cond4_ebm.out 2>&1
echo "COND4 rc=$?"
python3 synth_verify/analyze_beliefs.py --trace logs/cond4_trace.jsonl --history logs/cond4_final_beliefs_history.jsonl --out logs/cond4_belief_analysis.html > logs/cond4_analysis.out 2>&1
echo "ANALYSIS rc=$?"
kill $VPID 2>/dev/null
echo "=== COND4 JOB DONE ==="
