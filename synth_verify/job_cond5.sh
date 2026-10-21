#!/bin/bash
# Condition 5 (track record, counted memory) on seeds 0,1,2, plus the
# no-memory baseline on seeds 1,2 so the headline comparison has 3 seeds.
# One vLLM server (base Qwen3-8B) for all five runs.
set -x
export HF_HOME=/data/hvardhan2/hf-cache
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /data/hvardhan2/jobs/synth
pkill -f "vllm serve" 2>/dev/null; sleep 3
fuser -k 8001/tcp 2>/dev/null; sleep 2
/data/hvardhan2/vllm-venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --max-model-len 8192 > vllm_cond5.log 2>&1 &
VPID=$!
echo "vllm pid $VPID, waiting for health"
HEALTHY=0
for i in $(seq 1 120); do
  if curl -sf --max-time 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo "vllm healthy after $i tries"; HEALTHY=1; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then
    echo "vllm died, tail:"; tail -20 vllm_cond5.log; break
  fi
  sleep 10
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "VLLM NEVER BECAME HEALTHY - aborting"; tail -30 vllm_cond5.log
  kill $VPID 2>/dev/null; exit 1
fi
python3 synth_verify/run_cond.py --condition 5 --n 100 --seed 0 --log logs/cond5_trace_s0.jsonl --record-out logs/cond5_final_record_s0.txt > logs/cond5_track_record_s0.out 2>&1
echo "COND5 s0 rc=$?"
python3 synth_verify/run_cond.py --condition 5 --n 100 --seed 1 --log logs/cond5_trace_s1.jsonl --record-out logs/cond5_final_record_s1.txt > logs/cond5_track_record_s1.out 2>&1
echo "COND5 s1 rc=$?"
python3 synth_verify/run_cond.py --condition 5 --n 100 --seed 2 --log logs/cond5_trace_s2.jsonl --record-out logs/cond5_final_record_s2.txt > logs/cond5_track_record_s2.out 2>&1
echo "COND5 s2 rc=$?"
python3 synth_verify/run_cond.py --condition 1 --n 100 --seed 1 --log logs/cond1_trace_s1.jsonl > logs/cond1_no_memory_s1.out 2>&1
echo "COND1 s1 rc=$?"
python3 synth_verify/run_cond.py --condition 1 --n 100 --seed 2 --log logs/cond1_trace_s2.jsonl > logs/cond1_no_memory_s2.out 2>&1
echo "COND1 s2 rc=$?"
kill $VPID 2>/dev/null
echo "=== COND5 JOB DONE ==="
