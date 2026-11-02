#!/usr/bin/env bash
# T1 Step 2: poll admission metrics every 10 s for 8 min -> diag-metrics.log
cd /home/chris/tools/hyperqwen/bench-results/sparsekv-21b-20261102
: > diag-metrics.log
for i in $(seq 1 48); do
  ts=$(date +%H:%M:%S)
  run=$(curl -s -m 3 http://127.0.0.1:18025/metrics 2>/dev/null | grep "^vllm:num_requests_running" | awk '{print $2}')
  wait=$(curl -s -m 3 http://127.0.0.1:18025/metrics 2>/dev/null | grep "^vllm:num_requests_waiting" | awk '{print $2}')
  gpu=$(curl -s -m 3 http://127.0.0.1:18025/metrics 2>/dev/null | grep "^vllm:gpu_cache_usage_perc" | awk '{print $2}')
  echo "$ts running=${run:-NA} waiting=${wait:-NA} gpu_cache_usage=${gpu:-NA}" >> diag-metrics.log
  sleep 10
done
echo done >> diag-metrics.log
