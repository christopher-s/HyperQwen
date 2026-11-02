#!/bin/bash
# kld_serve.sh <ref|tier>  — Black Monolith GPU2 adaptation of upstream eval/kld/kld_serve.sh.
#
# Differences from upstream (ported for our stack):
# - engines come from ~/bin/qwen38-sparse-gpu2-on (sparse) and the PRODUCTION
#   :18021 dense engine (ref) instead of their launch/ tree;
# - MTP stays ON in the ref/tier configs (our serving state; their study ran MTP
#   off — recorded deviation, plan Task 5 Step 1). If the logits processor is
#   rejected under spec decode, the toggle arm is the fallback comparison;
# - SPARSE_TIER_HOST_BITS=8 SPARSE_TIER_HOST_GB=18 per Chris (no 4-bit KV data);
# - MAX_SEQS=2 (our target), MAX_LEN=262144, GPU_UTIL=0.94, port 18025.
set -euo pipefail
CFG=$1
KLD_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$CFG" in
  ref)
    # Dense reference = the PRODUCTION qwopus engine on :18021 (untouched).
    # This script does not start it; run_kld.sh asserts its health instead.
    curl -s -m 5 http://127.0.0.1:18021/health >/dev/null || { echo "production :18021 must be up" >&2; exit 1; }
    echo "ref = production dense engine :18021"
    ;;
  tier)
    # Sparse tier engine on :18025 via the Task 4 wrapper (all knobs inside it).
    /home/chris/bin/qwen38-sparse-gpu2-on on
    for i in $(seq 1 60); do
      curl -s -m 2 http://127.0.0.1:18025/health >/dev/null 2>&1 && break
      sleep 10
    done
    curl -s -m 5 http://127.0.0.1:18025/health >/dev/null || { echo "sparse engine failed to become healthy" >&2; exit 1; }
    echo "tier = sparse engine :18025 (2x262K, int8 host tier 18 GB)"
    ;;
  *)
    echo "unknown config $CFG (use ref|tier)" >&2
    exit 2
    ;;
esac
