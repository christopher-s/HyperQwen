#!/bin/bash
# run_kld.sh — Black Monolith GPU2 KLD battery driver (plan Task 6 Step 1).
#
# Compares the sparse tier engine (:18025, via qwen38-sparse-gpu2-on) against the
# PRODUCTION dense engine (:18021) on the same card. Runs AFTER both engines are
# up; does not stop either (production is never touched before Task 7).
#
# Arms:
#   ref     — generate references on the dense engine (temperature 0, 256 tokens)
#   forced  — replay the same token paths on the sparse engine, recording KLD
#   toggle  — fallback when logits processors are rejected under MTP:
#             same-server dense/sparse switch comparison (upstream KLD_TOGGLE=1)
#
# Usage: run_kld.sh [--samples DIR] [--out DIR]
set -euo pipefail
KLD_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="/app/venv/bin/python"   # runs inside the 032-sparse-port image via run_in_image.sh
SAMPLES="${SAMPLES:-$KLD_DIR/data}"
OUT="${OUT:-$HOME/tools/hyperqwen/bench-results/sparsekv-gpu2-$(date +%Y%m%d)/kld}"
mkdir -p "$SAMPLES" "$OUT"

echo "== 0. engine health"
curl -s -m 5 http://127.0.0.1:18021/health >/dev/null || { echo "dense :18021 down"; exit 1; }
curl -s -m 5 http://127.0.0.1:18025/health >/dev/null || { echo "sparse :18025 down (start it: qwen38-sparse-gpu2-on on)"; exit 1; }
echo "   both engines healthy"

echo "== 1. build samples (if absent)"
if [[ ! -f "$SAMPLES/samples.json" ]]; then
  echo "   run build_samples.py first (see README-kld.md)"; exit 1
fi

echo "== 2. references on the dense engine"
"$VENV_PY" "$KLD_DIR/kld_client.py" --role ref --port 18021 --data "$SAMPLES" --out "$OUT/ref"

echo "== 3. forced replay on the sparse engine (KLD)"
"$VENV_PY" "$KLD_DIR/kld_client.py" --role forced --mode tier --port 18025 \
  --data "$SAMPLES" --out "$OUT/forced" --ref-dir "$OUT/ref"

echo "== 4. analyze"
"$VENV_PY" "$KLD_DIR/kld_analyze.py" "$SAMPLES" "$OUT" --md "$OUT/results.md" --json "$OUT/results.json"
echo "   results: $OUT/results.md"
