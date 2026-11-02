# WINDOW-RUNBOOK — sparsekv GPU2 single-session execution

One scheduled GPU2 session executes everything that needs the card. GPU0 is
never touched at any point. Production `:18021` stays UP the whole time
(the dense reference IS the production engine).

Prereqs (already done, see ledger): upstream pinned d2d876d; variant image
`hyperqwen-pthfa:032-sparse-port` (patched vllm + pthfa + sparsekv module,
inertness verified); wrapper `~/bin/qwen38-sparse-gpu2-on`; KLD harness
ported; fit corpus staged (`sparsekv-port/fitdata/`).

Estimated wall time: 6–9 h total (fit ~10 min, boot ~5 min, KLD ~80 min,
conc ladder ~30 min, 262K probe ~15 min, soak 2 h, analysis ~20 min).

## 0. Pre-flight (5 min)

```bash
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
# GPU2 (GPU-2d65d36c…) must be the only card this runbook touches.
# GPU0 (GPU-798fc506…) must show its production engine untouched throughout.
free -g          # need ~20 GB available for the pinned tier
curl -s -m 3 http://127.0.0.1:18021/health >/dev/null && echo "production :18021 UP"
```

## 1. Basis fit on qwopus-heretic keys (~10 min, GPU2)

```bash
docker run --rm --name sparse-fit --runtime=nvidia --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES="GPU-2d65d36c-8c19-0b80-0f4a-40743d551530" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e MODEL_DIR=/models/qwopus-heretic-W4A16-AutoRound \
  -e NIAH_JSONL=/fitdata/niah_32k_samples.jsonl \
  -e VLLM_SRC=/fitdata/vllm-src \
  -e WORK=/out/work -e OUT=/out/basis27-qwopus.pt -e OUT_ESSAY=/out/basis27-qwopus-essay.pt \
  -v /mnt/500gb-ssd/ai/models:/models:ro \
  -v /home/chris/tools/hyperqwen/sparsekv-port/fitdata:/fitdata:ro \
  -v /home/chris/tools/hyperqwen/sparsekv-port:/out \
  hyperqwen-pthfa:032-sparse-port \
  bash -c "cd /port 2>/dev/null; cp -r /port/upstream/sparsekv/tools /tmp/tools 2>/dev/null; \
           PYTHONPATH=/tmp/tools/.. /app/venv/bin/python /tmp/tools/fit_basis27.py grams && \
           /app/venv/bin/python /tmp/tools/fit_basis27.py fit"
```

Then: `ls -la sparsekv-port/basis27-qwopus.pt` (~4.2 MB expected), record the
printed held-out key-energy/logit-energy fractions in NOTES.md.

## 2. Sparse boot + smoke (~15 min)

```bash
~/bin/qwen38-sparse-gpu2-on on
docker logs -f hyperqwen-sparse-gpu2 &   # watch for: host pool 17.x GB, GPU pool blocks,
                                         # NO "allocated > reserved"
curl -s http://127.0.0.1:18025/health
# 3-probe smoke: greedy arithmetic, facts list, 150K needle
```

Fail-safe: if boot fails on memory math, drop `SPARSE_TIER_HOST_GB` 18→14 in
the wrapper and retry once; below that, stop and report.

## 3. KLD battery (~80 min)

```bash
SAMPLES=sparsekv-port/eval/kld/data bash sparsekv-port/eval/kld/run_kld.sh
```

(Before this, one-time: build samples per upstream `build_samples.py` using the
corpus in `fitdata/`.) Record results.md. If the logits processor is rejected
under MTP: switch to the toggle arm (`KLD_TOGGLE=1`, `SPARSE_CUTOFF` per-request
dense/sparse) and note the deviation in RESULTS.md.

## 4. MTP acceptance + two-stream throughput (~30 min)

- `curl -s :18025/metrics | grep spec_decode` before/after a 20K-token gen;
  same prompts against :18021 for the dense side. Acceptance floor: 0.45
  (production today 0.60–0.69).
- Two concurrent 100K-context recall requests; record wall times, combined
  tok/s vs the production single-stream figure (84–92 tok/s per stream).

## 5. 262K capacity + recall probe (~15 min)

One 262,144-token prefill (cold ~7 min expected), needle at 90% depth +
code-line recall. Record prefill wall, spill stats, retrieval verdict.

## 6. Two-hour two-stream soak (thinking on, temp 1.0, effort medium)

Two agents, 200K+ contexts, alternating chat turns (production shape via the
router's effort defaults is NOT in play — direct to :18025 with
`chat_template_kwargs {enable_thinking: true, reasoning_effort: "medium"}`).
Gates: 0 preemptions, 0 empty outputs, 0 non-stop finishes, no unterminated
`<think>`, `MemAvailable ≥ 8 GB` (watch with `free -g` every 60 s), tier miss
rate stable ~14–16%.

## 7. RESULTS.md + decision

Write `bench-results/sparsekv-gpu2-<date>/RESULTS.md`: every gate from the
plan (KLD ≥ 95% top-1 at 128/192K, MTP acceptance ≥ 0.45, combined ≥ 1.6×,
soak clean, RAM floor held), GPU0-untouched attestation (nvidia-smi before and
after), rollback notes. Commit + push. Chris decides: adopt / register as
tray option / decline (uninstall via `sparsekv/install.sh --uninstall`).

## Abort / rollback at any point

```bash
~/bin/qwen38-sparse-gpu2-on off     # stop sparse engine; production untouched
docker rm -f hyperqwen-sparse-gpu2  # if the wrapper is unhappy
# full removal (only if Chris declines): sparsekv/install.sh --uninstall in-image
```
