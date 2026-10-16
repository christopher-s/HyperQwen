# Black Monolith deployment roadmap

Working notes for the twin RTX 3090 SuperQwen deployment served from this repo.
Research basis: 2026-10-16 audit (sources in the local research store, final report
`2026-10-16-hyperqwen-looping-enhancements`).

Current state: vLLM 0.31.0, PTH_FA int8 per-token-head KV (PR #325 branch head),
MTP k=3, PIECEWISE CUDA graphs, ASYNC_SCHED=0, MAX_SEQS=1, 240K window,
`hyperqwen_router.py` on :8020 with xhigh effort default, repetition_penalty 1.08,
max_tokens cap 8192.

## P1

1. [ ] Integrate PR #326 (`pr50021-conv-bound`) and run the MTP A/B.
   Upstream fix for the GDN speculative causal_conv1d bound (issue #320).
   Arms: current MTP, patched MTP, SPEC=off. Battery: 18 synthetic requests x 4,
   raw token IDs preserved, finish reasons recorded. DFlash2 is validated
   upstream; MTP + PTH_FA on sm86 TP=1 is not.
2. [ ] Reasoning-policy A/B after the engine patch.
   Checkpoint author defaults unspecified effort to medium with an anti-repeat
   stop rule on explicit xhigh. Router currently forces xhigh. Measure repeated
   n-grams, output tokens, length finishes, malformed tool calls, completion.

## P2

3. [ ] `repetition_detection` circuit breaker via the router.
   First arm: {"max_pattern_size": 32, "min_pattern_size": 1, "min_count": 8}.
   Log every trigger with request ID, lane, prompt/generated tokens, finish
   reason, detector settings, prefix-cache state, preceding request ID.
4. [ ] Diagnostic capture on every loop-class finish (length, repetition,
   repeated-token threshold): rewritten request body, raw output token IDs,
   last 256 prompt IDs, effort, sampler values, spec mode, target GPU.

## P3

5. [ ] GPU0 PCIe x4 UVA characterization (EMBED_UVA=1 measured on x16 only,
   issue #281 basis). Prefill/image/embedding A/B x4 vs x16.
6. [ ] Re-evaluate PR #329 (DYNAMIC_K, DFlash2 4-stream +7.7%) if the fleet
   moves to DFlash2 batching.
7. [ ] PR #332 seat-budget logging once merged upstream.
8. [ ] Swift 1.0/1.5 checkpoint experiment after the engine fault is closed.
9. [ ] Retire `/home/chris/tools/qwen38-27b-rtx3090` (pre-rename stale clone).

## Conventions

- `main` tracks `origin/main` (this fork). Working code lands on
  `black-monolith/router`; upstream sync via `upstream` remote.
- Never commit bench data (`bench-quality-data/`, `quantize.log`) or any
  credential material. Tuning-only `.env` files are allowed.
