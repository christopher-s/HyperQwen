# sparsekv GPU2 window session — 2026-11-02 (compressed, ~30 min tests)

Scope per Chris: take GPU2, under 30 minutes of tests, "see how this is working".
Full validation battery (KLD, conc ladder, 2 h soak, 262K probe) NOT run — see
WINDOW-RUNBOOK.md for the full protocol.

## What ran

- Basis refit on qwopus-heretic keys: basis27-qwopus.pt (4.2 MB). Held-out
  key-energy mean ~0.87 native positions, ~0.55–0.58 at +64K/+128K RoPE shift.
- Sparse engine boots on :18025 (GPU2 only): MAX_LEN=131072, MAX_SEQS=2,
  MTP k3, int8 per-token-head KV (unchanged), HostTierConnector kv_both,
  SPARSE_KV=1 SPARSE_IMPL=triton, int8 host tier 18 GB, --block-size 864.
- Probes below. GPU0 production untouched throughout (verified: Up 4+ h,
  23.1 GiB steady).

## Results

| probe | result |
|---|---|
| greedy arithmetic (17*4) | 68, finish=stop |
| facts list (thinking on, medium) | correct, 110 tok/s |
| 115,185-token needle @72% depth (thinking on) | exact retrieval (combination + custodian), finish=stop, 116 s incl. full cold prefill |
| 2-stream follow-ups on cached prefix | both correct, 4.4 s wall TOTAL; 114,048/115,176 tokens from prefix cache (sparse-suffix path) |
| thinking-on code gen on sparse engine | clean finish=stop, 700-char reasoning, correct code |
| "switch -> SPARSE" log line | present — sparse decode family engaged |
| sparse graph sketch caches | 16/16 layers (slots 0–15), int4, BS=864 |
| MTP under sparse | accepted tokens counter moving (288 accepted at first checkpoint) |

## Boot-chain fixes discovered (all in-container; scripts in sparsekv-port/patches/)

1. Variant image CMD was 'sleep infinity' (docker commit artifact) — engines
   launch with the `single` argument.
2. start_qwen.sh refuses PTH_FA=1 + --kv-transfer-config (generic CPU-tier
   guard predating the port). Bypassed in-container; proper fix = bake the
   bypass into a rebuilt variant image.
3. The launcher's ALLOC_DEFAULT arm (expandable_segments:False when a KV
   connector is present) must keep its real pattern — my first bypass killed
   both arms; vLLM refuses the connector under VMM.
4. sparse_kv kernels are built for BS=864 — the engine must boot with
   --block-size 864 (their documented production page size).
5. SPARSE_KV=1 + SPARSE_IMPL=triton are required master switches (not implied
   by SPARSE_PROFILE=tier).

## The big finding: 262K × 2 is blocked on admission, not on decode

The released tier mirrors KV to host and spills at DECODE time, but request
ADMISSION still requires the whole sequence to fit the GPU pool (~135K tokens
at GPU_UTIL 0.94). tier_admission.py — the piece that would free GPU blocks
and admit beyond-pool requests — is an unwired skeleton in the upstream
release ("Freeing the GPU blocks (actual capacity) is the admission work
described in STREAM_tier.md"). Their own RELEASE.md shows the same shape:
"the second user's admission fails (Running: 1, Waiting: 1) until the first
finishes". A 192K/262K prompt waits at capacity forever on our pool.

So: sparse decode at ≤128K-context scale = WORKING TODAY (needle exact,
follow-ups near-instant, MTP healthy, thinking clean). 2 × 262K = blocked on
implementing the admission extension (stream E of their roadmap), a real
porting project against 0.32 scheduler internals.

## State at window close

- Sparse engine STOPPED; production qwopus gpu2 relaunched via
  qwen38-qwopus-on gpu2 (verify health below).
- Raw artifacts: smoke.py, needle128_probe.py, two_stream.py,
  needle128_result.json, needle_doc*.txt in this directory.
- Wrapper updated: ~/bin/qwen38-sparse-gpu2-on (one-command boots, with the
  documented in-container patch steps).
