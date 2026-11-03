# Performance lever experiments — sparse tier engine, 2026-11-03

Baseline for every comparison: tonight's verified numbers on the 17 GB tier boot
(see BENCH-262K.md): 2 × 262K cold prefill ~670 tok/s aggregate, single 190K
cold ~790 tok/s, decode 24.3–44.3 tok/s per stream contended, hot-buffer miss
rate 10.1%.

Method: one lever per boot, same 200K benchmark pair (bench200_A/B.txt, needles
3318-502-779|MARIBEL and 9904-271-665|OSWALD) plus the single 190K control,
bench2stream.py as the ruler. Each lever gets: hypothesis, change, boot log
evidence, results, verdict, keep/revert. Cold-prefill numbers require the
engine to have never seen the docs — first run on each boot is the cold one.

---

## L1 — max_num_batched_tokens 2048 → 4096

**Hypothesis.** Prefill chunk count halves (121 → 61 chunks for a 248K doc);
per-chunk fixed costs (tier staging, sketch, scheduler round-trip) halve with
it. Expected +20–50% aggregate prefill throughput.

**Change.** EXTRA_ARGS gains `--max-num-batched-tokens 4096` on the boot.

**Risk.** Activation VRAM spike during prefill; CUDAGRAPH PIECEWISE re-captures.
Watch: boot guard, VRAM under dual prefill, first-pass correctness.

**Results (measured 2026-11-03 ~03:00, this boot).**

Dual 262K cold (bench262_L1_results.json):

| | baseline 2048 | L1 4096 | delta |
|---|---|---|---|
| total wall | 743.2 s | 772.3 s | +3.9% worse |
| aggregate prefill | 669 tok/s | 644 tok/s | −3.9% |
| stream walls | A 730.2 / B 457.2 | A 511.0 / B 760.8 | order flipped |
| decode per stream | 44.3 / 24.3 tok/s | 24.9 / 40.9 tok/s | swapped seats |
| warm re-read | A 5.7 / B 272.9 s | A 246.6 / B 3.6 s | eviction followed the first stream |

Single 190K control (L1_single190k.json): 254.0 s for 193,081 tokens =
**760 tok/s vs 796 baseline (−4.5%)**, needle PASS.

VRAM under dual prefill: 23.56 GiB (+256 MiB vs 2048) — fits.

**Verdict: REVERT.** Chunk count was not the bottleneck at these depths; the
serialization point is tier staging, not the scheduler loop. Slightly worse
across both tests (within noise, but consistently in the wrong direction).

STATUS: measured, reverting to 2048.

---

## L3 — async_scheduling 0 → 1

**Hypothesis.** Overlapping scheduler CPU work with GPU execution helps
decode-heavy and mixed phases; typical expectation 5–15% on decode. It had been
disabled during stability debugging and never retried on the working tier.

**Change.** Boot env `ASYNC_SCHED=1`, batched tokens back at 2048 (L1 reverted).
Verified in boot config: `'async_scheduling': True`.

**Results (measured 2026-11-03 ~04:00, this boot, bench262_L3_results.json).**

| | baseline (async=0) | L3 (async=1) | delta |
|---|---|---|---|
| dual 262K total wall | 743.2 s | 747.9 s | +0.6% (flat) |
| aggregate prefill | 669 tok/s | 665 tok/s | flat |
| stream walls | A 730.2 / B 457.2 | A 418.4 / B 736.1 | order effect dominates |
| decode per stream | 44.3 / 24.3 tok/s | 26.4 / 41.0 tok/s | seat-swapped, flat |
| accuracy | 4/4 | 4/4 | = |

A's 418.4 s cold is the fastest individual stream we've recorded (baseline A:
730.2), but its stream partner paid the difference — the sum is the honest
number, and it didn't move.

**Verdict: KEEP ASYNC_SCHED=0.** No measurable gain on this workload; no
reason to move a stability-relevant default off its proven value.

STATUS: measured, keeping 0.

---

## L2 — SPARSE_TIER_HOT_FRAC 0.10 → 0.15

**Hypothesis.** Miss rate 10.1% at 0.10; upstream's lab curve still improves at
0.15. Fewer host round-trips per decode step → higher decode tok/s. Costs
+~900 MB VRAM taken from the GPU pool (142 → ~128 blocks).

**Change.** Boot env `SPARSE_TIER_HOT_FRAC=0.15`, ASYNC_SCHED back to 0.

**Boot evidence.** Tier came up as predicted: hot buffer 2685 MB (was 1808),
GPU pool 115 blocks (was 142) — the trade happened exactly as costed.

**Results (measured 2026-11-03 ~04:40, this boot, bench262_L2_results.json).**

| | baseline 0.10 | L2 0.15 | delta |
|---|---|---|---|
| dual 262K total wall | 743.2 s | 746.9 s | flat |
| aggregate prefill | 669 tok/s | 666 tok/s | flat |
| decode per stream | 44.3 / 24.3 tok/s | 35.3 / 29.4 tok/s | sum 68.6 vs 64.7 — flat/noise |
| warm re-reads | A 5.7 / B 272.9 s | A 4.3 / B 295.2 s | high-variance every boot |
| accuracy | 4/4 | 4/4 | = |

Warm re-read variance note: across three boots the second stream's warm re-read
was 272.9 / 3.6 / 295.2 s with host-cache eviction present in one run (evict=331)
and absent in another (evict=0, alloc 576 < 593). This is a race between the
first stream's teardown and the second stream's re-read, not a lever effect.

**Verdict: REVERT to 0.10.** Miss-rate reduction didn't surface as decode gain
on this workload (decode was host-BW-bound in aggregate, and the hot buffer
already holds the shared-hot set). At 0.15 we'd pay 27 fewer GPU pool blocks
permanently for nothing measurable. Upstream's 0.15 win was measured on their
presets, single-request, different access pattern.

STATUS: measured, reverting to 0.10.

---

## Final configuration decision

All three knobs returned to the canonical boot: 2048 batched tokens,
ASYNC_SCHED=0, HOT_FRAC=0.10, HOST_GB=17. The canonical configuration survived
every challenge: the binding constraints at 2 × 262K are tier staging
serialization during prefill (L4-class code work, not a knob) and host
bandwidth (already pinned-memory optimal). Restoring the canonical engine now.

## Remaining known upside (not knobs — engineering)

- **Tier pre-staging depth tuning** (upstream roadmap item): prefetch KV to GPU
  before the model needs it; would attack the ~670 vs ~1,080 tok/s contention
  gap directly. Code work in tier_connector, 1–2 sessions.
- **Host decode-side batching of sketch fetches**: the 24–44 tok/s contended
  decode is bounded by ~6.5 GB/s DMA; batching fetches raises effective BW.
  Code work, riskier.
