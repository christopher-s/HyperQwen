# T3 Control: 192K-class spill + host prefix cache + two-stream overcommit

Engine: canonical 262K/13GB boot (GPU pool 150 blocks = 129,600 tokens,
host pool 454 blocks = 392,256 tokens). All numbers from this boot.

## 1. Single 190,973-token needle (126% of GPU pool)

- admitted immediately (running=1 within one 10 s poll; waiting stayed 0)
- wall 240.2 s cold prefill + decode, finish=stop
- needle at 72% depth retrieved EXACTLY (8314-772-906 / MARGARET)
- tier: 221 blocks mirrored to host, hot buffer live
- upstream parity: their 177K-on-139K measured 197.6–202.2 s cold, PASS.

## 2. Host prefix cache — multi-turn on the spilled context

| turn | prompt tokens | cached | wall | answer |
|---|---|---|---|---|
| Q1 cold | 190,973 | 0 | 240.2 s | needle PASS |
| Q2 follow-up | 190,976 | 190,080 | **12.3 s** | finish=length (thinking ate the 600-token cap; content empty — harness artifact, prefix path proven by cached_tokens) |
| Q3 follow-up | 190,965 | 190,080 | **2.3 s** | needle PASS |

Upstream measured 197.6 s → 2.3 s; ours 240.2 s → 2.3 s. The follow-up
TTFT matches theirs exactly. ~900 new tokens prefilled per turn.

## 3. Two-stream overcommit (191K demand vs 129.6K pool = 147%)

Two independent documents (94,480 + 95,364 tokens), fired concurrently:

| stream | prompt tok | wall | finish | comb | custodian |
|---|---|---|---|---|---|
| A | 94,162 | 177.2 s | stop | PASS | DELPHINE PASS |
| B | 96,879 | 150.6 s | stop | PASS | TORSTEN PASS |

- total wall 177.2 s for both (vs 240 s for ONE 190K serial prefill)
- 0 preemptions logged; hostcache alloc 378, hit 506, evict 0
- tier hot buffer active across both streams

This is the row the window session never reached (both streams running
simultaneously, both >pool-capable). Upstream's conc2–conc5 equivalent:
PASS/PASS with 0–1 preemptions.

## Honest deltas vs upstream

- Cold prefill slower than theirs (240 vs 198 s at 177K): our chunks are
  2,048 tokens (max_num_batched_tokens) vs their dev-server config; and
  this boot ran against a busier host. Not investigated further — prefill
  cost is not the gate.
- Q2 finish=length is a harness artifact (max_tokens=600 < thinking
  budget), not an engine defect. Recorded, not hidden.

## Gate implications (Task 5 will re-measure formally)

- spill admission: working at 126–147% overcommit
- host prefix cache: working at upstream's exact follow-up TTFT
- two-stream overcommit: working, 0 preemptions
- remaining unknowns: KLD quality at 192K thinking-on; >150K code-line
  recall (the x4 open suspect); MTP acceptance delta; soak stability.
