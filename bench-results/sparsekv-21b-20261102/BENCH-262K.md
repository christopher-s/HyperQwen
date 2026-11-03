# 2 × 262K concurrent-stream benchmark (quick gate) vs our current setup

Engine: MAX_LEN=262144, MAX_SEQS=2, HOST_GB=17 (host pool 17.05 GB = 593 blocks =
512,352 tokens = 2 × 256,176-token contexts). GPU pool 142 blocks = 122,688 tokens.
Booted with 33.0 GiB MemAvailable; the RAM guard accepted 17 GB. 0 preemptions
throughout. All numbers from bench200_results.json / bench262_results.json and
engine logs of this boot.

## Headline: 2 × 262K CONCURRENT STREAMS PASS

Two independent ~248,748-token documents (96.9% and 96.9% of the 262,144 RoPE
cap), fired simultaneously — 497,497 total prompt tokens against a 122,688-token
GPU pool (406% overcommit; host carries the rest):

| stream | cold wall | prompt tok | finish | needle | warm re-read | decode |
|---|---|---|---|---|---|---|
| A | 730.2 s | 248,749 | stop | PASS (exact) | 5.7 s (247,104 cached) | 44.3 tok/s |
| B | 457.2 s | 248,748 | stop | PASS (exact) | 272.9 s* | 24.3 tok/s |

*Stream B's warm follow-up cost 272.9 s because the host cache EVICTED part of
B's KV while A's cold prefill churned the pool (hostcache evict 331 blocks,
593 capacity, 986 allocated across the run). A's warm hit was clean. Expected
behavior at 103% host-pool utilization; not a defect, but honest to report.

Aggregate cold prefill: 497,497 tokens in 743 s wall = **~670 tok/s aggregate**
(A and B shared the engine; A finished at 730 s, B at 457 s — B's window
overlapped A's, so per-stream effective rates were ~341 and ~544 tok/s).

Decode (300-token no-think continuation on a fully cached 248.7K context):
24.3–44.3 tok/s per stream while the other stream idled. Single-stream decode
reference on this boot-class: ~50 tok/s class (MTP on, thinking off). Decode
degrades roughly with concurrent load, as expected.

## Comparison: 200K concurrent round (same boot, same protocol)

Two ~199,128-token documents, 398,256 total (307% overcommit):

| stream | cold wall | pass | warm | decode |
|---|---|---|---|---|
| A | 5.6 s* | PASS | 4.3 s | 34.7 tok/s |
| B | 5.3 s* | PASS | 4.5 s | 36.2 tok/s |

*Timing caveat, recorded honestly: the 200K docs had already been prefilled
server-side by two killed earlier attempts, so the "cold" numbers are cache-hit
numbers. The honest cold-prefill rate for ~200K-class content on this engine is
the earlier measured family: ~1,080 tok/s aggregate (T3: 191K tokens in 177 s
on the 13 GB boot). At that rate a genuine 2 × 199K cold concurrent round would
land around 6 minutes wall.

## 200K vs 262K vs previous canonical (13 GB / 2 × ~196K)

| config | max contexts | VRAM | host RAM | aggregate prefill | concurrent >pool | status |
|---|---|---|---|---|---|---|
| 13 GB tier (overnight) | 2 × ~196K (392K tok) | 22.7 GiB | 13.05 GB | ~1,080 tok/s | 191K/129.6K = 147% | verified T1–T3 |
| 17 GB tier (this bench) | 2 × 256,176 (512K tok) | 22.8–23.3 GiB | 17.05 GB | ~670 tok/s at 2×262K | 497K/122.7K = 406% | verified this run |

VRAM barely moved (22.7 → 23.3 GiB under dual-prefill load, 95% of the card);
the extra 4 GB tier lives in system RAM. RAM during the dual 262K prefill:
35.0 GiB used, 10.98 GiB available (floor 8 held throughout).

## Accuracy

Every question in every round answered from the document exactly: 262K round
4/4 needle+custodian PASS (2 cold + 2 warm), 200K round 4/4 PASS. Tier verify
counters: 0 diffs. select_full_reused [853, 840] shows the selection cache
reusing full-selection results; hot buffer across the run: 46.3M hits /
5.2M misses (89.9% hit) — within the ≤16%-miss fallback band.

## What it cost / how to reproduce

- Chris closed Slack + Brave (+3.1 GiB) before the boot.
- Replacing the 13 GB engine returned its pinned 13.05 GB → 33.0 GiB available
  at boot; guard accepted 17 GB with ~8 GiB to spare.
- Boot command: `QWEN38_SPARSE_HOST_GB=17 ~/bin/qwen38-sparse-gpu2-on`
  (then in-container guard/alloc patches + restart, as the wrapper prints).
- Bench: `bench2stream.py <docA> <docB> <combA|custA> <combB|custB> <out.json>`
  (bench262_A/B.txt = full 770K-char corpus, B reversed to avoid prefix reuse).

## Bottom line

2 × 262K concurrent streams work on this box tonight: both 248.7K documents
admitted and answered exactly, 406% GPU-pool overcommit carried by the 17 GB
host tier, zero preemptions, VRAM flat at ~23.3 GiB, RAM floor never breached.
The cost of the bigger tier shows up as prefill-rate dilution when two huge
cold prefills contend (~670 vs ~1,080 tok/s aggregate) and as host-cache
eviction when combined demand exceeds the host pool — both visible in the
numbers above and neither fatal.
