# T2 Sizing: the canonical sparse-gpu2 boot config

**Status: MEASURED. The RAM guard fired twice during this task and is the
binding constraint, not arithmetic.**

- HOST_GB=17 refused: "would leave less than 8.0 GB of MemAvailable (22.0 GB)"
- HOST_GB=30 refused (deliberate negative test): "(24.7 GB)"
- HOST_GB=13 boots and holds: canonical.

**Ruling in force** (ledger, Task 2): Chris removed ComfyUI from the box
("Ignore comfyui. It isn't loaded", 2026-11-02). The plan's 192K/10 GiB
numbers are superseded. This task targets **2 × 262K** directly.

## RAM arithmetic (45 GiB box) — MEASURED at boot, not assumed

| attempt | result |
|---|---|
| HOST_GB=17 | REFUSED: MemAvailable 22.0, needs 17+8=25 |
| HOST_GB=13 | BOOTS: `host pool 13.05 GB (454 blocks)`, MemAvailable 10 after load |
| HOST_GB=30 (negative test) | REFUSED: MemAvailable 24.7, needs 30+8=38 |

13 GiB = 454 blocks × 864 = 392,256 host tokens = **2 × ~196K contexts**
(MAX_LEN stays 262144: single 262K streams work; 2 × 262K = 16.2 GiB tier
needs ~4 GiB more headroom than the desktop currently yields).

| item | GiB | source |
|---|---|---|
| total | 45.0 | `free -g` |
| engine RSS (ONE engine; sparse replaces the gpu2 dense engine at adoption) | ~7.5 | upstream STREAM_tier measured 5–6 + 1.8; our production engines match |
| OS + desktop + services | ~3.0 | upstream's figure, matches our box |
| host KV tier (canonical) | **13.05** | boot log, this task |
| MemAvailable floor (the ported guard refuses below this) | 8.0 | `HOST_HEADROOM_GB=8` |
| **slack at boot** | **~10** | `free -g` available after load |

## GPU arithmetic

| item | value | source |
|---|---|---|
| host-indexed sketch | 524,288 host tokens × 4.2 KB ≈ **2.1 GiB** | STREAM_tier §3 "4.2 KB of GPU memory per host-pool token" |
| hot buffer | H = max(2C, 0.10 × 262,144) = 26,214 rows/slot ≈ **1.8 GB** | measured 19,660 rows = 1,357 MB at 192K, scales linearly |
| admission need per request | 6 blocks ≈ 5,184 tokens ≈ 0.18 GiB | 21b plan T4 (spec cap; pre-spec the spill frees under pressure) |
| sketch growth vs 192K boot | pool shrinks ~0.7 GiB vs the measured 174-block (150,336 tok) pool | arithmetic; still ≥ 20× the admission cap |

## The canonical config (wrapper `~/bin/qwen38-sparse-gpu2-on`)

```
MAX_LEN=262144  MAX_SEQS=2  GPU_UTIL=0.94  PORT=18025
SPARSE_KV=1  SPARSE_IMPL=triton  SPARSE_TIER=1          # T1: master switch, was missing
SPARSE_PROFILE=tier  SPARSE_SKETCH_BITS=4
SPARSE_TIER_HOST_BITS=8  SPARSE_TIER_HOST_GB=13
SPARSE_TIER_HOT_FRAC=0.10  SPARSE_TIER_SPILL=1
SPARSE_TIER_HOST_MAMBA_SLOTS=8  SPARSE_TIER_HOST_MAMBA_PER_REQ=2
SPARSE_STATS_SUB=4  SPARSE_SELECT_EVERY=2  SPARSE_MEM_RESERVE=1
EXTRA_ARGS="--block-size 864 --no-scheduler-reserve-full-isl --kv-transfer-config {...HostTierConnector...}"
```

Measured at this config (boot-262k.log): `tier: runtime mode=SPILL: 16
sparse layers in group(s) [3] ... host pool 13.05 GB (454 blocks), GPU pool
150 blocks (=129,600 tokens), hot 26214 rows/slot (1808 MB total), C=3136,
tail=1088`; 16 sketch lines; 0 `allocated > reserved`; available 10 GB.

Watermark: default (`num_gpu_blocks // 10`). Fallbacks: misses > 16% →
`SPARSE_TIER_HOT_FRAC=0.15`; guard refuses → lower HOST_GB (measured floor
~14 GiB tier at tonight's desktop load).

## Verification (T2 Steps 2–3)

- Boot log contract: `tier: runtime mode=SPILL ... host pool 17.xx GB ...
  GPU pool N blocks`, 16 sketch lines, no `allocated > reserved`,
  `switch -> SPARSE`, `free -g` available ≥ 8 during load.
- Negative test: `SPARSE_TIER_HOST_GB=30` must refuse boot with the
  `host_pool_blocks()` MemoryError line.
