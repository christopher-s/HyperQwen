# T1 Diagnosis: the window-session 192K stall

## Root cause (confirmed live)

`SPARSE_TIER=1` was missing from every window-session boot env. `enabled()`
(tier_connector.py:48) gates the entire tier on that single variable: without
it, `bind_kv_cache_manager` returns early (no spill hook), `register_kv_caches`
never builds TierRuntime (no host mirror), and `tier_dims` still runs for
buffer accounting — but nothing functional engages. Boot logs showed exactly
that shape: `switch -> SPARSE` (decode, gated on SPARSE_KV) present; zero
`tier:` lines. The 192K prompt then sat at ordinary no-tier capacity waiting:
pool ~135K tokens < 192K demand. The "admission gap" was our boot-env bug,
not a missing upstream feature.

## Fix + proof (this boot)

Added `-e SPARSE_TIER=1`, MAX_LEN=196608, `--no-scheduler-reserve-full-isl`.
Boot log: `tier: runtime mode=SPILL: 16 sparse layers in group(s) [3] ...
host pool 10.04 GB (349 blocks), GPU pool 174 blocks ... spill hook on KV
group 3 (FullAttentionManager): frees blocks [1, (processed-1088)//864),
policy pressure (watermark 19 free blocks); host prefix cache
HostTierFullAttentionManager`. MTP drafter = window layer.

Probe: 190,973-token needle (126% of the 150,336-token GPU pool):

- admitted immediately: `running=1.0, waiting=0.0` within one poll (10 s);
- wall 244.7 s (cold prefill over the spill boundary), finish=stop;
- needle at 72% depth retrieved EXACTLY (8314-772-906 / MARGARET);
- tier stats at step 164: mirrored_blocks 221, hot buffer hit 667,077 /
  miss 92,175 / resident 1,852,380; hostcache alloc 221 hit 0 (first turn);
  GDN checkpoint saved (boundary 190,080).

Upstream parity: their 177K-on-139K run measured 199 s TTFT, 96.9 tok/s
decode, needle PASS — ours matches shape and magnitude.

## Consequences for the plan

1. The "minimal positional tier" (monkeypatch spill) WORKS on 0.32 as-shipped
   once the env is right. Task 3's control is achievable without new code.
2. Task 4's spec/manager work remains valuable for what the monkeypatch path
   cannot do: pool sizing from the capped admission (GPU pool currently sized
   for full sequences minus spill-time freeing), sentinel-based concurrent
   admission of two >pool requests, and a standing upgrade-safe structure.
   Upstream's own docs mark the monkeypatch as gap (e) to replace with the
   spec.
3. `--no-scheduler-reserve-full-isl` is REQUIRED for beyond-pool admission
   (upstream runs it too; with it, admission checks only the next chunk).

## Ruling recorded

plan progress.md: boot at MAX_LEN=196608 (supersedes the plan's 131072 —
at 131072 no legal prompt can exceed the pool, so the diagnostic was
unreachable); SPARSE_TIER=1 added to the canonical boot env (wrapper update
lands in Task 2's commit).
