# 21b Task 1 diagnosis — no patch needed

The 192K admission stall was NOT a code gap. `SPARSE_TIER=1` was missing from
the boot env, so the entire tier (spill hook, host mirror, TierRuntime) was
inert (`enabled()`, tier_connector.py:48). With the variable set, the shipped
minimal positional tier works on 0.32: 190,973-token needle admitted instantly
and answered exactly at 126% of the GPU pool (bench-results/sparsekv-21b-20261102/DIAG.md).

The only required boot env additions vs the window session:
  SPARSE_TIER=1  (master tier switch — serves the spill hook + host mirror)
  --no-scheduler-reserve-full-isl  (chunk-wise admission; upstream runs it too)

Wrapper (~/bin/qwen38-sparse-gpu2-on) gets both in Task 2's commit. The
in-container guard/alloc patches are unchanged and still required.
