# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU miss-rate simulation of the hot-buffer policies on synthetic traces (stream E).

python -m sparsekv.tier_sim [--ctx 131072] [--steps 512]
Compares exact LRU (not graph-friendly: needs a sort/top-k over H rows per step) against the device policy in
tier.py (windowed CLOCK; age=0 is FIFO-with-skip), using the actual reference implementation gather_rows_tiered_ref.
All top-k picks are treated as non-resident (worst case: every old block spilled; sinks/window stay on GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier import HotLayer, NEG_AGE, gather_rows_tiered_ref  # noqa: E402
from sparsekv.tier_trace import gen_trace, overlap_stats  # noqa: E402


def lru_misses(tr: np.ndarray, N: int, H: int, warm: int) -> float:
    last = np.full(N, -1, dtype=np.int64)
    cached = np.zeros(N, dtype=bool)
    count = 0
    miss_tot = pick_tot = 0
    for t, P in enumerate(tr):
        hit = cached[P]
        mp = P[~hit]
        last[P[hit]] = t
        need = mp.size - (H - count)
        if need > 0:
            cpos = np.flatnonzero(cached & (last < t))
            ev = cpos[np.argpartition(last[cpos], need - 1)[:need]]
            cached[ev] = False
            count -= need
        cached[mp] = True
        last[mp] = t
        count += mp.size
        if t >= warm:
            miss_tot += mp.size
            pick_tot += P.size
    return miss_tot / pick_tot


def clock_misses(tr: np.ndarray, N: int, H: int, warm: int, age: int, C: int) -> float:
    hot = HotLayer(kv=None, tag=torch.full((1, H), -1, dtype=torch.int32),
                   last=torch.full((1, H), NEG_AGE, dtype=torch.int32),
                   pos2row=torch.full((1, N), -1, dtype=torch.int32), hand=torch.zeros(1, dtype=torch.int32))
    K = tr.shape[1]
    gpu_bt = torch.full((1, N // 864 + 2), -1, dtype=torch.int32)
    slot = torch.zeros(1, dtype=torch.int32)
    nv = torch.tensor([K], dtype=torch.int32)
    miss_tot = pick_tot = 0
    for t, P in enumerate(tr):
        idx = torch.zeros((1, C), dtype=torch.int32)
        idx[0, :K] = torch.from_numpy(P)
        h, m, _ = _ref_nocopy(gpu_bt, idx, nv, hot, slot, t + 1, age)
        if t >= warm:
            miss_tot += m
            pick_tot += K
    return miss_tot / pick_tot


def _ref_nocopy(gpu_bt, idx, nv, hot, slot, step, age):
    # gather_rows_tiered_ref needs BS from a tensor; pass a dummy host tensor carrying BS=864
    dummy = torch.empty((1, 1, 864, 1), dtype=torch.int8)
    return gather_rows_tiered_ref(None, gpu_bt, dummy, None, idx, nv, None, None, hot, slot, step, age, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[32768, 131072])
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--hot", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--ages", type=int, nargs="+", default=[0, 2, 8])
    ap.add_argument("--preset", nargs="+", default=["lab", "brief"])
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--C", type=int, default=2048 + 64 + 1024)
    a = ap.parse_args()
    out = []
    for preset in a.preset:
        for N in a.ctx:
            tr = gen_trace(N, a.steps + a.warm, K=a.K, preset=preset, seed=N)
            st = overlap_stats(tr[a.warm:], N)
            for hf in a.hot:
                H = max(2 * a.C, int(hf * N))
                row = {"preset": preset, "ctx": N, "hot_frac": hf, "H": H, **{k: round(v, 3) for k, v in st.items()},
                       "lru": round(lru_misses(tr, N, H, a.warm), 4)}
                for age in a.ages:
                    row[f"clock_age{age}"] = round(clock_misses(tr, N, H, a.warm, age, a.C), 4)
                print(json.dumps(row), flush=True)
                out.append(row)
    return out


if __name__ == "__main__":
    main()
