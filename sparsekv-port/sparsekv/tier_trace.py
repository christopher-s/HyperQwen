# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Synthetic sketch-selection traces for the host-tier simulations and benchmarks (stream E).

Model of one layer's top-k picks over an answer (positions in [sink, N - win), K picks per step):
  * a fraction `overlap` (lab: 0.61) of the previous step's picks is kept;
  * each new pick is, with probability `q`, a re-reference drawn from the picks of the last `R` steps that are not
    in the current set (recency-limited working set), otherwise a fresh token drawn from a slowly drifting focus
    (a `focus` fraction of the document, moved by `drift` tokens per step) with probability `pf`, else uniform.
q / R / pf are calibrated in tier_sim.py so that an exact-LRU hot buffer of 10% of the context misses ~9.6% of the
non-resident picks (the lab number, 9B, 512-token answers), while the step-to-step overlap stays at 61%.
A second preset ("brief") follows the brief's "working set ~10-11% of context per 512-token answer" literally:
all re-references and fresh picks stay inside a fixed 10.5% working set.
"""
from __future__ import annotations

import numpy as np

PRESETS = {
    # calibrated: LRU@10% ~9.6% misses (see tier_sim.py output in STREAM_tier.md)
    "lab": dict(overlap=0.61, q=0.875, R=64, pf=0.5, focus=0.05, drift=8.0),
    # stress: LRU@10% ~20% misses (worse than the lab's worst case, 16%)
    "heavy": dict(overlap=0.61, q=0.82, R=200, pf=0.3, focus=0.05, drift=8.0),
    # literal reading of the brief: whole answer stays inside a 10.5% working set
    "brief": dict(overlap=0.61, q=0.0, R=1, pf=1.0, focus=0.105, drift=0.0),
}


def gen_trace(N: int, steps: int, K: int = 2048, sink: int = 64, win: int = 1024, seed: int = 0, preset="lab",
              **kw) -> np.ndarray:
    """Returns int32 [steps, K] top-k picks (no duplicates within a row)."""
    p = dict(PRESETS[preset]); p.update(kw)
    rng = np.random.default_rng(seed)
    lo, hi = sink, N - win
    span = hi - lo
    K = min(K, span)
    fsz = max(1, int(p["focus"] * N))
    fstart = float(rng.integers(lo, max(lo + 1, hi - fsz)))
    cur = rng.choice(span, size=K, replace=False) + lo
    hist: list[np.ndarray] = [cur]
    out = np.empty((steps, K), dtype=np.int32)
    for t in range(steps):
        nkeep = int(round(p["overlap"] * K))
        keep = rng.choice(cur, size=nkeep, replace=False)
        need = K - nkeep
        member = np.zeros(N, dtype=bool)
        member[cur] = True  # dropped picks of the previous step are not re-picked now (keeps overlap at `overlap`)
        new = []
        nre = rng.binomial(need, p["q"]) if p["q"] > 0 else 0
        if nre:
            pool = np.unique(np.concatenate(hist[-p["R"]:]))
            pool = pool[~member[pool]]
            if pool.size:
                pick = rng.choice(pool, size=min(nre, pool.size), replace=False)
                member[pick] = True
                new.append(pick)
        got = sum(a.size for a in new)
        while got < need:
            k = need - got
            nf = rng.binomial(k, p["pf"])
            f0 = int(fstart)
            cand = np.concatenate([
                rng.integers(f0, min(hi, f0 + fsz), size=nf * 2 + 8),
                rng.integers(lo, hi, size=(k - nf) * 2 + 8),
            ])
            cand = np.unique(cand)
            cand = cand[~member[cand]]
            rng.shuffle(cand)
            cand = cand[:k]
            member[cand] = True
            new.append(cand)
            got += cand.size
        cur = np.concatenate([keep] + new).astype(np.int64)
        out[t] = cur
        hist.append(cur)
        if len(hist) > p["R"] + 1:
            hist.pop(0)
        fstart += p["drift"]
        if fstart + fsz >= hi:
            fstart = lo
    return out


def overlap_stats(tr: np.ndarray, N: int) -> dict:
    ov = [np.intersect1d(tr[i], tr[i + 1]).size / tr.shape[1] for i in range(len(tr) - 1)]
    return {"overlap": float(np.mean(ov)), "touched_frac": float(np.unique(tr).size / N)}
