# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Correctness of sparsekv.tier: Triton gather_rows_tiered vs gather_rows_tiered_ref, multi-step, mixed residency.

CUDA_VISIBLE_DEVICES=1 python sparsekv/test_tier.py
Checks per step: staging rows identical, hot-buffer state (tag/last/pos2row/hand) identical, hot rows hold the right
host data, and the staged rows equal the ground-truth KV of each picked token. Also replays the step in a CUDA graph.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier import (HostKVMirror, HotBuffer, TierScratch, gather_rows_tiered,  # noqa: E402
                           gather_rows_tiered_ref)
from sparsekv.tier_trace import gen_trace  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
HKV, BS, W = 4, 864, 520
L, NREQ = 2, 2
CTX = [20000, 13000]
KSEL, SINK, WIN = 512, 64, 1024
C = KSEL + SINK + WIN
H = 2 * C
MAXBLK = max(CTX) // BS + 1
NB = NREQ * MAXBLK + 1


def build():
    mirror = HostKVMirror(L, NREQ * MAXBLK, HKV, BS, W, device=dev)
    hot = HotBuffer(L, 3, H, HKV, W, max_ctx=MAXBLK * BS, device=dev, age=2)
    # strided cache views, like vLLM's hybrid allocator (block stride > Hkv*BS*W because pages are shared/padded)
    PAD = 4096
    kvs = []
    for _ in range(L):
        raw = torch.randint(-128, 127, (NB, HKV * BS * W + PAD), dtype=torch.int8, device=dev)
        kvs.append(raw[:, : HKV * BS * W].view(NB, HKV, BS, W))
    assert not kvs[0].is_contiguous()
    gpu_bt = torch.full((NREQ, MAXBLK), -1, dtype=torch.int32, device=dev)
    host_bt = torch.full((NREQ, MAXBLK), -1, dtype=torch.int32, device=dev)
    full_bt = torch.zeros((NREQ, MAXBLK), dtype=torch.int32, device=dev)  # ground truth GPU blocks
    for r in range(NREQ):
        nblk = (CTX[r] + BS - 1) // BS
        gbl = list(range(1 + r * MAXBLK, 1 + r * MAXBLK + nblk))
        hbl = mirror.alloc(nblk)
        full_bt[r, :nblk] = torch.tensor(gbl, dtype=torch.int32)
        host_bt[r, :nblk] = torch.tensor(hbl, dtype=torch.int32)
        for l in range(L):
            mirror.mirror_blocks(l, kvs[l], gbl, hbl)
        # residency: block 0 resident for r=0 (sinks), tail 2 blocks resident, a middle block resident too
        res = {nblk - 1, nblk - 2, nblk // 2} | ({0} if r == 0 else set())
        for b in range(nblk):
            if b in res:
                gpu_bt[r, b] = gbl[b]
    torch.cuda.synchronize()
    return mirror, hot, kvs, gpu_bt, host_bt, full_bt


def make_idx(step, traces):
    idx = torch.zeros((NREQ, C), dtype=torch.int32)
    nv = torch.zeros(NREQ, dtype=torch.int32)
    for r in range(NREQ):
        N = CTX[r]
        top = traces[r][step]
        sinks = np.arange(SINK)
        winr = np.arange(N - WIN, N)
        row = np.concatenate([top, sinks, winr])
        idx[r, :row.size] = torch.from_numpy(row.astype(np.int32))
        nv[r] = row.size if r == 0 else row.size - 7  # second request: a few padded columns
    return idx.to(dev), nv.to(dev)


def main():
    m1, hot1, kvs, gpu_bt, host_bt, full_bt = build()
    hot2 = HotBuffer(L, 3, H, HKV, W, max_ctx=MAXBLK * BS, device=dev, age=2)
    slot = torch.tensor([2, 0], dtype=torch.int32, device=dev)
    traces = [gen_trace(CTX[r], 40, K=KSEL, sink=SINK, win=WIN, seed=r) for r in range(NREQ)]
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(NREQ * NBS, dtype=torch.int32, device=dev).view(NREQ, NBS)
    stg1 = torch.zeros((NREQ * NBS, HKV, BS, W), dtype=torch.int8, device=dev)
    stg2 = torch.zeros_like(stg1)
    scratch = TierScratch(NREQ, C, dev)
    tot = np.zeros(3, dtype=np.int64)
    for step in range(30):
        idx, nv = make_idx(step, traces)
        hot1.next_step(); hot2.next_step()
        st = int(hot2.step.item())
        for l in range(L):
            gather_rows_tiered(kvs[l], gpu_bt, m1.dev[l], host_bt, idx, nv, stg1, sbt, hot1, l, slot, scratch)
            c = gather_rows_tiered_ref(kvs[l], gpu_bt, m1.dev[l], host_bt, idx, nv, stg2, sbt, hot2.layer(l), slot,
                                       st, hot2.age)
            tot += c
            torch.cuda.synchronize()
            for r in range(NREQ):
                n = int(nv[r])
                j = torch.arange(n, device=dev)
                a = stg1[sbt[r, j // BS].long(), :, j % BS]
                b = stg2[sbt[r, j // BS].long(), :, j % BS]
                assert torch.equal(a, b), f"staging mismatch step {step} layer {l} req {r}"
                pos = idx[r, :n].long()
                truth = kvs[l][full_bt[r, pos // BS].long(), :, pos % BS]
                assert torch.equal(a, truth), f"staging != ground truth step {step} layer {l} req {r}"
            for name in ("tag", "last", "pos2row", "hand"):
                assert torch.equal(getattr(hot1, name)[l], getattr(hot2, name)[l]), f"{name} mismatch {step} {l}"
            # every valid hot row holds the right data
            for s in (0, 2):
                tg = hot1.tag[l, s].long()
                ok = tg >= 0
                if ok.any():
                    r = 0 if s == 2 else 1
                    p = tg[ok]
                    truth = kvs[l][full_bt[r, p // BS].long(), :, p % BS]
                    assert torch.equal(hot1.kv[l, s][ok], truth), f"hot data wrong step {step} layer {l} slot {s}"
    print(f"eager OK: 30 steps x {L} layers x {NREQ} reqs; hit/miss/resident = {tot.tolist()}; "
          f"device stats {hot1.stats.tolist()}")

    # CUDA graph: capture one step, replay with new idx, compare against eager on a cloned state
    idx_s, nv_s = make_idx(30, traces)
    g = torch.cuda.CUDAGraph()
    snap = {k: getattr(hot1, k).clone() for k in ("kv", "tag", "last", "pos2row", "hand", "step")}
    with torch.cuda.graph(g):
        hot1.next_step()
        for l in range(L):
            gather_rows_tiered(kvs[l], gpu_bt, m1.dev[l], host_bt, idx_s, nv_s, stg1, sbt, hot1, l, slot, scratch)
    # capture does not execute; restore anyway to be safe
    for k, v in snap.items():
        getattr(hot1, k).copy_(v)
    for step in range(30, 38):
        idx, nv = make_idx(step, traces)
        idx_s.copy_(idx); nv_s.copy_(nv)
        g.replay()
        hot2.next_step()
        st = int(hot2.step.item())
        for l in range(L):
            gather_rows_tiered_ref(kvs[l], gpu_bt, m1.dev[l], host_bt, idx, nv, stg2, sbt, hot2.layer(l), slot, st,
                                   hot2.age)
        torch.cuda.synchronize()
        for name in ("tag", "last", "pos2row", "hand"):
            assert torch.equal(getattr(hot1, name), getattr(hot2, name)), f"graph: {name} mismatch step {step}"
        for r in range(NREQ):
            n = int(nv[r]); j = torch.arange(n, device=dev)
            assert torch.equal(stg1[sbt[r, j // BS].long(), :, j % BS], stg2[sbt[r, j // BS].long(), :, j % BS])
    print("graph replay OK: 8 steps match the reference")

    # Track B adapter: physical token rows -> HND staging
    from sparsekv.tier import gather_physical_rows, gather_physical_rows_ref
    R = 50000
    rows = torch.randint(-128, 127, (R, HKV * W), dtype=torch.int8, device=dev)
    phys = torch.randint(0, R, (NREQ, C), dtype=torch.int32, device=dev)
    phys[0, 5::17] = -1
    nv = torch.tensor([C, C - 100], dtype=torch.int32, device=dev)
    stg1.zero_(); stg2.zero_()
    gather_physical_rows(rows, phys, nv, stg1, sbt)
    gather_physical_rows_ref(rows, phys, nv, stg2, sbt)
    torch.cuda.synchronize()
    assert torch.equal(stg1, stg2), "gather_physical_rows mismatch"
    print("gather_physical_rows OK")
    m1.close()


if __name__ == "__main__":
    main()
