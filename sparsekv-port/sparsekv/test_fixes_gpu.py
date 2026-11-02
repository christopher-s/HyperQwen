# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""GPU regression tests for stream I review fixes (REVIEW.md G-ids).

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> python sparsekv/test_fixes_gpu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv import triton_ops as T  # noqa: E402

dev = "cuda"


def test_g6_write_sketch_short_slot_mapping():
    """write_sketch with len(slot_mapping) < len(key) must only write the first len(slot_mapping) rows."""
    torch.manual_seed(6)
    HKV, D, R, NS = 4, 256, 64, 864 * 4
    basis = torch.linalg.qr(torch.randn(HKV, D, R, device=dev))[0].transpose(1, 2).contiguous()
    key = torch.randn(40, HKV, D, device=dev, dtype=torch.bfloat16)
    slots = torch.randperm(NS, device=dev)[:24].to(torch.int64)
    slots[5] = -1
    a_q8 = torch.full((NS + 1, HKV, R), 77, dtype=torch.int8, device=dev)
    a_sc = torch.full((NS + 1, HKV), 7.0, dtype=torch.float16, device=dev)
    b_q8, b_sc = a_q8.clone(), a_sc.clone()
    T.write_sketch(key, slots, basis, a_q8, a_sc)                      # key longer than mapping
    T.write_sketch(key[:24].contiguous(), slots, basis, b_q8, b_sc)    # explicit truncation
    torch.cuda.synchronize()
    assert torch.equal(a_q8, b_q8) and torch.equal(a_sc, b_sc), "rows beyond len(slot_mapping) were written"
    touched = torch.zeros(NS + 1, dtype=torch.bool, device=dev)
    touched[torch.where(slots < 0, NS, slots)] = True
    assert bool((a_q8[~touched] == 77).all()), "write outside the mapped slots"
    print("G-6 write_sketch short slot_mapping: OK")


def test_g10_host_bt_with_own_row_stride():
    """gather_rows_tiered with a padded host_bt (row stride != gpu_bt's) must match the reference and ground truth."""
    from sparsekv import test_tier as tt
    from sparsekv.tier import HotBuffer, TierScratch, gather_rows_tiered, gather_rows_tiered_ref
    from sparsekv.tier_trace import gen_trace
    torch.manual_seed(10)
    m1, hot1, kvs, gpu_bt, host_bt, full_bt = tt.build()
    wide = torch.full((tt.NREQ, tt.MAXBLK + 37), -1, dtype=torch.int32, device=dev)
    wide[:, : tt.MAXBLK] = host_bt
    hbt = wide[:, : tt.MAXBLK]                   # stride(0) = MAXBLK + 37 != gpu_bt.stride(0)
    assert hbt.stride(0) != gpu_bt.stride(0)
    hot2 = HotBuffer(tt.L, 3, tt.H, tt.HKV, tt.W, max_ctx=tt.MAXBLK * tt.BS, device=dev, age=2)
    slot = torch.tensor([2, 0], dtype=torch.int32, device=dev)
    traces = [gen_trace(tt.CTX[r], 6, K=tt.KSEL, sink=tt.SINK, win=tt.WIN, seed=r) for r in range(tt.NREQ)]
    BS, C = tt.BS, tt.C
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(tt.NREQ * NBS, dtype=torch.int32, device=dev).view(tt.NREQ, NBS)
    stg1 = torch.zeros((tt.NREQ * NBS, tt.HKV, BS, tt.W), dtype=torch.int8, device=dev)
    stg2 = torch.zeros_like(stg1)
    scratch = TierScratch(tt.NREQ, C, dev)
    for step in range(6):
        idx, nv = tt.make_idx(step, traces)
        hot1.next_step(); hot2.next_step()
        st = int(hot2.step.item())
        for l in range(tt.L):
            gather_rows_tiered(kvs[l], gpu_bt, m1.dev[l], hbt, idx, nv, stg1, sbt, hot1, l, slot, scratch)
            gather_rows_tiered_ref(kvs[l], gpu_bt, m1.dev[l], host_bt, idx, nv, stg2, sbt, hot2.layer(l), slot,
                                   st, hot2.age)
            torch.cuda.synchronize()
            for r in range(tt.NREQ):
                n = int(nv[r]); j = torch.arange(n, device=dev)
                a = stg1[sbt[r, j // BS].long(), :, j % BS]
                pos = idx[r, :n].long()
                truth = kvs[l][full_bt[r, pos // BS].long(), :, pos % BS]
                assert torch.equal(a, stg2[sbt[r, j // BS].long(), :, j % BS]), f"vs ref: step {step} l {l} r {r}"
                assert torch.equal(a, truth), f"vs truth: step {step} l {l} r {r}"
    m1.close()
    print("G-10 gather_rows_tiered with padded host_bt: OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL OK")
