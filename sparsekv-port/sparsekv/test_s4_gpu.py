# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""GPU tests for the S4 (4 streams per card) changes.

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> venv/bin/python sparsekv/test_s4_gpu.py
  int4 sketch: pack/unpack round trip; triton write_sketch == ref; triton select == ref select on int4 sketches;
               planted strong keys are selected with int4 as with int8.
  hot buffer:  a request on the padding slot (no hot KV rows) stages correct rows from host and caches nothing.
  suffix:      select_suffix (pseudo-requests of 16 rows) == ref.select over all suffix rows; causal pthfa prefill
               launch over the [topk | sinks | window] staging == fp32 attention over exactly the selected rows;
               copy_blocks == indexing (strided source).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv import ref as R  # noqa: E402
from sparsekv import triton_ops as T  # noqa: E402

dev = "cuda"
Hq, Hkv, D, BS, G, RK = 24, 4, 256, 864, 6, 64


def _basis(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.linalg.qr(torch.randn(D, RK, generator=g))[0].T for _ in range(Hkv)]).to(dev).contiguous()


def test_int4_pack_roundtrip():
    q = torch.randint(-8, 8, (1000, Hkv, RK), dtype=torch.int32)
    assert torch.equal(R.unpack_int4(R.pack_int4(q)), q)
    print("int4 pack/unpack round trip: OK")


def test_int4_write_sketch_matches_ref():
    torch.manual_seed(1)
    NS = BS * 3
    basis = _basis(1)
    key = torch.randn(500, Hkv, D, device=dev, dtype=torch.bfloat16)
    slots = torch.randperm(NS, device=dev)[:500].to(torch.int64)
    slots[7] = -1
    a8 = torch.zeros(NS + 1, Hkv, RK // 2, dtype=torch.int8, device=dev)
    asc = torch.zeros(NS + 1, Hkv, dtype=torch.float16, device=dev)
    b8, bsc = a8.clone(), asc.clone()
    T.write_sketch(key, slots, basis, a8, asc)
    R.write_sketch(key, slots, basis, b8, bsc)
    torch.cuda.synchronize()
    va, vb = R.unpack_int4(a8), R.unpack_int4(b8)
    diff = (va - vb).abs()
    assert int(diff.max()) <= 1, f"int4 sketch differs by {int(diff.max())} levels"
    assert float((diff > 0).float().mean()) < 1e-3, "too many rounding differences"
    assert torch.allclose(asc.float(), bsc.float(), rtol=1e-3, atol=1e-6)
    print(f"int4 write_sketch vs ref: OK (rounding-tie diffs {float((diff > 0).float().mean()):.2e})")


def _setup(seq, planted=200, seed=2):
    """Paged sketch caches (int8 and int4) of random keys with `planted` keys per request aligned with the queries."""
    torch.manual_seed(seed)
    nreq, QL = len(seq), 8
    maxblk = max(seq) // BS + 2
    NB = nreq * maxblk + 1
    basis = _basis(seed)
    bt = (1 + torch.arange(nreq * maxblk, device=dev, dtype=torch.int32)).view(nreq, maxblk).contiguous()
    query = torch.randn(nreq * QL, Hq, D, device=dev, dtype=torch.bfloat16)
    keys = torch.randn(NB * BS, Hkv, D, device=dev) * 0.5
    want = []
    for r, L in enumerate(seq):
        pos = torch.randperm(L - 64 - 1024, device=dev)[:planted] + 64
        want.append(set(pos.tolist()))
        qdir = query[r * QL:(r + 1) * QL].float().view(QL, Hkv, G, D).mean(dim=(0, 2))  # [Hkv, D]
        slots = bt[r, pos // BS].long() * BS + pos % BS
        keys[slots] = qdir[None] / qdir.norm(dim=-1, keepdim=True)[None] * 40.0
    keys = keys.to(torch.bfloat16)
    allslots = torch.arange(NB * BS, device=dev)
    sk = {}
    for bits in (8, 4):
        q8 = torch.zeros(NB * BS + 1, Hkv, RK if bits == 8 else RK // 2, dtype=torch.int8, device=dev)
        sc = torch.zeros(NB * BS + 1, Hkv, dtype=torch.float16, device=dev)
        T.write_sketch(keys, allslots, basis, q8, sc)
        sk[bits] = (q8, sc)
    qsl = torch.arange(0, (nreq + 1) * QL, QL, dtype=torch.int32, device=dev)
    sl = torch.tensor(seq, dtype=torch.int32, device=dev)
    return query, qsl, sl, bt, basis, sk, want


def test_int4_select_matches_ref_and_finds_planted():
    seq = [40000, 25000]
    query, qsl, sl, bt, basis, sk, want = _setup(seq)
    got = {}
    for bits in (8, 4):
        q8, sc = sk[bits]
        it, nt = T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
        ir, nr = R.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_query_len=8)
        torch.cuda.synchronize()
        assert torch.equal(nt, nr)
        for r in range(len(seq)):
            a, b = set(it[r, :2048].tolist()), set(ir[r, :2048].tolist())
            ov = len(a & b) / 2048
            assert ov > 0.97, f"int{bits} triton vs ref top-k overlap {ov:.3f}"
            hit = len(a & want[r]) / len(want[r])
            assert hit > 0.99, f"int{bits}: planted keys selected {hit:.3f}"
            got[(bits, r)] = a
    ov84 = [len(got[(8, r)] & got[(4, r)]) / 2048 for r in range(len(seq))]
    print(f"int4 select: triton == ref (overlap > 0.97), planted keys found; int8 vs int4 top-k overlap "
          f"{[round(x, 3) for x in ov84]}: OK")


def test_int4_select_graph_capture():
    seq = [30000, 5000]
    query, qsl, sl, bt, basis, sk, _ = _setup(seq, planted=50, seed=3)
    q8, sc = sk[4]
    i0, n0 = T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        i1, n1 = T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(n0, n1) and torch.equal(i0, i1)
    print("int4 select under CUDA-graph capture == eager: OK")


def test_hot_padding_slot_has_no_kv_rows():
    from sparsekv.tier import HostKVMirror, HotBuffer, TierScratch, gather_rows_tiered
    torch.manual_seed(5)
    HK, W, L = 4, 520, 1
    ctx, ksel, sink, win = 9000, 256, 64, 1024
    C = ksel + sink + win
    maxblk = ctx // BS + 1
    NB = maxblk + 1
    kv = torch.randint(-128, 127, (NB, HK, BS, W), dtype=torch.int8, device=dev)
    mirror = HostKVMirror(L, maxblk, HK, BS, W, device=dev)
    gbl = list(range(1, 1 + maxblk))
    hbl = mirror.alloc(maxblk)
    mirror.mirror_blocks(0, kv, gbl, hbl)
    torch.cuda.synchronize()
    hot = HotBuffer(L, 2, 2 * C, HK, W, max_ctx=maxblk * BS, device=dev, kv_slots=1)  # slot 1 = padding slot
    assert hot.kv.shape[1] == 1
    gpu_bt = torch.full((1, maxblk), -1, dtype=torch.int32, device=dev)  # everything spilled
    host_bt = torch.tensor([hbl], dtype=torch.int32, device=dev)
    pos = torch.cat([torch.randperm(ctx - sink - win, device=dev)[:ksel] + sink, torch.arange(sink, device=dev),
                     torch.arange(ctx - win, ctx, device=dev)]).to(torch.int32)[None].contiguous()
    nv = torch.tensor([C], dtype=torch.int32, device=dev)
    nbs = -(-C // BS)
    st = torch.zeros(nbs, HK, BS, W, dtype=torch.int8, device=dev)
    sbt = torch.arange(nbs, dtype=torch.int32, device=dev)[None].contiguous()
    slot = torch.tensor([1], dtype=torch.int32, device=dev)
    hot.next_step()
    gather_rows_tiered(kv, gpu_bt, mirror.dev[0], host_bt, pos, nv, st, sbt, hot, 0, slot, TierScratch(1, C, dev))
    torch.cuda.synchronize()
    j = torch.arange(C, device=dev)
    p = pos[0].long()
    truth = kv[torch.tensor(gbl, device=dev)[p // BS], :, p % BS]
    assert torch.equal(st[sbt[0, j // BS].long(), :, j % BS], truth), "padding-slot rows staged wrong"
    assert bool((hot.pos2row[0, 1] == -1).all()) and bool((hot.tag[0, 1] == -1).all()), "padding slot cached rows"
    print("hot buffer padding slot (no KV rows): stages from host, caches nothing: OK")


def _views(kv):
    num_blocks, nkv, bs, content = kv.shape
    padded = content // 2
    base = torch.tensor([], dtype=torch.float32, device=kv.device).set_(kv.untyped_storage())
    st = [x // 4 for x in kv.stride()]
    off = kv.storage_offset() // 4
    ks = torch.as_strided(base, (num_blocks, bs, nkv), (st[0], st[2], st[1]), off + D // 4)
    vs = torch.as_strided(base, (num_blocks, bs, nkv), (st[0], st[2], st[1]), off + (padded + D) // 4)
    kc, vc = kv.transpose(1, 2).split(padded, dim=-1)
    return kc[..., :D], vc[..., :D], ks, vs


def test_select_suffix_matches_ref():
    torch.manual_seed(7)
    L, ql = 30000, 200
    maxblk = L // BS + 2
    basis = _basis(7)
    bt = (1 + torch.arange(maxblk, device=dev, dtype=torch.int32))[None].contiguous()
    NB = maxblk + 1
    keys = (torch.randn(NB * BS, Hkv, D, device=dev) * 0.5).to(torch.bfloat16)
    q = torch.randn(ql, Hq, D, device=dev, dtype=torch.bfloat16)
    planted = (torch.randperm(L - 64 - 1024, device=dev)[:100] + 64)
    qdir = q.float().view(ql, Hkv, G, D)[:ql // 2].mean(dim=(0, 2))   # only the first half of the rows "ask" for them
    keys[bt[0, planted // BS].long() * BS + planted % BS] = (qdir / qdir.norm(dim=-1, keepdim=True) * 40).to(keys.dtype)
    for bits in (8, 4):
        q8 = torch.zeros(NB * BS + 1, Hkv, RK if bits == 8 else RK // 2, dtype=torch.int8, device=dev)
        sc = torch.zeros(NB * BS + 1, Hkv, dtype=torch.float16, device=dev)
        T.write_sketch(keys, torch.arange(NB * BS, device=dev), basis, q8, sc)
        sl = torch.tensor([L], dtype=torch.int32, device=dev)
        it, nt = T.select_suffix(q, [ql], [L], sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16)
        qsl = torch.tensor([0, ql], dtype=torch.int32, device=dev)
        ir, nr = R.select(q, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_query_len=ql)
        torch.cuda.synchronize()
        assert int(nt[0]) == int(nr[0]) == 2048 + 64 + 1024
        a, b = set(it[0, :2048].tolist()), set(ir[0, :2048].tolist())
        ov = len(a & b) / 2048
        hit = len(a & set(planted.tolist())) / 100
        assert ov > 0.95 and hit > 0.99, (bits, ov, hit)
        assert torch.equal(it[0, 2048:].cpu(), ir[0, 2048:].cpu())
        print(f"select_suffix int{bits} ({ql} rows): top-k overlap with ref {ov:.3f}, planted found {hit:.2f}: OK")


def test_suffix_causal_attention():
    """Segmented suffix (S4): segments [172, 128] of one request, each with its own selection and causal end, in ONE
    pthfa prefill launch over the staging == fp32 attention of every row over exactly its segment's selected rows."""
    from pth_flash_attn import ops
    from pth_flash_attn.planner import plan_launch
    torch.manual_seed(8)
    W_ = 2 * (D + 4)
    L, ksel, sink, win = 20000, 1024, 64, 1024
    segs = [172, 128]
    ql = sum(segs)
    ends = [L - segs[1], L]
    C = ksel + sink + win
    maxblk = L // BS + 2
    NB = maxblk + 1
    kv = torch.randint(-127, 128, (NB, Hkv, BS, W_), dtype=torch.int8, device=dev)
    k, v, ks, vs = _views(kv)
    ks.copy_(torch.rand_like(ks) * 0.02 + 0.005)
    vs.copy_(torch.rand_like(vs) * 0.02 + 0.005)
    bt1 = (1 + torch.arange(maxblk, device=dev, dtype=torch.int32))[None]
    bt = bt1.expand(2, -1).contiguous()
    basis = _basis(8)
    q8 = torch.zeros(NB * BS + 1, Hkv, RK, dtype=torch.int8, device=dev)
    sc = torch.zeros(NB * BS + 1, Hkv, dtype=torch.float16, device=dev)
    kd = (k.float() * ks.unsqueeze(-1)).reshape(NB * BS, Hkv, D).to(torch.bfloat16)
    T.write_sketch(kd, torch.arange(NB * BS, device=dev), basis, q8, sc)
    q = torch.randn(ql, Hq, D, device=dev, dtype=torch.bfloat16)
    sl = torch.tensor(ends, dtype=torch.int32, device=dev)
    idx, nv = T.select_suffix(q, segs, ends, sl, bt, q8, sc, basis, ksel, sink, win, 1.0 / 16)
    nbs = -(-C // BS)
    st = torch.zeros(2 * nbs, Hkv, BS, W_, dtype=torch.int8, device=dev)
    sbt = torch.arange(2 * nbs, dtype=torch.int32, device=dev).view(2, nbs)
    T.gather_rows(kv, bt, idx, nv, st, sbt)
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    plan = plan_launch(num_tokens=ql, num_reqs=2, max_query_len=max(segs), G=G, Hkv=Hkv, D=D, num_sms=nsm,
                       is_prefill=True, ws_token_splits=0)
    assert plan.num_splits == 1
    out = torch.empty_like(q)
    ws = torch.empty(1, device=dev)
    a_, b_, c_, d_ = _views(st)
    qsl = torch.tensor([0, segs[0], ql], dtype=torch.int32, device=dev)
    ops.pth_flash_attn_fwd(out, q, a_, b_, c_, d_, sbt, nv, qsl, max(segs), plan.wm, plan.num_splits, 1.0 / 16,
                           ws, ws, 0, True)
    kdq, vdq = k.float() * ks.unsqueeze(-1), v.float() * vs.unsqueeze(-1)
    errs = []
    for j, (q0, n) in enumerate([(0, segs[0]), (segs[0], segs[1])]):
        pos = idx[j].long()
        assert torch.equal(pos[C - win:].cpu(), torch.arange(ends[j] - win, ends[j])), "window = last WIN tokens"
        kk, vv = kdq[bt1[0, pos // BS].long(), pos % BS], vdq[bt1[0, pos // BS].long(), pos % BS]
        qf = q[q0:q0 + n].float().view(n, Hkv, G, D)
        logits = torch.einsum("ihgd,jhd->ihgj", qf, kk) / 16
        cols = torch.arange(C, device=dev)
        mask = cols[None, :] <= (C - n + torch.arange(n, device=dev))[:, None]
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        ref = torch.einsum("ihgj,jhd->ihgd", torch.softmax(logits, -1), vv).reshape(n, Hq, D)
        errs.append((out[q0:q0 + n].float() - ref).abs().max().item() / ref.abs().max().item())
    assert max(errs) < 2e-2, errs
    print(f"segmented suffix causal attention == fp32 reference per segment (rel err {max(errs):.1e}): OK")


def test_select_reuse():
    seq = [40000, 25000]
    query, qsl, sl, bt, basis, sk, _ = _setup(seq, planted=50, seed=11)
    q8, sc = sk[8]
    store = torch.zeros(5, 2048, dtype=torch.int32, device=dev)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=dev)
    full = torch.zeros(2, dtype=torch.int32, device=dev)
    i0, n0 = T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
    i1, n1 = T.select(query, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8,
                      reuse=full, topk_store=store, slots=slots)
    assert torch.equal(i0, i1) and torch.equal(store[3], i0[0, :2048]) and torch.equal(store[1], i0[1, :2048])
    # next step: different query, 6 more tokens; row 0 reuses, row 1 selects again
    q2 = torch.randn_like(query)
    sl2 = sl + 6
    ru = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    def run():
        return T.select(q2, qsl, sl2, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8,
                        reuse=ru, topk_store=store, slots=slots)
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    ref1, _ = T.select(q2, qsl, sl2, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=8)
    store_bak = store.clone()
    with torch.cuda.stream(s_):
        run()
    torch.cuda.current_stream().wait_stream(s_)
    store.copy_(store_bak)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        i2, n2 = run()
    store.copy_(store_bak)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(i2[0, :2048], i0[0, :2048]), "reuse row must keep the stored top-k"
    assert torch.equal(i2[0, 2048:2112].cpu(), torch.arange(64))
    assert torch.equal(i2[0, 2112:].cpu(), torch.arange(seq[0] + 6 - 1024, seq[0] + 6)), "window from the new length"
    assert torch.equal(i2[1], ref1[1]) and torch.equal(store[1], ref1[1, :2048]), "full row selects + stores"
    assert torch.equal(store[3], i0[0, :2048])
    print("selection reuse (per-slot store, window rebuilt, mixed rows, graph replay): OK")


def _real_rows(nb, hk=4, bs=BS):
    """Plausible pthfa int8 rows: int8 values with outliers, positive fp32 scales."""
    kv = torch.randint(-60, 61, (nb, hk, bs, 520), dtype=torch.int8, device=dev)
    kv[..., ::37] = torch.randint(-127, 128, kv[..., ::37].shape, dtype=torch.int8, device=dev)
    f = kv.view(torch.int32)
    f[..., 64] = (torch.rand(nb, hk, bs, device=dev) * 0.02 + 0.001).view(torch.int32)
    f[..., 129] = (torch.rand(nb, hk, bs, device=dev) * 0.02 + 0.001).view(torch.int32)
    return kv


def _deq_float(rows):
    v = torch.cat([rows[..., :256], rows[..., 260:516]], -1).float()
    s = rows.view(torch.int32)[..., [64, 129]].view(torch.float32)
    return torch.cat([v[..., :256] * s[..., :1], v[..., 256:] * s[..., 1:]], -1)


def test_host4_kernels_match_ref_and_roundtrip():
    from sparsekv.tier import (W4H, copy_blocks_from_host4, dequant_rows4_ref, quant_blocks_to_host4,
                               quant_rows4_ref, alloc_pinned, uva_view)
    torch.manual_seed(12)
    kv = _real_rows(6)
    host = alloc_pinned((5, 4, BS, W4H))
    hdev = uva_view(host)
    si = torch.tensor([4, 1, 5], dtype=torch.int32, device=dev)
    di = torch.tensor([0, 3, 2], dtype=torch.int32, device=dev)
    quant_blocks_to_host4(kv, si, hdev, di)
    torch.cuda.synchronize()
    ref = quant_rows4_ref(kv[si.long()])
    got = host[di.cpu().long()].to(dev)
    nib = lambda x: torch.cat([(x[..., :128].to(torch.int32) << 28) >> 28, x[..., :128].to(torch.int32) >> 4], -1)
    dn = (nib(got) - nib(ref)).abs()
    assert int(dn.max()) <= 1 and float((dn > 0).float().mean()) < 1e-3, "int4 quant differs from ref"
    out = torch.zeros(4, 4, BS, 520, dtype=torch.int8, device=dev)
    copy_blocks_from_host4(out, torch.tensor([1, 3], dtype=torch.int32, device=dev), hdev,
                           torch.tensor([0, 3], dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    exp = dequant_rows4_ref(host[[0, 3]].to(dev))
    d8 = (out[[1, 3]].to(torch.int32) - exp.to(torch.int32))
    vals = torch.cat([d8[..., :256], d8[..., 260:516]], -1).abs()
    assert int(vals.max()) <= 1, "dequant values differ from ref"
    a, b = _deq_float(out[[1, 3]]), _deq_float(kv[[4, 1]])
    rel = ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()
    assert rel < 0.12, rel
    print(f"int4 host tier: quant/dequant kernels == ref, round-trip rel err {rel:.3f}: OK")


def test_gather_tiered_with_int4_host():
    from sparsekv.tier import (W4H, HotBuffer, TierScratch, alloc_pinned, dequant_rows4_ref, gather_rows_tiered,
                               quant_blocks_to_host4, uva_view)
    torch.manual_seed(13)
    ctx, ksel, sink, win = 9000, 256, 64, 1024
    C = ksel + sink + win
    maxblk = ctx // BS + 1
    kv = _real_rows(maxblk + 1)
    host = alloc_pinned((1, maxblk, 4, BS, W4H))
    hdev = uva_view(host)
    gbl = torch.arange(1, 1 + maxblk, dtype=torch.int32, device=dev)
    hbl = torch.arange(maxblk, dtype=torch.int32, device=dev)
    quant_blocks_to_host4(kv, gbl, hdev[0], hbl)
    torch.cuda.synchronize()
    truth4 = dequant_rows4_ref(host[0].to(dev))                       # [maxblk, 4, BS, 520] as fetched from host
    gpu_bt = torch.full((1, maxblk), -1, dtype=torch.int32, device=dev)
    gpu_bt[0, -2:] = gbl[-2:]                                          # last 2 blocks resident
    host_bt = hbl[None].contiguous()
    nbs = -(-C // BS)
    sbt = torch.arange(nbs, dtype=torch.int32, device=dev)[None].contiguous()
    for kv_slots, slot_id in ((2, 0), (1, 1)):                         # real slot (cached) / padding slot
        hot = HotBuffer(1, 2, 2 * C, 4, 520, max_ctx=maxblk * BS, device=dev, kv_slots=kv_slots)
        slot = torch.tensor([slot_id], dtype=torch.int32, device=dev)
        for step in range(2):                                          # step 0: misses, step 1: hits (real slot)
            g = torch.Generator(device="cpu").manual_seed(100 + step)
            pos = torch.cat([torch.randperm(ctx - sink - win, generator=g)[:ksel].to(dev) + sink,
                             torch.arange(sink, device=dev), torch.arange(ctx - win, ctx, device=dev)]).to(torch.int32)
            pos = pos[None].contiguous()
            st = torch.zeros(nbs, 4, BS, 520, dtype=torch.int8, device=dev)
            hot.next_step()
            gather_rows_tiered(kv, gpu_bt, hdev[0], host_bt, pos, torch.tensor([C], dtype=torch.int32, device=dev),
                               st, sbt, hot, 0, slot, TierScratch(1, C, dev))
            torch.cuda.synchronize()
            p = pos[0].long()
            blk, off = p // BS, p % BS
            resident = blk >= maxblk - 2
            exp = torch.where(resident[:, None, None], kv[gbl[blk].long(), :, off], truth4[blk, :, off])
            j = torch.arange(C, device=dev)
            got = st[sbt[0, j // BS].long(), :, j % BS]
            if not torch.equal(got, exp):
                bad = (got != exp).any(-1).any(-1)
                gi, ei = got.to(torch.int32), exp.to(torch.int32)
                vd = torch.cat([(gi - ei)[..., :256], (gi - ei)[..., 260:516]], -1).abs()
                sc_eq = torch.equal(got.view(torch.int32)[..., [64, 129]], exp.view(torch.int32)[..., [64, 129]])
                print(f"DIAG kv_slots={kv_slots} step={step}: bad rows {int(bad.sum())}/{C} (resident "
                      f"{int((bad & resident).sum())}, host {int((bad & ~resident).sum())}), max value diff "
                      f"{int(vd.max())}, frac {float((vd > 0).float().mean()):.2e}, scales equal {sc_eq}; "
                      f"first bad cols {bad.nonzero()[:5, 0].tolist()}; zero rows in got "
                      f"{int((got == 0).all(-1).all(-1).sum())}")
                assert int(vd.max()) <= 1 and sc_eq and int(bad[resident].sum()) == 0, (kv_slots, step)
    print("tiered gather with int4 host (misses, hits, padding slot) == dequantised host rows / resident rows: OK")


def test_copy_blocks():
    from sparsekv.tier import copy_blocks
    torch.manual_seed(9)
    raw = torch.randint(-128, 127, (12, Hkv * BS * 520 + 4096), dtype=torch.int8, device=dev)
    src = raw[:, : Hkv * BS * 520].view(12, Hkv, BS, 520)   # strided, like vLLM's hybrid pages
    dst = torch.zeros(9, Hkv, BS, 520, dtype=torch.int8, device=dev)
    si = torch.tensor([3, 7, 0, 11], dtype=torch.int32, device=dev)
    di = torch.tensor([1, 2, 5, 8], dtype=torch.int32, device=dev)
    copy_blocks(dst, di, src, si)
    torch.cuda.synchronize()
    assert torch.equal(dst[di.long()], src[si.long()]) and int(dst[0].abs().sum()) == 0
    print("copy_blocks (strided source): OK")


if __name__ == "__main__":
    test_copy_blocks()
    test_host4_kernels_match_ref_and_roundtrip()
    test_gather_tiered_with_int4_host()
    test_select_reuse()
    test_select_suffix_matches_ref()
    test_suffix_causal_attention()
    test_int4_pack_roundtrip()
    test_int4_write_sketch_matches_ref()
    test_int4_select_matches_ref_and_finds_planted()
    test_int4_select_graph_capture()
    test_hot_padding_slot_has_no_kv_rows()
    print("ALL OK")
