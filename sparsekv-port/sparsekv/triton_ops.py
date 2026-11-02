# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Fast, CUDA-graph-capturable versions of the sparse-KV ops (same signatures as ref.py).

No host syncs, no data-dependent shapes: every grid / buffer size depends only on (Tq, nreq, maxblk) and the
constants. Per-request work is bounded on device by seq_lens.

select pipeline (5 launches + top-k):
  1. _qproj_kernel    q[Tq,Hq,D] @ basis^T -> qn[Tq,Hq,64] (int8 or fp16, per-row scale qs incl. scale*log2e)
  2. _stats_kernel    split-K online softmax stats (max, sum-exp) per (req, kv head, verify row x q-head)
  3. _combine_kernel  reduce the splits -> m, 1/l per row; top-k row length (seq_len - win, or 0 for short reqs)
  4. _score_kernel    pooled score per token in [sink, seq_len - win) into a [nreq, maxblk*BS] fp32 buffer
                      (sum over rows, q-heads and the 4 kv heads); [0, sink) = -inf
  5. top-k            vllm top_k_per_row_decode (default) or torch.topk
  6. _assemble_kernel idx = [topk | sinks | window], n_valid; short requests -> arange(seq_len)
"""
import os

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

BS = 864
LOG2E = 1.4426950408889634

# "radix" (Triton radix select, below), "per_row" (vllm top_k_per_row_decode) or "torch"
TOPK_IMPL = os.environ.get("SPARSE_TOPK_IMPL", "radix")
# 1 = int8 x int8 tensor-core scoring (query projection quantised per row), 0 = fp16 x fp16 (fp32 accumulate)
QUANT_Q = os.environ.get("SPARSE_QUANT_Q", "1") == "1"
# S4: normaliser pass samples every STATS_SUB-th tile of the bulk (1 = exact, the contract behaviour)
STATS_SUB = int(os.environ.get("SPARSE_STATS_SUB", "1"))

_NSM = {}
# launch tuning (num_warps / num_stages per kernel); bench/prof scripts override these
CFG = {"stats_warps": 2, "stats_stages": 3, "score_warps": 8, "score_stages": 2, "split_a": 4, "split_b": 2,
       "block_a": 128, "block_b": 128,           # measured best on 3090 (sweep in prof.py ksweep, 150K B=1/2)
       "rx_digits": 8, "rx_block": 1024, "rx_split": 2,
       "exact_rows": True, "half_exp": False}   # exact_rows: 48 rows as 32+16 tiles; half_exp: f16x2 ex2
# S4: launch config for int4 sketches (run_s4/cfgsweep.out, 3090, 4 x 246K, STATS_SUB=4: 0.98 -> 0.78 ms per layer-step)
CFG4 = {"stats_warps": 2, "split_a": 8, "block_a": 64, "score_warps": 4, "split_b": 2, "block_b": 128}


def _num_sms(dev):
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _NSM:
        _NSM[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _NSM[i]


# --------------------------------------------------------------------------------------------- write_sketch
@triton.jit
def _write_sketch4_kernel(key_ptr, sk0, sk1, slot_ptr, basis_ptr, q8_ptr, sc_ptr, T, DUMMY,
                          HKV: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_TT: tl.constexpr,
                          BLOCK_D: tl.constexpr):
    # S4 int4 sketch: scale amax/7, byte j = dim j (low nibble) | dim j + R/2 (high nibble)
    R2: tl.constexpr = R // 2
    pt = tl.program_id(0)
    h = tl.program_id(1)
    t = pt * BLOCK_TT + tl.arange(0, BLOCK_TT)
    ok = t < T
    rr = tl.arange(0, R2)
    lo = tl.zeros([BLOCK_TT, R2], dtype=tl.float32)
    hi = tl.zeros([BLOCK_TT, R2], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        dd = d0 + tl.arange(0, BLOCK_D)
        k = tl.load(key_ptr + t.to(tl.int64)[:, None] * sk0 + h * sk1 + dd[None, :], mask=ok[:, None], other=0.0)
        b_lo = tl.load(basis_ptr + h * R * D + rr[:, None] * D + dd[None, :])
        b_hi = tl.load(basis_ptr + h * R * D + (rr + R2)[:, None] * D + dd[None, :])
        lo += tl.dot(k.to(tl.float32), tl.trans(b_lo), input_precision="ieee")
        hi += tl.dot(k.to(tl.float32), tl.trans(b_hi), input_precision="ieee")
    s = tl.maximum(tl.max(tl.abs(lo), 1), tl.max(tl.abs(hi), 1)) / 7.0
    safe = tl.where(s > 0, s, 1.0)
    ql = tl.minimum(tl.maximum(libdevice.rint(lo / safe[:, None]), -7.0), 7.0)
    qh = tl.minimum(tl.maximum(libdevice.rint(hi / safe[:, None]), -7.0), 7.0)
    ql = tl.where(s[:, None] > 0, ql, 0.0).to(tl.int32)
    qh = tl.where(s[:, None] > 0, qh, 0.0).to(tl.int32)
    packed = qh * 16 + (ql & 15)
    slot = tl.load(slot_ptr + t, mask=ok, other=-1).to(tl.int64)
    slot = tl.where(slot < 0, DUMMY, slot)
    tl.store(q8_ptr + slot[:, None] * (HKV * R2) + h * R2 + rr[None, :], packed.to(tl.int8), mask=ok[:, None])
    tl.store(sc_ptr + slot * HKV + h, s.to(tl.float16), mask=ok)


@triton.jit
def _write_sketch_kernel(key_ptr, sk0, sk1, slot_ptr, basis_ptr, q8_ptr, sc_ptr, T, DUMMY,
                         HKV: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_TT: tl.constexpr,
                         BLOCK_D: tl.constexpr):
    pt = tl.program_id(0)
    h = tl.program_id(1)
    t = pt * BLOCK_TT + tl.arange(0, BLOCK_TT)
    ok = t < T
    rr = tl.arange(0, R)
    acc = tl.zeros([BLOCK_TT, R], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        dd = d0 + tl.arange(0, BLOCK_D)
        k = tl.load(key_ptr + t.to(tl.int64)[:, None] * sk0 + h * sk1 + dd[None, :], mask=ok[:, None], other=0.0)
        bmat = tl.load(basis_ptr + h * R * D + rr[:, None] * D + dd[None, :])
        acc += tl.dot(k.to(tl.float32), tl.trans(bmat), input_precision="ieee")
    s = tl.max(tl.abs(acc), 1) / 127.0
    safe = tl.where(s > 0, s, 1.0)
    q = libdevice.rint(acc / safe[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    q = tl.where(s[:, None] > 0, q, 0.0)
    slot = tl.load(slot_ptr + t, mask=ok, other=-1).to(tl.int64)
    slot = tl.where(slot < 0, DUMMY, slot)
    tl.store(q8_ptr + slot[:, None] * (HKV * R) + h * R + rr[None, :], q.to(tl.int8), mask=ok[:, None])
    tl.store(sc_ptr + slot * HKV + h, s.to(tl.float16), mask=ok)


def write_sketch(key, slot_mapping, basis_l, sk_q8, sk_scale):
    T, Hkv, D = key.shape
    # G-6: never read slot_mapping past its end (a shorter mapping than key would scatter garbage slots);
    # same rule as ref.write_sketch: n = min(len(key), len(slot_mapping)). Shape-only, graph-safe.
    T = min(T, slot_mapping.shape[0])
    if T == 0:
        return
    R = basis_l.shape[1]
    assert key.stride(2) == 1 and sk_q8.is_contiguous() and sk_scale.is_contiguous()
    basis_l = basis_l.contiguous()
    BLOCK_TT = 16
    kern = _write_sketch4_kernel if sk_q8.shape[-1] == R // 2 else _write_sketch_kernel
    kern[(triton.cdiv(T, BLOCK_TT), Hkv)](
        key, key.stride(0), key.stride(1), slot_mapping, basis_l, sk_q8, sk_scale, T, sk_q8.shape[0] - 1,
        HKV=Hkv, R=R, D=D, BLOCK_TT=BLOCK_TT, BLOCK_D=64, num_warps=4)


# --------------------------------------------------------------------------------------------- select
@triton.jit
def _qproj_kernel(q_ptr, sq0, sq1, basis_ptr, qn_ptr, qs_ptr, qscale,
                  HQ: tl.constexpr, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                  BLOCK_D: tl.constexpr, QUANT: tl.constexpr):
    # one program per (query token, q-head): qr[64] = basis[h] @ q (fp32 FMA, exact vs the fp32 reference)
    i = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1)
    h = hq // G
    rr = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        dd = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + i * sq0 + hq * sq1 + dd).to(tl.float32)
        bmat = tl.load(basis_ptr + h * R * D + rr[:, None] * D + dd[None, :])
        acc += tl.sum(bmat * q[None, :], 1)
    acc = acc * qscale
    qs = tl.max(tl.abs(acc), 0) / 127.0
    safe = tl.where(qs > 0, qs, 1.0)
    qn = acc / safe
    row = i * HQ + hq
    if QUANT:
        tl.store(qn_ptr + row * R + rr, libdevice.rint(qn).to(tl.int8))
    else:
        tl.store(qn_ptr + row * R + rr, qn.to(tl.float16))
    tl.store(qs_ptr + row, qs)


@triton.jit
def _i2f_exact(x):
    # exact int32 -> fp32 for |x| < 2^22 using full-rate integer/float adds (I2F is quarter-rate on Ampere).
    # |int8 . int8| over 64 dims <= 127*127*64 < 2^20.
    return (x + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0


@triton.jit
def _ex2(x, HALF: tl.constexpr):
    # exp2; HALF: packed fp16 ex2.approx.f16x2 (2 results per MUFU op), returned as fp32
    if HALF:
        y = tl.inline_asm_elementwise("ex2.approx.f16x2 $0, $1;", "=r,r", [x.to(tl.float16)],
                                      dtype=tl.float16, is_pure=True, pack=2)
        return y.to(tl.float32)
    else:
        return tl.exp2(x)


@triton.jit
def _q_rows(qn_ptr, qs_ptr, q0, ql, L, h, m, kk, HQ: tl.constexpr, G: tl.constexpr, R: tl.constexpr,
            QMAX: tl.constexpr, ACTIVE: tl.constexpr):
    # rows m -> (verify row r = m // G, q-head h*G + m % G); returns int8/fp16 rows, scales, causal limit
    r = m // G
    g = m % G
    row_ok = (r < ql) & (r < QMAX) & ACTIVE
    qrow = (q0 + r).to(tl.int64) * HQ + h * G + g
    qn = tl.load(qn_ptr + qrow[:, None] * R + kk[None, :], mask=row_ok[:, None], other=0)
    qs = tl.load(qs_ptr + qrow, mask=row_ok, other=0.0)
    lim = tl.where(row_ok, L - ql + r, -1)                # row r sees t <= lim
    return qn, qs, lim


@triton.jit
def _load_k(sk_ptr, sks_ptr, slot, tok_ok, h, kk, HKV: tl.constexpr, R: tl.constexpr, MASKED: tl.constexpr,
            SK4: tl.constexpr = False):
    if SK4:
        # S4 int4 rows: unpack to [BLOCK_T, R] in interleaved order (dim j, dim j + R/2, ...); the query rows are
        # loaded with the same permutation (_perm_dims), so the dot product is unchanged.
        R2: tl.constexpr = R // 2
        k2 = tl.arange(0, R2)
        if MASKED:
            b = tl.load(sk_ptr + slot[:, None] * (HKV * R2) + h * R2 + k2[None, :], mask=tok_ok[:, None], other=0)
            zs = tl.load(sks_ptr + slot * HKV + h, mask=tok_ok, other=0.0).to(tl.float32)
        else:
            b = tl.load(sk_ptr + slot[:, None] * (HKV * R2) + h * R2 + k2[None, :])
            zs = tl.load(sks_ptr + slot * HKV + h).to(tl.float32)
        x = b.to(tl.int32)
        k = tl.reshape(tl.join(((x << 28) >> 28).to(tl.int8), (x >> 4).to(tl.int8)), [b.shape[0], R])
        return k, zs
    if MASKED:
        k = tl.load(sk_ptr + slot[:, None] * (HKV * R) + h * R + kk[None, :], mask=tok_ok[:, None], other=0)
        zs = tl.load(sks_ptr + slot * HKV + h, mask=tok_ok, other=0.0).to(tl.float32)
    else:
        k = tl.load(sk_ptr + slot[:, None] * (HKV * R) + h * R + kk[None, :])
        zs = tl.load(sks_ptr + slot * HKV + h).to(tl.float32)
    return k, zs


@triton.jit
def _perm_dims(kk, R: tl.constexpr, SK4: tl.constexpr):
    """Query-row dim order matching _load_k (identity for int8 sketches)."""
    if SK4:
        return (kk // 2) + (kk % 2) * (R // 2)
    return kk


@triton.jit
def _logits(qn, qs, k, zs, QUANT: tl.constexpr):
    # [rows, BLOCK_T] log2-domain logits
    if QUANT:
        s = _i2f_exact(tl.dot(qn, tl.trans(k)))
    else:
        s = tl.dot(qn, tl.trans(k.to(tl.float16)))
    return s * (qs[:, None] * zs[None, :])


@triton.jit
def _online(s, m_i, l_i, HALF: tl.constexpr):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = _ex2(s - m_use[:, None], HALF)
    l_i = l_i * tl.exp2(m_i - m_use) + tl.sum(p, 1)
    return m_new, l_i


@triton.jit
def _stats_kernel(qn_ptr, qs_ptr, qsl_ptr, sl_ptr, bt_ptr, bt_s0, sk_ptr, sks_ptr, wsm_ptr, wsl_ptr,
                  C_TOT, NSPLIT, ru_ptr,
                  HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, R: tl.constexpr, BS_: tl.constexpr,
                  MA: tl.constexpr, MB: tl.constexpr, MBS: tl.constexpr, QMAX: tl.constexpr,
                  BLOCK_T: tl.constexpr, QUANT: tl.constexpr, HALF: tl.constexpr, SK4: tl.constexpr = False,
                  SUB: tl.constexpr = 1, RU: tl.constexpr = False):
    # rows split into two power-of-2 groups A=[0,MA) and B=[MA,MA+MB) (e.g. 48 = 32 + 16; MB = 0 -> unused)
    # RU (S4 selection reuse): requests with ru[b] = 1 keep last step's top-k -> no work here
    # S4 SUB > 1: the unmasked bulk is sampled every SUB-th tile and its sum-exp scaled by SUB (normaliser estimate;
    # the causal tail stays exact). Only row weights in the pooled score change.
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    b = pid // HKV
    h = pid % HKV
    L = tl.load(sl_ptr + b)
    Lw = tl.where(L > C_TOT, L, 0)                     # short request -> no work
    if RU:
        Lw = tl.where(tl.load(ru_ptr + b) != 0, 0, Lw)
    q0 = tl.load(qsl_ptr + b)
    ql = tl.load(qsl_ptr + b + 1) - q0
    kk = tl.arange(0, R)
    kq = _perm_dims(kk, R, SK4)
    ma = tl.arange(0, MA)
    mb = MA + tl.arange(0, MBS)
    qa, qsa, lima = _q_rows(qn_ptr, qs_ptr, q0, ql, L, h, ma, kq, HQ, G, R, QMAX, True)
    qb, qsb, limb = _q_rows(qn_ptr, qs_ptr, q0, ql, L, h, mb, kq, HQ, G, R, QMAX, MB > 0)
    chunk = tl.cdiv(tl.cdiv(Lw, NSPLIT), BLOCK_T) * BLOCK_T
    start = sp * chunk
    end = tl.minimum(start + chunk, Lw)
    # bulk = full tiles every row sees (no masks; rows >= ql are dropped in _combine_kernel)
    be = tl.maximum(tl.minimum(end, Lw - ql + 1), start)
    bulk_end = start + ((be - start) // BLOCK_T) * BLOCK_T
    m_a = tl.full([MA], float("-inf"), tl.float32)
    l_a = tl.zeros([MA], dtype=tl.float32)
    m_b = tl.full([MBS], float("-inf"), tl.float32)
    l_b = tl.zeros([MBS], dtype=tl.float32)
    bt_row = bt_ptr + b.to(tl.int64) * bt_s0
    for t0 in range(start, bulk_end, BLOCK_T * SUB):
        t = t0 + tl.arange(0, BLOCK_T)
        blk = tl.load(bt_row + t // BS_)
        slot = blk.to(tl.int64) * BS_ + t % BS_
        k, zs = _load_k(sk_ptr, sks_ptr, slot, t, h, kk, HKV, R, False, SK4)
        m_a, l_a = _online(_logits(qa, qsa, k, zs, QUANT), m_a, l_a, HALF)
        if MB > 0:
            m_b, l_b = _online(_logits(qb, qsb, k, zs, QUANT), m_b, l_b, HALF)
    if SUB > 1:
        l_a = l_a * SUB
        l_b = l_b * SUB
    for t0 in range(bulk_end, end, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tok_ok = t < end
        blk = tl.load(bt_row + t // BS_, mask=tok_ok, other=0)
        slot = blk.to(tl.int64) * BS_ + t % BS_
        k, zs = _load_k(sk_ptr, sks_ptr, slot, tok_ok, h, kk, HKV, R, True, SK4)
        s = _logits(qa, qsa, k, zs, QUANT)
        s = tl.where((t[None, :] <= lima[:, None]) & tok_ok[None, :], s, float("-inf"))
        m_a, l_a = _online(s, m_a, l_a, HALF)
        if MB > 0:
            s = _logits(qb, qsb, k, zs, QUANT)
            s = tl.where((t[None, :] <= limb[:, None]) & tok_ok[None, :], s, float("-inf"))
            m_b, l_b = _online(s, m_b, l_b, HALF)
    o = (pid * NSPLIT + sp) * (MA + MB)
    tl.store(wsm_ptr + o + ma, m_a)
    tl.store(wsl_ptr + o + ma, l_a)
    if MB > 0:
        tl.store(wsm_ptr + o + mb, m_b)
        tl.store(wsl_ptr + o + mb, l_b)


@triton.jit
def _combine_kernel(wsm_ptr, wsl_ptr, mz_ptr, qsl_ptr, sl_ptr, lens_ptr, C_TOT, WIN, NSPLIT, ru_ptr,
                    HKV: tl.constexpr, G: tl.constexpr, MT: tl.constexpr, MT_PAD: tl.constexpr,
                    QMAX: tl.constexpr, NSP_PAD: tl.constexpr, RU: tl.constexpr = False):
    # mz = max + log2(sum-exp) per row (the full log2-normaliser); +inf for rows that do not exist
    pid = tl.program_id(0)
    b = pid // HKV
    h = pid % HKV
    sp = tl.arange(0, NSP_PAD)
    m = tl.arange(0, MT_PAD)
    mok = m < MT
    o = (pid * NSPLIT + sp[:, None]) * MT + m[None, :]
    ok = (sp < NSPLIT)[:, None] & mok[None, :]
    ms = tl.load(wsm_ptr + o, mask=ok, other=float("-inf"))
    ls = tl.load(wsl_ptr + o, mask=ok, other=0.0)
    mf = tl.max(ms, 0)
    mu = tl.where(mf == float("-inf"), 0.0, mf)
    lf = tl.sum(ls * tl.exp2(ms - mu[None, :]), 0)
    ql = tl.load(qsl_ptr + b + 1) - tl.load(qsl_ptr + b)
    r = m // G
    row_ok = (r < ql) & (r < QMAX) & (lf > 0)
    mz = tl.where(row_ok, mu + tl.log2(tl.where(lf > 0, lf, 1.0)), float("inf"))
    tl.store(mz_ptr + pid * MT + m, mz, mask=mok)
    if h == 0:
        L = tl.load(sl_ptr + b)
        ln = tl.where(L > C_TOT, L - WIN, 0)
        if RU:
            ln = tl.where(tl.load(ru_ptr + b) != 0, 0, ln)   # reuse: empty top-k range -> radix does no work
        tl.store(lens_ptr + b, ln)


@triton.jit
def _score_group(qn_ptr, qs_ptr, mz_ptr, q0, ql, L, b, h, m, kk, k, zs, HQ: tl.constexpr, HKV: tl.constexpr,
                 G: tl.constexpr, R: tl.constexpr, MT: tl.constexpr, QMAX: tl.constexpr, QUANT: tl.constexpr,
                 HALF: tl.constexpr, ACTIVE: tl.constexpr):
    qn, qs, lim = _q_rows(qn_ptr, qs_ptr, q0, ql, L, h, m, kk, HQ, G, R, QMAX, ACTIVE)
    mz = tl.load(mz_ptr + (b * HKV + h) * MT + m, mask=m < MT, other=float("inf"))
    s = _logits(qn, qs, k, zs, QUANT)
    if HALF:   # +15 keeps p*2^15 in fp16's normal range (p <= 1 -> <= 32768); undone at the store
        return tl.sum(_ex2(s - (mz[:, None] - 15.0), HALF), 0)
    else:
        return tl.sum(tl.exp2(s - mz[:, None]), 0)


@triton.jit
def _score_kernel(qn_ptr, qs_ptr, qsl_ptr, sl_ptr, bt_ptr, bt_s0, sk_ptr, sks_ptr, mz_ptr, sc_ptr, hist_ptr,
                  C_TOT, NSPLIT, W, ru_ptr,
                  HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, R: tl.constexpr, BS_: tl.constexpr,
                  MA: tl.constexpr, MB: tl.constexpr, MBS: tl.constexpr, QMAX: tl.constexpr,
                  BLOCK_T: tl.constexpr, QUANT: tl.constexpr, HALF: tl.constexpr,
                  SINK: tl.constexpr, SINK_PAD: tl.constexpr, WIN: tl.constexpr, FILL_TAIL: tl.constexpr,
                  SK4: tl.constexpr = False, RU: tl.constexpr = False):
    # tokens in [SINK, L - WIN) are visible to every verify row (q_len <= WIN): no causal mask needed.
    b = tl.program_id(0)
    sp = tl.program_id(1)
    L = tl.load(sl_ptr + b)
    long = L > C_TOT
    hi = tl.where(long, L - WIN, SINK)
    if RU:
        hi = tl.where(tl.load(ru_ptr + b) != 0, SINK, hi)   # reuse: nothing to score
    q0 = tl.load(qsl_ptr + b)
    ql = tl.load(qsl_ptr + b + 1) - q0
    kk = tl.arange(0, R)
    kq = _perm_dims(kk, R, SK4)
    ma = tl.arange(0, MA)
    mb = MA + tl.arange(0, MBS)
    n = hi - SINK
    chunk = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK_T) * BLOCK_T
    start = SINK + sp * chunk
    end = tl.minimum(start + chunk, hi)
    bt_row = bt_ptr + b.to(tl.int64) * bt_s0
    out_row = sc_ptr + b.to(tl.int64) * W
    for t0 in range(start, end, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tok_ok = t < end
        blk = tl.load(bt_row + t // BS_, mask=tok_ok, other=0)
        slot = blk.to(tl.int64) * BS_ + t % BS_
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        for h in tl.static_range(HKV):
            k, zs = _load_k(sk_ptr, sks_ptr, slot, tok_ok, h, kk, HKV, R, True, SK4)
            acc += _score_group(qn_ptr, qs_ptr, mz_ptr, q0, ql, L, b, h, ma, kq, k, zs, HQ, HKV, G, R,
                                MA + MB, QMAX, QUANT, HALF, True)
            if MB > 0:
                acc += _score_group(qn_ptr, qs_ptr, mz_ptr, q0, ql, L, b, h, mb, kq, k, zs, HQ, HKV, G, R,
                                    MA + MB, QMAX, QUANT, HALF, True)
        if HALF:
            acc = acc * (1.0 / 32768.0)
        tl.store(out_row + t, acc, mask=tok_ok)
    if sp == 0:
        j = tl.arange(0, SINK_PAD)
        tl.store(out_row + j, tl.full([SINK_PAD], float("-inf"), tl.float32), mask=j < SINK)
    if FILL_TAIL:   # torch.topk path: -inf on [hi, W)
        tchunk = tl.cdiv(tl.cdiv(W - hi, NSPLIT), BLOCK_T) * BLOCK_T
        ts = hi + sp * tchunk
        te = tl.minimum(ts + tchunk, W)
        for t0 in range(ts, te, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            tl.store(out_row + t, tl.full([BLOCK_T], float("-inf"), tl.float32), mask=t < te)


@triton.jit
def _assemble_kernel(tk_ptr, sl_ptr, idx_ptr, nv_ptr, C_TOT, KSEL, SINK, WIN, CUT, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pb = tl.program_id(1)
    L = tl.load(sl_ptr + b)
    long = L > C_TOT
    j = pb * BLOCK + tl.arange(0, BLOCK)
    tk = tl.load(tk_ptr + b.to(tl.int64) * KSEL + j, mask=j < KSEL, other=0)
    v_long = tl.where(j < KSEL, tk, tl.where(j < KSEL + SINK, j - KSEL, L - WIN + (j - KSEL - SINK)))
    v_short = tl.where(j < L, j, 0)
    v = tl.where(long, v_long, v_short)
    tl.store(idx_ptr + b.to(tl.int64) * C_TOT + j, v.to(tl.int32), mask=j < C_TOT)
    if pb == 0:
        nv = tl.where(long, C_TOT, L)
        nv = tl.where(L < CUT, 0, nv)            # dense_cutoff: backend runs dense pthfa for this request
        tl.store(nv_ptr + b, nv.to(tl.int32))


# --------------------------------------------------------------------------------------------- radix top-k
# Exact top-k over scores[b, lo:lens[b]) (non-negative fp32 -> uint32 bit order == value order).
# 3 digit levels (11/11/10 bits): histogram -> pick digit -> ... -> compaction. state[b] = [prefix, need, cnt_gt, cnt_eq]
@triton.jit
def _rx_hist_kernel(sc_ptr, W, lens_ptr, lo, hist_ptr, state_ptr, NSPLIT,
                    SHIFT: tl.constexpr, NBITS: tl.constexpr, LEVEL: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    sp = tl.program_id(1)
    NB: tl.constexpr = 1 << NBITS
    hi = tl.load(lens_ptr + b)
    prefix = tl.load(state_ptr + b * 4).to(tl.uint32)
    n = tl.maximum(hi - lo, 0)
    chunk = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK) * BLOCK
    start = lo + sp * chunk
    end = tl.minimum(start + chunk, hi)
    row = sc_ptr + b.to(tl.int64) * W
    acc = tl.zeros([2 * NB], dtype=tl.int32)
    for t0 in range(start, end, BLOCK):
        t = t0 + tl.arange(0, BLOCK)
        ok = t < end
        key = tl.load(row + t, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
        if LEVEL > 0:
            ok = ok & ((key >> (SHIFT + NBITS)) == prefix)
        d = ((key >> SHIFT) & (NB - 1)).to(tl.int32)
        d = tl.where(ok, d, NB)                       # invalid -> overflow bin
        acc += tl.histogram(d, 2 * NB)
    j = tl.arange(0, 2 * NB)
    tl.atomic_add(hist_ptr + b * NB + j, acc, mask=(j < NB) & (acc > 0), sem="relaxed")


@triton.jit
def _rx_pick_kernel(hist_ptr, state_ptr, KSEL, NBITS: tl.constexpr, LEVEL: tl.constexpr):
    b = tl.program_id(0)
    NB: tl.constexpr = 1 << NBITS
    j = tl.arange(0, NB)
    rev = tl.load(hist_ptr + b * NB + (NB - 1 - j))   # rev[j] = count of digit NB-1-j
    tl.store(hist_ptr + b * NB + j, tl.zeros([NB], dtype=tl.int32))   # reset for the next level
    if LEVEL == 0:
        need = KSEL
        prefix = 0
    else:
        need = tl.load(state_ptr + b * 4 + 1)
        prefix = tl.load(state_ptr + b * 4)
    cs = tl.cumsum(rev, 0)
    hit = cs >= need
    jj = tl.min(tl.where(hit, j, NB), 0)              # first digit (from the top) reaching need
    jj = tl.minimum(jj, NB - 1)
    above = tl.sum(tl.where(j < jj, rev, 0), 0)
    d = NB - 1 - jj
    tl.store(state_ptr + b * 4, (prefix << NBITS) | d)
    tl.store(state_ptr + b * 4 + 1, need - above)


@triton.jit
def _rx_count_kernel(sc_ptr, W, lens_ptr, lo, state_ptr, cnt_ptr, NSPLIT, BLOCK: tl.constexpr):
    # per (row, split): number of keys > threshold and == threshold (for deterministic, index-ordered compaction)
    b = tl.program_id(0)
    sp = tl.program_id(1)
    hi = tl.load(lens_ptr + b)
    thr = tl.load(state_ptr + b * 4).to(tl.uint32)
    n = tl.maximum(hi - lo, 0)
    chunk = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK) * BLOCK
    start = lo + sp * chunk
    end = tl.minimum(start + chunk, hi)
    row = sc_ptr + b.to(tl.int64) * W
    cgt = tl.zeros([BLOCK], dtype=tl.int32)
    ceq = tl.zeros([BLOCK], dtype=tl.int32)
    for t0 in range(start, end, BLOCK):
        t = t0 + tl.arange(0, BLOCK)
        ok = t < end
        key = tl.load(row + t, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
        cgt += (ok & (key > thr)).to(tl.int32)
        ceq += (ok & (key == thr)).to(tl.int32)
    tl.store(cnt_ptr + (b * NSPLIT + sp) * 2, tl.sum(cgt, 0))
    tl.store(cnt_ptr + (b * NSPLIT + sp) * 2 + 1, tl.sum(ceq, 0))


@triton.jit
def _rx_compact_kernel(sc_ptr, W, lens_ptr, lo, state_ptr, cnt_ptr, out_ptr, KSEL, NSPLIT,
                       BLOCK: tl.constexpr, NSP_PAD: tl.constexpr):
    # Deterministic: keys > thr in index order, then the lowest-index keys == thr (ties broken by index).
    b = tl.program_id(0)
    sp = tl.program_id(1)
    hi = tl.load(lens_ptr + b)
    thr = tl.load(state_ptr + b * 4).to(tl.uint32)
    need_eq = tl.load(state_ptr + b * 4 + 1)
    n_gt = KSEL - need_eq
    jj = tl.arange(0, NSP_PAD)
    before = jj < sp
    pgt = tl.sum(tl.load(cnt_ptr + (b * NSPLIT + jj) * 2, mask=before, other=0), 0)
    peq = tl.sum(tl.load(cnt_ptr + (b * NSPLIT + jj) * 2 + 1, mask=before, other=0), 0)
    n = tl.maximum(hi - lo, 0)
    chunk = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK) * BLOCK
    start = lo + sp * chunk
    end = tl.minimum(start + chunk, hi)
    row = sc_ptr + b.to(tl.int64) * W
    orow = out_ptr + b * KSEL
    for t0 in range(start, end, BLOCK):
        t = t0 + tl.arange(0, BLOCK)
        ok = t < end
        key = tl.load(row + t, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
        gt = (ok & (key > thr)).to(tl.int32)
        eq = (ok & (key == thr)).to(tl.int32)
        pos = pgt + tl.cumsum(gt, 0) - 1
        tl.store(orow + pos, t, mask=(gt > 0) & (pos < KSEL))
        pos = peq + tl.cumsum(eq, 0) - 1
        tl.store(orow + n_gt + pos, t, mask=(eq > 0) & (pos < need_eq))
        pgt += tl.sum(gt, 0)
        peq += tl.sum(eq, 0)


RADIX_LEVELS = {8: ((24, 8), (16, 8), (8, 8), (0, 8)), 11: ((21, 11), (10, 11), (0, 10))}


def radix_topk(scores, lens, lo, ksel, nsplit=None, block=None, digits=None, hist=None):
    """Exact top-ksel indices of scores[b, lo:lens[b]) (values must be >= 0), deterministic: output is the keys above
    the k-th value in ascending index order, then the lowest-index ties. Rows with fewer than ksel candidates produce
    garbage (callers ignore them). Any ksel. Graph-capturable (no host syncs; integer atomics only)."""
    nreq, W = scores.shape
    dev = scores.device
    digits = digits or CFG["rx_digits"]
    block = block or CFG["rx_block"]
    NSP = nsplit or max(1, triton.cdiv(CFG["rx_split"] * _num_sms(dev), nreq))
    nbins = 1 << max(n for _, n in RADIX_LEVELS[digits])
    have_l0 = hist is not None           # level-0 histogram already built (fused into _score_kernel, 8-bit digits)
    if have_l0:
        assert digits == 8
    else:
        hist = torch.zeros(nreq * nbins + nreq * 4, dtype=torch.int32, device=dev)
    state = hist[nreq * nbins:]
    out = torch.empty(nreq, ksel, dtype=torch.int32, device=dev)
    for level, (shift, nbits) in enumerate(RADIX_LEVELS[digits]):
        if not (level == 0 and have_l0):
            _rx_hist_kernel[(nreq, NSP)](scores, W, lens, lo, hist, state, NSP, SHIFT=shift, NBITS=nbits, LEVEL=level,
                                         BLOCK=block, num_warps=4)
        _rx_pick_kernel[(nreq,)](hist, state, ksel, NBITS=nbits, LEVEL=level, num_warps=4)
    cnt = torch.empty(nreq * NSP * 2, dtype=torch.int32, device=dev)
    _rx_count_kernel[(nreq, NSP)](scores, W, lens, lo, state, cnt, NSP, BLOCK=block, num_warps=4)
    _rx_compact_kernel[(nreq, NSP)](scores, W, lens, lo, state, cnt, out, ksel, NSP, BLOCK=block,
                                    NSP_PAD=_next_pow2(NSP), num_warps=4)
    return out


def _row_groups(rows, exact=True):
    """Split `rows` (verify rows x G) into <= 2 power-of-2 tiles >= 16 (48 -> 32 + 16) instead of padding."""
    full = max(16, _next_pow2(rows))
    if not exact or rows <= 16:
        return full, 0
    a = 1 << (rows.bit_length() - 1)
    if a == rows:
        return a, 0
    b = max(16, _next_pow2(rows - a))
    return (a, b) if a + b < full else (full, 0)


def _next_pow2(x):
    return 1 << max(0, (int(x) - 1).bit_length())


@triton.jit
def _topk_reuse_kernel(tk_ptr, store_ptr, slot_ptr, ru_ptr, KSEL, BLOCK: tl.constexpr):
    # S4: reuse rows load last step's top-k of their slot; the others store theirs for the next step
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = j < KSEL
    s = tl.load(slot_ptr + b).to(tl.int64)
    if tl.load(ru_ptr + b) != 0:
        tl.store(tk_ptr + b.to(tl.int64) * KSEL + j, tl.load(store_ptr + s * KSEL + j, mask=m), mask=m)
    else:
        tl.store(store_ptr + s * KSEL + j, tl.load(tk_ptr + b.to(tl.int64) * KSEL + j, mask=m), mask=m)


def select(query, query_start_loc, seq_lens, block_table, sk_q8, sk_scale, basis_l,
           ksel=2048, sink=64, win=1024, scale=1.0 / 16, max_q_len=None, topk_impl=None, quant=None,
           block_t=None, nsplit_a=None, nsplit_b=None, debug=None, half_exp=None, dense_cutoff=0,
           reuse=None, topk_store=None, slots=None):
    """max_q_len: optional static bound on per-request query rows (e.g. num_spec+1). Default Tq - nreq + 1.
    dense_cutoff: requests with seq_len < dense_cutoff get n_valid = 0 and no selection work (graph-safe: decided on
    device per request); the caller runs dense attention for them (see dense_sparse_seq_lens). 0 = contract behaviour.
    reuse/topk_store/slots (S4, all or none): int32 [nreq] flags; rows with reuse = 1 skip scoring and take the top-k
    stored for their slot (topk_store [nslots, ksel] int32, persistent per layer) at the last full selection; the
    sinks and window are always rebuilt from the current seq_len. Other rows store their new top-k."""
    Tq, Hq, D = query.shape
    nreq = seq_lens.shape[0]
    Hkv, R, _ = basis_l.shape
    SK4 = sk_q8.shape[-1] == R // 2   # S4 int4 sketch (packed two dims per byte)
    cfg = dict(CFG, **CFG4) if SK4 else CFG
    G = Hq // Hkv
    C = ksel + sink + win
    C_EFF = max(C, int(dense_cutoff) - 1)     # selection work only for seq_len > C and seq_len >= dense_cutoff
    dev = query.device
    W = block_table.shape[1] * BS
    quant = QUANT_Q if quant is None else quant
    topk_impl = topk_impl or TOPK_IMPL
    if topk_impl == "per_row" and ksel > 2048:   # DSA decode top-k is built for k <= 2048 (crashes above)
        topk_impl = "radix"
    qmax = int(max_q_len) if max_q_len else max(1, Tq - nreq + 1)
    assert qmax <= win, "verify rows must fit in the window"
    MA, MB = _row_groups(qmax * G, CFG["exact_rows"])
    MT = MA + MB
    MBS = MB if MB > 0 else 16
    half = CFG["half_exp"] if half_exp is None else half_exp
    assert query.stride(2) == 1 and block_table.stride(1) == 1
    assert sk_q8.is_contiguous() and sk_scale.is_contiguous()
    basis_l = basis_l.contiguous()
    nsm = _num_sms(dev)

    qn = torch.empty(Tq, Hq, R, dtype=torch.int8 if quant else torch.float16, device=dev)
    qs = torch.empty(Tq, Hq, dtype=torch.float32, device=dev)
    _qproj_kernel[(Tq, Hq)](query, query.stride(0), query.stride(1), basis_l, qn, qs, scale * LOG2E,
                            HQ=Hq, G=G, R=R, D=D, BLOCK_D=64, QUANT=quant, num_warps=4)

    RU = reuse is not None
    assert RU == (topk_store is not None) == (slots is not None)
    ru = reuse if RU else seq_lens
    NSA = nsplit_a or max(1, triton.cdiv(cfg["split_a"] * nsm, nreq * Hkv))
    wsm = torch.empty(nreq * Hkv * NSA * MT, dtype=torch.float32, device=dev)
    wsl = torch.empty_like(wsm)
    _stats_kernel[(nreq * Hkv, NSA)](qn, qs, query_start_loc, seq_lens, block_table, block_table.stride(0),
                                     sk_q8, sk_scale, wsm, wsl, C_EFF, NSA, ru,
                                     HQ=Hq, HKV=Hkv, G=G, R=R, BS_=BS, MA=MA, MB=MB, MBS=MBS, QMAX=qmax,
                                     BLOCK_T=block_t or cfg["block_a"], QUANT=quant, HALF=half, SK4=SK4, SUB=STATS_SUB,
                                     RU=RU, num_warps=cfg["stats_warps"], num_stages=cfg["stats_stages"])
    mz = torch.empty(nreq * Hkv * MT, dtype=torch.float32, device=dev)
    lens = torch.empty(nreq, dtype=torch.int32, device=dev)
    _combine_kernel[(nreq * Hkv,)](wsm, wsl, mz, query_start_loc, seq_lens, lens, C_EFF, win, NSA, ru,
                                   HKV=Hkv, G=G, MT=MT, MT_PAD=_next_pow2(MT), QMAX=qmax, NSP_PAD=_next_pow2(NSA),
                                   RU=RU, num_warps=4)

    NSB = nsplit_b or max(1, triton.cdiv(cfg["split_b"] * nsm, nreq))
    scores = torch.empty(nreq, W, dtype=torch.float32, device=dev)
    fill_tail = topk_impl == "torch"
    hist = scores   # unused placeholder (the fused level-0 histogram was removed: measured slower)
    _score_kernel[(nreq, NSB)](qn, qs, query_start_loc, seq_lens, block_table, block_table.stride(0),
                               sk_q8, sk_scale, mz, scores, hist, C_EFF, NSB, W, ru,
                               HQ=Hq, HKV=Hkv, G=G, R=R, BS_=BS, MA=MA, MB=MB, MBS=MBS, QMAX=qmax,
                               BLOCK_T=block_t or cfg["block_b"], QUANT=quant, HALF=half, SINK=sink, SINK_PAD=_next_pow2(max(sink, 1)), WIN=win,
                               FILL_TAIL=fill_tail, SK4=SK4, RU=RU, num_warps=cfg["score_warps"],
                               num_stages=cfg["score_stages"])

    if debug is not None:
        debug["scores"] = scores
    if topk_impl == "per_row":
        from vllm import _custom_ops as vops
        topk = torch.empty(nreq, ksel, dtype=torch.int32, device=dev)
        vops.top_k_per_row_decode(scores, 1, lens, topk, nreq, scores.stride(0), scores.stride(1), ksel)
    elif topk_impl == "radix":
        topk = radix_topk(scores, lens, sink, ksel)
    else:
        k_eff = min(ksel, W)
        topk = scores.topk(k_eff, dim=1, sorted=False).indices.to(torch.int32)
        if k_eff < ksel:
            topk = torch.nn.functional.pad(topk, (0, ksel - k_eff))
    if RU:
        assert topk_impl != "torch" and topk_store.shape[1] == ksel and topk.is_contiguous()
        _topk_reuse_kernel[(nreq, triton.cdiv(ksel, 1024))](topk, topk_store, slots, reuse, ksel, BLOCK=1024,
                                                            num_warps=4)
    idx = torch.empty(nreq, C, dtype=torch.int32, device=dev)
    n_valid = torch.empty(nreq, dtype=torch.int32, device=dev)
    _assemble_kernel[(nreq, triton.cdiv(C, 1024))](topk, seq_lens, idx, n_valid, C, ksel, sink, win,
                                                   int(dense_cutoff), BLOCK=1024,
                                                   num_warps=4)
    return idx, n_valid


def select_suffix(query, q_lens, seq_lens_host, seq_lens, block_table, sk_q8, sk_scale, basis_l, ksel=2048, sink=64,
                  win=1024, scale=1.0 / 16, rows_per_part=16, score_group=32, quant=None, h2d=None):
    """S4: selection for a prefill SUFFIX (a follow-up's uncached tail, q_len up to a few thousand rows).

    Same score as select (softmax over tokens per (query row, q-head), summed over rows and heads, one score per token
    shared by the 4 KV heads), but the rows of each request are processed as pseudo-requests of `rows_per_part` rows
    (select keeps all rows of a request in one register tile): stats with exact causal limits per pseudo-request,
    scores of [sink, L - win) summed into the real request's row, one top-k, then the usual [topk | sinks | window]
    layout with window = the last `win` tokens (win >= every q_len, so the suffix rows attend each other exactly).
    Eager only (the prefill path): host lists q_lens / seq_lens_host describe the requests, `seq_lens` is their device
    copy. h2d(list) -> int32 device tensor (pinned, non-blocking); default torch.tensor (synchronous).
    Returns (idx int32 [nreq, ksel+sink+win], n_valid int32 [nreq])."""
    Tq, Hq, D = query.shape
    nreq = len(q_lens)
    Hkv, R, _ = basis_l.shape
    SK4 = sk_q8.shape[-1] == R // 2
    G = Hq // Hkv
    C = ksel + sink + win
    assert max(q_lens) <= win and sum(q_lens) == Tq, (q_lens, win, Tq)
    dev = query.device
    W = block_table.shape[1] * BS
    quant = QUANT_Q if quant is None else quant
    h2d = h2d or (lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev))
    P = int(rows_per_part)
    pq, pl_stats, pl_score, preq = [0], [], [], []
    q0 = 0
    for r in range(nreq):
        ql, L = int(q_lens[r]), int(seq_lens_host[r])
        for c0 in range(0, ql, P):
            n = min(P, ql - c0)
            pq.append(pq[-1] + n)
            pl_stats.append(L - ql + c0 + n)   # causal: row j of this part is query row c0 + j -> sees t <= L - ql + c0 + j
            pl_score.append(L)
            preq.append(r)
        q0 += ql
    npart = len(preq)
    meta = h2d(pq + pl_stats + pl_score + preq)
    pqsl = meta[: npart + 1]
    psl_stats = meta[npart + 1: 2 * npart + 1]
    psl_score = meta[2 * npart + 1: 3 * npart + 1]
    preq_d = meta[3 * npart + 1: 4 * npart + 1].long()
    pbt = block_table.index_select(0, preq_d).contiguous()
    basis_l = basis_l.contiguous()
    nsm = _num_sms(dev)
    qn = torch.empty(Tq, Hq, R, dtype=torch.int8 if quant else torch.float16, device=dev)
    qs = torch.empty(Tq, Hq, dtype=torch.float32, device=dev)
    _qproj_kernel[(Tq, Hq)](query, query.stride(0), query.stride(1), basis_l, qn, qs, scale * LOG2E,
                            HQ=Hq, G=G, R=R, D=D, BLOCK_D=64, QUANT=quant, num_warps=4)
    MA, MB = _row_groups(P * G, CFG["exact_rows"])
    MT = MA + MB
    MBS = MB if MB > 0 else 16
    NSA = max(1, triton.cdiv(CFG["split_a"] * nsm, npart * Hkv))
    wsm = torch.empty(npart * Hkv * NSA * MT, dtype=torch.float32, device=dev)
    wsl = torch.empty_like(wsm)
    _stats_kernel[(npart * Hkv, NSA)](qn, qs, pqsl, psl_stats, pbt, pbt.stride(0), sk_q8, sk_scale, wsm, wsl, C, NSA,
                                      psl_stats,
                                      HQ=Hq, HKV=Hkv, G=G, R=R, BS_=BS, MA=MA, MB=MB, MBS=MBS, QMAX=P,
                                      BLOCK_T=CFG["block_a"], QUANT=quant, HALF=False, SK4=SK4, SUB=STATS_SUB,
                                      num_warps=CFG["stats_warps"], num_stages=CFG["stats_stages"])
    mz = torch.empty(npart * Hkv * MT, dtype=torch.float32, device=dev)
    lens_p = torch.empty(npart, dtype=torch.int32, device=dev)
    _combine_kernel[(npart * Hkv,)](wsm, wsl, mz, pqsl, psl_stats, lens_p, C, win, NSA, psl_stats,
                                    HKV=Hkv, G=G, MT=MT, MT_PAD=_next_pow2(MT), QMAX=P, NSP_PAD=_next_pow2(NSA),
                                    num_warps=4)
    scores = torch.zeros(nreq, W, dtype=torch.float32, device=dev)
    ng = max(1, int(score_group))
    buf = torch.empty(min(ng, npart), W, dtype=torch.float32, device=dev)
    for g0 in range(0, npart, ng):
        g1 = min(npart, g0 + ng)
        sub = buf[: g1 - g0]
        NSB = max(1, triton.cdiv(CFG["split_b"] * nsm, g1 - g0))
        _score_kernel[(g1 - g0, NSB)](qn, qs, pqsl[g0:], psl_score[g0:], pbt[g0:], pbt.stride(0),
                                      sk_q8, sk_scale, mz[g0 * Hkv * MT:], sub, sub, C, NSB, W, pqsl,
                                      HQ=Hq, HKV=Hkv, G=G, R=R, BS_=BS, MA=MA, MB=MB, MBS=MBS, QMAX=P,
                                      BLOCK_T=CFG["block_b"], QUANT=quant, HALF=False, SINK=sink,
                                      SINK_PAD=_next_pow2(max(sink, 1)), WIN=win, FILL_TAIL=False, SK4=SK4,
                                      num_warps=CFG["score_warps"], num_stages=CFG["score_stages"])
        scores.index_add_(0, preq_d[g0:g1], sub)
    lens = torch.where(seq_lens > C, seq_lens - win, torch.zeros_like(seq_lens)).to(torch.int32)
    topk = radix_topk(scores, lens, sink, ksel)
    idx = torch.empty(nreq, C, dtype=torch.int32, device=dev)
    n_valid = torch.empty(nreq, dtype=torch.int32, device=dev)
    _assemble_kernel[(nreq, triton.cdiv(C, 1024))](topk, seq_lens, idx, n_valid, C, ksel, sink, win, 0, BLOCK=1024,
                                                   num_warps=4)
    return idx, n_valid


@triton.jit
def _split_sl_kernel(sl_ptr, qsl_ptr, nv_ptr, dense_ptr, sparse_ptr, usedense_ptr, NREQ, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    ok = i < NREQ
    L = tl.load(sl_ptr + i, mask=ok, other=1)
    ql = tl.load(qsl_ptr + i + 1, mask=ok, other=1) - tl.load(qsl_ptr + i, mask=ok, other=0)
    nv = tl.load(nv_ptr + i, mask=ok, other=1)
    dense = nv == 0
    tl.store(dense_ptr + i, tl.where(dense, L, ql), mask=ok)        # inactive rows: minimal legal length q_len
    tl.store(sparse_ptr + i, tl.where(dense, ql, nv), mask=ok)
    tl.store(usedense_ptr + i, dense.to(tl.int8), mask=ok)


def dense_sparse_seq_lens(seq_lens, query_start_loc, n_valid):
    """For dense_cutoff routing: (dense_seq_lens, staging_seq_lens, use_dense[int8]) per request, on device.
    Inactive side gets seq_len = q_len (pthfa requires seq_len >= q_len); merge the two pthfa outputs per token
    with torch.where (never arithmetic: the inactive side may read stale staging rows)."""
    nreq = seq_lens.shape[0]
    d = torch.empty_like(seq_lens)
    sp = torch.empty_like(seq_lens)
    u = torch.empty(nreq, dtype=torch.int8, device=seq_lens.device)
    _split_sl_kernel[(1,)](seq_lens, query_start_loc, n_valid, d, sp, u, nreq, BLOCK=_next_pow2(max(nreq, 2)))
    return d, sp, u


# --------------------------------------------------------------------------------------------- gather_rows
@triton.jit
def _gather_kernel(kv_ptr, s0, s1, s2, st_ptr, d0, d1, d2, bt_ptr, bt_s0, sbt_ptr, sbt_s0, idx_ptr, idx_s0, nv_ptr,
                   HKV: tl.constexpr, BS_: tl.constexpr, WMAIN: tl.constexpr, WTAIL: tl.constexpr,
                   WTAIL_PAD: tl.constexpr, TOKB: tl.constexpr):
    b = tl.program_id(0)
    pb = tl.program_id(1)
    nv = tl.load(nv_ptr + b)
    j = pb * TOKB + tl.arange(0, TOKB)
    ok = j < nv
    tok = tl.load(idx_ptr + b * idx_s0 + j, mask=ok, other=0)
    sblk = tl.load(bt_ptr + b.to(tl.int64) * bt_s0 + tok // BS_, mask=ok, other=0).to(tl.int64)
    soff = (tok % BS_).to(tl.int64)
    dblk = tl.load(sbt_ptr + b.to(tl.int64) * sbt_s0 + j // BS_, mask=ok, other=0).to(tl.int64)
    doff = (j % BS_).to(tl.int64)
    w = tl.arange(0, WMAIN)
    w2 = tl.arange(0, WTAIL_PAD)
    for h in tl.static_range(HKV):
        src = kv_ptr + sblk * s0 + h * s1 + soff * s2
        dst = st_ptr + dblk * d0 + h * d1 + doff * d2
        x = tl.load(src[:, None] + w[None, :], mask=ok[:, None])
        tl.store(dst[:, None] + w[None, :], x, mask=ok[:, None])
        if WTAIL > 0:
            m2 = ok[:, None] & (w2[None, :] < WTAIL)
            y = tl.load(src[:, None] + WMAIN + w2[None, :], mask=m2)
            tl.store(dst[:, None] + WMAIN + w2[None, :], y, mask=m2)


def gather_rows(kv, block_table, idx, n_valid, staging_kv, staging_bt, tokb=16):
    NB, Hkv, bs, Wb = kv.shape
    assert bs == BS and kv.stride(3) == 1 and staging_kv.stride(3) == 1
    assert Wb % 4 == 0 and kv.dtype == torch.int8
    kv32 = kv.view(torch.int32)            # last dim 2*(D+4) bytes -> words
    st32 = staging_kv.view(torch.int32)
    Ww = Wb // 4
    WMAIN = 1 << (Ww.bit_length() - 1)
    WTAIL = Ww - WMAIN
    nreq, C = idx.shape
    _gather_kernel[(nreq, triton.cdiv(C, tokb))](
        kv32, kv32.stride(0), kv32.stride(1), kv32.stride(2),
        st32, st32.stride(0), st32.stride(1), st32.stride(2),
        block_table, block_table.stride(0), staging_bt, staging_bt.stride(0), idx, idx.stride(0), n_valid,
        HKV=Hkv, BS_=BS, WMAIN=WMAIN, WTAIL=WTAIL, WTAIL_PAD=_next_pow2(max(WTAIL, 1)), TOKB=tokb, num_warps=4)
