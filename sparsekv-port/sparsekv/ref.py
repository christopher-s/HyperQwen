# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Plain-PyTorch reference of the sparse-KV ops (stream B's minimal version).

Signatures follow the sparse-KV kernel contract (see triton_ops.py). Everything is CUDA-graph
capturable: no host syncs (no .item(), no nonzero, no boolean-mask indexing) and
every shape depends only on (Tq, nreq, maxblk) and the host constants
(ksel, sink, win, block_size, max_query_len).

Memory in ``select`` is O(nreq * maxblk * BS * Hkv * r) transient, done one KV
head at a time to bound the fp32 logits.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 864
# Read by pth_flash_attn.sparse_kv (G-5): this module is CUDA-graph capturable (no host syncs).
GRAPH_SAFE = True
NEG = -1.0e30


def write_sketch(key, slot_mapping, basis_l, sk_q8, sk_scale) -> None:
    """Project key rows to rank-r int8 sketches and store them at their slots.

    key [T, Hkv, D] (bf16/fp16), slot_mapping [T'] int64 (-1 = padding),
    basis_l [Hkv, r, D] fp32, sk_q8 [NS+1, Hkv, r] int8, sk_scale [NS+1, Hkv] fp16.
    Rows beyond min(T, T') are ignored; slot -1 goes to the dummy last row.
    """
    n = min(key.shape[0], slot_mapping.shape[0])
    if n == 0:
        return
    k = key[:n].float()
    z = torch.einsum("thd,hrd->thr", k, basis_l)  # [n, Hkv, r]
    amax = z.abs().amax(dim=-1).clamp_min(1e-6)  # [n, Hkv]
    r = basis_l.shape[1]
    if sk_q8.shape[-1] == r // 2:  # S4 int4 sketch: byte j = dim j (low nibble) | dim j + r/2 (high nibble)
        scale = amax / 7.0
        q4 = torch.round(z / scale.unsqueeze(-1)).clamp_(-7, 7).to(torch.int32)
        q = pack_int4(q4)
    else:
        scale = amax / 127.0
        q = torch.round(z / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    dummy = sk_q8.shape[0] - 1
    sm = slot_mapping[:n].to(torch.int64)
    slots = torch.where(sm < 0, torch.full_like(sm, dummy), sm)
    sk_q8.index_copy_(0, slots, q)
    sk_scale.index_copy_(0, slots, scale.to(sk_scale.dtype))


def pack_int4(q4):
    """[..., r] int values in [-8, 7] -> [..., r/2] int8: byte j = q[j] (low nibble) | q[j + r/2] (high nibble)."""
    h = q4.shape[-1] // 2
    lo, hi = q4[..., :h].to(torch.int32), q4[..., h:].to(torch.int32)
    return (hi * 16 + (lo & 15)).to(torch.int8)


def unpack_int4(b):
    """Inverse of pack_int4: [..., r/2] int8 -> [..., r] int32 (sign-extended nibbles)."""
    x = b.to(torch.int32)
    return torch.cat([(x << 28) >> 28, x >> 4], dim=-1)


def sketch_values(sk_q8, r):
    """Sketch rows as [..., r] numbers (int8 or unpacked int4)."""
    return unpack_int4(sk_q8) if sk_q8.shape[-1] == r // 2 else sk_q8


def select(
    query,
    query_start_loc,
    seq_lens,
    block_table,
    sk_q8,
    sk_scale,
    basis_l,
    ksel: int = 2048,
    sink: int = 64,
    win: int = 1024,
    scale: float = 1.0 / 16,
    *,
    max_query_len: int | None = None,
    block_size: int = BLOCK_SIZE,
):
    """Pick C = ksel+sink+win token positions per request.

    Returns (idx int32 [nreq, C], n_valid int32 [nreq]); layout
    [topk (ascending) | sinks | window (in order)]. Requests with
    seq_len <= C get idx = arange(C) (entries >= seq_len clamped to 0) and
    n_valid = seq_len (exact).
    """
    dev = query.device
    nreq = seq_lens.shape[0]
    maxblk = block_table.shape[1]
    L = maxblk * block_size
    Hq, D = query.shape[1], query.shape[2]
    Hkv, r = basis_l.shape[0], basis_l.shape[1]
    G = Hq // Hkv
    C = ksel + sink + win
    Tq = query.shape[0]
    qmax = int(max_query_len) if max_query_len is not None else Tq
    qmax = max(1, min(qmax, Tq))

    qsl = query_start_loc[: nreq + 1].to(torch.int64)
    sl = seq_lens[:nreq].to(torch.int64)
    qlen = qsl[1:] - qsl[:-1]  # [nreq]
    j = torch.arange(qmax, device=dev)
    rows = (qsl[:-1, None] + j[None, :]).clamp_(0, max(Tq - 1, 0))  # [nreq, qmax]
    row_ok = (j[None, :] < qlen[:, None]).float()  # [nreq, qmax]
    q = query[rows.reshape(-1)].float().reshape(nreq, qmax, Hkv, G, D)

    # Sketch rows of every logical position of each request.
    pos = torch.arange(L, device=dev)
    blk = block_table[:nreq, :].to(torch.int64)[:, pos // block_size]  # [nreq, L]
    slots = blk * block_size + (pos % block_size)[None, :]
    slots = slots.clamp_(0, sk_q8.shape[0] - 1)
    tok_ok = pos[None, :] < sl[:, None]  # [nreq, L]

    score = torch.zeros(nreq, L, device=dev, dtype=torch.float32)
    for h in range(Hkv):
        qh = q[:, :, h]  # [nreq, qmax, G, D]
        qhat = torch.einsum("bjgd,rd->bjgr", qh, basis_l[h]).reshape(nreq, qmax * G, r)
        zh = sketch_values(sk_q8[:, h][slots], r).float()  # [nreq, L, r]
        sh = sk_scale[:, h][slots].float()  # [nreq, L]
        logits = torch.bmm(qhat, zh.transpose(1, 2)) * (sh[:, None, :] * scale)
        logits = logits.masked_fill(~tok_ok[:, None, :], NEG)
        p = torch.softmax(logits, dim=-1)  # [nreq, qmax*G, L]
        w = row_ok.repeat_interleave(G, dim=1)  # [nreq, qmax*G]
        score += torch.einsum("bk,bkl->bl", w, p)
        del logits, p, zh, sh

    cand = tok_ok & (pos[None, :] >= sink) & (pos[None, :] < (sl - win)[:, None])
    score = score.masked_fill(~cand, float("-inf"))
    k = min(ksel, L)
    top = torch.topk(score, k, dim=-1, sorted=False).indices  # [nreq, k]
    top = torch.sort(top, dim=-1).values
    if k < ksel:
        top = torch.cat([top, top[:, :1].expand(nreq, ksel - k)], dim=1)
    sinks = torch.arange(sink, device=dev)[None, :].expand(nreq, sink)
    window = (sl - win)[:, None] + torch.arange(win, device=dev)[None, :]
    sparse_idx = torch.cat([top, sinks, window], dim=1)  # [nreq, C]

    dense_idx = torch.arange(C, device=dev)[None, :].expand(nreq, C)
    dense_idx = torch.where(dense_idx < sl[:, None], dense_idx, torch.zeros_like(dense_idx))
    is_sparse = (sl > C)[:, None]
    idx = torch.where(is_sparse, sparse_idx, dense_idx).clamp_(0, L - 1)
    n_valid = torch.where(sl > C, torch.full_like(sl, C), sl)
    return idx.to(torch.int32), n_valid.to(torch.int32)


def gather_rows(kv, block_table, idx, n_valid, staging_kv, staging_bt) -> None:
    """Copy the selected cache rows (all KV heads, K|ks|V|vs) into the staging cache.

    kv [NB, Hkv, BS, W] int8, block_table [nreq, maxblk] i32, idx [nreq, C] i32,
    staging_kv [NBs, Hkv, BS, W] int8, staging_bt [nreq, nbs] i32 with
    nbs * BS >= C. Entries j >= n_valid are copied too (harmless: masked by
    seq_lens = n_valid in the attention launch).
    """
    del n_valid
    nreq, C = idx.shape
    BS = kv.shape[2]
    Hkv = kv.shape[1]
    dev = kv.device
    ii = idx.to(torch.int64)
    src_blk = torch.gather(block_table[:nreq].to(torch.int64), 1, ii // BS)  # [nreq, C]
    src_off = ii % BS
    j = torch.arange(C, device=dev)
    dst_blk = staging_bt[:nreq].to(torch.int64)[:, j // BS]  # [nreq, C]
    dst_off = (j % BS)[None, :].expand(nreq, C)
    h = torch.arange(Hkv, device=dev)[None, :]
    sb, so = src_blk.reshape(-1, 1), src_off.reshape(-1, 1)
    db, do = dst_blk.reshape(-1, 1), dst_off.reshape(-1, 1)
    staging_kv[db, h, do] = kv[sb, h, so]
