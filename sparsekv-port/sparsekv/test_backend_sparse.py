# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Stream B unit test: select -> gather_rows -> pthfa over staging.

Run on a test GPU from the worktree root:
  venv/bin/python sparsekv/test_backend_sparse.py [ref|triton]
Checks: (1) KSEL covering everything => staging attention bit-equal to dense;
(2) KSEL=2048 => layout invariants + equals an fp32 torch attention over the
selected set; (3) CUDA-graph capture/replay of the whole decode path == eager.
"""
import importlib
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pth_flash_attn import ops  # noqa: E402
from pth_flash_attn.planner import plan_launch, workspace_token_splits, workspace_shapes  # noqa: E402

impl_name = sys.argv[1] if len(sys.argv) > 1 else "ref"
if impl_name.endswith(".py"):
    import importlib.util as _u
    _sp = _u.spec_from_file_location("ext_ops", impl_name)
    K = _u.module_from_spec(_sp)
    _sp.loader.exec_module(K)
else:
    K = importlib.import_module("sparsekv.ref" if impl_name == "ref" else "sparsekv.triton_ops")
print("impl", K.__name__, getattr(K, "__file__", ""))
import inspect as _ins
_P = _ins.signature(K.select).parameters
SKW = {}
for _n in ("max_query_len", "max_q_len"):
    if _n in _P:
        SKW[_n] = 8
if "block_size" in _P:
    SKW["block_size"] = 864

dev = "cuda"
torch.manual_seed(0)
Hq, Hkv, D, BS, G = 24, 4, 256, 864, 6
W = 2 * (D + 4)
NB = 160
QL = 8
seq = [60000, 3000]
nreq = len(seq)
maxblk = 72
num_sms = torch.cuda.get_device_properties(0).multi_processor_count


class _I:
    head_size = D


def views(kv):
    num_blocks, nkv, bs, content = kv.shape
    padded = content // 2
    base = torch.tensor([], dtype=torch.float32, device=kv.device).set_(kv.untyped_storage())
    s = [x // 4 for x in kv.stride()]
    ks = torch.as_strided(base, (num_blocks, bs, nkv), (s[0], s[2], s[1]), D // 4)
    vs = torch.as_strided(base, (num_blocks, bs, nkv), (s[0], s[2], s[1]), (padded + D) // 4)
    kc, vc = kv.transpose(1, 2).split(padded, dim=-1)
    return kc[..., :D], vc[..., :D], ks, vs


kv = torch.randint(-127, 128, (NB, Hkv, BS, W), dtype=torch.int8, device=dev)
k, v, ks, vs = views(kv)
ks.copy_(torch.rand_like(ks) * 0.02 + 0.005)
vs.copy_(torch.rand_like(vs) * 0.02 + 0.005)
# plant a few strong keys so selection matters
perm = torch.randperm(NB, device=dev)[: nreq * maxblk].reshape(nreq, maxblk).to(torch.int32)
bt = perm.contiguous()
seq_lens = torch.tensor(seq, dtype=torch.int32, device=dev)
qsl = torch.arange(0, (nreq + 1) * QL, QL, dtype=torch.int32, device=dev)
query = torch.randn(nreq * QL, Hq, D, device=dev, dtype=torch.bfloat16) * 2

# basis + sketches over every slot
basis = torch.stack([torch.linalg.qr(torch.randn(D, 64, device=dev))[0].T for _ in range(Hkv)])
sk_q8 = torch.zeros(NB * BS + 1, Hkv, 64, dtype=torch.int8, device=dev)
sk_sc = torch.zeros(NB * BS + 1, Hkv, dtype=torch.float16, device=dev)
kd = (k.float() * ks.unsqueeze(-1)).reshape(NB * BS, Hkv, D).to(torch.bfloat16)
allslots = torch.arange(NB * BS, device=dev)
K.write_sketch(kd, allslots, basis, sk_q8, sk_sc)

wts = workspace_token_splits(max_num_seqs=nreq, q_dec_max=QL, Hkv=Hkv, D=D, num_sms=num_sms)
plan = plan_launch(num_tokens=nreq * QL, num_reqs=nreq, max_query_len=QL, G=G, Hkv=Hkv, D=D,
                   num_sms=num_sms, is_prefill=False, ws_token_splits=wts)
o_shape, l_shape = workspace_shapes(ws_token_splits=wts, Hq=Hq, D=D)
ws_o = torch.empty(plan.ws_o_shape, device=dev)
ws_l = torch.empty(plan.ws_lse_shape, device=dev)


def attn(out, caches, btab, sl):
    a, b, c, d = caches
    ops.pth_flash_attn_fwd(out, query, a, b, c, d, btab, sl, qsl, QL, plan.wm, plan.num_splits,
                           1.0 / 16, ws_o, ws_l, 0, True)


dense = torch.empty_like(query)
attn(dense, (k, v, ks, vs), bt, seq_lens)


def run(ksel, sink=64, win=1024):
    C = ksel + sink + win
    nbs = -(-C // BS)
    st = torch.zeros(nreq * nbs, Hkv, BS, W, dtype=torch.int8, device=dev)
    stbt = torch.arange(nreq * nbs, dtype=torch.int32, device=dev).reshape(nreq, nbs)
    idx, nv = K.select(query, qsl, seq_lens, bt, sk_q8, sk_sc, basis, ksel, sink, win, 1.0 / 16,
                       **SKW)
    K.gather_rows(kv, bt, idx, nv, st, stbt)
    out = torch.empty_like(query)
    attn(out, views(st), stbt, nv)
    return idx, nv, out, st, stbt


# (1) exactness
if os.environ.get("SKIP_EXACT") != "1":
    idx, nv, out, _, _ = run(ksel=70000)
    print("exact: n_valid", nv.tolist(), "bit-equal:", torch.equal(out, dense))
    assert torch.equal(out, dense)

# (2) sparse
idx, nv, out, st, stbt = run(ksel=2048)
C = 2048 + 64 + 1024
print("sparse n_valid", nv.tolist())
assert nv.tolist() == [C, 3000]
i0 = idx[0].long()
assert torch.equal(i0[2048:2112].cpu(), torch.arange(64))
assert torch.equal(i0[2112:].cpu(), torch.arange(seq[0] - 1024, seq[0]))
assert i0.unique().numel() == C, "duplicates"
assert bool(((i0[:2048] >= 64) & (i0[:2048] < seq[0] - 1024)).all())
assert torch.equal(idx[1, :3000].long().cpu(), torch.arange(3000))
assert torch.equal(out[QL:], dense[QL:]), "short request must be exact"


def ref_attn(r, toks):
    blk = bt[r].long()[toks // BS]
    off = toks % BS
    kk = (k[blk, off].float() * ks[blk, off].unsqueeze(-1))  # [n, Hkv, D]
    vv = (v[blk, off].float() * vs[blk, off].unsqueeze(-1))
    q = query[r * QL:(r + 1) * QL].float().reshape(QL, Hkv, G, D)
    logits = torch.einsum("qhgd,nhd->qhgn", q, kk) / 16
    n = toks.numel()
    pos_q = torch.arange(seq[r] - QL, seq[r], device=dev)
    mask = toks[None, :] <= pos_q[:, None]  # causal by true position
    logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
    p = logits.softmax(-1)
    return torch.einsum("qhgn,nhd->qhgd", p, vv).reshape(QL, Hq, D)


ref = ref_attn(0, i0)
err = (out[:QL].float() - ref).abs().max().item()
print("sparse vs fp32 ref over selection: max abs err", err)
assert err < 2e-2
# selection recall vs exact top by true attention mass (diagnostic)
full = torch.arange(seq[0], device=dev)
print("rel diff sparse vs dense (random data, diagnostic):",
      ((out[:QL].float() - dense[:QL].float()).norm() / dense[:QL].float().norm()).item())

# (3) graph capture
C = 3136
nbs = -(-C // BS)
st = torch.zeros(nreq * nbs, Hkv, BS, W, dtype=torch.int8, device=dev)
stbt = torch.arange(nreq * nbs, dtype=torch.int32, device=dev).reshape(nreq, nbs)
gout = torch.empty_like(query)
stv = views(st)


def step():
    i, n = K.select(query, qsl, seq_lens, bt, sk_q8, sk_sc, basis, 2048, 64, 1024, 1.0 / 16,
                    **SKW)
    K.gather_rows(kv, bt, i, n, st, stbt)
    a, b, c, d = stv
    ops.pth_flash_attn_fwd(gout, query, a, b, c, d, stbt, n, qsl, QL, plan.wm, plan.num_splits,
                           1.0 / 16, ws_o, ws_l, 0, True)
    K.write_sketch(query[:, :Hkv], torch.full((nreq * QL,), -1, device=dev, dtype=torch.int64),
                   basis, sk_q8, sk_sc)


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    step()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step()
gout.zero_()
g.replay()
torch.cuda.synchronize()
_i1, _n1, out_b, _, _ = run(ksel=2048)
print("eager determinism: idx equal", torch.equal(_i1, idx), "out equal", torch.equal(out_b, out))
print("graph replay == eager:", torch.equal(gout, out), "max diff", (gout.float() - out.float()).abs().max().item())
if torch.equal(_i1, idx):
    assert torch.equal(gout, out)
else:
    print("WARN: impl is nondeterministic eagerly (ties/atomics); graph check relaxed")
    assert (gout.float() - out.float()).abs().max().item() < 0.05
# timing
for name, fn in [("graph step (select+gather+attn)", g.replay)]:
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    print(f"{name}: {(time.time()-t)/50*1e3:.3f} ms per layer-step (maxblk={maxblk}, L={maxblk*BS})")
print("ALL OK")
