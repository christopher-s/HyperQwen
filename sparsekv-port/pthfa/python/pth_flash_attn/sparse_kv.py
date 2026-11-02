# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Part of sparsekv-hyperqwen; not part of upstream pth-flash-attn.
"""Query-aware sparse decode for ``PthFlashAttnImpl`` (``SPARSE_KV=1``).

Per full-attention layer with a basis slot:

* every forward writes rank-r int8 key sketches of ``key`` at
  ``attn_metadata.slot_mapping`` (prefill and decode; padding -> dummy row);
* the decode slice selects ``C = KSEL + SINK + WIN`` tokens per request
  (``sparsekv.select``), copies those cache rows into one persistent staging
  cache shared by all layers (``sparsekv.gather_rows``), and runs the stock
  pthfa launch over the staging cache (same plan, staging block table,
  ``seq_lens = n_valid``);
* the prefill slice and the MTP drafter layer stay dense.

Knobs: SPARSE_KV, SPARSE_BASIS (must exist unless SPARSE_ALLOW_RANDOM_BASIS=1), SPARSE_KSEL, SPARSE_SINK, SPARSE_WIN,
SPARSE_IMPL (ref|triton), SPARSEKV_ROOT (dir holding the ``sparsekv``
package; default the current directory = the HyperQwen repo).
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import sys
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def enabled() -> bool:
    return os.environ.get("SPARSE_KV", "0") == "1"


class _Cfg:
    def __init__(self) -> None:
        self.ksel = int(os.environ.get("SPARSE_KSEL", "2048"))
        self.sink = int(os.environ.get("SPARSE_SINK", "64"))
        self.win = int(os.environ.get("SPARSE_WIN", "1024"))
        self.C = self.ksel + self.sink + self.win
        self.impl_name = os.environ.get("SPARSE_IMPL", "triton")
        self.basis_path = os.environ.get("SPARSE_BASIS", os.path.join(sys.prefix, "share", "sparsekv", "basis27.pt"))
        root = os.environ.get("SPARSEKV_ROOT") or os.getcwd()
        self.root = root
        self.graph_safe = True  # triton_ops is capturable; ref modules declare GRAPH_SAFE
        if root not in sys.path:
            sys.path.insert(0, root)
        self.ops = self._load_ops()
        self._basis: dict[str, Any] | None = None
        self._skw: dict[tuple[int, int], dict[str, Any]] = {}

    def _load_ops(self):
        # G-5: no silent fallback. A failed triton_ops import used to fall back to sparsekv.ref, which (the
        # sk-kernel version) host-syncs and breaks CUDA-graph capture / stalls eager decode. ref must be explicit.
        if self.impl_name == "triton":
            try:
                tri = importlib.import_module("sparsekv.triton_ops")
            except Exception as e:  # noqa: BLE001
                raise ImportError(
                    f"sparse_kv: SPARSE_IMPL=triton but sparsekv.triton_ops failed to import "
                    f"(SPARSEKV_ROOT={self.root}): {e!r}; fix the tree or set SPARSE_IMPL=ref explicitly"
                ) from e
            logger.info("sparse_kv: using sparsekv.triton_ops (%s)", getattr(tri, "__file__", "?"))
            return tri
        if self.impl_name != "ref":
            raise ValueError(f"sparse_kv: SPARSE_IMPL={self.impl_name!r} (expected triton or ref)")
        ref = importlib.import_module("sparsekv.ref")
        self.graph_safe = bool(getattr(ref, "GRAPH_SAFE", False))
        # G-3: the ref path is a test oracle (fp32 [nreq, L, r] transients, ~250 MB per call at 150K), not for serving.
        logger.warning(
            "sparse_kv: using sparsekv.ref (%s, graph_safe=%s): test oracle only, slow and large transient "
            "memory at long max-model-len; do not use for speed or capacity runs",
            getattr(ref, "__file__", "?"), self.graph_safe,
        )
        return ref

    def select_kwargs(self, max_q: int, block_size: int) -> dict[str, Any]:
        """Static extras for ops.select; names differ between ref and triton_ops."""
        import inspect

        hit = self._skw.get((max_q, block_size))
        if hit is not None:
            return hit
        params = inspect.signature(self.ops.select).parameters
        kw: dict[str, Any] = {}
        for name in ("max_query_len", "max_q_len"):
            if name in params:
                kw[name] = int(max_q)
        if "block_size" in params:
            kw["block_size"] = int(block_size)
        elif block_size != getattr(self.ops, "BS", block_size):
            raise RuntimeError(f"sparse_kv: ops built for BS={self.ops.BS}, cache BS={block_size}")
        self._skw[(max_q, block_size)] = kw
        return kw

    def basis(self) -> dict[str, Any]:
        if self._basis is None:
            if os.path.exists(self.basis_path):
                d = torch.load(self.basis_path, map_location="cpu")
                self._basis = {
                    "layers": [int(x) for x in d["layers"]],
                    "basis": d["basis"].float(),
                    "src": self.basis_path,
                }
            else:
                # G-2: a missing/mistyped basis must not silently give a plausible-but-bad selector.
                if os.environ.get("SPARSE_ALLOW_RANDOM_BASIS", "0") != "1":
                    raise FileNotFoundError(
                        f"sparse_kv: SPARSE_BASIS={self.basis_path} does not exist; set it to the "
                        "basis file (e.g. <venv>/share/sparsekv/basis27.pt) or SPARSE_ALLOW_RANDOM_BASIS=1 "
                        "for plumbing-only runs"
                    )
                # Random-orthonormal placeholder for plumbing tests only.
                layers = list(range(3, 64, 4))
                g = torch.Generator().manual_seed(1234)
                b = torch.empty(len(layers), 4, 64, 256)
                for i in range(len(layers)):
                    for h in range(4):
                        qm, _ = torch.linalg.qr(torch.randn(256, 64, generator=g))
                        b[i, h] = qm.T
                self._basis = {"layers": layers, "basis": b, "src": "random-placeholder"}
                logger.warning(
                    "sparse_kv: %s missing, using a RANDOM orthonormal basis "
                    "(plumbing only)", self.basis_path
                )
            logger.info(
                "sparse_kv: basis %s, KSEL=%d SINK=%d WIN=%d C=%d impl=%s",
                self._basis["src"], self.ksel, self.sink, self.win, self.C,
                self.ops.__name__,
            )
        return self._basis


_CFG: _Cfg | None = None
# G-1: bytes taken out of vLLM's KV budget for buffers allocated after the memory profile; and bytes really allocated
_MEM = {"reserved": None, "allocated": 0, "warned": False}
# Staging cache shared by all layers on a device: {device: (kv, bt)}.
_STAGING: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}


def cfg() -> _Cfg:
    global _CFG
    if _CFG is None:
        _CFG = _Cfg()
    return _CFG


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class LayerSparse:
    """Per-layer sparse state, owned by one PthFlashAttnImpl."""

    def __init__(self, layer_name: str, max_num_seqs: int) -> None:
        self.layer_name = layer_name
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.active = False
        self.slot = -1
        m = _LAYER_RE.search(layer_name)
        if "mtp" in layer_name or m is None:
            return
        c = cfg()
        b = c.basis()
        idx = int(m.group(1))
        if idx not in b["layers"]:
            return
        self.slot = b["layers"].index(idx)
        self.basis_cpu = b["basis"][self.slot].contiguous()  # [Hkv, r, D]
        self.active = True
        self.key: tuple[int, tuple[int, ...]] | None = None
        self.basis_l: torch.Tensor | None = None
        self.sk_q8: torch.Tensor | None = None
        self.sk_scale: torch.Tensor | None = None

    # -- allocation -----------------------------------------------------------
    def ensure(self, kv_cache: torch.Tensor) -> None:
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        if self.key == key:
            return
        if _capturing():
            raise RuntimeError(
                f"sparse_kv: {self.layer_name}: first sight of a KV cache inside a "
                "CUDA-graph capture; buffers must exist before capture"
            )
        nb, hkv, bs, _w = kv_cache.shape
        r = self.basis_cpu.shape[1]
        dev = kv_cache.device
        self.sk_q8 = None
        self.sk_scale = None
        tier = _tier_runtime()
        rows = tier.sketch_rows(nb, bs) if tier is not None else nb * bs + 1
        rq = r if sketch_bits() == 8 else r // 2  # S4: SPARSE_SKETCH_BITS=4 packs two int4 dims per byte
        self.sk_q8 = torch.zeros(rows, hkv, rq, dtype=torch.int8, device=dev)
        self.sk_scale = torch.zeros(rows, hkv, dtype=torch.float16, device=dev)
        if tier is not None:
            tier.register_sketch(self.slot, self.sk_q8, self.sk_scale)
            # S4 selection reuse: last full top-k per hot slot (incl. the padding slot)
            self.topk_store = torch.zeros(tier.max_reqs + 1, cfg().ksel, dtype=torch.int32, device=dev)
        self.basis_l = self.basis_cpu.to(dev)
        self.key = key
        _account(self.sk_q8.numel() + 2 * self.sk_scale.numel())
        _ensure_staging(kv_cache, self.max_num_seqs)
        logger.info(
            "sparse_kv: %s (slot %d): sketch cache %d rows x %d heads x %d (int%d) "
            "(%.1f MiB)", self.layer_name, self.slot, rows, hkv, r, sketch_bits(),
            (self.sk_q8.numel() + 2 * self.sk_scale.numel()) / 2**20,
        )

    # -- per forward ----------------------------------------------------------
    def write(self, key: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        cfg().ops.write_sketch(key, slot_mapping, self.basis_l, self.sk_q8, self.sk_scale)

    def select_and_stage(self, part, query: torch.Tensor, kv_cache: torch.Tensor, scale: float):
        """Return (staging_kv, staging_bt[:nreq], n_valid) for a decode slice."""
        c = cfg()
        if not c.graph_safe and _capturing():
            raise RuntimeError(
                f"sparse_kv: {c.ops.__name__} host-syncs (no GRAPH_SAFE) and cannot run inside CUDA-graph "
                "capture; use SPARSE_IMPL=triton, a capturable ref, or --enforce-eager"
            )
        nreq = part.seq_lens.shape[0]
        t0, t1 = part.token_start, part.token_end
        q = query[t0:t1]
        tier = _tier_runtime()
        sk_bt = (tier.sketch_block_table(self.layer_name, part.block_table[:nreq], part.seq_lens)
                 if tier is not None else part.block_table)
        extra = tier.select_reuse_args(nreq, self.topk_store) if tier is not None else {}
        idx, n_valid = c.ops.select(
            q, part.query_start_loc, part.seq_lens, sk_bt,
            self.sk_q8, self.sk_scale, self.basis_l,
            c.ksel, c.sink, c.win, scale,
            **c.select_kwargs(part.max_query_len, kv_cache.shape[2]), **extra,
        )
        st_kv, st_bt = _STAGING[kv_cache.device]
        if nreq > st_bt.shape[0]:
            raise RuntimeError(
                f"sparse_kv: decode slice of {nreq} requests exceeds staging for "
                f"{st_bt.shape[0]}"
            )
        bt = st_bt[:nreq]
        lg = tier.group_of(self.layer_name) if tier is not None else None
        if lg is not None:
            # stream E host tier (SPARSE_TIER=1 + HostTierConnector): picks of spilled blocks come from the GPU hot
            # buffer or pinned host RAM (UVA); resident picks as in gather_rows
            tier.gather(lg[0], lg[1], kv_cache, part.block_table, part.seq_lens, idx, n_valid, st_kv, bt,
                        ref_gather=c.ops.gather_rows)
        else:
            c.ops.gather_rows(kv_cache, part.block_table, idx, n_valid, st_kv, bt)
        return st_kv, bt, n_valid


def _tier_runtime():
    if os.environ.get("SPARSE_TIER", "0") != "1":
        return None
    try:
        from sparsekv import tier_connector
    except Exception as e:  # noqa: BLE001
        logger.warning("sparse_kv: SPARSE_TIER=1 but sparsekv.tier_connector failed to import: %s", e)
        return None
    return tier_connector.runtime()


def _ensure_staging(kv_cache: torch.Tensor, max_num_seqs: int) -> None:
    dev = kv_cache.device
    _nb, hkv, bs, w = kv_cache.shape
    nbs = -(-cfg().C // bs)
    have = _STAGING.get(dev)
    if have is not None and have[0].shape[1:] == (hkv, bs, w) and have[1].shape == (max_num_seqs, nbs):
        return
    if _capturing():
        raise RuntimeError("sparse_kv: staging allocation inside a CUDA-graph capture")
    st_kv = torch.zeros(max_num_seqs * nbs, hkv, bs, w, dtype=kv_cache.dtype, device=dev)
    _account(st_kv.numel() * st_kv.element_size())
    st_bt = torch.arange(max_num_seqs * nbs, dtype=torch.int32, device=dev).reshape(
        max_num_seqs, nbs
    )
    _STAGING[dev] = (st_kv, st_bt)
    logger.info(
        "sparse_kv: staging cache %d reqs x %d blocks (%.1f MiB)", max_num_seqs, nbs,
        st_kv.numel() / 2**20,
    )


def staging_views(impl, st_kv: torch.Tensor, attr: str = "_sparse_st_views") -> tuple[torch.Tensor, ...]:
    """(k, v, k_scale, v_scale) views of a staging cache, as Impl._caches builds."""
    cached = getattr(impl, attr, None)
    if cached is not None and cached[0] == st_kv.data_ptr():
        return cached[1]
    num_blocks, nkv, block_size, content = st_kv.shape
    dsz = st_kv.element_size()
    padded_hs = content // 2
    hs = impl.head_size
    raw = st_kv.untyped_storage()
    base_f32 = torch.tensor([], dtype=torch.float32, device=st_kv.device).set_(raw)
    s = st_kv.stride()

    def f32(e: int) -> int:
        assert (e * dsz) % 4 == 0
        return e * dsz // 4

    base = f32(st_kv.storage_offset())
    ks = torch.as_strided(base_f32, (num_blocks, block_size, nkv),
                          (f32(s[0]), f32(s[2]), f32(s[1])), base + f32(hs))
    vs = torch.as_strided(base_f32, (num_blocks, block_size, nkv),
                          (f32(s[0]), f32(s[2]), f32(s[1])), base + f32(padded_hs + hs))
    kc, vc = st_kv.transpose(1, 2).split(padded_hs, dim=-1)
    views = (kc[..., :hs], vc[..., :hs], ks, vs)
    setattr(impl, attr, (st_kv.data_ptr(), views))
    return views


# ---------------------------------------------------------------------------
# G-1: memory reserve. Sketch caches, the staging cache and (SPARSE_TIER=1) the tier's GPU buffers are allocated at
# the first forward / KV registration, i.e. AFTER vLLM sized the KV pool from its memory profile. Without a reserve
# a production-like gpu_memory_utilization OOMs at capture warm-up. Worker.determine_available_memory is wrapped
# (installed from PthFlashAttnImpl.__init__, i.e. in the worker, before profiling) to subtract an upper bound:
#   sketches  <= budget * f / (1 + f),  f = (r + 2) / W   (per pool token and sparse layer: Hkv*(r+2) B of sketch vs
#             >= Hkv*W B of that layer's own KV; W = 2*(D+4) int8 row incl. the inline fp32 scale)
#   + staging (max_num_seqs * ceil(C/BS) blocks) + sparsekv.tier_connector.gpu_extra_bytes when SPARSE_TIER=1.
# SPARSE_MEM_RESERVE=0 disables (reproduce old runs); SPARSE_MEM_RESERVE_BYTES=<n> overrides the estimate.
# ---------------------------------------------------------------------------
def gpu_reserve_bytes(vllm_config, kv_budget: int) -> tuple[int, dict[str, int]]:
    c = cfg()
    b = c.basis()
    nl = len(b["layers"])
    r = int(b["basis"].shape[2])
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    hkv = int(mc.get_num_kv_heads(pc))
    d = int(mc.get_head_size())
    w = 2 * (d + 4)
    bs = int(getattr(vllm_config.cache_config, "block_size", None) or 864)
    max_seqs = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    rb = sketch_row_bytes(r)  # per token and KV head: int8 r (or int4 r/2) + fp16 scale
    f = rb / w
    parts = {
        "staging": max_seqs * (-(-c.C // bs)) * hkv * bs * w,
    }
    if os.environ.get("SPARSE_TIER", "0") == "1":
        from sparsekv import tier_connector
        parts.update(tier_connector.gpu_extra_bytes(vllm_config, nl, hkv, w, r, bs, c.C))
        if os.environ.get("SPARSE_TIER_VERIFY", "0") == "1":
            parts["tier_verify_buf"] = parts["staging"]
    # S4: the GPU-slot sketch region scales with the pool that is LEFT after the other reserves (it used to be sized
    # from the whole budget: ~0.5 GiB too much at 128K x 4). f = sketch bytes per pool byte of a sparse layer's KV.
    rest = max(0, kv_budget - sum(parts.values()))
    parts = {"sketch": int(rest * f / (1.0 + f)) + nl * hkv * rb, **parts}
    total = sum(parts.values())
    override = os.environ.get("SPARSE_MEM_RESERVE_BYTES")
    if override:
        total = int(override)
        parts = {"override": total}
    return total, parts


def sketch_bits() -> int:
    """SPARSE_SKETCH_BITS: 8 (default, int8 rank-r sketch) or 4 (S4: int4, two values per byte; half the GPU bytes)."""
    b = int(os.environ.get("SPARSE_SKETCH_BITS", "8"))
    if b not in (4, 8):
        raise ValueError(f"sparse_kv: SPARSE_SKETCH_BITS={b} (expected 4 or 8)")
    return b


def sketch_row_bytes(r: int) -> int:
    """Sketch bytes per token and KV head (values + fp16 scale)."""
    return (r if sketch_bits() == 8 else r // 2) + 2


def install_memory_reserve() -> None:
    if os.environ.get("SPARSE_MEM_RESERVE", "1") != "1":
        logger.warning("sparse_kv: SPARSE_MEM_RESERVE=0: sparse buffers are NOT budgeted in the KV pool size")
        return
    try:
        from vllm.v1.worker import gpu_worker
    except Exception as e:  # noqa: BLE001
        logger.warning("sparse_kv: cannot install the memory reserve (%s)", e)
        return
    wk = gpu_worker.Worker
    if getattr(wk.determine_available_memory, "_sparse_kv_reserve", False):
        return
    orig = wk.determine_available_memory

    @functools.wraps(orig)
    def determine_available_memory(self) -> int:
        avail = int(orig(self))
        res, parts = gpu_reserve_bytes(self.vllm_config, avail)
        if res >= avail:
            raise RuntimeError(
                f"sparse_kv: sparse buffers need ~{res / 2**30:.2f} GiB but the KV budget is only "
                f"{avail / 2**30:.2f} GiB; raise --gpu-memory-utilization / --kv-cache-memory or lower max-model-len"
            )
        _MEM["reserved"] = res
        logger.info(
            "sparse_kv: reserving %.2f GiB of the %.2f GiB KV budget for buffers allocated after profiling (%s)",
            res / 2**30, avail / 2**30, ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in parts.items()),
        )
        if hasattr(self, "available_kv_cache_memory_bytes"):
            self.available_kv_cache_memory_bytes = avail - res
        return avail - res

    determine_available_memory._sparse_kv_reserve = True
    wk.determine_available_memory = determine_available_memory


def _account(nbytes: int) -> None:
    _MEM["allocated"] += int(nbytes)
    res = _MEM["reserved"]
    if res is not None and _MEM["allocated"] > res and not _MEM["warned"]:
        _MEM["warned"] = True
        logger.warning(
            "sparse_kv: sparse buffers allocated %.2f GiB > reserved %.2f GiB (estimate too low; set "
            "SPARSE_MEM_RESERVE_BYTES)", _MEM["allocated"] / 2**30, res / 2**30,
        )


# ---------------------------------------------------------------------------
# Per-step dense/sparse switch (stage 4).
#
# The decision is made on the host from the batch's max_seq_len (an int vLLM
# already has) and a per-GPU / per-batch-size cutoff table. FULL decode graphs
# are captured twice (dense family + sparse family; vLLM patch
# patches_switch/sparse-switch-dual-graphs.patch) and the graph manager replays
# the family chosen for this step, so the switch costs nothing per step.
# Eager / piecewise steps read the same decision from the metadata.
#
# SPARSE_SWITCH=0 disables it (always sparse, one graph family, as before).
# SPARSE_CUTOFF overrides the table: "44000" (all batch sizes) or
# "1:44000,2:29000" (by real request count; the largest key <= n applies).
# SPARSE_CUTOFF=0 means always sparse. Real host-tier spill (SPARSE_TIER_SPILL=1)
# forces always-sparse: spilled blocks are not readable by the dense kernel.
# ---------------------------------------------------------------------------

# Default cutoffs from stream A's crossover bench (STREAM_kernel.md, a834af8).
_DEFAULT_CUTOFFS = {
    "3090 Ti": {1: 44000, 2: 29000},
    "3090": {1: 20000, 2: 19000},
}
_FAMILY: str | None = None  # "dense" / "sparse" while capturing one family
_STEP_SPARSE = True
_CUTOFFS: dict[int, int] | None = None
_SWITCH_LOG = os.environ.get("SPARSE_SWITCH_LOG", "1") == "1"
_LAST_LOGGED: bool | None = None


def switch_enabled() -> bool:
    return enabled() and os.environ.get("SPARSE_SWITCH", "1") == "1"


def cutoffs() -> dict[int, int]:
    """Cutoff table {real batch size: min max_seq_len for sparse}; {} = always sparse."""
    global _CUTOFFS
    if _CUTOFFS is not None:
        return _CUTOFFS
    table: dict[int, int] = {}
    env = os.environ.get("SPARSE_CUTOFF", "").strip()
    if not switch_enabled() or os.environ.get("SPARSE_TIER_SPILL", "0") == "1":
        table = {}
    elif env:
        if ":" in env:
            for item in env.split(","):
                k, v = item.split(":")
                table[int(k)] = int(v)
        elif int(env) > 0:
            table = {1: int(env)}
    else:
        name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
        for key in sorted(_DEFAULT_CUTOFFS, key=len, reverse=True):  # "3090 Ti" before "3090"
            if key in name:
                table = dict(_DEFAULT_CUTOFFS[key])
                break
    table = {k: v for k, v in table.items() if v > 0}
    _CUTOFFS = table
    logger.info("sparse_kv: dense/sparse switch cutoffs %s (%s)",
                table or "none: always sparse", "env" if env else "per-GPU default")
    return table


def cutoff(nreq: int) -> int:
    t = cutoffs()
    if not t:
        return 0
    keys = [k for k in sorted(t) if k <= nreq]
    return t[keys[-1]] if keys else t[min(t)]


def dual_graphs() -> bool:
    """Capture two FULL graph families (only if some step can go dense)."""
    return switch_enabled() and bool(cutoffs())


def decide(max_seq_len: int, nreq: int) -> bool:
    """Host decision for this step (called by the target builder); remembered for replay."""
    global _STEP_SPARSE, _LAST_LOGGED
    _STEP_SPARSE = (not switch_enabled()) or max_seq_len >= cutoff(max(1, nreq))
    if _SWITCH_LOG and _STEP_SPARSE != _LAST_LOGGED and nreq > 0 and max_seq_len < 10**7:
        _LAST_LOGGED = _STEP_SPARSE
        logger.info("sparse_kv: switch -> %s (max_seq_len=%d, reqs=%d, cutoff=%d)",
                    "SPARSE" if _STEP_SPARSE else "dense", max_seq_len, nreq, cutoff(max(1, nreq)))
    return _STEP_SPARSE


def step_sparse() -> bool:
    return _STEP_SPARSE


def set_family(family: str | None) -> None:
    global _FAMILY
    _FAMILY = family


def decode_sparse(attn_metadata) -> bool:
    """Run this forward's decode slice sparse? Family being captured wins."""
    if _FAMILY is not None:
        return _FAMILY == "sparse"
    return bool(getattr(attn_metadata, "sparse_decode", True))
