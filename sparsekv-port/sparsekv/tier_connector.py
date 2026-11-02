# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sparsekv/tier_connector.py -- run the host tier inside the vLLM server, loaded as an external KV connector.

    --kv-transfer-config '{"kv_connector":"HostTierConnector","kv_connector_module_path":"sparsekv.tier_connector",
                           "kv_role":"kv_both"}'      (plus SPARSE_KV=1 SPARSE_TIER=1)

Mode "shadow spill" (the only mode so far; no scheduler change):
  * every full block of the 16 sparse layers is mirrored (DMA) to a pinned host pool as it fills -- host block id ==
    GPU block id, so the host pool has the GPU pool's size;
  * the sparse decode (stream B's compaction path, pth_flash_attn/sparse_kv.py) then TREATS every mirrored block
    between the sink block and the tail window as spilled: those picks come from the GPU hot buffer or from host RAM
    through UVA (sparsekv.tier.gather_rows_tiered). The GPU copy stays allocated, so nothing else changes.
  This exercises the real data path in the real server: decode speed with host fetches, and exactness (the staged
  bytes are identical, so greedy outputs must equal SPARSE_TIER=0). Freeing the GPU blocks (actual capacity) is the
  admission work described in STREAM_tier.md / tier_admission.py.

Knobs: SPARSE_TIER=1, SPARSE_TIER_HOT_FRAC (0.10 of max_model_len, >= 2C rows), SPARSE_TIER_TAIL (tokens kept
"resident" at the end, default WIN + 64), SPARSE_TIER_SINK_BLOCKS (1), SPARSE_TIER_MARGIN (64 tokens: a block is
mirrored only when it ends this far below num_computed_tokens, so async/spec-decode in-flight tokens never land in a
mirrored block).
"""
from __future__ import annotations

import collections
import dataclasses
import itertools
import os
import re
import sys
from typing import Any

import numpy as np
import torch

from sparsekv.tier import (HostKVMirror, HotBuffer, TierScratch, alloc_pinned, copy_blocks, copy_blocks_from_host4,
                           gather_rows_tiered, host_row_width, quant_blocks_to_host4)

try:
    from vllm.v1.core.kv_cache_utils import get_block_hash, make_block_hash_with_group_id
except Exception:  # noqa: BLE001
    get_block_hash = make_block_hash_with_group_id = None

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_RT: "TierRuntime | None" = None


def enabled() -> bool:
    return os.environ.get("SPARSE_TIER", "0") == "1"


def runtime() -> "TierRuntime | None":
    return _RT


def tier_dims(vllm_config, bs: int, C: int, spill: bool) -> dict:
    """Sizes TierRuntime allocates (shared with gpu_extra_bytes so the memory reserve cannot drift)."""
    max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
    max_len = int(vllm_config.model_config.max_model_len)
    maxblk = -(-max_len // bs) + 16  # >= the runner block-table width (it pads)
    hot_frac = float(os.environ.get("SPARSE_TIER_HOT_FRAC", "0.10"))
    H = max(2 * C, int(hot_frac * max_len))
    HB = None
    if spill:
        # same rule as host_pool_blocks (SPARSE_TIER_HOST_GB / _HOST_TOKENS / default capped at _HOST_GB_CAP)
        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        block_bytes = (int(mc.get_num_kv_heads(pc)) * bs * host_row_width(2 * (int(mc.get_head_size()) + 4))
                       * len(sparse_layer_ids()))
        HB = host_pool_blocks(max_len, max_reqs, bs, block_bytes, check_ram=False)
    return {"max_reqs": max_reqs, "max_len": max_len, "maxblk": maxblk, "H": H, "HB": HB}


def gpu_extra_bytes(vllm_config, nlayers: int, hkv: int, w: int, r: int, bs: int, C: int) -> dict:
    """GPU bytes the tier allocates after vLLM's memory profile (G-1/G-17): hot buffer, scratch, and in SPILL mode
    the host-indexed sketch region and the prefill staging cache. Used by sparse_kv.gpu_reserve_bytes."""
    spill = os.environ.get("SPARSE_TIER_SPILL", "0") == "1"
    d = tier_dims(vllm_config, bs, C, spill)
    nslots = d["max_reqs"] + 1  # + the CUDA-graph padding slot: bookkeeping only, no hot KV rows (S4)
    out = {
        "tier_hot": nlayers * (d["max_reqs"] * d["H"] * hkv * w + nslots * (d["H"] * 8 + d["maxblk"] * bs * 4)),
        "tier_scratch": (d["max_reqs"] + 8) * C * 17,
    }
    if spill:
        out["tier_sketch_host_rows"] = nlayers * d["HB"] * bs * hkv * sketch_row_bytes(r)
        out["tier_prefill_staging"] = (d["maxblk"] + 1) * hkv * bs * w
        # S4: suffix staging lives in the prefill staging cache (sparse suffixes run before dense chunks per layer)
        out["tier_prefill_staging"] = max(d["maxblk"] + 1, suffix_staging_blocks(C, bs)) * hkv * bs * w
    return out


def suffix_sparse_enabled() -> bool:
    """S4: follow-up suffixes (uncached prefill tails <= SPARSE_TIER_SUFFIX_MAX tokens) of requests with spilled blocks
    attend sparsely (select over the sketch + tiered gather of only the picked rows) instead of staging every block."""
    return os.environ.get("SPARSE_TIER_SUFFIX_SPARSE", "1") == "1"


def suffix_cfg() -> dict:
    """Suffix knobs (env; SPARSE_TIER_SUFFIX_CFG=<json file> overrides them at runtime, re-read when it changes -- dev
    tuning without a reboot). A suffix is split into segments that each get their own selection and causal end: the
    last SEG_LAST rows (the new question) in segments of SEG_Q rows, the uncached document tail before it in SEG-row
    segments. One summed selection over ~1.5K rows let the tail rows drown the question's picks (s4a128n4)."""
    e = os.environ.get
    return {"sparse": e("SPARSE_TIER_SUFFIX_SPARSE", "1") == "1", "max": int(e("SPARSE_TIER_SUFFIX_MAX", "2048")),
            "ksel": int(e("SPARSE_TIER_SUFFIX_KSEL", "4096")),  # tuned (run_s4/tune.out): 6/8 code lines vs 2/8
            "seg": int(e("SPARSE_TIER_SUFFIX_SEG", "256")), "seg_last": int(e("SPARSE_TIER_SUFFIX_SEG_LAST", "128")),
            "seg_q": int(e("SPARSE_TIER_SUFFIX_SEG_Q", "16"))}


def suffix_segments(ql: int, cfg: dict | None = None) -> list[int]:
    c = cfg or suffix_cfg()
    tail = min(ql, c["seg_last"])
    out = []
    while tail > 0:  # question rows, SEG_Q at a time, from the end
        n = min(c["seg_q"], tail)
        out.insert(0, n)
        tail -= n
    rem = ql - sum(out)
    while rem > 0:
        n = min(c["seg"], rem)
        out.insert(0, n)
        rem -= n
    return out


def suffix_staging_blocks(C: int, bs: int, cfg: dict | None = None) -> int:
    """Staging blocks for one suffix (requests are staged one after the other, in pf_kv): C' rows per segment."""
    c = cfg or suffix_cfg()
    win = int(os.environ.get("SPARSE_WIN", "1024"))
    ksel = int(os.environ.get("SPARSE_KSEL", "2048"))
    return len(suffix_segments(c["max"], c)) * -(-(C - ksel + c["ksel"]) // bs)


def sketch_row_bytes(r: int) -> int:
    """Sketch bytes per token and KV head; mirrors pth_flash_attn.sparse_kv.sketch_row_bytes (SPARSE_SKETCH_BITS)."""
    return (r if int(os.environ.get("SPARSE_SKETCH_BITS", "8")) == 8 else r // 2) + 2


def _log(msg, *a):
    try:
        from vllm.logger import init_logger
        init_logger("vllm.sparsekv.tier").info(msg, *a)
    except Exception:  # noqa: BLE001
        print(msg % a if a else msg, flush=True)


class TierRuntime:
    """Worker-side tier state for all sparse layers of one GPU.

    SPARSE_TIER_SPILL=0 (shadow): host block id == GPU block id, GPU blocks are never freed.
    SPARSE_TIER_SPILL=1 (real spill): the scheduler frees GPU blocks of the sparse group in
      [SINK_BLOCKS, (processed - TAIL) // BS) (HostTierConnector.bind_kv_cache_manager). Every such block has been
      mirrored to its own host block before (mirroring runs before the forward of the step that frees it, and the
      host copy is stream-ordered before any reuse). The worker treats as spilled every mirrored block outside
      [0, SINK_BLOCKS) and the TAIL -- a superset of what the scheduler freed -- so stale GPU block ids are never read:
        * sparse decode: hot buffer / host rows (gather_rows_tiered), sketch rows from a host-indexed sketch region;
        * dense prefill of a sparse layer: all blocks staged into a scratch cache (host DMA + GPU copies);
        * other layers of the group (MTP drafter): sink + tail window only (compacted block table).
    """

    def __init__(self, kv_caches: dict[str, torch.Tensor], kv_cache_config, vllm_config):
        ksel = int(os.environ.get("SPARSE_KSEL", "2048"))
        sink = int(os.environ.get("SPARSE_SINK", "64"))
        win = int(os.environ.get("SPARSE_WIN", "1024"))
        self.C = ksel + sink + win
        self.spill = os.environ.get("SPARSE_TIER_SPILL", "0") == "1"
        basis_layers = sparse_layer_ids()
        self.layers: dict[str, tuple[int, int]] = {}
        group_of = {}
        for g, grp in enumerate(kv_cache_config.kv_cache_groups):
            for name in grp.layer_names:
                group_of[name] = g
        for name in kv_caches:
            m = _LAYER_RE.search(name)
            if "mtp" in name or m is None or int(m.group(1)) not in basis_layers:
                continue
            self.layers[name] = (basis_layers.index(int(m.group(1))), group_of[name])
        assert len(self.layers) == len(basis_layers), f"tier: found {len(self.layers)} sparse layers"
        self.groups = sorted({g for _, g in self.layers.values()})
        if self.spill:
            assert len(self.groups) == 1, f"tier spill needs all sparse layers in one KV group, got {self.groups}"
        self.window_layers = {n for n in kv_caches
                              if group_of.get(n) in self.groups and n not in self.layers}
        any_kv = next(kv_caches[n] for n in self.layers)
        NB, HKV, BS, W = any_kv.shape
        self.NB, self.BS = NB, BS
        dims = tier_dims(vllm_config, BS, self.C, self.spill)
        self.max_reqs, self.max_len = dims["max_reqs"], dims["max_len"]
        maxblk = dims["maxblk"]  # >= the runner block-table width (it pads)
        self.maxblk = maxblk
        L = len(basis_layers)
        self.L = L
        self.kv_by_slot = [None] * L
        for name, (l, _) in self.layers.items():
            self.kv_by_slot[l] = kv_caches[name]
        self.tail = int(os.environ.get("SPARSE_TIER_TAIL", str(win + 64)))
        self.sink_blocks = int(os.environ.get("SPARSE_TIER_SINK_BLOCKS", "1"))
        self.margin = int(os.environ.get("SPARSE_TIER_MARGIN", "64"))
        assert self.tail >= self.margin + 16
        H = dims["H"]
        dev = any_kv.device
        self.dev = dev
        self.WH = host_row_width(W)  # S4: 272 with SPARSE_TIER_HOST_BITS=4 (int4 host tier), else W
        self.host4 = self.WH != W
        if self.spill:
            self.HB = host_pool_blocks(self.max_len, self.max_reqs, BS, HKV * BS * self.WH * L)
            assert self.HB == dims["HB"], (self.HB, dims["HB"])  # the G-1 reserve was sized from dims
        else:
            assert not self.host4, "SPARSE_TIER_HOST_BITS=4 needs SPARSE_TIER_SPILL=1"
            self.HB = NB
        self.mirror = HostKVMirror(L, self.HB, HKV, BS, self.WH, device=dev)
        self.host_free = list(range(self.HB - 1, -1, -1))
        nslots = self.max_reqs + 1  # slot max_reqs = dummy for CUDA-graph padding rows
        self.hot = HotBuffer(L, nslots, H, HKV, W, max_ctx=maxblk * BS, device=dev, kv_slots=self.max_reqs)
        self.scratch = TierScratch(self.max_reqs + 8, self.C, dev)
        self.batch_slots = torch.full((self.max_reqs + 8,), self.max_reqs, dtype=torch.int32, device=dev)
        self.mirrored = torch.zeros((max(self.groups) + 1, nslots), dtype=torch.int32, device=dev)
        self.host_bt = torch.full((nslots, maxblk), -1, dtype=torch.int32, device=dev)
        # SPILL: blocks [SINK_BLOCKS, freed_upto[slot]) were freed by the scheduler (reported in the connector meta)
        self.freed_upto = torch.zeros(nslots, dtype=torch.int32, device=dev)
        self.barange = torch.arange(maxblk, dtype=torch.int32, device=dev)
        self.sketch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.verify = os.environ.get("SPARSE_TIER_VERIFY", "0") == "1"
        self.vbuf = None
        self.vcount = torch.zeros(2, dtype=torch.int64, device=dev)  # [differing bytes, compared bytes]
        # G-17: prefill staging cache for one full-length request, allocated here (runtime init, before graph
        # capture) and counted in sparse_kv.gpu_reserve_bytes; grows only if one prefill slice needs more.
        self.pf_kv = (torch.empty((max(maxblk + 1, suffix_staging_blocks(self.C, BS)), HKV, BS, W), dtype=any_kv.dtype,
                                  device=dev) if self.spill else None)
        # GDN/mamba checkpoint host store (worker half): per mamba group, per layer state tensor, [slots, ...] pinned
        self.mamba_items: dict[int, list[torch.Tensor]] = {}
        self.mamba_done: set[int] = set()  # G-23: save ids whose GPU->host copy is enqueued (read by the scheduler)
        self.fin_done: set[int] = set()  # G-25: finish-time tail copy ids enqueued (read by the scheduler)
        self.mamba_host: dict[int, list[torch.Tensor]] = {}
        ms = host_mamba_slots() if self.spill else 0
        if ms > 0:
            nbytes = 0
            for g, grp in enumerate(kv_cache_config.kv_cache_groups):
                if type(grp.kv_cache_spec).__name__ != "MambaSpec":
                    continue
                items = []
                for name in grp.layer_names:
                    t = kv_caches.get(name)
                    if t is None:
                        continue
                    items.extend(t if isinstance(t, (list, tuple)) else [t])
                self.mamba_items[g] = items
                nbk = kv_cache_config.num_blocks
                # vLLM state tensors may hold several kernel blocks per scheduler block (see
                # copy_kv_cache_blocks_inplace): block b = rows [b*k, (b+1)*k)
                self.mamba_k = {id(it): it.shape[0] // nbk for it in items} | getattr(self, "mamba_k", {})
                self.mamba_host[g] = [alloc_pinned((ms, it.shape[0] // nbk, *it.shape[1:]), it.dtype) for it in items]
                _log("tier: mamba group %d: %d state tensors, first shapes %s strides %s, num_blocks %d", g,
                     len(items), [tuple(t.shape) for t in items[:2]], [t.stride() for t in items[:2]], nbk)
                nbytes += sum(h.numel() * h.element_size() for h in self.mamba_host[g])
            _log("tier: mamba checkpoint host store: groups %s, %d slots, %.2f GB pinned (%.1f MB per checkpoint)",
                 sorted(self.mamba_items), ms, nbytes / 1e9, nbytes / ms / 1e6)
        # S4: pinned host ring for host->device uploads (non-blocking; the old pageable copies synced every step)
        self._ring = [[torch.empty(1 << 12, dtype=torch.int64).pin_memory(), None] for _ in range(16)]
        self._ring_i = 0
        self.sync_slots = os.environ.get("SPARSE_TIER_SYNC_SLOTS", "0") == "1"  # A/B: old synchronous slot upload
        # S4 selection reuse: SPARSE_SELECT_EVERY=N -> a decoding request runs the full selection every N-th step and
        # reuses its top-k (per hot slot, per layer) in between; flags are per batch row, decided here on the host
        self.select_every = max(1, int(os.environ.get("SPARSE_SELECT_EVERY", "1")))
        self.dec_q = int(os.environ.get("DRAFT_TOKENS", "7")) + 1
        self.batch_reuse = torch.ones((self.max_reqs + 8,), dtype=torch.int32, device=dev)
        self._sel_counts = [0, 0]  # [full, reused] decode-row selections
        # S4: prefill plan (per step, shared by the 16 sparse layers) and sparse-suffix buffers
        self.ksel, self.sink, self.win = ksel, sink, win
        self.sfx = suffix_cfg()
        self.suffix_sparse = self.spill and self.sfx["sparse"]
        self.suffix_max = self.sfx["max"]
        self._sfx_file, self._sfx_mtime = os.environ.get("SPARSE_TIER_SUFFIX_CFG"), None
        self._pp_key = None
        self._pp = None
        self._sub_plans: dict = {}
        self.sf_stats = torch.zeros(3, dtype=torch.int64, device=dev)  # suffix gathers: [hit, host rows, resident]
        self.sf_scratch = None
        self.pad_slots = None
        self.reqs: dict[str, dict] = {}
        self.free_slots = list(range(self.max_reqs - 1, -1, -1))
        self.batch_rows: list[str | None] = []
        self.stats = {"steps": 0, "mirrored_blocks": 0, "prefill_staged_blocks": 0, "host_blocks_peak": 0,
                      "suffix_sparse": 0, "suffix_tokens": 0}
        _log("tier: kv cache view shape %s strides %s contiguous=%s", tuple(any_kv.shape), any_kv.stride(),
             any_kv.is_contiguous())
        _log("tier: runtime mode=%s: %d sparse layers in group(s) %s, window layers %s, host pool %.2f GB "
             "(%d blocks), GPU pool %d blocks, hot %d rows/slot (%.0f MB total), C=%d, tail=%d",
             "SPILL" if self.spill else "shadow", L, self.groups, sorted(self.window_layers),
             self.mirror.nbytes / 1e9, self.HB, NB, H, self.hot.nbytes / 1e6, self.C, self.tail)

    # ---------------------------------------------------------------- sketch registry (called by sparse_kv)
    def sketch_rows(self, nb: int, bs: int) -> int:
        """Rows the sketch cache needs: GPU slots + (spill) a host-indexed region [NB*BS, (NB+HB)*BS)."""
        return (nb + (self.HB if self.spill else 0)) * bs + 1

    def register_sketch(self, slot_l: int, sk_q8: torch.Tensor, sk_scale: torch.Tensor) -> None:
        self.sketch[slot_l] = (sk_q8, sk_scale)

    # ---------------------------------------------------------------- per step (CPU, before the forward)
    def start_step(self, so, request_ids: list[str] | None, meta=None):
        G = max(self.groups) + 1
        for rid in so.finished_req_ids or ():
            self._drop(rid)
        for nr in so.scheduled_new_reqs:
            self._drop(nr.req_id)
            self._new(nr.req_id, nr.block_ids, nr.num_computed_tokens)
            # S4: uncached prompt tokens at admission (prompt minus prefix/host hit): only a short remainder (a
            # follow-up turn) may take the sparse-suffix prefill; cold prompts stay dense chunk by chunk
            ptoks = getattr(nr, "prompt_token_ids", None)
            if ptoks is not None:
                self.reqs[nr.req_id]["uncached"] = len(ptoks) - int(nr.num_computed_tokens)
        cr = so.scheduled_cached_reqs
        # G-13: sk-port's hisparse-oct-fixes removes SchedulerOutput.block_table_updates; the real-spill design does
        # not need it (freed ranges come via the connector meta), so treat a missing field as "no updates".
        updates = getattr(so, "block_table_updates", None) or {}
        for i, rid in enumerate(cr.req_ids):
            nb = cr.new_block_ids[i]
            if rid in cr.resumed_req_ids or rid not in self.reqs:
                self._drop(rid)
                self._new(rid, nb or tuple([] for _ in range(G)), cr.num_computed_tokens[i])
                continue
            st = self.reqs[rid]
            if rid in updates:
                st["blocks"] = {g: list(updates[rid][g]) for g in self.groups}
            elif nb is not None:
                for g in self.groups:
                    st["blocks"][g].extend(nb[g])
            st["ncomp"] = cr.num_computed_tokens[i]
        for rid, n in (so.num_scheduled_tokens or {}).items():
            if rid in self.reqs:
                self.reqs[rid]["nsched"] = n
        if self.spill:
            self._apply_host_meta(meta)
            if meta is not None and self.mamba_items:
                self._apply_mamba_meta(meta)
        else:
            self._shadow_mirror()
        # batch order -> slots (attention metadata rows follow the input-batch order)
        slots = [self.max_reqs] * self.batch_slots.numel()
        self.batch_rows = list(request_ids or [])
        for i, rid in enumerate(self.batch_rows[: len(slots)]):
            st = self.reqs.get(rid)
            if st is not None:
                slots[i] = st["slot"]
        reuse = self._reuse_flags(len(slots))
        if self.sync_slots:
            self.batch_slots.copy_(torch.tensor(slots, dtype=torch.int32))  # pageable source: host-synchronous (old)
            if reuse is not None:
                self.batch_reuse.copy_(torch.tensor(reuse, dtype=torch.int32))
        elif reuse is None:
            self.batch_slots.copy_(self.h2d(slots, torch.int32))  # pinned, stream-ordered, no host sync
        else:
            dv = self.h2d(slots + reuse, torch.int32)
            self.batch_slots.copy_(dv[:len(slots)])
            self.batch_reuse.copy_(dv[len(slots):])
        self.hot.next_step()
        self.stats["steps"] += 1
        if self.stats["steps"] % 50 == 0 or so.finished_req_ids or (meta is not None and meta.nhit):
            if _HOST_CACHE is not None:
                self.stats["hostcache"] = dict(_HOST_CACHE.stats, used=_HOST_CACHE.n - len(_HOST_CACHE.free))
            if getattr(self, "select_every", 1) > 1:
                self.stats["select_full_reused"] = list(self._sel_counts)
            if _MAMBA_HOST is not None:
                self.stats["gdn_store"] = dict(_MAMBA_HOST.stats)
            _log("tier: stats %s hot[hit,miss,resident]=%s suffix[hit,host,resident]=%s verify[diff,compared]=%s",
                 self.stats, self.hot.stats.tolist(), self.sf_stats.tolist(), self.vcount.tolist())

    def _reuse_flags(self, n: int):
        """Per batch row: 1 = reuse last step's decode top-k. Only for a request that ran a full or reused decode
        selection in the previous step with a long context (its stored top-k is valid), at most N-1 times in a row.
        Padding rows: 1 (no work)."""
        if getattr(self, "select_every", 1) <= 1:
            return None
        step = self.stats["steps"]
        flags = [1] * n
        for i, rid in enumerate(self.batch_rows[:n]):
            st = self.reqs.get(rid)
            if st is None:
                continue
            dec = 0 < st["nsched"] <= self.dec_q
            if not dec:
                st["sel_step"] = -1
                flags[i] = 0
                continue
            ok = (st.get("sel_step") == step - 1 and st.get("sel_len", 0) > self.C + 64
                  and st.get("since", 0) < self.select_every - 1)
            flags[i] = 1 if ok else 0
            st["since"] = st.get("since", 0) + 1 if ok else 0
            st["sel_step"] = step
            st["sel_len"] = st["ncomp"]
            self._sel_counts[int(ok)] += 1
        return flags

    def h2d(self, xs, dtype=torch.int32) -> torch.Tensor:
        """Host list -> new device tensor through a pinned ring buffer (non_blocking; no host sync unless the ring
        slot's previous copy has not run yet, which needs 16 uploads queued ahead of the GPU)."""
        n = len(xs)
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        ent = self._ring[self._ring_i]
        if ent[1] is not None:
            ent[1].synchronize()
        if ent[0].numel() < n:
            ent[0] = torch.empty(max(n, 2 * ent[0].numel()), dtype=torch.int64).pin_memory()
        hb = ent[0][:n]
        if n:
            hb.copy_(torch.as_tensor(np.asarray(xs, dtype=np.int64)))
        out = torch.empty(n, dtype=torch.int64, device=self.dev)
        out.copy_(hb, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        ent[1] = ev
        return out if dtype == torch.int64 else out.to(dtype)

    def _copy_blocks(self, g, gids, hids):
        """DMA GPU blocks (all sparse layers of group g) + their sketch rows into host blocks (compute stream)."""
        rows = None
        ids4 = None
        for l in range(self.L):
            if self.layers_group(l) != g:
                continue
            kv = self.kv_by_slot[l]
            if self.host4:  # quantise on the GPU and write straight into pinned host memory (UVA)
                if ids4 is None:
                    dv = self.h2d(list(gids) + list(hids), torch.int32)
                    ids4 = (dv[:len(gids)], dv[len(gids):])
                quant_blocks_to_host4(kv, ids4[0], self.mirror.dev[l], ids4[1])
            else:
                for gb, hb in zip(gids, hids):
                    self.mirror.host[l, hb].copy_(kv[gb], non_blocking=True)
            if self.spill and l in self.sketch:
                q8, sc = self.sketch[l]
                if rows is None:  # same rows for every layer: one upload
                    ar = torch.arange(self.BS, device=self.dev)
                    rows = ((self.h2d(gids, torch.int64)[:, None] * self.BS + ar).reshape(-1),
                            ((self.h2d(hids, torch.int64) + self.NB)[:, None] * self.BS + ar).reshape(-1))
                rows_s, rows_d = rows
                q8[rows_d] = q8[rows_s]
                sc[rows_d] = sc[rows_s]
        self.stats["mirrored_blocks"] += len(gids)

    def _shadow_mirror(self):
        for rid, st in self.reqs.items():
            done = max(0, (st["ncomp"] - self.margin) // self.BS)
            for g in self.groups:
                m0 = st["mirrored"][g]
                n = min(done, len(st["blocks"][g]))
                if n <= m0:
                    continue
                gids = st["blocks"][g][m0:n]
                self._copy_blocks(g, gids, gids)
                st["mirrored"][g] = n
                self.mirrored[g, st["slot"]] = n

    def _run_mamba_saves(self):
        """Copy offered boundary states to host SPARSE_TIER_MAMBA_SAVE_DELAY (2) steps after the offer, at the start of
        a step (before its forward). Align mode writes a checkpoint block during the following step's state
        preprocessing, so the offered step's own post-forward point is too early.
        G-23: the offered blocks are NOT protected by the request (it may finish within the delay, and the V2 runner's
        update_requests zeroes re-handed-out blocks before this hook), so the scheduler holds a reference on them until
        the save id appears in mamba_done (the copy is enqueued; later zeroing is stream-ordered after it)."""
        delay = int(os.environ.get("SPARSE_TIER_MAMBA_SAVE_DELAY", "2"))
        q = getattr(self, "_msave_q", [])
        keep = []
        for age, slot, blocks, sid in q:
            age += 1
            if age < delay:
                keep.append((age, slot, blocks, sid))
                continue
            self._copy_save(slot, blocks, sid)
        self._msave_q = keep

    def _copy_save(self, slot, blocks, sid):
        for g, b in blocks.items():
            for it, hb in zip(self.mamba_items[g], self.mamba_host[g]):
                k = self.mamba_k[id(it)]
                hb[slot].copy_(it[b * k:(b + 1) * k], non_blocking=True)
        self.mamba_done.add(sid)
        self.stats["mamba_saved"] = self.stats.get("mamba_saved", 0) + 1

    def run_saves_now(self, pred) -> None:
        """Run the queued saves matching pred(slot, sid) now, leaving the others' ages alone (G-23: forcing used to
        call _run_mamba_saves, which also aged every other queued save, so one could be copied a step early)."""
        keep = []
        for age, slot, blocks, sid in getattr(self, "_msave_q", []):
            if pred(slot, sid):
                self._copy_save(slot, blocks, sid)
            else:
                keep.append((age, slot, blocks, sid))
        self._msave_q = keep

    def load_mamba_now(self, slot: int, blocks: dict) -> None:
        self.run_saves_now(lambda s, i: s == slot)
        for g, b in blocks.items():
            for it, hb in zip(self.mamba_items[g], self.mamba_host[g]):
                k = self.mamba_k[id(it)]
                it[b * k:(b + 1) * k].copy_(hb[slot], non_blocking=True)
        self.stats["mamba_restored"] = self.stats.get("mamba_restored", 0) + 1

    def _apply_mamba_meta(self, meta):
        for slot, blocks, hit in getattr(meta, "mamba_compares", ()):
            torch.cuda.synchronize()
            res = {}
            for g, b in blocks.items():
                eq = sum(int(torch.equal(it[b].cpu(), hb[slot][0] if hb.dim() > it.dim() - 1 + 1 else hb[slot]))
                         for it, hb in zip(self.mamba_items[g], self.mamba_host[g]))
                res[g] = f"{eq}/{len(self.mamba_items[g])}"
            _log("tier: mamba compare at hit %d: host slot %d vs GPU blocks %s: equal pages %s", hit, slot, blocks, res)
        self._run_mamba_saves()
        self._msave_q = getattr(self, "_msave_q", []) + [(0, s, b, i) for s, b, i in getattr(meta, "mamba_saves", ())]
        loaded: dict[int, int] = {}
        want = {slot for slot, _ in getattr(meta, "mamba_loads", ())}
        if want:  # a restore needs a not-yet-run save: run it now
            self.run_saves_now(lambda s, i: s in want)
        for slot, blocks in getattr(meta, "mamba_loads", ()):
            for g, b in blocks.items():
                for it, hb in zip(self.mamba_items[g], self.mamba_host[g]):
                    k = self.mamba_k[id(it)]
                    it[b * k:(b + 1) * k].copy_(hb[slot], non_blocking=True)
                loaded[b] = g
            self.stats["mamba_restored"] = self.stats.get("mamba_restored", 0) + 1
        # vLLM applied this step's copy-on-write block copies (checkpoint -> running state block) in
        # update_requests, BEFORE this hook; redo the ones whose source we just filled from host.
        so = getattr(meta, "so", None)
        for src, dst in (getattr(so, "kv_cache_block_copies", None) or ()) if loaded else ():
            g = loaded.get(int(src))
            if g is not None:
                for it in self.mamba_items[g]:
                    k = self.mamba_k[id(it)]
                    it[int(dst) * k:(int(dst) + 1) * k].copy_(it[int(src) * k:(int(src) + 1) * k], non_blocking=True)
                self.stats["mamba_cow_redone"] = self.stats.get("mamba_cow_redone", 0) + 1

    def _apply_host_meta(self, meta):
        """SPILL: the scheduler-side HostCache decided host blocks. maps = (logical block, host block) pairs to
        record (host-cache hits need no copy); copies = pairs whose GPU block must be DMA'd now."""
        g = self.groups[0]
        if meta is None:
            return
        for rid, pairs in meta.maps.items():
            st = self.reqs.get(rid)
            if st is None or not pairs:
                continue
            dv = self.h2d([b for b, _ in pairs] + [h for _, h in pairs], torch.int64)
            self.host_bt[st["slot"]].index_copy_(0, dv[: len(pairs)], dv[len(pairs):].to(torch.int32))
            for b, h in pairs:
                st["host"][b] = h
        for rid, pairs in meta.copies.items():
            st = self.reqs.get(rid)
            if st is None or not pairs:
                continue
            gids = [st["blocks"][g][b] for b, _ in pairs]
            assert all(x != 0 for x in gids), f"tier: copy of a null block requested for {rid}: {pairs}"
            self._copy_blocks(g, gids, [h for _, h in pairs])
        for cid, gids, hids in getattr(meta, "fin_copies", ()):  # G-25: finished requests' tail blocks
            self._copy_blocks(g, gids, hids)
            self.fin_done.add(cid)
            self.stats["fin_mirrored_blocks"] = self.stats.get("fin_mirrored_blocks", 0) + len(gids)
        for rid, upto in meta.freed.items():
            st = self.reqs.get(rid)
            if st is not None:
                self.freed_upto[st["slot"]].fill_(upto)  # fill_ with a host scalar: no host sync
                st["freed"] = upto
        self.stats["host_hit_blocks"] = self.stats.get("host_hit_blocks", 0) + meta.nhit

    def layers_group(self, l: int) -> int:
        return self.groups[0] if len(self.groups) == 1 else next(g for (ll, g) in self.layers.values() if ll == l)

    def _new(self, rid, block_ids, ncomp):
        slot = self.free_slots.pop()
        self.hot.reset_slot(slot)
        self.mirrored[:, slot] = 0
        self.host_bt[slot].fill_(-1)
        self.freed_upto[slot] = 0
        self.reqs[rid] = {"slot": slot, "blocks": {g: list(block_ids[g]) for g in self.groups}, "ncomp": ncomp,
                          "nsched": 0, "mirrored": {g: 0 for g in self.groups}, "host": {}}

    def _drop(self, rid):
        st = self.reqs.pop(rid, None)
        if st is not None:
            self.free_slots.append(st["slot"])

    # ---------------------------------------------------------------- per layer (inside the forward / graphs)
    def _spilled(self, group, block_table, seq_lens, slots):
        """Blocks to read from host: host-mapped and (null in the GPU table, or outside sink + tail)."""
        b = self.barange[: block_table.shape[1]][None, :]
        old = (b >= self.sink_blocks) & ((b + 1) * self.BS <= (seq_lens[:, None] - self.tail))
        if not self.spill:
            mir = self.mirrored[group].index_select(0, slots)
            return old & (b < mir[:, None])
        hmap = self._host_bt(block_table, slots) >= 0
        freed = (b >= self.sink_blocks) & (b < self.freed_upto.index_select(0, slots)[:, None])
        return hmap & ((block_table == 0) | freed)

    def _host_bt(self, block_table, slots):
        if not self.spill:
            return block_table
        return self.host_bt.index_select(0, slots)[:, : block_table.shape[1]].contiguous()

    def sketch_block_table(self, layer_name, block_table, seq_lens):
        """Block table for sparsekv.select: spilled blocks read the host-indexed sketch region."""
        lg = self.layers.get(layer_name)
        if lg is None or not self.spill:
            return block_table
        slots = self.batch_slots[: block_table.shape[0]]
        sp = self._spilled(lg[1], block_table, seq_lens, slots)
        return torch.where(sp, self._host_bt(block_table, slots) + self.NB, block_table).contiguous()

    def select_reuse_args(self, nreq: int, store: torch.Tensor) -> dict:
        """Extra kwargs for triton_ops.select in the decode path (empty unless SPARSE_SELECT_EVERY > 1)."""
        if self.select_every <= 1:
            return {}
        return {"reuse": self.batch_reuse[:nreq], "topk_store": store, "slots": self.batch_slots[:nreq]}

    def gather(self, slot_l: int, group: int, kv, block_table, seq_lens, idx, n_valid, st_kv, st_bt,
               ref_gather=None):
        nreq = idx.shape[0]
        slots = self.batch_slots[:nreq]
        bt = block_table[:nreq]
        sp = self._spilled(group, bt, seq_lens, slots)
        gbt = torch.where(sp, torch.full_like(bt, -1), bt).contiguous()
        gather_rows_tiered(kv, gbt, self.mirror.dev[slot_l], self._host_bt(bt, slots), idx, n_valid, st_kv, st_bt,
                           self.hot, slot_l, slots, self.scratch)
        if self.verify and ref_gather is not None and not self.spill:
            # SPARSE_TIER_VERIFY=1 (shadow mode only): stage again with the plain GPU gather and count differing
            # bytes in the valid columns -- must stay 0
            if self.vbuf is None or self.vbuf.shape != st_kv.shape:
                self.vbuf = torch.zeros_like(st_kv)
            ref_gather(kv, bt, idx, n_valid, self.vbuf, st_bt)
            C = idx.shape[1]
            j = torch.arange(C, device=idx.device)
            blk = st_bt.long()[:, j // self.BS]
            off = (j % self.BS)[None, :].expand(nreq, C)
            a = st_kv[blk, :, off]
            b = self.vbuf[blk, :, off]
            valid = (j[None, :] < n_valid[:, None])[:, :, None, None]
            self.vcount[0] += ((a != b) & valid).sum()
            self.vcount[1] += valid.sum() * a.shape[2] * a.shape[3]

    def window_part(self, layer_name, part, req_start: int):
        """Group layers without sparse decode (MTP drafter): drop null blocks (host-cache hits) and spilled blocks,
        keep the rest in order (exact attention over the kept blocks; keys carry RoPE). Graph-safe."""
        if not self.spill or layer_name not in self.window_layers:
            return part
        sl = part.seq_lens
        n = sl.shape[0]
        bt = part.block_table[:n]
        slots = self.batch_slots[req_start: req_start + n]
        W = bt.shape[1]
        b = self.barange[:W][None, :]
        full = (b + 1) * self.BS <= sl[:, None]  # only full blocks may be dropped
        freed = (b >= self.sink_blocks) & (b < self.freed_upto.index_select(0, slots)[:, None])
        drop = full & ((bt == 0) | freed)
        key = drop.to(torch.int32) * W + b  # kept blocks first, original order
        order = torch.argsort(key, dim=1)
        cbt = torch.gather(bt, 1, order).contiguous()
        ndrop = drop.sum(dim=1).to(sl.dtype)
        return dataclasses.replace(part, block_table=cbt, seq_lens=sl - ndrop * self.BS)

    def stage_prefill(self, layer_name, kv, part, req_start: int):
        """Dense prefill for a sparse layer whose request has spilled blocks: build a scratch cache with every
        block of the prefill requests (spilled ones from host, the rest from the GPU cache) and return
        (scratch_kv, patched_part), or None when nothing in this slice is spilled. Eager only (prefill slices are
        never in FULL graphs)."""
        lg = self.layers.get(layer_name)
        if lg is None or not self.spill:
            return None
        l, g = lg
        n = part.seq_lens.shape[0]
        plan = []
        cpu_bt = part.block_table[:n].cpu()  # eager prefill path: a sync here is acceptable
        for r in range(n):
            bi = req_start + r
            rid = self.batch_rows[bi] if bi < len(self.batch_rows) else None
            st = self.reqs.get(rid) if rid is not None else None
            if st is None:
                plan.append(None)
                continue
            seq = st["ncomp"] + st["nsched"]
            nblk = -(-seq // self.BS)
            fu = st.get("freed", 0)
            spl = {b for b in st["host"] if b < nblk and (
                int(cpu_bt[r, b]) == 0 or self.sink_blocks <= b < fu)}
            plan.append((st, nblk, spl))
        if not any(p is not None and p[2] for p in plan):
            return None
        need = sum(p[1] for p in plan if p is not None) + 1
        if self.pf_kv is None or self.pf_kv.shape[0] < need:
            _log("tier: WARNING prefill staging grows to %d blocks (%.0f MB) outside the memory reserve", need,
                 need * kv[0].numel() / 1e6)
            self.pf_kv = torch.empty((max(need, self.maxblk + 1), *kv.shape[1:]), dtype=kv.dtype, device=kv.device)
        bt = part.block_table[:n].clone()
        nxt = 1
        h4d, h4s = [], []
        for r, p in enumerate(plan):
            if p is None:
                continue
            st, nblk, spl = p
            res_src, res_dst = [], []
            for b in range(nblk):
                d = nxt + b
                if b in spl:
                    if self.host4:
                        h4d.append(d); h4s.append(st["host"][b])
                    else:
                        self.pf_kv[d].copy_(self.mirror.host[l, st["host"][b]], non_blocking=True)
                    self.stats["prefill_staged_blocks"] += 1
                else:
                    res_src.append(int(cpu_bt[r, b])); res_dst.append(d)
            if res_src:
                self.pf_kv[torch.tensor(res_dst, device=kv.device)] = kv[torch.tensor(res_src, device=kv.device)]
            bt[r, :nblk] = torch.arange(nxt, nxt + nblk, dtype=bt.dtype, device=bt.device)
            nxt += nblk
        if h4d:
            copy_blocks_from_host4(self.pf_kv, torch.tensor(h4d, dtype=torch.int32, device=kv.device), self.mirror.dev[l],
                                   torch.tensor(h4s, dtype=torch.int32, device=kv.device))
        return self.pf_kv, dataclasses.replace(part, block_table=bt)

    # ---------------------------------------------------------------- S4: prefill of sparse layers (eager path)
    def _reload_sfx(self):
        if not self._sfx_file:
            return
        try:
            mt = os.stat(self._sfx_file).st_mtime
        except OSError:
            return
        if mt == self._sfx_mtime:
            return
        self._sfx_mtime = mt
        try:
            import json
            new = dict(suffix_cfg(), **json.load(open(self._sfx_file)))
        except Exception as e:  # noqa: BLE001
            _log("tier: suffix cfg %s unreadable (%s); keeping %s", self._sfx_file, e, self.sfx)
            return
        self.sfx = new
        self.suffix_sparse = self.spill and bool(new["sparse"])
        _log("tier: suffix cfg reloaded from %s: %s", self._sfx_file, new)

    def prefill_plan(self, part, req_start: int):
        """Per-step host plan of the prefill slice (cached: the 16 sparse layers share it). For each request:
        'plain' (no spilled block: normal dense attention over the GPU cache), 'sparse' (follow-up suffix: select +
        tiered gather of the picked rows only) or 'dense' (long chunk: every block staged, batched, no host sync).
        Built from the scheduler outputs the tier already tracks (block lists, host map, freed boundary), so no
        device->host copy. Returns None when the slice cannot be planned (caller falls back to stage_prefill)."""
        key = (self.stats["steps"], req_start, part.token_start, part.token_end)
        if self._pp_key == key:
            return self._pp
        self._pp_key, self._pp = key, None
        if not self.spill:
            return None
        self._reload_sfx()
        sfx = self.sfx
        g = self.groups[0]
        n = part.seq_lens.shape[0]
        plan, qoff = [], 0
        for r in range(n):
            bi = req_start + r
            rid = self.batch_rows[bi] if bi < len(self.batch_rows) else None
            st = self.reqs.get(rid) if rid is not None else None
            if st is None:
                return None
            ql = int(st["nsched"])
            L = int(st["ncomp"]) + ql
            nblk = -(-L // self.BS)
            blocks = st["blocks"][g]
            if len(blocks) < nblk or ql <= 0:
                return None
            fu = st.get("freed", 0)
            spl = [b for b in sorted(st["host"]) if b < nblk and (blocks[b] == 0 or self.sink_blocks <= b < fu)]
            if not spl:
                mode = "plain"
            elif (self.suffix_sparse and ql <= sfx["max"] and L - ql > self.C
                  and st.get("uncached", 1 << 30) <= sfx["max"]):
                mode = "sparse"
            else:
                mode = "dense"
            plan.append({"r": r, "rid": rid, "st": st, "ql": ql, "L": L, "nblk": nblk, "spl": spl, "qoff": qoff,
                         "mode": mode, "segs": suffix_segments(ql, sfx) if mode == "sparse" else [ql]})
            qoff += ql
        if qoff != part.token_end - part.token_start:
            if not getattr(self, "_pp_warned", False):
                self._pp_warned = True
                _log("tier: prefill plan mismatch (%d planned vs %d slice tokens): legacy staging", qoff,
                     part.token_end - part.token_start)
            return None
        if all(p["mode"] == "plain" for p in plan):
            self._pp = plan
            return plan
        # one upload for the step: per request [0, ql] (sub-launch query_start_loc) and L; dense staging lists
        small, hdst, hsrc, gdst, gsrc = [], [], [], [], []
        nxt = 1
        for p in plan:
            # per request: segment query_start_loc [0, s1, s1+s2, ..., ql], then each segment's causal end
            p["so"] = len(small)
            acc, ends, tot = [0], [], 0
            for n in p["segs"]:
                tot += n
                acc.append(tot)
                ends.append(p["L"] - p["ql"] + tot)
            p["seg_ends"] = ends
            small += acc + ends
            if p["mode"] == "dense":
                spl = set(p["spl"])
                blocks = p["st"]["blocks"][g]
                for b in range(p["nblk"]):
                    if b in spl:
                        hdst.append(nxt + b); hsrc.append(p["st"]["host"][b])
                    else:
                        gdst.append(nxt + b); gsrc.append(blocks[b])
                p["pf0"] = nxt
                nxt += p["nblk"]
        need = max(nxt, max((len(p["segs"]) * -(-(self.C - self.ksel + sfx["ksel"]) // self.BS)
                             for p in plan if p["mode"] == "sparse"), default=0))
        if need > (self.pf_kv.shape[0] if self.pf_kv is not None else 0):
            nxt = need
            _log("tier: WARNING prefill staging grows to %d blocks outside the memory reserve", nxt)
            any_kv = self.kv_by_slot[0]
            self.pf_kv = torch.empty((max(nxt, self.maxblk + 1), *any_kv.shape[1:]), dtype=any_kv.dtype,
                                     device=self.dev)
        nh, ng = len(hdst), len(gdst)
        dv = self.h2d(small + hdst + hsrc + gdst + gsrc, torch.int32)
        o = len(small)
        self._pp_h = (dv[o:o + nh], dv[o + nh:o + 2 * nh])
        self._pp_g = (dv[o + 2 * nh:o + 2 * nh + ng], dv[o + 2 * nh + ng:o + 2 * nh + 2 * ng])
        W = part.block_table.shape[1]
        for p in plan:
            ns = len(p["segs"])
            p["qsl_d"] = dv[p["so"]:p["so"] + ns + 1]
            p["L_d"] = dv[p["so"] + ns + 1:p["so"] + 2 * ns + 1]
            if p["mode"] == "dense":
                bt = torch.zeros((1, W), dtype=part.block_table.dtype, device=self.dev)
                bt[0, :p["nblk"]] = torch.arange(p["pf0"], p["pf0"] + p["nblk"], dtype=bt.dtype, device=self.dev)
                p["bt_d"] = bt
        self.stats["prefill_staged_blocks"] += nh * self.L
        ns = [p for p in plan if p["mode"] == "sparse"]
        self.stats["suffix_sparse"] += len(ns)
        self.stats["suffix_tokens"] += sum(p["ql"] for p in ns)
        if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1":
            _log("tier: prefill plan step %d: %s", self.stats["steps"],
                 [(p["rid"][-8:], p["mode"], p["ql"], p["L"], len(p["spl"])) for p in plan])
        self._pp = plan
        return plan

    def _sub_plan(self, impl, ql: int, nreq: int = 1, maxq: int | None = None):
        maxq = ql if maxq is None else maxq
        key = (ql, nreq, maxq, getattr(impl, "window", 0) or 0)
        plan = self._sub_plans.get(key)
        if plan is None:
            from pth_flash_attn.planner import plan_launch
            plan = plan_launch(num_tokens=ql, num_reqs=nreq, max_query_len=maxq, G=impl.num_heads // impl.num_kv_heads,
                               Hkv=impl.num_kv_heads, D=impl.head_size,
                               num_sms=torch.cuda.get_device_properties(self.dev).multi_processor_count,
                               is_prefill=True, ws_token_splits=int(getattr(impl, "_ws_token_splits", 0) or 0),
                               window=key[3])
            self._sub_plans[key] = plan
        return plan

    def prefill_layer(self, impl, layer_name, sp, kv, part, req_start, query, output, caches, causal, views) -> bool:
        """S4 prefill of one sparse layer. Returns False if the caller should use the legacy path (stage_prefill);
        True when the prefill slice's attention has been written to `output`."""
        lg = self.layers.get(layer_name)
        if lg is None or not self.spill:
            return False
        plan = self.prefill_plan(part, req_start)
        if plan is None:
            return False
        if all(p["mode"] == "plain" for p in plan):
            impl._run_slice(part, query, output, caches, causal)
            return True
        l, g = lg
        # sparse suffixes first: they stage into pf_kv[0:...], which the dense chunks' copy then overwrites (stream order)
        order = [p for p in plan if p["mode"] == "sparse"] + [p for p in plan if p["mode"] != "sparse"]
        copied = False
        for p in order:
            if p["mode"] == "dense" and not copied:
                (copy_blocks_from_host4 if self.host4 else copy_blocks)(self.pf_kv, self._pp_h[0], self.mirror.dev[l],
                                                                        self._pp_h[1])
                copy_blocks(self.pf_kv, self._pp_g[0], kv, self._pp_g[1])
                copied = True
            r, ql = p["r"], p["ql"]
            t0 = part.token_start + p["qoff"]
            ns, mq = len(p["segs"]), max(p["segs"])
            sub = dict(plan=self._sub_plan(impl, ql, ns, mq), token_start=t0, token_end=t0 + ql, max_query_len=mq,
                       query_start_loc=p["qsl_d"])
            if p["mode"] == "sparse":
                st_kv, st_bt, nv = self._suffix_stage(sp, l, g, kv, part, req_start, p, query[t0:t0 + ql], impl.scale)
                impl._run_slice(dataclasses.replace(part, seq_lens=nv, block_table=st_bt, **sub), query, output,
                                views(impl, st_kv, attr="_tier_pf_views"), causal)
            elif p["mode"] == "dense":
                impl._run_slice(dataclasses.replace(part, seq_lens=p["L_d"], block_table=p["bt_d"], **sub), query,
                                output, views(impl, self.pf_kv, attr="_tier_pf_views"), causal)
            else:
                impl._run_slice(dataclasses.replace(part, seq_lens=part.seq_lens[r:r + 1],
                                                    block_table=part.block_table[r:r + 1], **sub),
                                query, output, caches, causal)
        return True

    def _suffix_stage(self, sp, l, g, kv, part, req_start, p, q, scale):
        """Select the suffix's rows over the sketch (spilled blocks read the host-indexed sketch region) and gather only
        those rows: resident from the GPU cache, the rest from pinned host RAM (UVA, zero-copy). The padding slot is
        used for the hot buffer so a suffix neither evicts nor fills decode rows."""
        from sparsekv import triton_ops
        r, ns = p["r"], len(p["segs"])
        bt = part.block_table[r:r + 1].expand(ns, -1).contiguous()  # every segment reads the request's blocks
        slots = self.batch_slots[req_start + r:req_start + r + 1].expand(ns).contiguous()
        spm = self._spilled(g, bt, p["L_d"], slots)
        hbt = self._host_bt(bt, slots)
        sk_bt = torch.where(spm, hbt + self.NB, bt).contiguous()
        ks = int(self.sfx["ksel"])
        Cs = ks + self.sink + self.win
        idx, nv = triton_ops.select_suffix(q, p["segs"], p["seg_ends"], p["L_d"], sk_bt, sp.sk_q8, sp.sk_scale,
                                           sp.basis_l, ks, self.sink, self.win, scale, h2d=self.h2d)
        gbt = torch.where(spm, torch.full_like(bt, -1), bt).contiguous()
        nbs = -(-Cs // self.BS)
        if self.sf_scratch is None or self.sf_scratch.code.shape[0] < ns or self.sf_scratch.code.shape[1] != Cs:
            self.sf_scratch = TierScratch(max(ns, 16), Cs, self.dev)  # eager path: grows outside graphs
            self.pad_slots = torch.full((max(ns, 16),), self.max_reqs, dtype=torch.int32, device=self.dev)
            self._sf_bt = torch.arange(max(ns, 16) * nbs, dtype=torch.int32, device=self.dev).view(-1, nbs)
        st_bt = self._sf_bt[:ns]
        gather_rows_tiered(kv, gbt, self.mirror.dev[l], hbt, idx, nv, self.pf_kv, st_bt, self.hot, l,
                           self.pad_slots[:ns], self.sf_scratch, stats=self.sf_stats)
        return self.pf_kv, st_bt, nv

    def group_of(self, layer_name: str) -> tuple[int, int] | None:
        return self.layers.get(layer_name)


HOST_HEADROOM_GB = 8.0


def host_pool_blocks(max_len: int, max_reqs: int, bs: int, block_bytes: int, check_ram: bool = True) -> int:
    """Host KV pool size in blocks (scheduler and worker must agree).

    Knobs (first one set wins):
      SPARSE_TIER_HOST_GB      pinned host RAM for the KV mirror, in GB (all 16 sparse layers)
      SPARSE_TIER_HOST_TOKENS  capacity in tokens
    Default: max_model_len x max_num_seqs tokens (every running request can be fully spilled), capped at
    SPARSE_TIER_HOST_GB_CAP (16 GB). The result is refused if it would leave less than HOST_HEADROOM_GB of
    MemAvailable (the VM's two servers share 78 GB).
    """
    if os.environ.get("SPARSE_TIER_HOST_GB"):
        nb = int(float(os.environ["SPARSE_TIER_HOST_GB"]) * 1e9) // block_bytes
    elif os.environ.get("SPARSE_TIER_HOST_TOKENS"):
        nb = -(-int(os.environ["SPARSE_TIER_HOST_TOKENS"]) // bs)
    else:
        nb = -(-(max_len * max_reqs) // bs)
        cap = int(float(os.environ.get("SPARSE_TIER_HOST_GB_CAP", "16")) * 1e9) // block_bytes
        nb = min(nb, cap)
    nb += max_reqs
    if not check_ram:  # the worker (which allocates) checks; the scheduler is created after that allocation
        return nb
    try:
        avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
        if nb * block_bytes > avail - HOST_HEADROOM_GB * 1e9:
            raise MemoryError(f"tier: host KV pool {nb * block_bytes / 1e9:.1f} GB would leave less than "
                              f"{HOST_HEADROOM_GB} GB of MemAvailable ({avail / 1e9:.1f} GB); lower SPARSE_TIER_HOST_GB")
    except StopIteration:
        pass
    return nb


def host_mamba_slots() -> int:
    return int(os.environ.get("SPARSE_TIER_HOST_MAMBA_SLOTS", "8"))


def single_process_engine(vllm_config) -> bool:
    """G-21: host GDN restore is only race-free when the scheduler and the model runner share a process and a CUDA
    stream (UniProc executor): the host->GPU load is then enqueued at scheduling time, before any later zeroing of
    the block. The multi-process fallback (load applied in start_step, after update_requests) is refused."""
    pc = getattr(vllm_config, "parallel_config", None)
    backend = getattr(pc, "distributed_executor_backend", None)
    world = int(getattr(pc, "world_size", 1) or 1)
    return backend == "uni" and world == 1


def sparse_layer_ids() -> list[int]:
    bp = os.environ.get("SPARSE_BASIS", os.path.join(sys.prefix, "share", "sparsekv", "basis27.pt"))
    if os.path.exists(bp):
        return [int(x) for x in torch.load(bp, map_location="cpu")["layers"]]
    return list(range(3, 64, 4))


# ---------------------------------------------------------------------------------------------------- connector
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata,
                                                                   KVConnectorRole, SupportsHMA)
except Exception:  # noqa: BLE001  (importable without vLLM for docs/tests)
    KVConnectorBase_V1 = object
    KVConnectorMetadata = object
    KVConnectorRole = None

    class SupportsHMA:  # noqa: D101
        pass


class HostTierMeta(KVConnectorMetadata):
    def __init__(self, so=None):
        self.so = so
        self.maps: dict[str, list[tuple[int, int]]] = {}  # req -> (logical block, host block), incl. host hits
        self.copies: dict[str, list[tuple[int, int]]] = {}  # req -> subset of maps whose GPU block must be copied
        self.nhit = 0
        self.freed: dict[str, int] = {}  # req -> scheduler freed blocks [SINK_BLOCKS, upto)
        self.mamba_saves: list[tuple[int, dict[int, int], int]] = []  # (host slot, {group: GPU block}, save id)
        self.mamba_loads: list[tuple[int, dict[int, int]]] = []  # (host slot, {mamba group: GPU block}) host->GPU
        # G-25: (copy id, [GPU block], [host block]) of finished requests' unmirrored tail blocks; the GPU blocks are
        # held by the scheduler until the worker reports the copy id in TierRuntime.fin_done
        self.fin_copies: list[tuple[int, list[int], list[int]]] = []


class HostCache:
    """Scheduler-side host block pool with a prefix cache: block hash -> host block (refcounted, LRU eviction of
    unreferenced cached blocks). A host block's content is complete before any step that can read it, because its
    copy is enqueued on the compute stream before the forward of the step that publishes the mapping."""

    def __init__(self, nblocks: int):
        self.n = nblocks
        self.free = list(range(nblocks - 1, -1, -1))
        self.h2b: dict = {}
        self.b2h: dict = {}
        self.ref = [0] * nblocks
        self.lru: "collections.OrderedDict[int, bool]" = collections.OrderedDict()
        self.stats = {"alloc": 0, "hit": 0, "evict": 0}

    def lookup(self, h):
        return None if h is None else self.h2b.get(h)

    def acquire(self, hb: int) -> None:
        if self.ref[hb] == 0:
            self.lru.pop(hb, None)
        self.ref[hb] += 1
        self.stats["hit"] += 1

    def alloc(self, h) -> int:
        if self.free:
            hb = self.free.pop()
        elif self.lru:
            hb, _ = self.lru.popitem(last=False)
            old = self.b2h.pop(hb, None)
            if old is not None and self.h2b.get(old) == hb:
                del self.h2b[old]
            self.stats["evict"] += 1
        else:
            raise MemoryError("tier: host KV pool exhausted (all host blocks referenced); "
                              "raise SPARSE_TIER_HOST_TOKENS")
        self.ref[hb] = 1
        if h is not None and h not in self.h2b:
            self.h2b[h] = hb
            self.b2h[hb] = h
        self.stats["alloc"] += 1
        return hb

    def release(self, hb: int) -> None:
        self.ref[hb] -= 1
        if self.ref[hb] == 0:
            if hb in self.b2h:
                self.lru[hb] = True
            else:
                self.free.append(hb)


_HOST_CACHE: "HostCache | None" = None


class MambaHostStore:
    """Scheduler-side store of GDN/mamba checkpoint states in host RAM, keyed by the raw block hash of the
    boundary block (one slot holds the page of every mamba group). LRU over SPARSE_TIER_HOST_MAMBA_SLOTS slots."""

    def __init__(self, nslots: int):
        self.n = nslots
        self.h2s: "collections.OrderedDict" = collections.OrderedDict()
        self.free = list(range(nslots - 1, -1, -1))
        self.pending_saves: list = []
        self.next_sid = 0
        self.pending_loads: list = []
        self.pending_compares: list = []
        self.stats = {"saved": 0, "restored": 0, "evict": 0, "pinned": 0, "unpinned": 0, "flushed_at_finish": 0}

    def lookup(self, h):
        s = self.h2s.get(h)
        if s is not None:
            self.h2s.move_to_end(h)
        return s

    def save(self, h, blocks: dict) -> int | None:
        """Register h -> slot and queue the GPU->host copy; returns its save id (None if h is already stored)."""
        if h in self.h2s:
            self.h2s.move_to_end(h)
            return None
        if self.free:
            slot = self.free.pop()
        else:
            _, slot = self.h2s.popitem(last=False)
            self.stats["evict"] += 1
        self.h2s[h] = slot
        sid = self.next_sid
        self.next_sid += 1
        self.pending_saves.append((slot, dict(blocks), sid))
        self.stats["saved"] += 1
        return sid


_MAMBA_HOST: "MambaHostStore | None" = None


def make_mamba_hit_manager_cls(base):
    """Mamba manager whose prefix lookup restores a host-saved checkpoint when the GPU copy was evicted: it takes
    free GPU blocks (one per mamba group), registers them under the boundary hash, queues a host->GPU load (done by
    the worker before the next forward) and returns them as an ordinary cached hit."""

    class HostTierMambaManager(base):
        @classmethod
        def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                   drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
            args = (block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec, drop_eagle_block,
                    alignment_tokens, dcp_world_size, pcp_world_size)
            computed, hit = super().find_longest_cache_hit(*args)
            ms = _MAMBA_HOST
            bs = kv_cache_spec.block_size
            if (ms is not None and hit > 0 and os.environ.get("SPARSE_TIER_DEBUG_MAMBA_CMP", "0") == "1"
                    and alignment_tokens == bs):
                h = block_hashes[hit // bs - 1]
                slot = ms.h2s.get(h)
                if slot is not None:  # GPU hit that is also host-saved: ask the worker to compare contents
                    ms.pending_compares.append((slot, {gid: c[-1].block_id for gid, c in
                                                       zip(kv_cache_group_ids, computed)}, hit))
            if ms is None or not ms.h2s or alignment_tokens != bs or block_pool.hash_block_size != bs:
                return computed, hit
            n = min(max_length // bs, len(block_hashes))
            for i in range(n - 1, hit // bs - 1, -1):
                h = block_hashes[i]
                slot = ms.lookup(h)
                if slot is None:
                    continue
                if _RT is None or not _RT.mamba_items:
                    # G-21: no same-process worker -> the load could only be applied after update_requests zeroes
                    # / CoW-copies the block; refuse the restore instead of risking a corrupted GDN state.
                    if not getattr(cls, "_g21_warned", False):
                        cls._g21_warned = True
                        _log("tier: host GDN restore REFUSED (G-21): no same-process tier runtime (multi-process "
                             "executor?); falling back to recompute")
                    break
                if block_pool.get_num_free_blocks() < len(kv_cache_group_ids) + int(
                        os.environ.get("SPARSE_TIER_MAMBA_RESTORE_MARGIN", "2")):
                    ms.stats["refused_nofree"] = ms.stats.get("refused_nofree", 0) + 1
                    _log("tier: host GDN restore deferred: %d free GPU blocks (boundary %d, GPU hit %d)",
                         block_pool.get_num_free_blocks(), (i + 1) * bs, hit)
                    break
                blks = block_pool.get_new_blocks(len(kv_cache_group_ids))
                for gid, blk in zip(kv_cache_group_ids, blks):
                    block_pool._insert_block_hash(make_block_hash_with_group_id(h, gid), blk,
                                                  num_tokens=(i + 1) * bs)
                target = {gid: blk.block_id for gid, blk in zip(kv_cache_group_ids, blks)}
                # Same process (single-GPU engine): enqueue the host->GPU copy now. vLLM's state preprocessing
                # reads the hit checkpoint before the connector's pre_forward, so a load there is too late;
                # this copy is ordered before the next step on the stream.
                _RT.load_mamba_now(slot, target)
                block_pool.free_blocks(blks)  # hashed, ref 0: cached and evicted last
                ms.stats["restored"] += 1
                c2, h2 = super().find_longest_cache_hit(*args)
                if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1":
                    _log("tier: mamba lookup max %d: GPU hit %d -> host restore at %d -> hit %d", max_length, hit,
                         (i + 1) * bs, h2)
                return c2, h2
            if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1" and max_length >= bs:
                _log("tier: mamba lookup max %d: GPU hit %d, no host checkpoint in (%d, %d] (%d stored)", max_length,
                     hit, hit, n * bs, len(ms.h2s))
            return computed, hit

    HostTierMambaManager.__name__ = "HostTier" + base.__name__
    return HostTierMambaManager


def make_hit_manager_cls(base):
    """Full-attention manager whose prefix-cache lookup also accepts host-cached blocks (returned as the null block,
    like sliding-window hits): the worker reads them from host RAM."""

    class HostTierHitManager(base):
        @classmethod
        def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                   drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
            hc = _HOST_CACHE
            bs = kv_cache_spec.block_size
            if (hc is None or alignment_tokens != bs or dcp_world_size != 1 or pcp_world_size != 1
                    or block_pool.hash_block_size != bs):
                if hc is not None and not getattr(cls, "_warned", False):
                    cls._warned = True
                    _log("tier: host prefix hits DISABLED for this lookup (alignment %d, block %d, hash block %d)",
                         alignment_tokens, bs, block_pool.hash_block_size)
                return super().find_longest_cache_hit(block_hashes, max_length, kv_cache_group_ids, block_pool,
                                                      kv_cache_spec, drop_eagle_block, alignment_tokens,
                                                      dcp_world_size, pcp_world_size)
            computed = tuple([] for _ in kv_cache_group_ids)
            host_only = os.environ.get("SPARSE_TIER_DEBUG_FA_HOST_ONLY", "0") == "1"  # test: ignore GPU-cached hits
            for h in itertools.islice(block_hashes, max_length // bs):
                cached = None if (host_only and h in hc.h2b) else block_pool.get_cached_block(h, kv_cache_group_ids)
                if cached:
                    for c, x in zip(computed, cached):
                        c.append(x)
                elif h in hc.h2b:
                    for c in computed:
                        c.append(block_pool.null_block)
                else:
                    break
            # S4: prefer the host copy for the middle of the prefix (sink block .. tail window). A returning request
            # otherwise re-acquires every GPU-cached block of its document (its cold prefill left ~60 at 250K) plus
            # its GDN pages, and the 4th of 4 x 250K stayed Waiting at ~50 % pool use (follow-up TTFT 18 s). Decode and
            # the sparse suffix read null (host) blocks anyway; only sink + tail stay GPU-resident.
            if os.environ.get("SPARSE_TIER_PREFER_HOST", "1") == "1" and computed[0]:
                sink_b = int(os.environ.get("SPARSE_TIER_SINK_BLOCKS", "1"))
                tail_tok = int(os.environ.get("SPARSE_TIER_TAIL", str(int(os.environ.get("SPARSE_WIN", "1024")) + 64)))
                last = len(computed[0]) - (-(-tail_tok // bs) + 1)
                hashes = list(itertools.islice(block_hashes, len(computed[0])))
                for i in range(sink_b, max(sink_b, last)):
                    if computed[0][i] is not block_pool.null_block and hashes[i] in hc.h2b:
                        for c in computed:
                            c[i] = block_pool.null_block
                        cls._preferred = getattr(cls, "_preferred", 0) + 1
            hit = len(computed[0]) * bs
            if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1" and max_length >= bs:
                nh = sum(1 for x in computed[0] if x is block_pool.null_block)
                _log("tier: FA lookup max %d: hit %d blocks (%d host, %d GPU) of %d", max_length, len(computed[0]), nh,
                     len(computed[0]) - nh, max_length // bs)
            if drop_eagle_block and hit > 0:
                hit -= bs
            nb = -(-hit // bs)
            for c in computed:
                del c[nb:]
            return computed, hit

    HostTierHitManager.__name__ = "HostTier" + base.__name__
    return HostTierHitManager


class HostTierSpillPolicy:
    """Scheduler-side spill policy for the KV group of the sparse layers. Frees GPU blocks
    [SINK_BLOCKS, (processed - TAIL) // BS) -- every such block already has a host copy (HostTierConnector._plan_req)
    -- either always ("eager") or while free GPU blocks are below the watermark ("pressure", default), and records
    the freed boundary for the worker.

    vLLM 0.31's manager classes refuse ``__class__`` reassignment (their layout is fixed), so the policy is bound to
    the live manager instance as its ``remove_skipped_blocks`` (the class method stays the fallback chain)."""

    def __init__(self, conn, tail: int, sink_blocks: int, policy: str, watermark: int):
        self.conn, self.tail, self.sink_blocks, self.policy, self.watermark = conn, tail, sink_blocks, policy, watermark

    def install(self, mgr) -> None:
        base_rsb = type(mgr).remove_skipped_blocks
        pol = self

        def remove_skipped_blocks(request_id, processed_computed_tokens, num_prompt_tokens=None):
            base_rsb(mgr, request_id, processed_computed_tokens, num_prompt_tokens)
            pol.maybe_spill(mgr, request_id, processed_computed_tokens)

        mgr.remove_skipped_blocks = remove_skipped_blocks
        mgr.tier_spill_policy = self
        orig_pop = mgr.pop_blocks_for_free
        conn = self.conn

        def pop_blocks_for_free(request_id):
            conn.safe.pop(request_id, None)  # G-16: finish or preemption: the next incarnation starts unmirrored
            return orig_pop(request_id)

        mgr.pop_blocks_for_free = pop_blocks_for_free  # mgr.free() goes through it too

    def maybe_spill(self, mgr, request_id: str, processed_computed_tokens: int) -> None:
        conn = self.conn
        # G-16: never free a block whose host copy was not enqueued in an earlier step. A block freed and handed out
        # again in this same scheduling pass is zeroed (needs_kv_cache_zeroing, hybrid GDN model) / CoW'd by the
        # worker's update_requests BEFORE start_load_kv would copy it -> zeros on the host.
        last = min((processed_computed_tokens - self.tail) // mgr.block_size, conn.safe.get(request_id, 0))
        if last <= self.sink_blocks or last <= conn.freed.get(request_id, 0):
            return
        if self.policy == "pressure" and mgr.block_pool.get_num_free_blocks() >= self.watermark:
            return
        self.free_range(mgr, request_id, self.sink_blocks, last)
        conn.freed[request_id] = last
        conn.freed_dirty.add(request_id)

    @staticmethod
    def free_range(mgr, request_id: str, first: int, last: int) -> None:
        """G-18b: like vLLM's _remove_blocks_in_range, but skip null blocks (host prefix hits) instead of stopping at
        the first one, so GPU blocks below a host-hit block are freed too."""
        blocks = mgr.req_to_blocks.get(request_id)
        if not blocks:
            return
        null = mgr._null_block
        out = []
        for i in range(min(last, len(blocks)) - 1, first - 1, -1):
            if blocks[i] is null or blocks[i] == null:
                continue
            out.append(blocks[i])
            blocks[i] = null
        if out:
            mgr.block_pool.free_blocks(out)


class HostTierConnector(KVConnectorBase_V1, SupportsHMA):
    """Scheduler side: host block pool + prefix cache (SPILL), prefix-hit hook, spill hook; forces a pre-forward
    hook and ships the scheduler output + host mappings to the worker. Worker side: TierRuntime.
    SupportsHMA keeps vLLM's hybrid KV cache manager on (required for the GDN hybrid)."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._kvc = kv_cache_config
        self.spill = enabled() and os.environ.get("SPARSE_TIER_SPILL", "0") == "1"
        self.hc = None
        self.req_obj: dict = {}
        self.sreq: dict[str, dict] = {}
        self.margin = int(os.environ.get("SPARSE_TIER_MARGIN", "64"))
        self.freed: dict[str, int] = {}
        self.freed_dirty: set[str] = set()
        # G-16: per request, blocks [0, safe) whose host copy was planned in an EARLIER step's connector meta (so the
        # worker enqueued it before this step's update_requests can zero a re-handed-out block). Reset when the
        # request's GPU blocks are popped (finish / preemption).
        self.safe: dict[str, int] = {}
        self.hash_k = 1  # G-18a: request block hashes per 864-token block (set in bind_kv_cache_manager)
        self.resumed = 0
        self.preempted = 0
        if self.spill and role == KVConnectorRole.SCHEDULER:
            global _HOST_CACHE
            g = self._sparse_group()
            self.bs = self._kvc.kv_cache_groups[g].kv_cache_spec.block_size
            max_len = int(vllm_config.model_config.max_model_len)
            max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
            spec = self._kvc.kv_cache_groups[g].kv_cache_spec
            nsparse = len(sparse_layer_ids())
            row = spec.num_kv_heads * host_row_width(2 * (spec.head_size + 4))  # host bytes per token (2080 / 1088 B)
            self.hc = HostCache(host_pool_blocks(max_len, max_reqs, self.bs, row * self.bs * nsparse,
                                                 check_ram=False))  # == TierRuntime.HB
            _HOST_CACHE = self.hc
            global _MAMBA_HOST
            if host_mamba_slots() > 0:
                if single_process_engine(vllm_config):
                    _MAMBA_HOST = MambaHostStore(host_mamba_slots())
                else:
                    _log("tier: host GDN checkpoint store DISABLED (G-21): executor backend %s, world size %s is not "
                         "a single-process engine; host GDN restores are only safe with the UniProc executor",
                         getattr(vllm_config.parallel_config, "distributed_executor_backend", None),
                         getattr(vllm_config.parallel_config, "world_size", None))
        self._coord = None
        self._mamba_gids: list[int] = []
        self._block_pool = None
        # G-23: save id -> GPU blocks we hold a reference on until the worker has enqueued that save's copy
        self._mholds: dict[int, list] = {}
        # G-25: finish-time tail mirror: copy id -> held GPU blocks; copies waiting for the next connector meta
        self._fholds: dict[int, list] = {}
        self._fin_pending: list = []
        self._fin_next = 0

    def _sparse_group(self) -> int:
        ids = sparse_layer_ids()
        groups = set()
        for g, grp in enumerate(self._kvc.kv_cache_groups):
            for name in grp.layer_names:
                m = _LAYER_RE.search(name)
                if m and "mtp" not in name and int(m.group(1)) in ids:
                    groups.add(g)
        assert len(groups) == 1, f"tier spill: sparse layers span groups {groups}"
        return groups.pop()

    # ---- scheduler side
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.req_obj[request.request_id] = request
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.req_obj[request.request_id] = request
        return None

    def _drop_req(self, rid):
        st = self.sreq.pop(rid, None)
        if st is not None and self.hc is not None:
            # G-25: release deepest blocks first, so LRU eviction of the host pool trims a cached prefix from its
            # tail (the hit shrinks) instead of evicting block 0 first (which kills the whole prefix hit).
            for _, hb in sorted(st["host"].items(), reverse=True):
                self.hc.release(hb)

    def _bhash(self, hashes, b):
        # G-18a: key of logical block b = the (chained) hash of its LAST hash-block, i.e. of tokens [0, (b+1)*BS).
        k = self.hash_k
        i = (b + 1) * k - 1 if k else -1
        return hashes[i] if 0 <= i < len(hashes) else None

    def _plan_req(self, meta, rid, nc, new):
        if new:
            self._drop_req(rid)
            self.sreq[rid] = {"host": {}, "mirrored": 0}
            if self.freed.pop(rid, None):  # preempted/resumed: fresh block list
                pass
            self.freed_dirty.add(rid)
        st = self.sreq.get(rid)
        req = self.req_obj.get(rid)
        if st is None or req is None:
            return
        hashes = req.block_hashes
        k = self.hash_k

        def bhash(b):
            # G-18a: key of logical block b = the (chained) hash of its LAST hash-block, i.e. of tokens [0, (b+1)*BS).
            # hashes[b] is only right when hash_block_size == block_size; k = None disables sharing/lookup.
            i = (b + 1) * k - 1 if k else -1
            return hashes[i] if 0 <= i < len(hashes) else None

        maps, copies = meta.maps.setdefault(rid, []), meta.copies.setdefault(rid, [])
        if new:  # host-cache hits of the prefix (null GPU blocks) must be mapped before the first forward
            for b in range(nc // self.bs):
                hb = self.hc.lookup(bhash(b))
                if hb is not None:
                    self.hc.acquire(hb)
                    st["host"][b] = hb
                    maps.append((b, hb))
                    meta.nhit += 1
        eligible = max(0, (nc - self.margin) // self.bs)
        for b in range(st["mirrored"], eligible):
            if b in st["host"]:
                continue
            h = bhash(b)
            hb = self.hc.lookup(h)
            if hb is not None:  # same tokens already on host: share, no copy
                self.hc.acquire(hb)
                maps.append((b, hb))
            else:
                hb = self.hc.alloc(h)
                maps.append((b, hb))
                copies.append((b, hb))
            st["host"][b] = hb
        st["mirrored"] = max(st["mirrored"], eligible)

    def build_connector_meta(self, scheduler_output):
        so = scheduler_output
        so.has_sync_kv_loads = True  # run start_load_kv BEFORE the forward
        # the worker needs this step's scheduler output (V2 pre_forward does not forward it to start_load_kv)
        meta = HostTierMeta(so=so)
        if self.hc is not None:
            for rid in getattr(so, "preempted_req_ids", None) or ():
                self.preempted += 1
                _log("tier: request %s preempted (total %d)", rid, self.preempted)
            for nr in so.scheduled_new_reqs:
                if nr.req_id in self.sreq:  # V2 runner: a resumed (preempted) request comes back as new
                    _log("tier: request %s re-admitted after preemption with %d computed tokens (prefix/host hit)",
                         nr.req_id, nr.num_computed_tokens)
                self._plan_req(meta, nr.req_id, nr.num_computed_tokens, True)
            cr = so.scheduled_cached_reqs
            for i, rid in enumerate(cr.req_ids):
                self._plan_req(meta, rid, cr.num_computed_tokens[i], rid in cr.resumed_req_ids)
            for rid in cr.resumed_req_ids:
                self.resumed += 1
                _log("tier: request %s resumed after preemption (num_computed %d, total resumes %d)", rid,
                     cr.num_computed_tokens[cr.req_ids.index(rid)] if rid in cr.req_ids else -1, self.resumed)
            for rid in self.freed_dirty:
                meta.freed[rid] = self.freed.get(rid, 0)
            self.freed_dirty.clear()
            # G-16: copies planned in THIS meta only become safe-to-free for the next scheduling pass
            for rid, st in self.sreq.items():
                self.safe[rid] = st["mirrored"]
        self._release_fin_holds()
        if self._fin_pending:
            meta.fin_copies, self._fin_pending = self._fin_pending, []
        self._release_mamba_holds()
        self._save_mamba_offers(so)
        if _MAMBA_HOST is not None:
            meta.mamba_saves, _MAMBA_HOST.pending_saves = _MAMBA_HOST.pending_saves, []
            meta.mamba_loads, _MAMBA_HOST.pending_loads = _MAMBA_HOST.pending_loads, []
            meta.mamba_compares, _MAMBA_HOST.pending_compares = _MAMBA_HOST.pending_compares, []
        return meta

    def _save_mamba_offers(self, so):
        """Save GDN/mamba boundary states offered this step (vLLM hands connectors the exact align-mode checkpoint
        blocks; content is final after this step's forward, so the worker copies them in wait_for_save). Only the
        last boundaries inside the prompt are kept: those are what a follow-up turn hits."""
        if _MAMBA_HOST is None or not self._mamba_gids:
            return
        st = getattr(so, "kv_connector_block_state", None)
        offers = getattr(st, "boundary_state_offloads", None) if st is not None else None
        if not offers:
            return
        bs = self.bs
        keep = int(os.environ.get("SPARSE_TIER_HOST_MAMBA_PER_REQ", "2"))
        for rid, entries in offers.items():
            req = self.req_obj.get(rid)
            if req is None:
                continue
            per_b: dict[int, dict[int, int]] = {}
            for gid, block_id, boundary in entries:
                per_b.setdefault(boundary, {})[gid] = block_id
            for boundary, blocks in sorted(per_b.items()):
                if (boundary % bs or len(blocks) != len(self._mamba_gids)
                        or not (req.num_prompt_tokens - keep * bs < boundary <= req.num_prompt_tokens)
                        or boundary // bs > len(req.block_hashes)):
                    continue
                sid = _MAMBA_HOST.save(req.block_hashes[boundary // bs - 1], blocks)
                if sid is not None and self._block_pool is not None:
                    # G-23: pin the checkpoint blocks until the delayed copy is enqueued; otherwise a finished
                    # request's block can be re-handed-out and zeroed (update_requests) before it is copied.
                    held = [self._block_pool.blocks[b] for b in blocks.values()]
                    self._block_pool.touch(held)
                    self._mholds[sid] = (rid, held)
                    _MAMBA_HOST.stats["pinned"] += 1
                if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1":
                    _log("tier: mamba offer saved %s boundary %d blocks %s store %s", rid, boundary, blocks,
                         _MAMBA_HOST.stats)

    def _release_mamba_holds(self):
        """G-23: drop the references of saves whose GPU->host copy the worker has enqueued (same process: _RT)."""
        if not self._mholds or self._block_pool is None or _RT is None:
            return
        done = [sid for sid in self._mholds if sid in _RT.mamba_done]
        for sid in done:
            _RT.mamba_done.discard(sid)
            self._block_pool.free_blocks(self._mholds.pop(sid)[1])
            if _MAMBA_HOST is not None:
                _MAMBA_HOST.stats["unpinned"] += 1
        if len(self._mholds) > 64:
            _log("tier: WARNING %d GDN checkpoint saves still pinned (worker not running saves?)", len(self._mholds))

    def _mirror_tail_at_finish(self, request, block_ids):
        """G-25: mirroring is planned at scheduling time from num_computed_tokens - MARGIN, so the blocks computed in
        a request's LAST step(s) -- the final prefill chunk when the request ends right after its prompt (max_tokens
        1, immediate EOS, abort) -- never reach the host. Once the GPU prefix cache evicts them, the host chain stops
        below the prompt-end GDN checkpoint and the next turn on that document gets no hit at all (cached=0, full
        re-prefill). Here the remaining full blocks up to the processed prompt/decode end get host blocks (prefix
        keys) and a copy in the next connector meta; their GPU blocks are held (ref+1) until the worker has
        enqueued the copy, so they cannot be re-handed-out and zeroed first (same reasoning as G-16/G-23)."""
        if (self.hc is None or _RT is None or self._block_pool is None
                or os.environ.get("SPARSE_TIER_MIRROR_AT_FINISH", "1") != "1"):
            return
        rid = request.request_id
        st, req = self.sreq.get(rid), self.req_obj.get(rid)
        if st is None or req is None or not block_ids:
            return
        g = self._sparse_group_id
        gblocks = block_ids[g] if isinstance(block_ids, tuple) else block_ids
        processed = int(request.num_computed_tokens) - int(getattr(request, "num_in_flight_tokens", 0) or 0)
        upto = max(0, processed - self.margin) // self.bs  # same rule as _plan_req (in-flight spec tokens)
        if getattr(request, "num_output_tokens", 0) >= 1:  # a sampled token: the whole prompt's KV is written
            upto = max(upto, int(request.num_prompt_tokens) // self.bs)
        upto = min(len(gblocks), upto)
        gids, hids, lbs = [], [], []
        for b in range(upto):
            if b in st["host"]:
                continue
            h = self._bhash(req.block_hashes, b)
            gid = int(gblocks[b])
            if h is None or gid == 0:
                break
            if self.hc.lookup(h) is not None:  # already on host (shared prefix): nothing to copy
                continue
            try:
                hb = self.hc.alloc(h)
            except MemoryError:
                break
            st["host"][b] = hb  # released (LRU-cached) by _drop_req right after
            gids.append(gid)
            hids.append(hb)
            lbs.append(b)
        if not gids:
            return
        held = [self._block_pool.blocks[x] for x in gids]
        self._block_pool.touch(held)
        cid = self._fin_next
        self._fin_next += 1
        self._fholds[cid] = held
        self._fin_pending.append((cid, gids, hids))
        self.fin_mirrored = getattr(self, "fin_mirrored", 0) + len(gids)
        if os.environ.get("SPARSE_TIER_DEBUG", "0") == "1" or self.fin_mirrored == len(gids):
            _log("tier: finish-time tail mirror for %s: %d blocks [%d..%d] (processed %d, prompt %d, total %d)", rid,
                 len(gids), lbs[0], lbs[-1], processed,
                 request.num_prompt_tokens, self.fin_mirrored)

    def _release_fin_holds(self):
        if not self._fholds or _RT is None or self._block_pool is None:
            return
        done = [cid for cid in self._fholds if cid in _RT.fin_done]
        for cid in done:
            _RT.fin_done.discard(cid)
            self._block_pool.free_blocks(self._fholds.pop(cid))
        if len(self._fholds) > 64:
            _log("tier: WARNING %d finish-time tail copies still held (worker not running steps?)", len(self._fholds))

    def request_finished(self, request, block_ids):
        # G-23: a finished request's last step is already enqueued, so its checkpoints are final: copy its pending
        # saves now (same process/stream) and drop the holds, so idle engines do not keep blocks pinned (and
        # reset_prefix_cache still works).
        mine = {sid for sid, (rid, _) in self._mholds.items() if rid == request.request_id}
        if mine and _RT is not None and _RT.mamba_items:
            pending = {i for _, _, _, i in getattr(_RT, "_msave_q", [])} & mine
            if pending and _MAMBA_HOST is not None:
                _MAMBA_HOST.stats["flushed_at_finish"] += len(pending)
            _RT.run_saves_now(lambda s, i: i in mine)
            self._release_mamba_holds()
        self._mirror_tail_at_finish(request, block_ids)
        self.freed.pop(request.request_id, None)
        self.safe.pop(request.request_id, None)
        self._drop_req(request.request_id)
        self.req_obj.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    def bind_kv_cache_manager(self, kv_cache_manager) -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        if not self.spill:
            return
        g = self._sparse_group()
        coord = kv_cache_manager.coordinator
        mgr = coord.single_type_managers[g]
        policy = os.environ.get("SPARSE_TIER_SPILL_POLICY", "pressure")
        wm = int(os.environ.get("SPARSE_TIER_WATERMARK_BLOCKS", "0")) or max(8, mgr.block_pool.num_gpu_blocks // 10)
        use_hits = os.environ.get("SPARSE_TIER_HOST_PREFIX", "1") == "1"
        tail = int(os.environ.get("SPARSE_TIER_TAIL", str(int(os.environ.get("SPARSE_WIN", "1024")) + 64)))
        sink_blocks = int(os.environ.get("SPARSE_TIER_SINK_BLOCKS", "1"))
        bs = mgr.block_size
        hbs = int(getattr(mgr.block_pool, "hash_block_size", bs) or bs)
        self.hash_k = bs // hbs if bs % hbs == 0 else None
        if self.hash_k != 1:
            _log("tier: hash block %d vs block %d: host blocks keyed by hash index (b+1)*%s-1%s", hbs, bs,
                 self.hash_k, "" if self.hash_k else " -> DISABLED (not a divisor): no host sharing")
        HostTierSpillPolicy(self, tail, sink_blocks, policy, wm).install(mgr)
        self._sparse_group_id = g
        self._coord = coord
        self._block_pool = kv_cache_manager.block_pool
        self._mamba_gids = [i for i, grp in enumerate(self._kvc.kv_cache_groups)
                            if type(grp.kv_cache_spec).__name__ == "MambaSpec"]
        # G-23 holds touch blocks of kv_cache_manager.block_pool by id: the mamba groups must allocate from it
        assert all(coord.single_type_managers[g].block_pool is self._block_pool for g in self._mamba_gids), \
            "tier: mamba groups use a separate block pool; GDN checkpoint holds (G-23) would pin the wrong blocks"
        mhit = "off"
        if _MAMBA_HOST is not None and hasattr(coord, "attention_groups"):
            for i, sg in enumerate(coord.attention_groups):
                if type(sg.spec).__name__ == "MambaSpec":
                    coord.attention_groups[i] = sg._replace(manager_cls=make_mamba_hit_manager_cls(sg.manager_cls))
                    mhit = coord.attention_groups[i].manager_cls.__name__
        _log("tier: mamba groups %s, host checkpoint store %s (%d slots)", self._mamba_gids, mhit,
             _MAMBA_HOST.n if _MAMBA_HOST else 0)
        hit = "off"
        if use_hits and hasattr(coord, "attention_groups"):
            for i, sg in enumerate(coord.attention_groups):
                if g in sg.group_ids:
                    coord.attention_groups[i] = sg._replace(manager_cls=make_hit_manager_cls(sg.manager_cls))
                    hit = coord.attention_groups[i].manager_cls.__name__
        _log("tier: SPILL scheduler hook on KV group %d (%s): frees blocks [%d, (processed-%d)//%d), policy %s "
             "(watermark %d free blocks); host prefix cache %s (%d host blocks)", g, type(mgr).__name__, sink_blocks,
             tail, bs, policy, wm, hit, self.hc.n if self.hc else -1)

    # ---- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        global _RT
        if enabled():
            _RT = TierRuntime(kv_caches, self._kvc, self._vllm_config)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        if _RT is None:
            return
        meta = self._get_connector_metadata()
        so = getattr(meta, "so", None)
        if so is None:
            raise RuntimeError("tier: connector metadata carries no scheduler output")
        _RT.start_step(so, kwargs.get("request_ids"), meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return None

    def wait_for_save(self):
        return None
