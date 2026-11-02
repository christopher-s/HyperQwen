# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sparsekv/tier_admission.py -- SKELETON (not wired, not tested): let vLLM admit more context than the GPU KV pool.

Smallest viable path ("positional host tier"), see STREAM_tier.md for the full plan and the vLLM 0.31 code references.

  * The 16 sparse full-attention layers get their own spec, HostTierFullAttentionSpec. Its per-request GPU need is
    capped at sink blocks + tail window + in-flight tokens (the SlidingWindowSpec pattern), so the startup check
    (_check_enough_kv_cache_memory) and the admission gate (max_admission_blocks_per_request) stop requiring
    max_model_len to fit on the GPU.
  * HostTierFullAttentionManager allocates a HOST block for every GPU block (write-through mirror target) from a
    private host BlockPool (the HiSparseSourceManager.bind_host_pool pattern), returns the Mamba-style sentinel when
    the host pool is short, and in remove_skipped_blocks "spills" mirrored blocks between the sinks and the tail
    window: the GPU entry becomes the null block, the GPU block is freed, and the request goes into
    block_table_updates so the worker rewrites its table row (Model Runner V2 already applies overwrite=True rows,
    v1/worker/gpu/model_runner.py update_requests).
  * HostTierConnector carries per step: new (gpu_block -> host_block) mirror pairs and host block-table rows. The
    worker side owns the pinned host pool (HostKVMirror), mirrors filled blocks on a side stream, and serves:
      - decode: sparsekv.tier.gather_rows_tiered (null GPU entry -> hot buffer / host via UVA);
      - sketch: write_sketch/select index the sketch cache by HOST slot (host_bt*BS+off), so a spilled token keeps
        its sketch on the GPU after its GPU block is reused;
      - prefill over spilled blocks: stage host blocks into a per-layer GPU scratch pool and run dense pthfa over a
        patched block table (eager; prefill attention is outside the FULL decode graphs).
  * GDN/mamba groups, the MTP drafter layer (plain FullAttentionSpec group) and everything else are unchanged.

Nothing here is imported by the server yet. Class bodies list the exact behaviour to implement; methods that are
not implemented raise NotImplementedError so a premature wiring fails loudly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

TIER_ENABLED = os.environ.get("SPARSE_TIER", "0") == "1"
HOST_GIB = float(os.environ.get("SPARSE_TIER_HOST_GIB", "16"))
SPILL_POLICY = os.environ.get("SPARSE_TIER_SPILL", "pressure")  # pressure | eager
HOT_FRAC = float(os.environ.get("SPARSE_TIER_HOT_FRAC", "0.10"))


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ------------------------------------------------------------------------------------------------ scheduler side
def make_spec_class():
    """Build HostTierFullAttentionSpec lazily (vLLM import only inside the server)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    @dataclass(frozen=True, kw_only=True)
    class HostTierFullAttentionSpec(FullAttentionSpec):
        sink_tokens: int = 64
        tail_tokens: int = 1024 + 64  # SPARSE_WIN + verify tokens (k+1) + slack; must cover the selection window

        def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
            # sink block(s) + window tail + in-flight chunk, +1 for a window that straddles a block boundary
            n = cdiv(self.sink_tokens, self.block_size) + cdiv(
                min(self.tail_tokens + max_in_flight_tokens, max_model_len), self.block_size) + 1
            return min(n, cdiv(max_model_len, self.block_size))

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return self.max_admission_blocks_per_request(
                vllm_config.max_in_flight_tokens, vllm_config.model_config.max_model_len) * self.page_size_bytes

        # NOTE: dataclass __eq__ compares the class, so the MTP drafter's FullAttentionSpec stays in its own group.

    return HostTierFullAttentionSpec


def make_manager_class():
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    class HostTierFullAttentionManager(FullAttentionManager):
        """GPU blocks for sinks + tail only (under pressure); every block mirrored to a private host pool.

        To register: KVCacheSpecRegistry.register(HostTierFullAttentionSpec, HostTierFullAttentionManager) and add
        the spec to the isinstance tuple in get_manager_for_kv_cache_spec (single_type_kv_cache_manager.py ~2745)
        so max_admission_blocks_per_request reaches the manager.
        """

        def __init__(self, kv_cache_spec, **kwargs):
            super().__init__(kv_cache_spec, **kwargs)
            self.req_to_host_blocks: dict[str, list] = {}
            self.host_pool = None  # bound by bind_host_pool(num_host_blocks) after KV planning
            self.mirrored: dict[str, int] = {}  # req -> number of leading blocks whose mirror copy is ACKED
            self.table_updates: set[str] = set()
            self.new_mirror_pairs: list[tuple[int, int]] = []  # (gpu_block_id, host_block_id) to copy this step

        def bind_host_pool(self, num_blocks: int) -> None:
            # copy of HiSparseSourceManager.bind_host_pool (single_type_kv_cache_manager.py:2367): a
            # SharedEventQueueBlockPool(medium=MEDIUM_CPU, enable_caching=prefix caching) -- host blocks keep their
            # block hashes so a later turn can hit host-only blocks.
            raise NotImplementedError

        def get_num_blocks_to_allocate(self, request_id, num_tokens, *args, **kwargs) -> int:
            # 1. GPU need = super() (null entries count toward len(req_to_blocks), so a running request with spilled
            #    blocks only asks for new tail blocks; new requests are capped by max_admission_blocks_per_request).
            # 2. host need = cdiv(num_tokens, BS) - len(req_to_host_blocks[req]); if host pool short -> return
            #    coordinator.gpu_pool.num_gpu_blocks + 1 (admission sentinel, as HiSparseSourceManager :2412).
            raise NotImplementedError

        def allocate_new_blocks(self, request_id, num_tokens, *args, **kwargs):
            # super() for GPU blocks; allocate the same number of host blocks; queue (gpu, host) pairs. The worker
            # mirrors a pair once the GPU block is FULL (num_computed_tokens crosses its end) -- partial tail blocks
            # are always resident and never spilled.
            raise NotImplementedError

        def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
            # Spill window: blocks in [cdiv(sink, BS), (processed_computed_tokens - tail_tokens) // BS) whose mirror
            # is acked. Policy 'eager': spill all of them. Policy 'pressure': only while
            # block_pool.get_num_free_blocks() < watermark (HiSparse uses max(hot_cost, num_blocks // 10)), oldest
            # first. Spill = _remove_blocks_in_range-like loop that sets req_to_blocks[i] = null_block and frees
            # the GPU block (shared prefix blocks just drop a ref); add request_id to table_updates.
            raise NotImplementedError

        def find_longest_cache_hit(self, *args, **kwargs):
            # GPU hit first (super); then extend the hit with host-cached blocks (host pool hash lookup). Host-only
            # hit blocks get a null GPU entry + their host block id; the worker stages them for the prefill.
            raise NotImplementedError

        def free(self, request_id: str) -> None:
            # super().free (GPU) + free host blocks to the host pool (they stay cached, evictable).
            raise NotImplementedError

        def take_block_table_updates(self, req_blocks_fn) -> dict:
            out = {r: req_blocks_fn(r) for r in self.table_updates}
            self.table_updates.clear()
            return out

    return HostTierFullAttentionManager


@dataclass
class HostTierStepMeta:
    """Connector metadata for one scheduler step (scheduler -> worker)."""
    mirror_pairs: list[tuple[int, int]] = field(default_factory=list)  # (gpu_block, host_block) full blocks to DMA
    host_rows: dict[str, list[int]] = field(default_factory=dict)  # req_id -> host block ids appended this step
    new_reqs: list[str] = field(default_factory=list)  # reset hot-buffer slot
    finished_reqs: list[str] = field(default_factory=list)


def make_connector_class():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    class HostTierConnector(KVConnectorBase_V1):
        """Scheduler side: collects manager queues into HostTierStepMeta and sets scheduler_output.block_table_updates
        (exactly as HiSparseConnector.build_connector_meta does, connector.py:109).
        Worker side: register_kv_caches -> HostKVMirror (alloc_pinned, exact size, cudaHostRegister) for the 16
        sparse layers + TierRuntime; start_load_kv -> apply host rows, run mirror DMAs on the side stream (the next
        forward waits on its event before any block it frees can be rewritten); wait_for_save -> record event.
        Ack: get_finished-like report of mirrored GPU block ids; the manager spills only acked blocks (HiSparse's
        _complete_host_writes protocol, coordinator.py:705-766), which also makes async scheduling safe.
        """

        def build_connector_meta(self, scheduler_output):
            raise NotImplementedError

        def register_kv_caches(self, kv_caches):
            raise NotImplementedError

        def start_load_kv(self, forward_context, **kwargs):
            raise NotImplementedError

        def wait_for_layer_load(self, layer_name):
            return None

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            return None

        def wait_for_save(self):
            raise NotImplementedError

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False  # host-cache hits are resolved inside the manager, not via external loads

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return None

    return HostTierConnector


# --------------------------------------------------------------------------------------------------- worker side
class TierRuntime:
    """Worker-side state shared by the 16 sparse layers (owned by the connector worker, read by pthfa sparse_kv).

    host_bt   int32 [max_num_reqs, max_blocks] indexed by req_state_idx (same index as Model Runner V2 block
              tables); batch rows are gathered with idx_mapping into a static buffer (graph-safe).
    hot       sparsekv.tier.HotBuffer with slots = req_state_idx; rows = max(2*C, HOT_FRAC * max_model_len).
    mirror    sparsekv.tier.HostKVMirror [16, host_blocks, Hkv, BS, W].
    sketch    per layer int8 [host_blocks*BS + 1, Hkv, r] + fp16 scale, indexed by HOST slot (stable across spills,
              prefix-cacheable with the host block).

    Hook points in stream B's pthfa/python/pth_flash_attn/sparse_kv.py:
      LayerSparse.write : slot_mapping -> host slot mapping (host_bt[req, pos//BS]*BS + pos%BS), computed once per
                          step for all layers.
      select_and_stage  : select(..., block_table=host_bt_batch) so the sketch is read through host slots;
                          c.ops.gather_rows(...) -> tier.gather_rows_tiered(kv, block_table, mirror.dev[l],
                          host_bt_batch, idx, n_valid, st_kv, bt, hot, l, idx_mapping, scratch, null_block=0).
      prefill slice     : stage_prefill(l, ...) below before the dense pthfa call.
    """

    def __init__(self, *a, **k):
        raise NotImplementedError

    def stage_prefill(self, layer: int, block_table, num_computed_blocks):
        """Copy host-only blocks needed by this prefill chunk into a GPU scratch pool (double-buffered, prefetch
        layer l+1 on the copy stream) and return a patched block table. Cost: (spilled context) x 2080 B x 16 per
        chunk -- 150K spilled = 5 GB = ~0.4 s (x16) / ~0.8 s (x4) per 2048-token chunk."""
        raise NotImplementedError
