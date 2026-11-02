# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU regression tests for stream I review fixes (REVIEW.md G-ids). No GPU needed.

PYTHONPATH=<worktree>/pthfa/python:<worktree> python sparsekv/test_fixes_cpu.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pthfa", "python"))
sys.path.insert(0, ROOT)
os.environ.setdefault("SPARSEKV_ROOT", ROOT)


def _fresh_sparse_kv(**env):
    import importlib
    for k in list(os.environ):
        if k.startswith("SPARSE_"):
            del os.environ[k]
    os.environ["SPARSEKV_ROOT"] = ROOT
    os.environ.update(env)
    import pth_flash_attn.sparse_kv as sk
    assert sk.__file__.startswith(ROOT), sk.__file__
    sk = importlib.reload(sk)
    return sk


def test_g2_missing_basis_raises():
    sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSE_BASIS="/nonexistent/basis.pt")
    try:
        sk.cfg().basis()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing SPARSE_BASIS did not raise")
    sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSE_BASIS="/nonexistent/basis.pt",
                          SPARSE_ALLOW_RANDOM_BASIS="1")
    assert sk.cfg().basis()["src"] == "random-placeholder"
    print("G-2 missing basis raises (random only with SPARSE_ALLOW_RANDOM_BASIS=1): OK")


def test_g5_no_silent_ref_fallback():
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "sparsekv"))
    open(os.path.join(d, "sparsekv", "__init__.py"), "w").close()
    open(os.path.join(d, "sparsekv", "triton_ops.py"), "w").write("raise ImportError('broken on purpose')\n")
    open(os.path.join(d, "sparsekv", "ref.py"), "w").write("def select(*a, **k):\n    pass\n")  # no GRAPH_SAFE
    for m in [m for m in sys.modules if m == "sparsekv" or m.startswith("sparsekv.")]:
        del sys.modules[m]
    sys.path.insert(0, d)
    try:
        sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="triton", SPARSEKV_ROOT=d)
        try:
            sk.cfg()
        except ImportError as e:
            assert "SPARSE_IMPL=ref" in str(e)
        else:
            raise AssertionError("broken triton_ops silently fell back to ref")
        sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSEKV_ROOT=d)
        assert sk.cfg().graph_safe is False
    finally:
        sys.path.remove(d)
        for m in [m for m in sys.modules if m == "sparsekv" or m.startswith("sparsekv.")]:
            del sys.modules[m]
    sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref")
    assert sk.cfg().graph_safe is True, "worktree ref.py must declare GRAPH_SAFE"
    print("G-5 no silent ref fallback, GRAPH_SAFE flag: OK")


class _FakeHot:
    def next_step(self):
        pass

    def reset_slot(self, slot):
        pass


def _fake_runtime(spill=False):
    import torch
    from sparsekv.tier_connector import TierRuntime
    rt = TierRuntime.__new__(TierRuntime)
    rt.groups, rt.reqs, rt.max_reqs, rt.spill = [0], {}, 2, spill
    rt.batch_slots = torch.full((10,), 2, dtype=torch.int32)
    rt.hot = _FakeHot()
    rt.stats = {"steps": 0, "mirrored_blocks": 0}
    rt.mirrored = torch.zeros((1, 3), dtype=torch.int32)
    rt.host_bt = torch.full((3, 8), -1, dtype=torch.int32)
    rt.freed_upto = torch.zeros(3, dtype=torch.int32)
    rt.free_slots = [1, 0]
    rt.margin, rt.BS = 64, 864
    rt._shadow_mirror = lambda: None
    rt.sync_slots, rt.sf_stats = True, torch.zeros(3, dtype=torch.int64)  # S4 fields (CPU fake: synchronous upload)
    return rt


def test_g13_scheduler_output_without_block_table_updates():
    import types
    rt = _fake_runtime()
    cr = types.SimpleNamespace(req_ids=["a"], new_block_ids=[([1, 2],)], resumed_req_ids=set(),
                               num_computed_tokens=[100])
    so = types.SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[], scheduled_cached_reqs=cr,
                               num_scheduled_tokens={"a": 8})  # no block_table_updates attribute (sk-port series)
    rt.start_step(so, ["a"], None)
    rt.start_step(so, ["a"], None)  # second step: cached-request path that read so.block_table_updates
    assert rt.reqs["a"]["blocks"][0] == [1, 2, 1, 2] and int(rt.batch_slots[0]) == 0
    print("G-13 start_step without SchedulerOutput.block_table_updates: OK")


class _Blk:
    def __init__(self, i):
        self.block_id = i

    def __repr__(self):
        return f"B{self.block_id}"


class _FakePool:
    def __init__(self, n):
        self.num_gpu_blocks = n
        self.freed = []
        self.nfree = 0  # always under the watermark: pressure policy frees

    def get_num_free_blocks(self):
        return self.nfree

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


class _FakeMgr:
    block_size = 864

    def __init__(self, pool):
        self.block_pool = pool
        self._null_block = _Blk(0)
        self.req_to_blocks = {}

    def remove_skipped_blocks(self, request_id, processed, num_prompt_tokens=None):
        pass

    def pop_blocks_for_free(self, request_id):
        return self.req_to_blocks.pop(request_id, [])

    def free(self, request_id):
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    def _remove_blocks_in_range(self, request_id, first_block, last_block):  # copy of vLLM 0.31
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or first_block >= last_block:
            return
        last_block = min(last_block, len(blocks))
        freed = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] is self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)


def _fake_connector(nblocks=40, bs=864):
    import types
    from sparsekv import tier_connector as tc
    os.environ.update(SPARSE_KV="1", SPARSE_TIER="1", SPARSE_TIER_SPILL="1", SPARSE_TIER_HOST_PREFIX="0",
                      SPARSE_BASIS="/nonexistent")
    con = tc.HostTierConnector.__new__(tc.HostTierConnector)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(3, 64, 4)]
    con._kvc = types.SimpleNamespace(kv_cache_groups=[types.SimpleNamespace(layer_names=names,
                                                                          kv_cache_spec=types.SimpleNamespace())])
    con.spill, con.margin, con.bs = True, 64, bs
    con.hc = tc.HostCache(64)
    con.req_obj, con.sreq, con.freed, con.freed_dirty, con.safe = {}, {}, {}, set(), {}
    con.resumed = con.preempted = 0  # sk-tier stage 4 counters
    con._mholds, con._block_pool = {}, None  # G-23 hold state (set in __init__ / bind)
    con._fholds, con._fin_pending, con._fin_next = {}, [], 0  # G-25 finish-time tail mirror state
    pool = _FakePool(nblocks)
    mgr = _FakeMgr(pool)
    coord = types.SimpleNamespace(single_type_managers=[mgr])
    con.bind_kv_cache_manager(types.SimpleNamespace(coordinator=coord, block_pool=pool))
    return con, mgr, pool


def _so(new=(), cached=()):
    import types
    nr = [types.SimpleNamespace(req_id=r, num_computed_tokens=n) for r, n in new]
    cr = types.SimpleNamespace(req_ids=[r for r, _ in cached], num_computed_tokens=[n for _, n in cached],
                               resumed_req_ids=set())
    return types.SimpleNamespace(scheduled_new_reqs=nr, scheduled_cached_reqs=cr)


def test_g16_never_free_unmirrored_blocks():
    import types
    bs = 864
    con, mgr, pool = _fake_connector(bs=bs)
    blocks = [_Blk(i) for i in range(1, 31)]
    mgr.req_to_blocks["r"] = list(blocks)
    con.req_obj["r"] = types.SimpleNamespace(block_hashes=[("h", i) for i in range(30)])
    copied_before = set()  # GPU blocks whose host copy is in an EARLIER step's meta

    def step(processed, new):
        mgr.remove_skipped_blocks("r", processed)       # scheduler: allocate_slots -> spill hook
        for b in pool.freed:
            assert b.block_id in copied_before, f"freed {b} before its host copy was enqueued"
        meta = con.build_connector_meta(_so(new=[("r", processed)]) if new else _so(cached=[("r", processed)]))
        for lb, _hb in meta.copies.get("r", []):
            copied_before.add(blocks[lb].block_id)

    step(20 * bs, True)    # first step of a request with a 20-block prefix-cache hit: nothing may be freed yet
    assert pool.freed == [], pool.freed
    step(22 * bs, False)   # one chunk later: blocks copied in step 1 may go
    assert pool.freed, "nothing freed under pressure"
    n1 = len(pool.freed)
    step(25 * bs, False)
    assert len(pool.freed) > n1
    # preemption resets safety: after pop + re-add, the first scheduling pass frees nothing
    mgr.pop_blocks_for_free("r")
    pool.freed.clear()
    copied_before.clear()
    con.freed.pop("r", None)
    mgr.req_to_blocks["r"] = list(blocks)
    step(20 * bs, True)
    assert pool.freed == [], pool.freed
    print("G-16 spill hook frees only blocks mirrored in an earlier step: OK")


def test_g18a_host_keys_use_full_block_hash():
    import types
    bs = 864
    con, mgr, pool = _fake_connector(bs=bs)
    pool.hash_block_size = 432
    con.bind_kv_cache_manager(types.SimpleNamespace(coordinator=types.SimpleNamespace(single_type_managers=[mgr]),
                                                    block_pool=pool))  # re-bind with the small hash block
    assert con.hash_k == 2
    mgr.req_to_blocks["r"] = [_Blk(i) for i in range(1, 11)]
    con.req_obj["r"] = types.SimpleNamespace(block_hashes=[("h", i) for i in range(20)])
    con.build_connector_meta(_so(new=[("r", 8 * bs)]))
    assert set(con.hc.h2b) == {("h", 2 * b + 1) for b in range(7)}, sorted(con.hc.h2b)
    pool.hash_block_size = 500  # not a divisor: no keys at all
    con.bind_kv_cache_manager(types.SimpleNamespace(coordinator=types.SimpleNamespace(single_type_managers=[mgr]),
                                                    block_pool=pool))
    assert con.hash_k is None
    con.hc = __import__("sparsekv.tier_connector", fromlist=["x"]).HostCache(64)
    con.sreq.clear()
    con.build_connector_meta(_so(new=[("r", 8 * bs)]))
    assert con.hc.h2b == {}
    print("G-18a host block keys follow hash_block_size: OK")


def test_g18b_free_below_host_hit_null():
    import types
    bs = 864
    con, mgr, pool = _fake_connector(bs=bs)
    hb = con.hc.alloc(("h", 5))   # block 5 of this prefix is already on host (another request), unreferenced
    con.hc.release(hb)
    blocks = [_Blk(i) for i in range(1, 31)]
    blocks[5] = mgr._null_block   # host hit -> null GPU block
    mgr.req_to_blocks["r"] = list(blocks)
    con.req_obj["r"] = types.SimpleNamespace(block_hashes=[("h", i) for i in range(30)])
    mgr.remove_skipped_blocks("r", 20 * bs)
    con.build_connector_meta(_so(new=[("r", 20 * bs)]))
    mgr.remove_skipped_blocks("r", 22 * bs)
    freed = {b.block_id for b in pool.freed}
    assert {2, 3, 4, 5} <= freed, sorted(freed)  # logical blocks 1..4 (GPU ids 2..5) sit below the null at 5
    assert 1 not in freed  # sink block
    print("G-18b spill hook frees GPU blocks below a host-hit null block: OK")


def _vcfg(max_len=153600, max_seqs=2, bs=864):
    import types
    mc = types.SimpleNamespace(get_num_kv_heads=lambda pc: 4, get_head_size=lambda: 256, max_model_len=max_len)
    return types.SimpleNamespace(model_config=mc, parallel_config=None,
                                 cache_config=types.SimpleNamespace(block_size=bs),
                                 scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs))


def test_g1_memory_reserve_covers_post_profile_allocations():
    sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSE_BASIS="/nonexistent", SPARSE_ALLOW_RANDOM_BASIS="1")
    vc = _vcfg()
    budget = 8 * 2**30
    res, parts = sk.gpu_reserve_bytes(vc, budget)
    # pool the remaining budget would hold if ALL of it were the 16 sparse layers' KV (upper bound on tokens)
    kv = budget - res
    tokens = kv // (16 * 4 * 520)
    nb = tokens // 864
    sketch_actual = 16 * (nb * 864 + 1) * 4 * (64 + 2)
    staging_actual = 2 * 4 * 4 * 864 * 520
    assert parts["sketch"] >= sketch_actual and parts["staging"] >= staging_actual, (parts, sketch_actual)
    assert res < 0.13 * budget + 64 * 2**20, res
    # tier SPILL adds hot buffer, host-indexed sketch rows and prefill staging, computed from the same dims
    sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSE_BASIS="/nonexistent", SPARSE_ALLOW_RANDOM_BASIS="1",
                          SPARSE_TIER="1", SPARSE_TIER_SPILL="1")
    res2, parts2 = sk.gpu_reserve_bytes(vc, budget)
    from sparsekv.tier_connector import tier_dims
    d = tier_dims(vc, 864, 3136, True)
    assert parts2["tier_prefill_staging"] == (d["maxblk"] + 1) * 4 * 864 * 520
    # S4: hot KV rows for the 2 request slots only (the CUDA-graph padding slot has bookkeeping but no rows)
    assert parts2["tier_hot"] >= 16 * 2 * d["H"] * 4 * 520 and res2 > res
    assert parts2["tier_hot"] < 16 * 3 * d["H"] * 4 * 520
    # wrapper: subtracts from Worker.determine_available_memory and is idempotent
    from vllm.v1.worker import gpu_worker
    orig = gpu_worker.Worker.determine_available_memory
    try:
        gpu_worker.Worker.determine_available_memory = lambda self: budget
        sk = _fresh_sparse_kv(SPARSE_KV="1", SPARSE_IMPL="ref", SPARSE_BASIS="/nonexistent",
                              SPARSE_ALLOW_RANDOM_BASIS="1")
        sk.install_memory_reserve(); sk.install_memory_reserve()
        w = gpu_worker.Worker.__new__(gpu_worker.Worker)
        w.vllm_config = vc
        assert gpu_worker.Worker.determine_available_memory(w) == budget - res
    finally:
        gpu_worker.Worker.determine_available_memory = orig
    print("G-1/G-17 memory reserve (sketch/staging/tier) subtracted from the KV budget: OK")


def test_g24_install_upgrade_keeps_stored_copy_when_reverse_fails():
    """install.sh upgrade loop: a changed, previously applied patch that neither reverses nor is absent (mixed tree)
    must abort with the stored copy kept (old code deleted it, so --uninstall could no longer reverse it)."""
    import subprocess
    import tempfile
    d = tempfile.mkdtemp()
    venv = os.path.join(d, "venv")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    py = os.path.join(venv, "bin", "python")
    sp = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], check=True,
                        capture_output=True, text=True).stdout.strip()
    for pkg in ("vllm", "pth_flash_attn"):
        os.makedirs(os.path.join(sp, pkg))
    target = os.path.join(sp, "vllm", "foo.py")
    applied = os.path.join(venv, "share", "sparsekv", "applied", "vllm")
    os.makedirs(applied)
    stored = os.path.join(applied, "sparse-zz-test.patch")  # not in the repo's series: content differs from "current"
    open(stored, "w").write("--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n+b = 3\n c = 4\n")
    open(target, "w").write("a = 1\nb = 9\nc = 4\n")  # mixed: neither b = 2 (absent) nor b = 3 (applied)
    r = subprocess.run(["bash", os.path.join(ROOT, "sparsekv", "install.sh"), venv], capture_output=True, text=True)
    assert r.returncode == 1 and "Stored copy kept" in r.stderr, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert os.path.exists(stored), "stored copy deleted although the old patch could not be reversed"
    print("G-24 install.sh upgrade keeps the stored copy and aborts on a mixed tree: OK")


class _FakeMambaPool:
    hash_block_size = 864

    def __init__(self, n=40):
        self.blocks = [_Blk(i) for i in range(n)]
        for b in self.blocks:
            b.ref_cnt = 0
        self.free_q = list(self.blocks[1:])
        self.inserted = []

    def get_num_free_blocks(self):
        return len(self.free_q)

    def get_new_blocks(self, n):
        out, self.free_q = self.free_q[:n], self.free_q[n:]
        for b in out:
            b.ref_cnt += 1
        return out

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and b in self.free_q:
                self.free_q.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free_q.append(b)

    def _insert_block_hash(self, key, blk, num_tokens=None):
        self.inserted.append((key, blk.block_id))


def test_g21_refuse_host_gdn_restore_without_same_process_runtime():
    import types
    from sparsekv import tier_connector as tc
    assert tc.single_process_engine(types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(distributed_executor_backend="uni", world_size=1)))
    for be, ws in (("mp", 1), ("ray", 1), ("uni", 2), (None, 1)):
        assert not tc.single_process_engine(types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(distributed_executor_backend=be, world_size=ws))), be

    class _Base:
        @classmethod
        def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                   drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
            return tuple([] for _ in kv_cache_group_ids), 0

    old_ms, old_rt = tc._MAMBA_HOST, tc._RT
    try:
        tc._MAMBA_HOST = tc.MambaHostStore(4)
        hashes = [("h", i) for i in range(4)]
        tc._MAMBA_HOST.h2s[hashes[2]] = 0          # host checkpoint at the end of block 2
        tc._RT = None                              # multi-process: no worker runtime in the scheduler process
        pool = _FakeMambaPool()
        mgr = tc.make_mamba_hit_manager_cls(_Base)
        spec = types.SimpleNamespace(block_size=864)
        _, hit = mgr.find_longest_cache_hit(hashes, 4 * 864, [1, 2], pool, spec, False, 864)
        assert hit == 0 and not tc._MAMBA_HOST.pending_loads and not pool.inserted, "restore not refused"
    finally:
        tc._MAMBA_HOST, tc._RT = old_ms, old_rt
    print("G-21 host GDN restore refused without a same-process runtime: OK")


class _FakeRT:
    def __init__(self):
        self.mamba_done = set()
        self.mamba_items = {1: [], 2: []}


def test_g23_offered_gdn_blocks_pinned_until_saved():
    """A finished request's offered GDN checkpoint blocks must not be re-handed-out before the delayed save ran."""
    import types
    from sparsekv import tier_connector as tc
    bs = 864
    con, mgr, _ = _fake_connector(bs=bs)
    pool = _FakeMambaPool(40)
    con.hc = None                       # skip the KV-tier half of build_connector_meta
    con._mamba_gids = [1, 2]
    con._block_pool = pool
    con._mholds = getattr(con, "_mholds", {})
    old_ms, old_rt = tc._MAMBA_HOST, tc._RT
    try:
        tc._MAMBA_HOST = tc.MambaHostStore(4)
        rt = tc._RT = _FakeRT()
        a = pool.get_new_blocks(2)      # request A's checkpoint blocks (one per mamba group), owned by A
        con.req_obj["A"] = types.SimpleNamespace(num_prompt_tokens=4 * bs, block_hashes=[("h", i) for i in range(4)])
        offers = {"A": [(1, a[0].block_id, 4 * bs), (2, a[1].block_id, 4 * bs)]}
        so = types.SimpleNamespace(kv_connector_block_state=types.SimpleNamespace(boundary_state_offloads=offers),
                                   scheduled_new_reqs=[], scheduled_cached_reqs=None)
        meta = con.build_connector_meta(so)            # step N: offer -> save queued for the worker
        (slot, blocks, *rest), = meta.mamba_saves
        sid = rest[0] if rest else None
        pool.free_blocks(a)                            # A finishes (short answer) within the save delay
        empty = types.SimpleNamespace(kv_connector_block_state=None, scheduled_new_reqs=[], scheduled_cached_reqs=None)
        for _ in range(2):                             # steps N+1, N+2: B allocates everything that is free
            got = pool.get_new_blocks(pool.get_num_free_blocks())
            assert not set(b.block_id for b in got) & set(blocks.values()), "checkpoint block re-handed-out before save"
            con.build_connector_meta(empty)
        rt.mamba_done.add(sid)                         # worker enqueued the copy (start_step of N+2)
        con.build_connector_meta(empty)                # next scheduling pass releases the hold
        assert set(b.block_id for b in pool.free_q) >= set(blocks.values()), "hold not released after the save"
        assert sid not in rt.mamba_done and not con._mholds
    finally:
        tc._MAMBA_HOST, tc._RT = old_ms, old_rt
    print("G-23 offered GDN checkpoint blocks pinned until the delayed save is enqueued: OK")


class _FakeRTQ(_FakeRT):
    """Worker stand-in with the real queue logic of TierRuntime (copies recorded instead of done)."""

    def __init__(self):
        super().__init__()
        self._msave_q = []
        self.copied = []
        self.stats = {}

    def _copy_save(self, slot, blocks, sid):
        self.copied.append(sid)
        self.mamba_done.add(sid)


def test_g23_request_finished_runs_saves_now_and_unpins():
    import types
    from sparsekv import tier_connector as tc
    rt = _FakeRTQ()
    rt.run_saves_now = types.MethodType(tc.TierRuntime.run_saves_now, rt)
    rt._run_mamba_saves = types.MethodType(tc.TierRuntime._run_mamba_saves, rt)
    rt._msave_q = [(0, 0, {1: 5}, 10), (0, 1, {1: 6}, 11)]
    rt.run_saves_now(lambda s, i: i == 11)       # forcing one save must not age the other
    assert rt.copied == [11] and rt._msave_q == [(0, 0, {1: 5}, 10)], rt._msave_q
    rt._run_mamba_saves()                         # 1st regular step after the offer: age 1 < delay 2 -> not yet
    assert rt.copied == [11]
    rt._run_mamba_saves()
    assert rt.copied == [11, 10]
    # scheduler: finishing the request runs its pending saves at once and releases the holds
    bs = 864
    con, _, _ = _fake_connector(bs=bs)
    pool = _FakeMambaPool(40)
    con.hc, con._mamba_gids, con._block_pool, con._mholds = None, [1, 2], pool, {}
    old_ms, old_rt = tc._MAMBA_HOST, tc._RT
    try:
        tc._MAMBA_HOST = tc.MambaHostStore(4)
        rt2 = tc._RT = _FakeRTQ()
        rt2.run_saves_now = types.MethodType(tc.TierRuntime.run_saves_now, rt2)
        a = pool.get_new_blocks(2)
        con.req_obj["A"] = types.SimpleNamespace(num_prompt_tokens=4 * bs, block_hashes=[("h", i) for i in range(4)])
        offers = {"A": [(1, a[0].block_id, 4 * bs), (2, a[1].block_id, 4 * bs)]}
        so = types.SimpleNamespace(kv_connector_block_state=types.SimpleNamespace(boundary_state_offloads=offers),
                                   scheduled_new_reqs=[], scheduled_cached_reqs=None)
        meta = con.build_connector_meta(so)
        rt2._msave_q = [(0, s, b, i) for s, b, i in meta.mamba_saves]   # worker queued it (start_step N)
        pool.free_blocks(a)
        con.request_finished(types.SimpleNamespace(request_id="A"), [])
        assert rt2.copied == [meta.mamba_saves[0][2]] and not con._mholds
        assert {b.block_id for b in a} <= {b.block_id for b in pool.free_q}
    finally:
        tc._MAMBA_HOST, tc._RT = old_ms, old_rt
    print("G-23 forced saves do not age others; request_finished copies now and unpins: OK")


def test_g25_finish_mirrors_tail_blocks_and_lru_trims_from_tail():
    """A request that ends right after its prompt (max_tokens 1) had its last prefill chunk's blocks never mirrored:
    after GPU eviction the host chain stopped below the prompt-end GDN checkpoint -> cached=0 on the next turn."""
    import types
    from sparsekv import tier_connector as tc
    bs = 864
    con, mgr, _ = _fake_connector(bs=bs)
    pool = _FakeMambaPool(40)
    con._block_pool = pool
    old_rt = tc._RT
    try:
        rt = tc._RT = types.SimpleNamespace(fin_done=set(), mamba_done=set(), mamba_items={})
        P = 10 * bs + 300                      # 10 full prompt blocks
        gblk = pool.get_new_blocks(11)         # the request's GPU blocks (logical 0..10)
        hashes = [("h", i) for i in range(11)]
        req = types.SimpleNamespace(request_id="r", block_hashes=hashes, num_prompt_tokens=P, num_computed_tokens=P,
                                    num_in_flight_tokens=0, num_output_tokens=0)
        con.req_obj["r"] = req
        con.build_connector_meta(_so(new=[("r", 0)]))
        con.build_connector_meta(_so(cached=[("r", 8 * bs)]))   # last scheduled chunk starts at 8*bs: 7 mirrored
        assert set(con.sreq["r"]["host"]) == set(range(7)), con.sreq["r"]["host"]
        req.num_output_tokens = 1                                 # finished after one sampled token
        ids = ([b.block_id for b in gblk],)
        con.request_finished(req, ids)                            # scheduler frees the blocks right after
        assert set(con.hc.h2b) == {("h", i) for i in range(10)}, sorted(con.hc.h2b)  # whole prompt now keyed
        held = {b.block_id for blk in con._fholds.values() for b in blk}
        assert held == {gblk[b].block_id for b in (7, 8, 9)}, held
        pool.free_blocks(gblk)                                    # request's own refs go away
        assert not held & {b.block_id for b in pool.free_q}, "tail blocks re-handed-out before their host copy"
        meta = con.build_connector_meta(_so())
        (cid, gids, hids), = meta.fin_copies
        assert gids == [gblk[b].block_id for b in (7, 8, 9)] and len(hids) == 3
        rt.fin_done.add(cid)                                      # worker enqueued the copy
        con.build_connector_meta(_so())
        assert not con._fholds and held <= {b.block_id for b in pool.free_q}, "hold not released"
        # off switch = old behaviour
        os.environ["SPARSE_TIER_MIRROR_AT_FINISH"] = "0"
        con.hc = tc.HostCache(64)
        con.sreq.clear()
        con.req_obj["r"] = req
        con.build_connector_meta(_so(new=[("r", 8 * bs)]))
        con.request_finished(req, ids)
        assert len(con.hc.h2b) == 7, len(con.hc.h2b)
        os.environ.pop("SPARSE_TIER_MIRROR_AT_FINISH")
        # LRU: an evicted host block must be the deepest one of a released prefix, never block 0
        con.hc = tc.HostCache(8)
        con.sreq.clear()
        req2 = types.SimpleNamespace(request_id="q", block_hashes=[("q", i) for i in range(9)])
        con.req_obj["q"] = req2
        con.build_connector_meta(_so(new=[("q", 8 * bs + 100)]))  # 8 blocks mirrored = the whole host pool
        con._drop_req("q")
        con.hc.alloc(("other", 0))
        assert ("q", 0) in con.hc.h2b and ("q", 7) not in con.hc.h2b, sorted(con.hc.h2b)
    finally:
        tc._RT = old_rt
    print("G-25 finish-time tail mirror (held until copied) + host LRU trims prefixes from the tail: OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL OK")
