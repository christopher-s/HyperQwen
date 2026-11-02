# SparseKV 21b: Beyond-Pool Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the admission path upstream sparsekv designed but never shipped, so the GPU2 sparse engine admits requests whose KV exceeds the GPU pool — 2 concurrent long-context agents (~170–200K each on our RAM budget; 2 × 262K if ComfyUI yields RAM) with KV spilled to pinned host RAM.

**Architecture:** Add `HostTierFullAttentionSpec` + `HostTierFullAttentionManager` (the `tier_admission.py` skeleton, upstream's Track A) so the 16 sparse layers' KV group caps per-request GPU residency at sink + tail + in-flight chunks and everything else lives in the already-ported host tier (HostCache, HostTierSpillPolicy, host-indexed sketch, `gather_rows_tiered`, `stage_prefill` — all present and validated in the port). A 0.32-native vLLM patch emits the spec from `attention.py::get_kv_cache_spec` and registers the manager via `KVCacheSpecRegistry`; a diagnostic task first explains the window session's unexplained 192K stall so the spec work builds on a known-good baseline.

**Tech Stack:** vLLM 0.32.0 `black-monolith` 65-patch series (in-image, `/app/venv/...`), pthfa int8 KV backend (CUSTOM), Triton tier kernels (already in the image), Docker (engine on GPU2 UUID `GPU-2d65d36c-8c19-0b80-0f4a-40743d551530`, port 18025).

**Spec:** `sparsekv-port/upstream/sparsekv/STREAM_tier.md` (sections "Admission: what is implemented, and the full plan", "Stage 4", "Two concurrent long requests", "Host pool sizing") + `sparsekv-port/upstream/sparsekv/tier_admission.py` (the skeleton this plan wires) + `sparsekv-port/PLAN-sparsekv-gpu2.md` Tasks 5–6 (validation gates unchanged) + ROADMAP.md item 21b. 0.32 interface references (verified in-image 2026-11-02): `KVCacheSpecRegistry.register` (kv_cache_spec_registry.py:61), `SlidingWindowSpec.max_admission_blocks_per_request` (kv_cache_interface.py:869), spec emission `attention.py:627` full-attn branch (~:714), `allocate_slots` → `remove_skipped_blocks` ordering (kv_cache_manager.py:509-527), `apply_admission_cap` semantics (kv_cache_coordinator.py:219, single_type_kv_cache_manager.py:210).

## Global Constraints

- **GPU0 untouchable; production `:18021`/`:18020` never stopped.** All boots on GPU2 only (UUID above), port 18025. Verify `nvidia-smi` after every boot.
- **KV precision fixed: int8 everywhere.** Host mirror `SPARSE_TIER_HOST_BITS=8` (2,080 B/token/layer); the int4 sketch is scoring-only (Chris directive 2026-11-02).
- **RAM floor:** host tier sizing must leave `MemAvailable ≥ 8 GB` (the ported `host_pool_blocks()` enforces it; do not raise `SPARSE_TIER_HOST_GB` past ~10.5 GiB while ComfyUI (16 GB) is resident — measured budget 2026-11-02).
- **MTP k3 stays on** in every boot and measurement.
- **Every change to the served tree happens via a numbered patch in `sparsekv-port/patches/`** applied into the container (or baked into a rebuilt image); the upstream tree (`sparsekv-port/upstream/`, PIN `d2d876d`) is reference, never imported.
- **Basis:** `sparsekv-port/basis27-qwopus.pt` (fitted 2026-11-02). Never boot without it.
- **Measurements land in** `~/tools/hyperqwen/bench-results/sparsekv-21b-<date>/`; every task commits.

## Review Focus

1. **Selection quality >128K on x16** (upstream's open suspect): their 177K code-line recall FAILED on the x4 card in 2 of 4 runs; stream B validated retrieval only to 128K. Expect ≥150K code-line recall to pass on our x16 card. → Task 5 Step 2: 192K code-line + needle probes must PASS before any concurrency claim.
2. **Preemption thrash at a full pool** (two big prefills competing): freed blocks can be re-demanded by the other stream's next chunk; upstream saw "1 preemption, resume via host cache works" — expect 0 wedge states and logged, recoverable preemptions. → Task 5 Step 4: force-watermark run must complete both streams with logged preemption counts.
3. **The unexplained window-session stall** (192K sat at `waiting=capacity` with the flag live): porting the spec on top of an undiagnosed stall bakes in a bug. → Task 1 diagnoses it first with `SPARSE_TIER_DEBUG=1`; Task 3 Step 3 proves 192K admits pre-spec as the control.
4. **MTP verify + drafter at long range:** the drafter layer stays a plain FullAttentionSpec group (dense KV, capped by the pool); drafts at long range degrade but targets must stay exact. → Task 5 Step 5: MTP acceptance within 0.05 of the dense same-prompt run; zero non-stop finishes.
5. **RAM overshoot against ComfyUI:** the sketch grows with the host pool (4.2 KB/token GPU) and the mirror pins host RAM at boot. Expect `MemAvailable ≥ 8 GB` and pool shrinkage accounted (sketch 1.36 GiB at 339K tokens, measured arithmetic). → Task 2 Step 4: boot log must show host pool GB, sketch GiB, and `free -g ≥ 8` during the concurrency smoke.

---

### Task 1: Diagnostic window — explain the 192K stall on the current (pre-spec) stack

**Files:**
- Create: `~/tools/hyperqwen/bench-results/sparsekv-21b-<date>/DIAG.md`
- Create: `~/tools/hyperqwen/sparsekv-port/patches/21b-diag-README.md` (what was observed, no code changes unless the diagnosis lands in our patches)

**Interfaces:**
- Consumes: the working 128K boot recipe (wrapper `~/bin/qwen38-sparse-gpu2-on`, patches `apply-in-container-guard-bypass.py` + `apply-in-container-alloc-arm.py`).
- Produces: `DIAG.md` naming the exact mechanism that kept a 192K prompt at `waiting=capacity` with `--no-scheduler-reserve-full-isl` live; Task 3's control depends on it.

- [ ] **Step 1: Boot the sparse engine at 131072 with debug on**

Apply the two in-container patches, restart, and wait for health, exactly as the wrapper documents, plus `-e SPARSE_TIER_DEBUG=1 -e SPARSE_TIER_VERIFY=0`. Capture the full log to `bench-results/sparsekv-21b-<date>/boot-diag.log`. Expected lines: `tier: runtime mode=...` (TierRuntime init), host pool GB, `switch -> SPARSE`.

- [ ] **Step 2: Fire the 192K probe and watch admission**

Send the `needle_doc.txt` 192K request (the window session's exact probe) with the client in the background writing to `diag192.json`. Poll `/metrics` `num_requests_running/waiting` every 10 s for 8 min into `diag-metrics.log`.

- [ ] **Step 3: Read the evidence and name the mechanism**

Check, in order: (a) does any `tier:` log line appear after the request arrives (spill frees? mirror DMA?); (b) does `num_requests_running` ever go 1 (prefill started) or stay 0; (c) at the moment of stall, what does `docker logs` show for the last `Avg prompt throughput` line. Record the mechanism (candidates from the 2026-11-02 analysis: MTP lookahead slots vs chunk alloc at pool pressure; TierRuntime silent init failure; mamba align-chunk reservation inflation). If the fix is one env knob or a small patch to OUR patches, apply it, re-run Step 2, and expect: prompt admitted (`running=1`), prefill completes, needle PASS.

- [ ] **Step 4: Commit the diagnosis**

```bash
cd ~/tools/hyperqwen && git add bench-results/sparsekv-21b-*/ sparsekv-port/patches/21b-diag-README.md
git commit -m "sparsekv 21b task1: diagnose the 192K admission stall pre-spec"
```

### Task 2: Host-pool + GPU-budget sizing at our RAM numbers

**Files:**
- Modify: `~/bin/qwen38-sparse-gpu2-on` (final `MAX_LEN`, `SPARSE_TIER_HOST_GB`, `SPARSE_TIER_HOT_FRAC` values from this task)
- Create: `~/tools/hyperqwen/bench-results/sparsekv-21b-<date>/SIZING.md`

**Interfaces:**
- Consumes: Task 1's working 192K boot; the ported `host_pool_blocks()` RAM guard.
- Produces: the single canonical boot config later tasks use: `MAX_LEN`, `MAX_SEQS=2`, `SPARSE_TIER_HOST_GB`, `SPARSE_TIER_HOT_FRAC`, `SPARSE_TIER_WATERMARK_BLOCKS` (if overridden).

- [ ] **Step 1: Fix the target config by arithmetic (no boot)**

From the measured budget (45 GB box; ComfyUI 16; engines ~7.5; OS 3; floor 8 → ~10.5 GiB tier): host pool 10.5 GiB = 339K tokens = 2 × 169K, sketch cost 1.36 GiB GPU. Write `SIZING.md` with the chosen ceiling and its arithmetic: **`MAX_LEN=196608` (= 192 × 1024, their proven max minus the >128K selection risk is respected by the Task 5 Step 2 gate)**, `SPARSE_TIER_HOST_GB=10`, `SPARSE_TIER_HOT_FRAC=0.10` (x16 link needs less than their x4 0.15–0.20 recommendation; 0.15 is the recorded fallback if misses exceed 16%). Record every number with its source (this plan's Global Constraints + measured pool 7.44 GiB).

- [ ] **Step 2: Boot at the chosen config, assert the log contract**

Wrapper values updated; boot; health. Assert in the log: host pool line ≈ 10 GB, sketch cache lines (16 layers), no `allocated > reserved`, `switch -> SPARSE` present after a probe. `free -g` shows available ≥ 8.

- [ ] **Step 3: Negative test — the RAM guard fires**

Boot once with `SPARSE_TIER_HOST_GB=30` (deliberate overshoot). Expected: the `host_pool_blocks()` MemoryError refuses startup with the clear message; container exits; record the line. Revert to Step 2's config.

- [ ] **Step 4: Commit**

```bash
cd ~/tools/hyperqwen && git add -f bench-results/sparsekv-21b-*/SIZING.md && git add ~/bin/qwen38-sparse-gpu2-on sparsekv-port/ 2>/dev/null; git commit -m "sparsekv 21b task2: 192K sizing - 10 GiB host tier, hot 0.10, RAM guard verified"
```

### Task 3: 192K control — spill working pre-spec (the admission baseline)

**Files:**
- Create: `~/tools/hyperqwen/bench-results/sparsekv-21b-<date>/CONTROL.md`, `control192.json`, `control2x.json`

**Interfaces:**
- Consumes: Task 1 diagnosis + Task 2 config.
- Produces: the measured baseline the spec-based admission must beat or match: single 192K needle PASS + one forced-pressure two-stream run.

- [ ] **Step 1: Single 192K needle on the configured boot**

Reuse the Task 1 probe; `needle_retrieved=true`, `finish=stop`. Record wall time and tier hit/miss lines (expect ~9–16% miss at hot 0.10 per upstream's real traces).

- [ ] **Step 2: Multi-turn follow-ups on the 192K context**

Three sequential questions on the same document (their `run_tier` host-prefix pattern). Expected per upstream: TTFT ≤ ~5 s after the cold turn, needle/code/needle2 PASS (host prefix cache; our port has `SPARSE_TIER_HOST_PREFIX=1` default).

- [ ] **Step 3: Two-stream forced-pressure run**

Two different ~150K documents, two clients, watermark forced to 1 (`SPARSE_TIER_WATERMARK_BLOCKS=1` boot variant) to trigger spill early. Expected per upstream conc2–conc5: both PASS (code-line on the second stream is the known-flaky one — record honestly), 0–1 logged preemptions, follow-up TTFTs single-digit seconds.

- [ ] **Step 4: Commit the control record**

```bash
cd ~/tools/hyperqwen && git add -f bench-results/sparsekv-21b-*/ && git commit -m "sparsekv 21b task3: 192K control - spill+host-cache baseline measured"
```

### Task 4: Wire the spec — `HostTierFullAttentionSpec` + manager registration on 0.32

**Files:**
- Create: `~/tools/hyperqwen/sparsekv-port/sparsekv/tier_admission_032.py` (our 0.32 adaptation of the skeleton: `make_spec_class()` + `make_manager_class()` bodies filled in)
- Create: `~/tools/hyperqwen/sparsekv-port/patches/21b-host-tier-spec-032.patch` (vLLM tree: spec emission + registration + startup-check respect)
- Create: `~/tools/hyperqwen/sparsekv-port/sparsekv/tests/test_admission_cpu.py`
- Modify: `~/tools/hyperqwen/sparsekv-port/patches/` apply script (installs the new patch into a container)

**Interfaces:**
- Consumes: skeleton API (`make_spec_class`, `make_manager_class`, `max_admission_blocks_per_request` semantics), 0.32 interfaces verified in-image 2026-11-02: `KVCacheSpecRegistry.register(spec_cls, manager_cls, uniform_type_base_spec=FullAttentionSpec)`, `FullAttentionManager` (single_type_kv_cache_manager.py:738), `SlidingWindowSpec.max_admission_blocks_per_request` pattern (kv_cache_interface.py:869), spec emission at `attention.py::get_kv_cache_spec` full-attn branch (~line 714), `remove_skipped_blocks` called from `allocate_slots` before counting (kv_cache_manager.py:516), coordinator `get_num_blocks_to_allocate(apply_admission_cap=True)` only on the full-sequence path (kv_cache_coordinator.py:219), startup check `_check_enough_kv_cache_memory` (kv_cache_utils.py:1003, already skipped by our kvcheck patch).
- Produces: a boot where `max_model_len=196608` plans a KV pool sized for the capped admission (sink+tail+in-flight per request), per-request admission uses the cap, and the MTP drafter layer stays a plain `FullAttentionSpec` group. All manager/connector method names match the skeleton exactly.

- [ ] **Step 1: Write the failing CPU test**

`test_admission_cpu.py`: instantiate the spec (`block_size=864`, sink 64, tail 1092); assert `max_admission_blocks_per_request(max_in_flight_tokens=2048, max_model_len=196608) == 6` (= cdiv(64,864)=1 sink + cdiv(1092+2048,864)=4 tail + 1 straddle; 6 × 864 = 5,184 tokens) and `max_memory_usage_bytes` equals `6 * page_size_bytes`; assert the manager's `get_num_blocks_to_allocate` caps at 6 under `apply_admission_cap=True` and does not cap with `False`.

- [ ] **Step 2: Run it — expect ImportError/NotImplementedError**

Run: `docker exec hyperqwen-sparse-gpu2 /app/venv/bin/python -m pytest /basis/sparsekv/tests/test_admission_cpu.py -v` (against the ported tree, pre-implementation).
Expected: FAIL (module incomplete).

- [ ] **Step 3: Implement the spec class (fill `make_spec_class`)**

The skeleton's formula is the spec; port it verbatim with `tail_tokens = SPARSE_WIN(1024) + verify(k=3 → 4) + slack(64) = 1092`. Keep the dataclass frozen and `kw_only=True`; note the `__eq__`-by-class property that keeps the drafter in its own group.

- [ ] **Step 4: Implement the manager (fill `make_manager_class` bodies)**

Per skeleton docstrings: `get_num_blocks_to_allocate` (GPU need via super + host need + sentinel on host-pool-short), `allocate_new_blocks` (pairs queue), `remove_skipped_blocks` (spill window `[cdiv(sink,BS), (processed-tail)//BS)` pressure-gated), `find_longest_cache_hit` (GPU hit then host-hash extension), `free`, `take_block_table_updates`. Keep `bind_host_pool` raising (the ported connector's `HostCache` already owns host blocks — the manager consumes `(gpu, host)` pairs from it; delete the unused method rather than double-owning). Run Step 1's test: PASS.

- [ ] **Step 5: The vLLM patch — emission + registration**

Patch `attention.py::get_kv_cache_spec` full-attn branch: when `sparsekv` env `SPARSE_KV=1` + `SPARSE_TIER=1` + the layer is in the sparse group (16 FA target layers; the drafter layer keeps plain spec — detect via the existing backend's layer set, same mechanism `group_of()` uses in `sparse_kv.py`), return `HostTierFullAttentionSpec(...)`. Registration happens in the connector's `__init__` (scheduler+worker both): `KVCacheSpecRegistry.register(spec_cls, manager_cls, uniform_type_base_spec=FullAttentionSpec)` before model load. Patch must be a no-op without the env knobs (inertness invariant).

- [ ] **Step 6: Boot + group-shape assertion**

Boot at the Task 2 config. Assert in the log: the sparse group's spec type is `HostTierFullAttentionSpec`; pool sizing used the capped spec (GPU pool GROWS vs 7.44 GiB — the pool no longer reserves for 196K × 2); the MTP drafter group is unchanged; no `check_kv_cache_spec_registry` assertion. Run the 128K smoke (greedy + facts list): PASS.

- [ ] **Step 7: The 192K admission test, now via the spec**

Send the 192K needle. Expected: admitted immediately (running=1), needle PASS. Record pool usage % during prefill (`/metrics` `gpu_cache_usage`): the spec's promise is usage stays low while host carries the bulk.

- [ ] **Step 8: Commit**

```bash
cd ~/tools/hyperqwen && git add sparsekv-port/ && git commit -m "sparsekv 21b task4: HostTierFullAttentionSpec wired on 0.32 - admission capped at sink+tail+in-flight"
```

### Task 5: Validation battery (gate) at 2-long-stream

**Files:**
- Create: `~/tools/hyperqwen/bench-results/sparsekv-21b-<date>/RESULTS.md`, raw subdirs `kld/ conc/ soak/`

**Interfaces:**
- Consumes: Task 4 engine; `PLAN-sparsekv-gpu2.md` Task 5 harness (ported, unchanged); Task 2 config.
- Produces: the go/no-go record for ROADMAP 21b. **Gate thresholds (unchanged from the plan):** KLD(top-1, MTP-on, thinking-on arm) ≥ 95% at 128K/192K; MTP acceptance ≥ 0.45; 2-stream combined ≥ 1.6× single-stream dense at matched context; zero unrecoverable preemptions in the 2 h soak; `MemAvailable ≥ 8 GB` throughout; >150K code-line recall PASS (Review Focus 1).

- [ ] **Step 1: KLD matrix, MTP-on, thinking arms (32/64/128/192K)**

Run the ported harness (Task 5 of the original plan). Expected per upstream: top-1 ≥ 95.6% class at ≤128K; record 192K thinking-on (no upstream prior exists — Review Focus 1).

- [ ] **Step 2: 192K+ recall battery (the x16 selection-quality question)**

Needle @30%/code-line @60%/needle2 @75% on 177K and on 192K docs, 3 reps each. Gate: all PASS on x16 (their x4 failed 2 of 4 at 177K — if ours fails, STOP and record; the feature is not shippable at this context).

- [ ] **Step 3: Two-stream 2 × ~165K independent documents**

Conc ladder (existing `bench/conc_ladder.py` protocol) + the Task 3 Step 3 pattern at real scale: per-stream tok/s, combined vs single-stream dense at matched context (1.6× gate), follow-up TTFTs, preemption log.

- [ ] **Step 4: 262K capacity probe (single stream)**

One 262K prefill, needle @90%. Expected: PASS (RoPE cap = 262,144); record TTFT and spill stats. This is capacity-plus-recall, matching the original plan's framing.

- [ ] **Step 5: 2 h two-stream soak, thinking on, temp 1.0, effort medium**

Two agents on ~150K+ contexts alternating turns. Watch: 0 non-stop finishes, 0 unterminated `<think>`, MTP acceptance within 0.05 of the dense run, `MemAvailable ≥ 8 GB`, tier miss rate stable (~9–16% band).

- [ ] **Step 6: RESULTS.md + roadmap update + commit/push**

Gate verdict per threshold; GPU0-untouched attestation; update ROADMAP 21b checkbox with the measured outcome and the 21a-merge decision (the baked image now carries both). Push.

### Task 6: Bake and present (adoption is Chris's, per plan Task 7)

**Files:**
- Create: variant image `hyperqwen-pthfa:032-sparse-port-21b` (docker commit of the fully-patched container)
- Modify: `~/bin/qwen38-sparse-gpu2-on` (image name, drop the manual patch steps)
- Modify: `ROADMAP.md` item 21b status line

- [ ] **Step 1: Bake the image, boot from it clean-room, smoke**

New container from the committed image, NO manual patch steps, boot → health → 3-probe smoke + one 192K needle. PASS required (the "no manual patching" promise verified).

- [ ] **Step 2: Stop the engine, restore production GPU2, present the record to Chris**

`docker rm -f` sparse; `qwen38-qwopus-on gpu2`; health. Then the Task 7 decision framing from `PLAN-sparsekv-gpu2.md` (adopt / register-as-option / decline), now with 21b numbers.

## Rollback

- The vLLM patch is inert without `SPARSE_KV=1 SPARSE_TIER=1` (assert in Task 4 Step 6 by booting once with knobs off and diffing greedy outputs against the 128K smoke).
- Every boot config lives in git (`~/bin/qwen38-sparse-gpu2-on` + patches); production is never stopped before Task 6 Step 2.
- Worst case: `docker rm -f hyperqwen-sparse-gpu2` and the box is exactly as before 21b.
