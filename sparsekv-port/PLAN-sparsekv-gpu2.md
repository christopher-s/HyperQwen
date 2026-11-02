# SparseKV GPU2 Two-Stream 262K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port bradj00/sparsekv-hyperqwen (query-aware sparse KV decode + int4 host-RAM KV tier) onto the GPU2 qwopus-heretic engine so it serves **2 concurrent streams at 262,144-token context** (the model's native max) instead of today's 1 × 200K, with GPU0 untouched.

**Architecture:** A separate build venv carries the upstream sparsekv patches (rebased onto our 0.32 65-patch `black-monolith` series) plus a new Triton ops module and a host-tier KV connector. The sketch basis is refit on qwopus-heretic keys (the released `basis27.pt` was fitted on stock Qwen3.8-27B). A new tray-adjacent wrapper boots the sparse engine on a dedicated port (:18025) behind an optional router alias, and a gated A/B validates quality (KLD vs dense, WITH MTP on) and throughput (2-stream conc ladder) against the current dense engine before any swap.

**Tech Stack:** vLLM 0.32.0 (hyperqwen `black-monolith` @ c73dd35), pthfa int8 KV (PTH_FA), Triton 3.x kernels, PyTorch pinned-host UVA reads, lm-eval-free custom KLD harness (ported from `eval/kld/`).

**Spec:** roadmap item 21 in `~/tools/hyperqwen/ROADMAP.md` (the review record with the four fit blockers this plan resolves); upstream repo github.com/bradj00/sparsekv-hyperqwen @ `main` (2 commits); companion `sparsekv-lab` for design notes.

## Global Constraints

- **GPU0 is untouchable.** No command in this plan may target GPU0, stop `hyperqwen-qwopus-pthfa-gpu0`, or bind ports 18020/8020's GPU0 lane. Every build, boot, and benchmark runs against GPU2 (UUID `GPU-2d65d36c-8c19-0b80-0f4a-40743d551530`, PCI 0B:00.0, x16). Verify with `nvidia-smi` after every GPU-adjacent step.
- **GPU2 stays in production until Task 7.** The current dense engine (`:18021`, MAX_SEQS=1, MAX_LEN=200000) serves traffic. The sparse engine boots on **:18025** for all validation. Nothing on :18021 stops until the A/B gate passes and Chris approves the swap.
- **Scope: 2 streams only.** `MAX_SEQS=2`, `MAX_LEN=262144`, GPU_UTIL 0.94. No 4-stream configs, no GPU0 replication, no tray integration in this plan (tray integration is a follow-up after Chris adopts it).
- **MTP stays on.** Production parity requires SPEC=mtp k3. Their KLD numbers ran MTP-off; Task 6 re-measures with MTP on, which is the number that matters for our serving state.
- **Host RAM budget: ≤ 20 GB pinned.** 2 × 262,144 tokens × ~17.8 KB/token ≈ 9.3 GB tier + working slack. `SPARSE_TIER_HOST_GB=12` is the working value (headroom without starving ComfyUI's 16 GB resident). Boot must fail loudly if pinned allocation would push `MemAvailable` below 8 GB.
- **Every measurement lands in** `~/tools/hyperqwen/bench-results/sparsekv-gpu2-<date>/`.
- Model under test: `/mnt/500gb-ssd/ai/models/qwopus-heretic-W4A16-AutoRound` (same weights as production; the plan does not change weights).

## Review Focus

1. **Thinking-mode interaction (highest risk).** Their whole quality study ran thinking-off, temp 0. Our production traffic thinks (medium default, xhigh available) at temp 1.0. Sparse selection could interact with long reasoning chains differently than with 1024-token answers. → Task 6 Step 4 adds a thinking-on KLD arm and a long-form generation probe; the gate requires thinking-on numbers, not just their protocol's numbers.
2. **MTP draft acceptance under sparse verification.** The MTP drafter stays dense but the verifier attends sparsely; acceptance rate regression would silently erase the throughput win. → Task 6 Step 5 records MTP acceptance (today's baseline: 0.60–0.69) as a first-class gate metric.
3. **262K > their 250K ceiling.** All their measured configs top out at 250K contexts; KLD validation only reached 192K. The plan sizes for 262K but the quality gate samples at 192K (their proven max) plus one 262K capacity probe. → Tasks 5–6.
4. **Port-regression risk in the 65-patch series.** Their scheduler monkeypatches assume 0.31 internals. Any silently-wrong rebase corrupts output rather than crashing. → Task 4 Step 3 runs the dense-control (no `SPARSE_*`) battery from their `run_release/dense-control.sh` before any sparse boot; sparse boots are invalid until dense-on-new-venv matches dense-on-production within the established boot-variance band.
5. **RAM exhaustion killing other services.** Pinning 12 GB on a 45 GB box that also runs ComfyUI (16 GB), two engines, and the desktop can trigger the OOM killer against an innocent process. → Task 3 Step 2 adds the loud-fail guard; Task 6 Step 1 verifies `MemAvailable ≥ 8 GB` throughout the 2-stream soak.

---

### Task 1: Stage the upstream tree and isolate the build

**Files:**
- Create: `~/tools/hyperqwen/sparsekv-port/upstream/` (git clone of bradj00/sparsekv-hyperqwen, pinned to `main` HEAD at port time)
- Create: `~/tools/hyperqwen/sparsekv-port/NOTES.md` (port journal: every conflict, every decision)
- Create: venv `~/tools/hyperqwen/sparsekv-port/venv` (clone of the current serving venv recipe at 0.32.0 + `black-monolith` @ c73dd35 + 65 patches + pthfa, WITHOUT sparse patches)

**Interfaces:**
- Produces: `sparsekv-port/venv` — the base build every later task patches; `NOTES.md` — the conflict log Tasks 2–4 append to.

- [ ] **Step 1: Clone and pin upstream**

```bash
mkdir -p ~/tools/hyperqwen/sparsekv-port && cd ~/tools/hyperqwen/sparsekv-port
git clone https://github.com/bradj00/sparsekv-hyperqwen upstream
cd upstream && git rev-parse HEAD > ../UPSTREAM_PIN && cd ..
```

Record the hash in `NOTES.md`.

- [ ] **Step 2: Build the base venv on OUR stack**

Build the serving venv recipe (0.32.0 + 65 patches @ c73dd35 + pthfa int8 KV plugin) into `sparsekv-port/venv`, reusing the image build tooling already in the hyperqwen repo (the same steps that produce `hyperqwen-pthfa:032-b`, run in a venv instead of a container). Verify: `python -c "import vllm; print(vllm.__version__)"` prints 0.32.0 and `import pth_flash_attn` succeeds.

- [ ] **Step 3: Verify the base venv serves dense on GPU2**

Boot with the production flag set minus sparse (MAX_LEN=200000, MAX_SEQS=1, PTH_FA=1, SPEC=mtp, INT8_ACT=int8, CUDAGRAPH_MODE=PIECEWISE, GPU_UTIL=0.94, PORT=18025), pin `CUDA_VISIBLE_DEVICES` to GPU2's UUID. Health-check `:18025`, run one greedy arithmetic probe. Expected: clean answer, KV pool ≈ production's 216K. Stop the server. **GPU0 check: `nvidia-smi` shows no process on GPU0.**

- [ ] **Step 4: Commit the stage**

```bash
cd ~/tools/hyperqwen && git add sparsekv-port/UPSTREAM_PIN sparsekv-port/NOTES.md
git commit -m "sparsekv port: stage upstream tree + base venv (GPU2-only)"
```

### Task 2: Refit the sketch basis on qwopus-heretic keys

**Files:**
- Create: `~/tools/hyperqwen/sparsekv-port/basis27-qwopus.pt`
- Modify: `~/tools/hyperqwen/sparsekv-port/NOTES.md` (fit stats)

**Interfaces:**
- Consumes: upstream `sparsekv/tools/fit_basis27.py` + `q27loader.py`, NIAH sample JSONL (theirs expects `NIAH_JSONL`; generate from our long-context probe corpus), `MODEL_DIR=/mnt/500gb-ssd/ai/models/qwopus-heretic-W4A16-AutoRound`.
- Produces: `basis27-qwopus.pt` — rank-64 basis for Tasks 3–6; fit-metrics block for the A/B record.

- [ ] **Step 1: Generate the fitting corpus**

Build a NIAH-style JSONL from our own traffic shape: 3 long documents (2 code: vLLM source dump, hyperqwen patch series; 1 prose), cut at 32K/64K/128K/192K, one recall task each. Store under `sparsekv-port/fitdata/`.

- [ ] **Step 2: Run the fit on GPU2**

```bash
MODEL_DIR=/mnt/500gb-ssd/ai/models/qwopus-heretic-W4A16-AutoRound \
NIAH_JSONL=~/tools/hyperqwen/sparsekv-port/fitdata/samples.jsonl \
WORK=~/tools/hyperqwen/sparsekv-port/work OUT=~/tools/hyperqwen/sparsekv-port/basis27-qwopus.pt \
~/tools/hyperqwen/sparsekv-port/venv/bin/python upstream/sparsekv/tools/fit_basis27.py
```

Expected: basis file ~4.2 MB. **GPU0 must stay idle throughout (verify once mid-run).**

- [ ] **Step 3: Record fit stats and commit**

Log singular-value decay and reconstruction error into `NOTES.md`; compare against any stats their repo records for basis27.pt. Commit basis + notes.

```bash
cd ~/tools/hyperqwen && git add sparsekv-port/ && git commit -m "sparsekv port: refit sketch basis on qwopus-heretic keys"
```

### Task 3: Port the Triton ops, tier connector, and pthfa hook onto 0.32

**Files:**
- Create: `~/tools/hyperqwen/sparsekv-port/sparsekv/` (ported module: `triton_ops.py`, `tier.py`, `tier_connector.py`, `ref.py`, `install.sh`, `tests/`)
- Create: `~/tools/hyperqwen/sparsekv-port/patches/sparse-switch-dual-graphs-032.patch` (rebased)
- Create: `~/tools/hyperqwen/sparsekv-port/patches/sparse-tier-spill-kvcheck-032.patch` (rebased)
- Create: `~/tools/hyperqwen/sparsekv-port/patches/hyperqwen-base-032.diff` (rebased registration diff)
- Create: `~/tools/hyperqwen/sparsekv-port/patches/pthfa-sparse-kv-032.patch` (decode hook rebased onto our pthfa)
- Modify: `~/tools/hyperqwen/sparsekv-port/venv/` (apply via their `install.sh` pattern, adapted)
- Test: `~/tools/hyperqwen/sparsekv-port/sparsekv/tests/` (their tests, ported) + one new RAM-guard test

**Interfaces:**
- Consumes: Task 1's venv, Task 2's basis, upstream module sources.
- Produces: an installable module + patch set whose public surface (env knobs `SPARSE_PROFILE`, `SPARSE_SKETCH_BITS`, `SPARSE_TIER_HOST_BITS`, `SPARSE_TIER_HOST_GB`, `SPARSE_TIER_HOT_FRAC`, `SPARSE_TIER_SPILL`, `SPARSE_STATS_SUB`, `SPARSE_SELECT_EVERY`, `SPARSE_MEM_RESERVE`) matches upstream names exactly, so their launcher patterns and README commands port verbatim.

- [ ] **Step 1: Port the unit tests first (TDD)**

Copy `sparsekv/test_*.py` into the port tree; adapt imports. Run against `ref.py` only (pure PyTorch, no vLLM): `venv/bin/python -m pytest sparsekv/tests/ -v`. Expected: PASS on the reference implementations before any vLLM patching exists. Fix only import-path breakage in this step; behavior changes are Task 4 findings, not silent edits.

- [ ] **Step 2: Port `triton_ops.py` and `tier.py`/`tier_connector.py`**

Copy the module; adjust for 0.32 API changes the unit tests expose. New guard in `tier.py` (their code has none): at tier allocation, if projected pinned bytes would leave `MemAvailable < 8 GB`, raise instead of pinning.

```python
def _assert_ram_headroom(pinned_bytes_needed: int, floor_gb: float = 8.0) -> None:
    avail = _mem_available_bytes()
    if avail - pinned_bytes_needed < floor_gb << 30:
        raise RuntimeError(
            f"sparsekv tier: refusing to pin {pinned_bytes_needed>>30} GiB; "
            f"would leave < {floor_gb} GiB MemAvailable (now {avail>>30} GiB)"
        )
```

- [ ] **Step 3: Rebase the three vLLM patches + pthfa hook**

`git apply --3way` each upstream patch against the 0.32 tree; resolve conflicts in `NOTES.md` (expected hot spots: `v1/core/kv_cache_utils.py` and the scheduler — both files our series also touches). The pthfa `sparse_kv.py` decode hook ports onto our pthfa plugin's current source. Every conflict resolution is one commit with the conflict noted.

- [ ] **Step 4: Install into the port venv and verify inertness**

```bash
bash sparsekv-port/sparsekv/install.sh sparsekv-port/venv --basis sparsekv-port/basis27-qwopus.pt
```

Then boot dense (no `SPARSE_*` knobs) on :18025 and run their `run_release/dense-control.sh` battery. Expected: bit-identical behavior to Task 1 Step 3's boot (same greedy outputs). This proves the patched tree is inert with knobs off — the property that makes rollback trivial.

- [ ] **Step 5: Commit the port**

```bash
cd ~/tools/hyperqwen && git add sparsekv-port/ && git commit -m "sparsekv port: module + patches rebased onto 0.32 black-monolith"
```

### Task 4: Boot the sparse engine at the target config

**Files:**
- Create: `~/bin/qwen38-sparse-gpu2-on` / `-off` / `-status` (wrappers, mirroring `qwen38-qwopus-on` conventions: UUID-pinned, :18025, loud port-conflict failure)
- Create: `~/tools/hyperqwen/sparsekv-port/serve-sparse-gpu2.sh` (the boot command with all knobs)
- Test: boot-log assertions (host pool line, GPU pool blocks, no `allocated > reserved`)

**Interfaces:**
- Consumes: Task 3's installed venv + basis.
- Produces: a running sparse engine on :18025 with `SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=2 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4 SPARSE_TIER_HOST_BITS=4 SPARSE_TIER_HOST_GB=12 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 SPARSE_MEM_RESERVE=1 SPARSE_TIER_SPILL=1`, serving model id `qwen3.8-27b`.

- [ ] **Step 1: Write the boot wrapper**

Same skeleton as `~/bin/qwen38-qwopus-on` (gpu2 branch): GPU UUID pin, port 18025, volume-mount the model read-only, pass the `SPARSE_*` knobs. Wrapper refuses to start if :18025 is occupied and never touches :18020/:18021 containers.

- [ ] **Step 2: Boot and assert the log contract**

Boot; grep the log for `host pool 11.7 GB`-class line, `GPU pool` block count, and absence of `allocated > reserved`. Confirm `/v1/models` on :18025 serves `qwen3.8-27b`. Run the 3-probe smoke: greedy arithmetic, facts list, one 150K needle query with `SPARSE_CUTOFF` unset. Expected: all clean, tier stats lines show `hot[hit, miss]` counters moving.

- [ ] **Step 3: Two-stream concurrency smoke**

Fire 2 concurrent 100K-context recall requests (prefill one document, then two different needles from two clients). Expected: both complete; neither serializes behind the other beyond prefill; tier spill counters show host reads. Record wall times.

- [ ] **Step 4: RAM and GPU0 verification**

During the concurrency smoke: `free -g` shows `available ≥ 8`; `nvidia-smi` shows GPU0 with no compute processes. Log both into the record.

- [ ] **Step 5: Commit**

```bash
cd ~/tools/hyperqwen && git add -A sparsekv-port/ && git commit -m "sparsekv: GPU2 sparse engine boots at 2x262K target config"
```

### Task 5: Port the KLD quality harness

**Files:**
- Create: `~/tools/hyperqwen/sparsekv-port/eval/kld/` (their `kld_proc.py`, `build_samples.py`, `kld_client.py`, `kld_analyze.py`, adapted; launch scripts rewritten for our wrapper)
- Modify: `~/tools/hyperqwen/sparsekv-port/NOTES.md` ( deviations log)

**Interfaces:**
- Consumes: Task 3 venv, Task 4 engine, Task 2 basis.
- Produces: a runnable `run_kld.sh` emitting `results/results.md` + `results.json` in their schema, so Task 6 numbers are comparable to the upstream published tables.

- [ ] **Step 1: Port the harness scripts**

Adapt `kld_serve.sh` configs: `ref` = our dense production-style server (MTP ON, unlike their ref — noted deviation), `tier` = Task 4's engine. Keep their logits-processor interface (`--logits-processors kld_proc:KLDRecorder`); verify it loads on 0.32 (they flagged vLLM rejects custom logits processors under spec decode — if 0.32 still rejects it under MTP, implement the Task 6 MTP-on quality arm as the toggle-dense/sparse comparison instead and record the deviation).

- [ ] **Step 2: Build samples at our lengths**

Their 4 lengths (32/64/128/192K) × 5 docs × 3 tasks, using the Task 2 corpus. One added 262K capacity sample (needle at 90% depth, recall task) — quality claim for 262K is capacity-plus-recall, with the full KLD matrix staying ≤192K (their dense-reference ceiling).

- [ ] **Step 3: Commit the harness**

```bash
cd ~/tools/hyperqwen && git add sparsekv-port/eval/ && git commit -m "sparsekv: KLD harness ported for 0.32 + MTP-on deviation"
```

### Task 6: Run the validation battery (the gate)

**Files:**
- Create: `~/tools/hyperqwen/bench-results/sparsekv-gpu2-<date>/RESULTS.md` (the record)
- Create: `~/tools/hyperqwen/bench-results/sparsekv-gpu2-<date>/{kld,conc,soak}/` (raw)

**Interfaces:**
- Consumes: Task 4 engine, Task 5 harness, production engine :18021 as the dense reference.
- Produces: the go/no-go dataset. **Gate thresholds (set before running):** KLD(top-1, MTP-on) ≥ 95% at 128K/192K; MTP acceptance ≥ 0.45 (production floor today ~0.60; large drop = verifier interaction); 2-stream combined decode ≥ 1.6× single-stream dense; zero preemptions in a 2 h soak; `MemAvailable ≥ 8 GB` throughout.

- [ ] **Step 1: KLD matrix, MTP-on**

Run the Task 5 harness: sparse vs dense on GPU2, all lengths, both thinking-off and thinking-on arms (chat template kwargs; effort medium). Compare against the same-card dense reference. Expected per their data: top-1 ≥ 95.6% off-thinking; record whatever thinking-on shows (no upstream prior exists).

- [ ] **Step 2: MTP acceptance under sparse verification**

Read `spec_decode` counters from :18025 `/metrics` during a 20K-token generation batch. Compare with the production engine's counters under identical prompts. Record accept length and acceptance rate per 5K segment.

- [ ] **Step 3: Two-stream throughput**

Seeded conc ladder (the 2026-10-31/11-01 protocol) at 2 × 262K-capable config vs the production single-stream engine at 200K: per-stream tok/s, combined tok/s, TTFT for second-stream arrival mid-first-stream-generation. Record against the gate's 1.6× combined target.

- [ ] **Step 4: 262K capacity + recall probe**

One 262K prefill (their cold-prefill expectation: ~390 s at 244K; ours at 262K + slab), then needle at 90% depth + code-line recall. Expected: exact retrieval; note prefill wall time and tier spill stats.

- [ ] **Step 5: 2-hour two-stream soak with thinking on**

Two agents, 200K+ contexts, alternating turns at temp 1.0, effort medium (production shape). Watch: 0 preemptions, 0 empty outputs, 0 non-stop finishes, no unterminated `<think>` blocks, `MemAvailable ≥ 8 GB`, tier miss rate stable (their number: 14–16%).

- [ ] **Step 6: Write RESULTS.md and commit**

Gate verdict per threshold, all raw numbers, GPU0-untouched attestation (nvidia-smi history from each task), rollback notes. Commit + push.

```bash
cd ~/tools/hyperqwen && git add bench-results/sparsekv-gpu2-*/ && git commit -m "sparsekv GPU2 validation battery: results + gate verdict"
```

### Task 7: Adoption decision point (Chris gate — nothing automatic)

**Files:**
- Modify: `~/tools/hyperqwen/ROADMAP.md` (item 21 status update: measured-on-our-rig numbers replace review-only)
- Modify: `~/tools/hyperqwen/sparsekv-port/NOTES.md` (final port state)

- [ ] **Step 1: Present the record to Chris**

RESULTS.md verdict, with the decision framed as: adopt sparse as the new GPU2 default (2 × 262K), keep dense-1×200K as default and register sparse as a tray-selectable profile, or decline with numbers.

- [ ] **Step 2: Execute only the approved branch**

- **Adopt:** stop :18021 dense engine, boot sparse on :18021 (router alias unchanged; Hermes config gains `context_length: 262144` if adopting the new ceiling), update tray label to `Qwopus-Heretic ☢ · W4A16 · Q8 KV · Sparse-262K ×2` (or per Model Label Standard), update skill reference, retire :18025.
- **Register-as-option:** keep both engines; sparse stays on :18025 behind a router alias (`qwen-gpu2-sparse`); tray gains a GPU2 entry.
- **Decline:** `install.sh --uninstall`, delete port venv, roadmap item updated with the measured refusal numbers.

- [ ] **Step 3: Post-adoption verification (if adopted)**

`hermes -p` probe through the router, two concurrent sessions live, GPU0 engine still up with its own traffic unaffected, `systemctl --user is-active hyperqwen-router` untouched.

---

## Rollback (every state, all tasks)

- Port venv is inert without `SPARSE_*` knobs (Task 3 Step 4 proves it); `install.sh --uninstall` reverses to byte-identical pre-install.
- Production :18021 is never stopped before Task 7, so "rollback" before then = stop :18025.
- After adoption: dense rollback = `qwen38-qwopus-on gpu2` (the current wrapper, unchanged and retained).
