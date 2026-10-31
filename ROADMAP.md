# Black Monolith deployment roadmap

Working notes for the twin RTX 3090 SuperQwen deployment served from this repo.
Research basis: 2026-10-16 audit (sources in the local research store, final report
`2026-10-16-hyperqwen-looping-enhancements`).

Current state: vLLM 0.31.0, PTH_FA int8 per-token-head KV (PR #325 branch head),
MTP k=3, PIECEWISE CUDA graphs, ASYNC_SCHED=0, MAX_SEQS=1, 240K window,
`hyperqwen_router.py` on :8020 with xhigh effort default, repetition_penalty 1.08,
max_tokens cap 8192.

## P1

1. [ ] Integrate PR #326 (`pr50021-conv-bound`) and run the MTP A/B.
   Upstream fix for the GDN speculative causal_conv1d bound (issue #320).
   Arms: current MTP, patched MTP, SPEC=off. Battery: 18 synthetic requests x 4,
   raw token IDs preserved, finish reasons recorded. DFlash2 is validated
   upstream; MTP + PTH_FA on sm86 TP=1 is not.
2. [ ] Reasoning-policy A/B after the engine patch.
   Checkpoint author defaults unspecified effort to medium with an anti-repeat
   stop rule on explicit xhigh. Router currently forces xhigh. Measure repeated
   n-grams, output tokens, length finishes, malformed tool calls, completion.

## P2

3. [ ] `repetition_detection` circuit breaker via the router.
   First arm: {"max_pattern_size": 32, "min_pattern_size": 1, "min_count": 8}.
   Log every trigger with request ID, lane, prompt/generated tokens, finish
   reason, detector settings, prefix-cache state, preceding request ID.
4. [ ] Diagnostic capture on every loop-class finish (length, repetition,
   repeated-token threshold): rewritten request body, raw output token IDs,
   last 256 prompt IDs, effort, sampler values, spec mode, target GPU.

## P3

5. [ ] GPU0 PCIe x4 UVA characterization (EMBED_UVA=1 measured on x16 only,
   issue #281 basis). Prefill/image/embedding A/B x4 vs x16.
6. [ ] Re-evaluate PR #329 (DYNAMIC_K, DFlash2 4-stream +7.7%) if the fleet
   moves to DFlash2 batching.
7. [ ] PR #332 seat-budget logging once merged upstream.
8. [ ] Swift 1.0/1.5 checkpoint experiment after the engine fault is closed.
9. [ ] Retire `/home/chris/tools/qwen38-27b-rtx3090` (pre-rename stale clone).
10. [ ] At the next vLLM pin bump, re-evaluate every `Retires-when:` line in
    patches/ (script: grep "Retires-when" and diff each against the actual
    version; one already reads "internal prefill checkpoints, in 0.29" but our
    0.31.0 never reaches that path, so the mamba-align patch is still working
    here and its retirement premise is wrong).

## vLLM 0.32.0 watch (2026-10-16: 0.31.0 pinned 2026-10-02; 0.31.1 rc0 (10-06) and 0.32.0 rc0-rc3 open; 0.32.0 final not tagged; rc3 cut 17:25 PDT today)

Below is a rebase decision list, not cherry-picks: all land with a 0.31 -> 0.32 bump.

- #59973 quantized draft lm_head (NVFP4/FP8, 2026-10-15): +18% c=1 decode on a
  27B-class head; ours already ships the fast variant's int4 lm_head, so the
  arm is "NVFP4 head vs the shipped one", acceptance + tok/s.
- #53036 ragged Mamba2 decode, adaptive verification (2026-10-16): native,
  in-engine version of what #329/DYNAMIC_K does via --speculative-config;
  re-score item 6 (re-evaluate #329) on the 0.32 base. #61451 (skip bonus-logit
  for short drafts) rides along.
- #46963 NVFP4 paged KV on sm8x (2026-10-07): 3.32x pool capacity on a 3090
  analog. Gated on our head_dim (not 64), an A/B for accuracy vs bf16, and on
  whether the 4-bit store collides with the pthfa path.
- #57881 Mamba-family ops out of CUDA graphs (breakable-CG eager break,
  2026-10-10): the "illegal memory access" class that #34 already showed once
  on this card — a 0.32 stability candidate, no 0.31 lever.
- #60602 FP8 KV on Ampere sm_80/sm_86 (2026-10-13): 0.32 will natively cover
  what triton-spec-attn-fp8-kv hand-carries on 0.31; nothing to do, noted for
  the re-export.
- #61422 (shared Triton JIT warmup pool, 9-28% faster cold compile; boot-only)
  and #59735 (non-spec GDN on the plain path, +~1.5%) come along free;
  individually negligible here.

## 2026-10-31 upstream research: 155 commits ahead, staged upgrade plan

Status: researched 2026-10-31 (upstream/main f382342 = merge of PR #422; 78 PRs
merged since 2026-10-15, the Oct-30 wave headlined by the 0.32 port). NOT
STARTED: Chris is running the gpu unlock; begin only after it completes, and
re-baseline the A/B arms on post-unlock hardware.

Baseline pin: black-monolith 2be9c30 (2026-10-16), vLLM 0.31.0, PTH_FA int8 KV,
INT8_ACT + PREFILL_ATTN=int8, MTP k3, PIECEWISE, ASYNC_SCHED=0, MAX_SEQS=1,
200K Leminkozey twin (gpu0 :18020, gpu2 :18021). Working tree was dirty at
research time (ROADMAP.md additions, hyperqwen_router.py 2-line diff,
untracked quantize_superqwen.py) — snapshot before any branch work.

What upstream shipped that touches our profile:

- PR #415: vllm==0.32.0 port (lands #387 + 0.32 cuts of #404/#409). Verified on
  a 3090: C1 131.8 tok/s greedy / 121.7 default, tok/step 3.36, GSM8K 0.970
  (n=200). The Oct-30 re-cut series assumes this base.
- PR #418 (lands #396): pad 2-4 token GDN in_proj_ba GEMM batches. MTP-3 verify
  is exactly the 4-token cuBLAS cliff, hit in all 48 GDN layers every step.
  3090-measured: in_proj_ba 1.80 -> 0.24 ms/step; engine step -4.8%
  short-prompt, -3.8% at 25k cached. Knob VLLM_GDN_BA_PAD_SMALL_M=1 (default
  on; LoRA-safe gate included, we run no LoRA).
- PR #422 (lands #386): PTH_FA2_SLAB_PREFILL=1 bounded FA2 staging for long
  prefill on the INT8 cache. +6.0% prefill at 128K on the maintainer's 3090
  (754 vs 711 tok/s); author measured +7.2% at 131K, +7.8% at 150K. Fixed
  136 MiB workspace, about 3-4K tokens of our KV pool. The #422
  stream-takeover fix was required on 0.32; if trialled on 0.31 first, verify
  engagement is a no-op risk (instrument before trusting).
- PR #413 (lands #404): preserve shared pages when creating quantized KV scale
  views (two-line backport of vllm #64141). Correctness bug in our exact
  per-token-head int8 KV regime; PTH_FA inherits the affected methods, CUSTOM
  does not avoid it. Upstream reproduced first-request garble on the 0.31 pin
  (2+2 -> "4,0"). Candidate contributor to our gpu2 empty-first-attempt
  hiccup; re-test that symptom after adopting.
- PR #417 (lands #382 + #381, 0.32 cuts): async Mamba KV sizing (fixes
  one-block undersized pool admission, 218 admitted vs 219 needed at 163K) and
  continuous usage streaming while Qwen buffers tool arguments (our Bifrost
  stream-idle exposure during agent tool calls).
- #409: startup refusal of contexts beyond drafter positional capacity (we sit
  inside 262K; protective only).

KV capacity: no upstream change adds pool tokens at fixed 200K. The slab
workspace costs ~136 MiB; #382 only corrects undersized admission. Capacity
remains governed by the 2026-10-17 leminkozey-capacity experiment.

Not adopted, with reasons: Cyrillic draft vocab (#411/#423 — RU-only benefit;
full head measured ~2% slower on English), kv-cache-sharing branch (stale,
281 commits behind upstream/main), DYNAMIC_K #329 (dflash2-only), SPEC_ATTN=0
guidance (CTX=fast profile only), gpu-unlock field claims of 2x (single-card
data shows gains only at C4-C8: +5-11% DFlash2 C4/C8 per issue #402 A/B,
greedy C8 +19.7% per the #414 recompute; C1/C2 within run spread).

Watch list (no action yet): #423 ready-made Cyrillic draft vocab; #424 retain
SWA history at sparse Mamba offload boundaries (becomes relevant only if a CPU
KV-offload tier returns; fixes a DFlash2 offload-lookup zero-hit confirmed on
f382342); #425 interior GDN checkpoint export on 0.32.

### Plan A: 0.31 cherry-picks (spare lane, after gpu unlock lands)

29. [x] Snapshot the dirty working tree (commit or stash ROADMAP.md,
    hyperqwen_router.py, quantize_superqwen.py) before any branch work.
    (Working-tree changes left uncommitted by agreement; carried intact
    through the branch work instead of stashed.)
30. [x] After the gpu unlock completes, re-baseline: one conc_ladder pass per
    lane on the stock image; these post-unlock numbers are the A/B control.
    (Unlock confirmed 2026-10-31: both 3090s 250 W cap, max_limit 375 W,
    max SM 2100 MHz. Baselines in bench-results/unlock-rebaseline-20261031/.)
31. [x] Cherry-pick the 0.31 cuts of #404 (KV scale-view page preservation) and
    #396 (gdn-ba-pad-small-m) onto pth-flash-attn-int8-kv; rebuild; deploy one
    lane. (Done as branch unlock-upgrade-a: #332 seat-budget dependency added,
    then #396 and #404; image hyperqwen-pthfa:unlock-a, in-container gate
    passed fuzz 0. Deployed gpu0 first, then gpu2 after gates.)
32. [x] A/B upgraded lane vs post-unlock control: conc_ladder --salt for
    timing, plus the standing verification set (greedy arithmetic,
    FIRST-request content check, near-ceiling retrieval, long generation,
    vision probe). Accept on step-time gain in the -4 to -5% band with zero
    first-request anomalies and loop-free output; otherwise roll back.
    (ADOPTED 2026-10-31, with a corrected band: measured ms/pass -1.5 to
    -2.4% at our 4K-ctx profile — upstream's -4.8% was short-prompt; decode
    +3.6-9.3%; all gates clean: first-request "68" both lanes, KV pool
    unchanged 203,831, no loops, QUARTZ-77 exact at 186,195 prompt tokens,
    0 preemptions at N=4. Full record:
    bench-results/unlock-rebaseline-20261031/RESULTS.md. pr326 retained as
    rollback. Probe artifacts logged, both pre-existing xhigh behaviors:
    content=null when max_tokens is exhausted by reasoning, and essay
    re-drafting inside reasoning inflating naive n-gram counts.)

Plan A status: both lanes ran hyperqwen-pthfa:unlock-a; superseded the same
day by Plan B. Next: Plan B (0.32).

## 2026-10-31 research: reasoning-effort policy + max_tokens default (items 2 & 15 prep)

### Measured state (probes against the live fleet, 2026-10-31)

- The served Leminkozey checkpoint's chat template: xhigh is the template
  default; its xhigh branch carries the generic correctness instruction and
  NOTHING else. The 512-token bound and anti-repetition sentence exist only in
  the retired SuperQwen template (verified by diffing both
  chat_template.jinja files). The router's stated justification for forcing
  xhigh is void on current weights; the live anti-loop defenses are
  ASYNC_SCHED=0, PIECEWISE, repetition_penalty 1.08.
- The router unconditionally overwrites all four effort fields with xhigh
  (hyperqwen_router.py:115-129). Measured: identical responses (55
  completion tokens each) for requests carrying medium/high/max/minimal.
  Caller effort selection is currently a no-op. Hermes config's
  reasoning_effort: medium for qwen-gpu0|gpu2 (config.yaml:221-222) is
  dead-lettered by this.
- vLLM 0.32 types reasoning_effort as
  Literal[none|minimal|low|medium|high|xhigh|max] in the chat protocol, so
  OpenAI-vocabulary efforts no longer 400 (gotcha 58's risk is gone at the
  protocol layer); the engine passes the value into the template and derives
  enable_thinking from none automatically when the caller sets no explicit
  flag. Live: effort=high via Bifrost answered 3/3.
- thinking_token_budget is a typed request parameter in our image
  (protocol.py:271 -> SamplingParams) and flows through chat completions.
- max_tokens injection: the router's 8192 default fires only when a request
  omits both max_tokens and max_completion_tokens. vLLM's no-cap behavior
  (get_max_tokens, entrypoints/serve/utils/api_utils.py) is
  max_model_len - input_length; the documented incident d309a671 (77,010
  generated tokens after a 122,990-token prompt, exactly exhausting the 200K
  window) is the failure the guard exists for. Bifrost's catalog advertises
  max_output_tokens 32768 for both lanes, so Bifrost-path traffic has a
  different ceiling than direct-to-router traffic.
- Coupling: reasoning tokens count inside max_tokens (2026-10-31 probes:
  content=null with completion_tokens 3000-8000, all spent in reasoning at
  xhigh). External evidence (zjshen 3090 agent study, 24 attempts): the
  output cap was the dominant lever (medium 8K -> 32K: 5/24 -> 20/24 agent
  deliveries); effort helped only after room existed (medium/32K 20/24 ->
  xhigh/32K 23/24, +32% median attempt time). Stock-model medium costs ~9
  accuracy points on hard non-agentic benchmarks (external, single-source).

### Decision (Chris, 2026-10-31): implement + A/B, and test 32,768 as a cap arm

36. [x] Document the effort policy and cap decision. (This section.)
37. [ ] Implement in the router behind an env kill-switch
    (QWEN_ROUTER_EFFORT_POLICY=v1):
    - effort passthrough with validation + nearest-level mapping:
      none->none; minimal,low->low; medium->medium;
      high,xhigh,max->xhigh; absent->medium (new default);
    - reject unknown chat_template_kwargs.reasoning_effort values via the
      same map instead of forwarding them to the template's raise path;
    - max_tokens injection default configurable
      (QWEN_ROUTER_MAX_TOKENS; A/B arms 16384 and 32768, see 38).
    Router restart is the deploy mechanism (it is a per-connection proxy; use
    a low-traffic window and check :8020 health after).
38. [x] Run the A/B on the gpu2 lane (2026-10-31 12:40-12:54, 4 arms x 3
    reps x 2 probe shapes; bench/effort_ab_20261031.py, raw record
    bench-results/unlock-rebaseline-20261031/effort-ab-20261031.txt):
    RESULTS — arm (a) legacy xhigh+8192: essay 3/3 finish=length,
    content_chars=0, 26.8-28.4K reasoning chars, 199-714 repeated 8-grams,
    ~89 s wasted per request. Arm (b) medium+16384: essay 3/3 finish=stop,
    5.0-5.6K content chars, ~1.1-1.2K reasoning chars, loops 0, ~22 s.
    Arm (c) medium+32768: essay 3/3 finish=stop, 5.2-5.6K content, loops 0
    (one rep had 58 from reasoning-space re-drafting), ~22 s. Arm
    (d) xhigh+32768: essay 3/3 finish=stop BUT 10.9-17.2K completion tokens
    with 33-47K reasoning chars vs ~6K content, loops8 1067-2282 (reasoning
    re-drafting), 112-171 s — 5x the wall time for +0.5K content chars.
    Tool calls: clean JSON args, zero malformed, all arms (the qwen3_coder
    path is unaffected by effort). ADOPTED: v1 policy (default medium) +
    16384 injected default. The 32K arm bought nothing over 16K at medium
    on these shapes and xhigh is contra-indicated for routine traffic on
    this checkpoint (its extra tokens go to reasoning re-drafting, not
    answer quality); 32K remains available to callers who set it explicitly
    and matches Bifrost's advertised max_output_tokens. Item 2 (reasoning
    A/B) and item 15 (output ceiling) are resolved by this record: medium
    default, 16,384 ceiling, per-request override honored.
39. [ ] After adoption: set Hermes config reasoning_effort for
    qwen-gpu0|gpu2 to the adopted default (it becomes live instead of
    dead-lettered), confirm the Bifrost max_output_tokens row matches the
    adopted cap, and update the ops skill's anti-loop section (the xhigh
    template-guard layer is retired on current weights).

Plan B status: both lanes run hyperqwen-pthfa:032-b (see item 35). Next work:
items 37-39, then the deferred Hermes tool-call integrity section (11-28),
with 11-14's re-capture on 0.32 first.

### Plan B: 0.32 bump (after Plan A evidence)

33. [x] Revalidate the anti-loop stack on 0.32 before touching profiles:
    ASYNC_SCHED=0 still required (vLLM #51599 merge state unverified),
    CUDAGRAPH_MODE=PIECEWISE, xhigh template guard, repetition_penalty 1.08.
    (Verified against upstream/main single-user/start_qwen.sh: ASYNC_SCHED=0
    remains the shipped default on long-spec paths and the launcher still maps
    it to --no-async-scheduling; PIECEWISE remains the mandated CG mode with
    the gotcha-37 warning that FULL corrupts one prompt in 128, MTP included.
    vLLM #51599 (async D2H accepted-token snapshot) exists per third-party
    source, but is moot for us while upstream keeps sync scheduling. xhigh
    guard and repetition_penalty live in our hyperqwen_router.py, which the
    merge carried untouched.)
34. [x] Take upstream main (#415 + re-cut series) onto black-monolith; enable
    PTH_FA2_SLAB_PREFILL=1 with the #422 takeover fix; re-run the 204K
    capacity stress (195,670+6,000 pattern) and the full verification set;
    confirm the 200K pool still fits at GPU_UTIL=0.93 minus the 136 MiB slab;
    slab off is the fallback on any regression.
    (DONE 2026-10-31 as branch unlock-upgrade-b, merge c73dd35 of
    upstream/main f382342 — 56-entry 0.32 series supersedes the Plan A 0.31
    picks; our router/.env/ROADMAP carried through; serve-404-served-names
    dropped per upstream's 0.32 re-export. Image hyperqwen-pthfa:032-b,
    in-container gates all PASS incl. verify 0 failures.
    SIZING CORRECTION vs the plan's guess: at GPU_UTIL=0.93 the slab
    workspace leaves only 197,472 tokens and the fixed 200,000 MAX_LEN
    refuses to boot (engine ValueError). GPU_UTIL=0.94 boots cleanly:
    203,831-token pool with slab reserved. Gotcha-4's failure band is 0.95+;
    0.94 keeps margin and long-run validation is still owed at depth.
    Adopted on BOTH lanes at 0.94 + slab. Measured: slab cold prefill 186K
    prompt 205 s gpu0 / 202 s gpu2 (~908/920 tok/s) vs 242-243 s no-slab
    (~769 tok/s), +16-18%; a second seed-777 192K sample ran 894 tok/s.
    Slab decode within noise (33.9-34.7 ms/pass; no regression). Gates:
    first-request 68 clean on both lanes, QUARTZ-77 exact at 186,195
    (pre-slab-adopt numbers) and again on both slab lanes, OBSIDIAN-42 exact
    at 192,376, vision ZENITH-77 exact read in 1.0 s (first probe's one
    wrong char was the probe's hand-drawn font, not the engine), N=4 zero
    preemptions, router :8020 serving qwen-gpu0/qwen-gpu2. Pool at 0.32
    without slab: 205,363 (+1,532 vs 0.31 same UTIL).
    Remaining debt: full 195,670+6,000 capacity stress and long-soak at 0.94
    not yet re-run; 0.31 fallbacks retained (unlock-a, pr326).)
35. [x] Update router/Bifrost metadata and tray labels only after both lanes
    pass; retain the 0.31 image and launchers for rollback.
    (Router :8020 verified serving qwen-gpu0/qwen-gpu2 after the swap; both
    advertise 200000 via the engines, no metadata edit needed — Bifrost rows
    already read context_length from the engines' /v1/models at the Oct-30
    fix. Tray labels unchanged by design: model identity, quant and window
    are identical; the engine version is not part of the label standard.
    NOTE for the tray: launcher env vars renamed QWEN38_SUPER_* ->
    QWEN38_PTHFA_* on 2026-10-31 (Chris: SUPER is a specific finetune, not
    useful in code); tests updated, 5 passed + 2 subtests. GPU_UTIL now
    env-overridable (QWEN38_PTHFA_GPU_UTIL, default 0.93; lanes run 0.94
    with slab). local-ai-control bin copy updated in step; commit pending
    with the next tray change. Rollback set: unlock-a (0.31+picks), pr326
    (pre-Plan-A), fp8 200K launchers.)

## Deferred roadmap: Hermes tool-call integrity and lane reliability

Status: documented 2026-10-18 for a later work session. No implementation was
authorized in the incident investigation.

### Verified incident evidence

- Bifrost request `dd3b7f7e-c634-494b-bb66-2310ddbcf0fa` completed successfully
  through `black-monolith/qwen-gpu0` but contained a malformed `patch` call. The
  normalized arguments leaked chat-template XML (`</parameterparameter=...>`) and
  ended without valid JSON. Hermes correctly refused to execute it.
- GPU0 produced 8 malformed calls among 582 logged calls in the inspected 72-hour
  window; GPU2 produced 0 among 527. Five GPU0 malformed calls were recorded as
  successful Bifrost transactions, so transport interruption is not required.
- Request `d309a671-0462-442a-90c5-9a6f398a6c4e` stopped mid-`patch` arguments.
  The engine then generated 77,010 tokens after a 122,990-token prompt, exactly
  exhausting the 200,000-token request window. Bifrost timed out after its
  configured 300-second stream-idle window.
- Both per-card engines run `--max-num-seqs 1`, while Bifrost admits concurrency
  4 with buffer 128. During the incident the engine reported one running request
  and up to three waiting/deferred requests; queued requests then hit the same
  stream-idle timeout.
- A GPU2 deployment invoked the shared `router-off` / `router-on` path. Active
  proxy handlers prevented graceful shutdown, systemd killed the router after
  its 90-second stop timeout, and Bifrost recorded GPU0 connection refusals on
  `192.168.4.27:8020` during the outage.
- Bifrost currently stores normalized `responses_output` and `tool_calls` but has
  `send_back_raw_request=0`, `send_back_raw_response=0`, and
  `store_raw_request_response=0`. Those settings do not preserve the original
  SSE bytes needed to distinguish malformed model XML from parser corruption.

### P1: isolate the malformed-tool boundary

11. [ ] Build a local, redacted reproduction harness from one stored failing
    request. Send it directly to GPU0 `:18020/v1/responses`, bypassing the shared
    router and Bifrost, and save the raw SSE bytes plus normalized output.
12. [ ] Run the same payload through `:8020` and Bifrost. Diff all three boundaries:
    model stream, router stream, and Bifrost `responses_output`/`tool_calls`.
13. [ ] Classify the defect from the raw capture:
    - malformed XML from the model -> revise the Qwen tool template or tool policy;
    - valid XML plus malformed normalized arguments -> patch or upgrade the
      vLLM `qwen3_coder` parser;
    - changed bytes at `:8020` -> fix the router proxy.
14. [ ] Add a deterministic regression fixture for the observed failure forms:
    leaked `</parameterparameter=...>` markup, missing scalar values such as
    `"replace_all":`, unterminated strings, and parallel batches containing one
    valid call plus one malformed call.

Acceptance: the captured raw boundary identifies the first component that emits
invalid tool arguments, and the regression fixture fails before its fix.

### P1: bound runaway generation

15. [ ] Add an explicit output ceiling for local-Qwen Responses requests that
    omit one. Start the experiment at 16,384 output tokens; test 8,192 and 16,384
    against long `patch`, `write_file`, `execute_code`, and `browser_exec` calls.
16. [ ] Preserve the existing 8,192 chat-completions cap. Ensure the Responses
    implementation writes the correct Responses field rather than relying on the
    chat-only `max_tokens` rewrite.
17. [ ] Record prompt tokens, generated tokens, finish reason, tool name, argument
    length, JSON validity, and whether the ceiling fired.

Acceptance: a malformed tool call cannot decode to the remaining 200K context
window, and valid large tool calls still complete at the selected ceiling.

### P1: one-lane admission control

18. [ ] Make router admission reflect each engine's `MAX_SEQS=1`: at most one
    forwarded inference request per GPU lane. Health, models, and metrics remain
    available while the lane is occupied.
19. [ ] When a lane is occupied, return a retryable busy response immediately with
    the lane and retry guidance. Do not hold a silent socket in the router or
    vLLM queue.
20. [ ] Add concurrency tests for GPU0 and GPU2 independently, including one busy
    lane while its sibling remains available.

Acceptance: no same-lane request waits silently long enough to hit Bifrost's
300-second stream-idle timeout; sibling-lane traffic remains independent.

### P1: graceful shared-router lifecycle

21. [ ] Remove shared-router restarts from per-card GPU0/GPU2 start and stop paths.
    Model changes keep `:8020` serving throughout.
22. [ ] Give the router explicit active-handler accounting and a drain operation.
    A deliberate router reload stops admitting new inference requests, waits for
    active handlers to finish within a bounded drain window, then exits cleanly.
23. [ ] Replace swallowed `IncompleteReadError`, `ConnectionResetError`, and
    `BrokenPipeError` paths with request-ID/lane/byte-count diagnostics while
    preserving client-disconnect handling.
24. [ ] Test SIGTERM during an active stream. Require either a complete response
    or an explicit retryable error, with no truncated success response and no
    systemd SIGKILL.

Acceptance: GPU2-only operations cause zero GPU0 connection failures; deliberate
router reloads finish without exceeding the user-unit stop timeout.

### P2: observability and production validation

25. [ ] Add correlation IDs across Hermes, Bifrost, router, and vLLM logs. Include
    model lane, upstream port, first-byte time, bytes relayed, queue rejection,
    finish reason, and malformed-argument classification.
26. [ ] Add a temporary, narrowly scoped raw-stream capture mode for one request ID.
    Redact secrets and request content by default; store only explicitly armed
    captures under `bench-results/tool-call-integrity/`.
27. [ ] Run a validation battery covering sequential tool turns, two simultaneous
    requests to one lane, simultaneous requests to both lanes, long-context
    patches, client cancellation, Bifrost idle timeout, and router drain/reload.
28. [ ] Keep Bifrost's 300-second stream-idle timeout unchanged during this work.
    Reconsider it only after parser, output ceiling, admission control, and router
    lifecycle fixes pass the validation battery.

Final acceptance:

- zero malformed tool arguments in the validation battery;
- zero silent same-lane queues;
- zero cross-lane interruption during per-card operations;
- no router shutdown reaching SIGKILL;
- failures carry a request ID and identify the first failing boundary.

### Execution order and rollback

Implement 11-14 first, because they identify the component that owns the malformed
arguments. Then implement 15-17, 18-20, 21-24, and finish with 25-28. Keep each
change separately revertible. Preserve the current router, launcher, Bifrost, and
image state until its replacement passes the relevant acceptance gate.

## Conventions

- `black-monolith` is THE deployment branch: all custom work (router, launchers,
  configs) plus merged upstream PRs. This is the checked-out branch that builds
  the serving image and backs the local AI tray's Qwen tier.
- `main` tracks `origin/main` (this fork's copy of upstream). Upstream sync via
  the `upstream` remote lands on `main` first, then merges into `black-monolith`.
- Upstream PRs (like #326) are merged into `black-monolith` after validation.
- Never commit bench data (`bench-quality-data/`, `quantize.log`) or any
  credential material. Tuning-only `.env` files are allowed.
