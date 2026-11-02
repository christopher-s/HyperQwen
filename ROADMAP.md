# Black Monolith deployment roadmap — OPEN work only

Working notes for the twin RTX 3090 Qwen deployment served from this repo.
Completed items live in ROADMAP-COMPLETED-2026-10-31.md (and are preserved with
their evidence there).

Current state (2026-10-31): vLLM 0.32.0 image `hyperqwen-pthfa:032-b` on both
lanes (gpu0 :18020, gpu2 :18021), PTH_FA int8 KV, INT8_ACT + PREFILL_ATTN=int8,
MTP k=3, PIECEWISE, ASYNC_SCHED=0, MAX_SEQS=1, PTH_FA2_SLAB_PREFILL=1,
GPU_UTIL=0.94, 203,831-token pools, MAX_LEN=200000. Router :8020 on the v1
effort policy (caller effort mapped to nearest level, absent -> medium,
QWEN_ROUTER_EFFORT_POLICY=legacy rollback) with QWEN_ROUTER_MAX_TOKENS=16384
injected default, repetition_penalty 1.08. black-monolith @ 0bee80a, pushed.

## Session close (2026-10-31 evening): quant/backend evaluation concluded

Chris is running Qwen as-is and will evaluate in daily use. Findings that
closed the open optimization questions, all measured or source-verified:

- FlashInfer + fp8 KV (vLLM #60532): tested TWICE on GPU2, declined on
  methodology-correct evidence (item 19). CUSTOM + PTH_FA int8 stays.
- Quant ranking for sm86 (researched against vLLM docs + 2026 guides +
  measured local tables): our stack is rank 1 on BOTH tables —
  W4A16 int4-g128 weights (Marlin; NVFP4/FP8/MXFP4 are the same weight-only
  class on Ampere, no gain) and int8-per-token-head KV read by PTH_FA
  (1.41x FA2-fp8 at 128K). W4A8 already deployed where it wins (prefill via
  INT8_ACT). Remaining headroom on this card is speculative acceptance and
  upstream kernel work, neither quant-shaped.
- Experimental KV tiers for "small perf cost, big pool, accuracy-neutral":
  KVarN k4v2 (+0.16% ppl, 2x pool, ~1.06x short / 2.13x deep decode) and
  int4_per_token_head (+0.3% ppl, 355k pool, Triton tax) are the two live
  options; both already available (CTX=huge, KV=int4pth). Chris declines
  denser-than-int8 KV for now (reasoning quality and predictability first);
  revisit only if a capacity wall is hit. TurboQuant 4bit_nc disqualified
  for our profile (chunked-prefill OOM past 32k prompts on 24 GB).
- gpu-unlock (Zagitta) verified ACTIVE on all three GPUs (boot log):
  doubles fp16/bf16 tensor rate; no effect on our int4/int8 decode paths
  (by design), benefits prefill/vision/bf16 arms — already banked.
- KV capacity levers if ever needed: GPU_UTIL 0.94 (0.95+ unstable),
  slab-off (+136 MiB), EMBED_UVA off, CTX=huge / KV=int4pth tiers.

Fleet state at close: both lanes serving, router healthy, GPU2 restored to
production after the A/Bs and verified (Application startup complete,
CUSTOM backend in engine argv). GPU0 was never touched during testing
(Elijah's session).

## Testing policy (Chris, 2026-10-31)

Chris tests Qwen himself during normal use and escalates issues here; long
automated soak/verification runs are the exception, run only when a change
cannot be observed interactively (capacity ceilings, memory pressure, timing
regimes). Items are tagged accordingly: [CHEAP] = minutes, [CHRIS] = covered
by normal use, [SOAK] = the only class that justifies long unattended runs.

## P1 — engine/stack debt from the 0.32 adoption

1. [ ] Full 204K capacity stress on the 0.94 + slab sizing: the standing
   195,670 prompt + 6,000 completion pattern plus near-ceiling exact retrieval,
   plus a long soak (>=4 h mixed traffic) at 0.94. [SOAK] — this is the one
   genuine soak: interactive use cannot reach 204K tokens casually, and
   gotcha-4's failure band is time/pressure dependent. Everything else about
   0.94+slab already passed interactive gates on 2026-10-31.
2. [x] At the next vLLM pin bump, re-evaluate every `Retires-when:` line in
    patches/. (DONE 2026-10-31: swept all 56 markers. The 0.32 port commit
    a342b1c re-exported the whole 53-file series against 0.32.0, which
    retired what was retireable at merge time (e.g. serve-404-served-names,
    dropped). Spot-verified the remaining live classes: api-root-health's
    "Retires-when: upstream PR" — no upstream equivalent exists yet and the
    patch demonstrably still works (engine GET / = 200, /api/status =
    {"status":"ok"}); auth-deny-default same class; quant-scale-view's
    premise (vllm #64141) is NOT in 0.32 so the patch remains load-bearing;
    the "stays while the knob exists" class is self-describing. Conclusion:
    nothing to retire now; next sweep at the 0.33+ bump.)
3. [ ] PR #326 MTP A/B: the fix is integrated (merge a2769dc, in both serving
   images) but the standing A/B battery (MTP vs SPEC=off, 18 synthetic
   requests x 4, raw token IDs, finish reasons) never ran against it.
   [CHRIS->CHEAP] — daily MTP use since integration is the real test; if
   Chris sees no spec-decode weirdness, close as superseded by the 0.32 gate
   results (GSM8K 0.970 n=200 upstream on this series). Escalation-triggered.

## P1 — Hermes tool-call integrity and lane reliability (items 11-28, paused
### at Chris's request 2026-10-31; resume here first)

Evidence base (verified incident record preserved verbatim in git history and
summarized here): Bifrost request dd3b7f7e carried malformed `patch` arguments
(leaked template XML, non-JSON tail) through a successful transaction; GPU0
produced 8 malformed calls among 582 in the audited 72 h window vs GPU2 0/527;
request d309a671 decoded 77,010 tokens after a 122,990-token prompt, exactly
exhausting the 200K window before Bifrost's 300 s stream-idle timeout; both
engines run MAX_SEQS=1 while Bifrost admits concurrency 4; a GPU2 deployment
restarted the shared router and SIGKILLed active GPU0 handlers; Bifrost does
not store raw SSE bytes (send_back_raw_request/response and
store_raw_request_response all 0).

Context updates since the incident: fleet moved to 0.32 (PR #417 ships
continuous usage streaming while Qwen buffers tool arguments, and #404's
page-preservation fix removes one candidate corruption source); the router's
effort forcing is gone (v1 policy). Re-capture before diagnosing: the symptom
may no longer reproduce.

4. [ ] Build a local, redacted reproduction harness from one stored failing
   request. Send it directly to GPU0 :18020, bypassing the shared router and
   Bifrost; save raw SSE bytes plus normalized output. [CHEAP] — a script and
   a handful of requests.
5. [ ] Run the same payload through :8020 and Bifrost. Diff all three
   boundaries: model stream, router stream, Bifrost responses_output/tool_calls.
   First re-check whether the symptom still reproduces at all on 0.32 + the
   v1 router (2026-10-31 A/B tool probes: 12/12 clean JSON args, no malformation).
   [CHEAP]. Fallback per testing policy: skip the harness entirely and let
   normal Hermes use be the probe — if Chris's daily tool traffic stays clean
   for a few days, close 4-7 as not-reproducing-on-0.32 and reopen only on
   escalation.
6. [ ] Classify the defect from the raw capture: malformed XML from the model
   -> revise tool template/policy; valid XML plus malformed normalized
   arguments -> patch or upgrade the vLLM qwen3_coder parser; changed bytes at
   :8020 -> fix the router proxy. [CHEAP] once 4-5 produce a capture; the fix
   itself is scoped by what the classification finds.
7. [ ] Add deterministic regression fixtures for the observed failure forms:
   leaked `</parameterparameter=...>` markup, missing scalar values such as
   `"replace_all":`, unterminated strings, parallel batches with one valid +
   one malformed call. [CHEAP] — pytest-style fixtures against the parser.

Acceptance: the captured raw boundary identifies the first component that
emits invalid tool arguments, and the regression fixture fails before its fix.

8. [ ] Responses-API output ceiling: confirm the Responses path carries the
   same bound now that the chat path defaults to 16,384 (the router JSON-rewrites
   chat bodies; /v1/responses traffic does NOT traverse the router — it goes
   engine-direct or via Bifrost). Verify where a ceiling is actually enforced
   for Responses and close the gap if any. Record prompt/generated tokens,
   finish reason, tool name, argument length, JSON validity, ceiling-fired.
   [CHEAP] — read the Responses request schema in the image, one or two probe
   requests; only a found gap leads to a small router/Bifrost-side fix.

Acceptance: a malformed tool call cannot decode to the remaining 200K window
on either API surface.

9. [x] One-lane admission control.
    (INVESTIGATED AND RESCOPED 2026-10-31: measured the exact failure shape —
    a 16K-cap essay generation holds the lane 37 s; a request arriving mid-
    generation waits as a vLLM Waiting req (engine log: Running: 1, Waiting: 1
    for ~30 s) and then completes normally (36 tokens, finish=stop). The
    silent-wait hazard is real only for waits approaching Bifrost's 300 s
    stream-idle window — reachable only when the occupied request itself
    decodes for minutes (longgen at 16K) AND the queued request arrives with
    <300 s of idle-budget left; vLLM streams nothing for a queued request, so
    Bifrost's idle clock runs on the waiting socket. Implemented the cheap
    80% fix instead of full admission control: /v1/responses in-flight
    clamp (item 8) bounds single-request lane holds to minutes, and the busy
    response is deferred until Chris reports an actual silent-stall incident.
    Reopen if a stall is ever observed.)
    REMAINING (deferred): router-side immediate retryable-busy response.
    ~30 lines in handle_client (track one in-flight inference per target
    port; on second request return 429 with Retry-After) + a two-curl test.
    Schedule when convenient, not urgent.

10. [ ] Graceful shared-router lifecycle: remove router restarts from per-card
    start/stop paths; explicit active-handler accounting + bounded drain;
    replace swallowed IncompleteReadError/ConnectionResetError/BrokenPipeError
    with request-ID/lane/byte-count diagnostics; SIGTERM-during-stream test
    (complete response or explicit retryable error; no truncated success; no
    SIGKILL). [CHEAP implementation, CHRIS-observed] — code change plus one
    deliberate reload while a stream runs; the cross-lane kill scenario is
    exactly what Chris hits in normal per-card operations.

Acceptance: GPU2-only operations cause zero GPU0 connection failures;
deliberate reloads finish inside the unit stop timeout.

11. [ ] Observability (P2): correlation IDs across Hermes/Bifrost/router/vLLM
    (lane, upstream port, first-byte time, bytes relayed, queue rejection,
    finish reason, malformed-argument classification); temporary narrowly
    scoped raw-stream capture mode for one request ID (redacted by default,
    stored under bench-results/tool-call-integrity/); validation battery
    (sequential tool turns, two simultaneous same-lane, simultaneous
    cross-lane, long-context patches, client cancellation, idle timeout,
    router drain/reload). [CHEAP parts: router-side correlation IDs and the
    capture mode. The full battery is [SOAK]-class and only worth running if
    4-10 land a real fix — otherwise drop it and let production traffic be
    the validation.]

Final acceptance: zero malformed tool arguments in the battery; zero silent
same-lane queues; zero cross-lane interruption during per-card operations; no
router shutdown reaching SIGKILL; failures carry a request ID and identify
the first failing boundary.

Execution order: 4-7 first (identify the owning component), then 8, 9, 10,
finish with 11. Keep each change separately revertible; preserve current
router/Bifrost/image state until replacements pass their acceptance gate.

## P2

12. [x] Loop-defense injection.
    (DONE 2026-10-31 via the router: every forwarded request that does not
    set its own now carries
    repetition_detection={"max_pattern_size":32,"min_pattern_size":1,
    "min_count":8}. Engine-native (vLLM 0.32 SamplingParams ->
    RepetitionDetectionParams), no launcher flag needed. VERIFIED LIVE:
    "repeat banana forever" prompt -> finish=repetition at 83 tokens in 1 s
    (a new finish reason emitted by the detector) instead of burning the
    16,384 budget; normal request unaffected (finish=stop, 27). A new
    loop-class finish reason exists now — item 13's capture should key on it.
    Per-request override preserved: callers sending their own
    repetition_detection keep it.)
13. [ ] Loop-class diagnostic capture (length/repetition finishes): rewritten
    request body, output token IDs, effort, sampler values, target GPU.
    Re-scope after 12: if the engine's detector ends loops early, the capture
    matters only for post-mortems, so a small log line may suffice.
    Note: the A/B already records loop-class data for bench runs; this item is
    about production traffic only.

## P3

14. [ ] GPU0 PCIe x4 UVA characterization (EMBED_UVA=1 measured on x16 only,
    issue #281 basis). Prefill/image/embedding A/B x4 vs x16.
15. [ ] Re-evaluate PR #329 (DYNAMIC_K, DFlash2 4-stream +7.7%) if the fleet
    moves to DFlash2 batching. Also re-score on the 0.32 base against
    #53036 (native ragged Mamba2 decode, adaptive verification), which
    supersedes it in-engine.
16. [ ] Swift 1.0/1.5 checkpoint experiment after the engine fault is closed.
    (External evidence 2026-10-31: Swift 1.5 targets up to 58.5% fewer
    thinking tokens with LiveCodeBench accuracy claims; evaluate against our
    v1 effort policy — it may be redundant now that medium is the default.)
    [CHRIS] — a checkpoint swap Chris would feel directly in daily use;
    decide only if medium-default still feels slow. No automated eval planned.
17. [x] Retire `/home/chris/tools/qwen38-27b-rtx3090` (pre-rename stale clone).
    (Already gone — directory does not exist on disk as of 2026-10-31; only
    doc mentions remain in docs/reproductions and single-user/README, which
    are historical records, not live references.)
18. [ ] IFEval (lm-eval) on qwopus-heretic with the Qwopus card protocol —
    PAUSED 2026-11-02 per Chris (2-4 h GPU2 estimate at MAX_SEQS=1 too long
    for the moment). FULL resume config + gotchas live in the skills repo:
    winston-skills `local-ai/control/local-ai-operations/references/
    qwopus-heretic-eval-roadmap.md` (canonical status board — consult that
    first). Short form: xhigh template, card sampling (temp 1.0, top_p
    0.95, top_k 20), max_gen_toks=8192 (IFEval default 1280 starves under
    xhigh reasoning), verbatim /v1/chat/completions base_url,
    tokenized_requests=false, seed 1234, output
    bench-results/qwopus-lmeval-<date>-xhigh/.

## Watch list (upstream, no action yet)

### 2026-10-31 vLLM upstream scan (0.32.0 pinned; main +1,369 commits ahead)

Fleet context: attention-backend CUSTOM (PTH_FA int8 per-token-head plugin),
MTP k=3, hybrid 48 GDN + 16 FA layers, sm_86, 24 GB. Scan filtered for that
profile.

18. [x] Scan documented (this section). Key finds:
    - vLLM #60532 (MERGED): FlashInfer fa2 uniform-batch spec-decode
      verification in FULL CUDA graphs on pre-SM90. Upstream SM80 Qwen3.8-27B
      NVFP4 MTP k=3: 73.6 -> 108.1 tok/s TP=1 (+47%), bitwise-same greedy
      output; no change without spec decode. OUR PATH DOES NOT USE IT: we pin
      attention-backend CUSTOM (PTH_FA plugin), so FlashInfer never runs.
      Item 19 is the A/B.
    - vLLM #62113 (OPEN, conflicting with main): 16-byte-aligned
      per-token-head KV slots. RTX 3080 int8_per_token_head kernel 1.25-1.96x
      faster, 128K TTFT -21%; cost +4.6% KV memory at head_size 256 (ours) --
      higher performance bought with a small KV spend, quant untouched.
      ACTION: re-check on rebase; port into our patch series when it lands.
    - vLLM #63944 (OPEN issue): int8_per_token_head scale-view FILL overwrites
      GDN state in hybrid models -> garbage first request after boot.
      CHECKED OUR IMAGE: our triton_attn.py _ensure_scale_caches builds views
      with NO fill (read from hyperqwen-pthfa:032-b), so we do not carry the
      bug. Watch: if upstream restructures view creation, re-verify
      first-request behavior at the next pin bump (neighbors our #404 patch).
    - Marginal merged: #64425 in_proj_ba aux-stream overlap (TPOT -2.7% C=1,
      bitwise identical; interacts with our #396/#418 GEMM patch -- A/B first),
      #64419 GDN prefill empty+in-place (-1.9% on no-compile base),
      #64509 fused norm+quant (NVFP4/FP8 consumers; we are W4A16 -- mostly
      inapplicable), #64482/#64650 skinny GEMM plans (SM100/SM103 only).
    - KV capacity/precision: NOTHING merged for NVIDIA sm86 that grows KV
      without reducing quant. ROCm got Octave 3-bit KV (#59774, AMD-only).
      bf16 KV would halve our pool; int8-per-token-head remains the right
      store. Realistic capacity levers stay GPU_UTIL (0.94; 0.95+ unstable),
      slab-off (+136 MiB), EMBED_UVA off.
19. [x] FlashInfer A/B on GPU2. (RAN TWICE 2026-10-31; GPU0 untouched --
    Elijah's lane stayed up the whole session.)
    ARM B1 (~19:28): FlashInfer + fp8 KV + CUDAGRAPH_MODE=PIECEWISE (env
    inherited). RESULT: 99.4 tok/s long-gen, 78.8-85.3 tok/s lists, KV pool
    215,625, first-request 68 clean. Initially read as "equivalent, declined."
    ARM B2 (~20:00) — REDO after Chris pushed back: B1 was methodologically
    flawed. I had pinned PIECEWISE, which is exactly the mode that blocks
    #60532's mechanism (the win is spec-decode VERIFY steps captured as FULL
    decode graphs, unlocked by FlashInfer's UNIFORM_BATCH CG support;
    resolve_cudagraph_mode_and_sizes downgrades spec-decode decode-FULL to
    PIECEWISE when min_cg_support < UNIFORM_BATCH). B2 = FlashInfer + fp8 KV
    with NO cudagraph pin (launcher default): engine resolved
    CUDAGraphMode.FULL_AND_PIECEWISE — the #60532 configuration, verified in
    the boot log. MEASURED B2: greedy 68 clean 0.3 s; facts lists 79.1 / 82.6
    / 85.3 tok/s (identical to B1 within noise); 16,384-token gen 104.0
    tok/s (vs 99.4 B1, +4.6%); 25K-prompt + 4K-gen 70.5 tok/s. Arm A
    reference 84.2 tok/s per-stream (v032-slab record), boot-to-boot MTP
    variance spans 83.0 -> 90.7.
    GATE FACTS (source-read from the image):
    - FlashInfer supported KV dtypes: auto/float16/bfloat16/fp8/fp8_e4m3/
      fp8_e5m2/nvfp4/nvfp4_4over6. int8_per_token_head is NOT supported;
      the per-token-head family (int8/int4/fp8) exists ONLY on TRITON_ATTN
      upstream and our CUSTOM plugin. There is no third option that pairs
      FlashInfer with a per-token-head store.
    - fp8 (block) KV is the ONLY FlashInfer pairing available to us; B2 is
      the correct and complete test of #60532 on our weights.
    CONCLUSION (now properly grounded): even with FULL_AND_PIECEWISE active
    and UNIFORM_BATCH engaged, FlashInfer+fp8 delivers 104 vs our ~84-92
    tok/s (within or near MTP boot variance) on short/medium generations,
    LOSES at depth (70.5 tok/s at 20K prompt vs PTH_FA's deeper-context
    advantage), loses 128K TTFT (129 s PTH_FA vs ~250 s fp8 tiers,
    upstream-HyperQwen measured), and sits in gotcha-40's Xid-31 fault tier
    (fp8 KV + MTP + prefix caching on 3090). #60532's +47% was NVFP4
    weights + a decode-bound C1 regime; on W4A16 + hybrid GDN + chat shapes
    the mechanism is active and produces ~nothing. CLOSING this as
    tested-and-declined is now evidence-based, twice over: same verdict
    with the mechanism properly armed. Re-test triggers unchanged:
    FlashInfer gains int8 per-token-head support, or weights move to NVFP4.
    ALSO LEARNED: the B1 essay content-null at 16K was the direct-to-engine
    path missing the router's effort default (medium) — reasoning ate the
    budget. Production traffic goes through the router and is unaffected.

- #424 retain SWA history at sparse Mamba offload boundaries: relevant only
  if a CPU KV-offload tier returns; fixes a DFlash2 offload-lookup zero-hit
  confirmed on f382342; unmerged, author-validated on H200/A4000.
- #425 interior GDN checkpoint export on 0.32: author holds it open on an
  unmet acceptance criterion (only parity observed, default-off); experimental.
- #423 ready-made Cyrillic draft vocab: RU-heavy traffic only; full head
  measured ~2% slower on English. Adopt only if RU workload appears.

### 2026-11-01 club-3090 scan (noonghunna/club-3090, master d7b35c5)

Scanned for methods/techniques beyond HyperQwen that would improve accuracy,
KV cache, or performance for our twin-3090 single-user profile. Repo: 0 open
PRs, ~30 open issues (mostly bench reports), engines: vLLM, SGLang,
llama.cpp, QEngine, Strata, PrismML.

20. [x] club-3090 scan findings. (All claims from club-3090 BENCHMARKS.md
    and issues as of 2026-11-01.)
    - QEngine 1.3.3.1 (`ghcr.io/bpx03/qengine`, PolyForm Noncommercial
      license): custom engine with a FINE-TUNED MTP head for Qwen3.8-27B
      (RedHat INT4 weights). Measured at TP=2: 147-157 narr tok/s decode
      (accept 3.28-3.41, TTFT 36-39 ms), 8-pack quality 108-123/150
      thinking-off, 132-137/150 thinking-on; promoted to
      Production-w/-caveats (#1892). The only >20%-decode item found
      anywhere in this scan. NOT a drop-in for us: different engine
      (llama.cpp-style /props API, no vLLM patch series, no router), different
      checkpoint, NC license (personal use fine). If Chris ever wants a
      second "fast lane", this is the candidate; it does not replace the
      PTH_FA vLLM lanes.
    - MAMBA_BLOCK_SIZE=8192 pool-recovery knob (club-3090 #1170): worth ~20%
      pool on their BATCH MTP arm (MAX_SEQS 4-32, align every 1,616 tokens).
      CHECKED OUR CONFIG: single-user MAX_SEQS=1; engine pads mamba pages by
      0.09% (boot log). Gain here ~= zero. vllm 0.32 carries
      cache_config.mamba_block_size if we ever batch. Not porting.
    - GPU-unlock A/B (#1755): prefill +14.5-18.4%, decode flat. Matches our
      earlier verification (unlock active here; benefits already banked).
    - BF16-unlock W4A16-vs-W4A8 re-measure (issue #1798): W4A8's prefill
      advantage narrowed +46/+37% -> +25-27%/+18-19% post-unlock; W4A16
      decode slightly ahead. Our W4A16 weights + INT8_ACT prefill capture
      exactly this post-unlock optimum. Confirms, changes nothing.
    - Strata 0.2.9 + Qwen3.8-Flash-Next (MoE sibling via CPU expert offload):
      125/159-165 tok/s at 524K context with YaRN x2, int8 KV, own MTP layer
      (#1888 same-release rematch). Different model family (Flash-Next MoE),
      needs ~77+ GB host RAM for the expert arena, IQ3_S weights. Not
      applicable to our dense-27B lanes; noted as the deep-context MoE path
      if that model family ever interests Chris.
    - Methodology note adopted for all future fleet benches: bench on a
      dedicated boot — club-3090 measured a gate-chained bench reading
      -25-27% (allocator/cudagraph state) on identical config.
    VERDICT: nothing to port to the PTH_FA vLLM lanes. The fleet remains at
    the measured optimum for this profile. QEngine recorded as the one real
    alternative engine; mamba-block-size ruled out with numbers.

### 2026-11-01 MTP draft-head spike (item 21)

21. [ ] Improve MTP acceptance via a fine-tuned draft head. Live telemetry
    (2026-11-01, both lanes): tok/step 1.87 (gpu0) / 1.89 (gpu2), accept
    62.4% / 63.1%, per-position decay IDENTICAL on both lanes
    (pos0 0.80 -> pos1 0.61 -> pos2 0.46-0.47). Same shape on two
    independently-used lanes = structural draft-head property, not config.
    Counterfactual: pos2 holding pos1's rate adds only ~+0.14 tok/step, so
    chain depth is not the lever; per-position head quality is.
    RESEARCHED (all verified 2026-11-01):
    - No public vLLM-loadable fine-tuned Qwen3.8-27B MTP head exists.
      HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3 is EXL3/exllamav3-only
      (sidecar size optimization, geomean accept 0.403-0.418, Apache-2.0);
      QEngine's model_mtp_ft.safetensors (849,400,392 B, ~8-9% more accepted
      drafts upstream-claimed) is a GitHub release blob for their engine,
      NC-licensed context, but same-base MTP head — inspectable.
    - The training path is turnkey: vllm-project/speculators supports MTP
      head fine-tuning per Tencent's FastMTP (arXiv 2509.1836): Convert
      (MTPConverter lifts native mtp.*) -> Train (teacher-forced multi-step,
      exp-decay weights [0.51,0.31,0.18] at k=3, only MTP layers trainable,
      embed/lm_head frozen, online hidden-state generation from a live vLLM
      server) -> Stitch (merge back into verifier checkpoint). ~15 tensors /
      ~800M params. quivent/qwen-mtp-corpus ladder: 79% untuned -> 85-90%
      self-distilled -> 95% multi-layer. Deployment = checkpoint-level mtp.*
      tensor swap (Red Hat FP8-MTP gist demonstrates the rewrite mechanism).
    - Documented negative result (pirateface Holodeck-Lounge): fine-tuning on
      the wrong distribution REGRESSED acceptance (49.75% vs 58.55%). Train
      on OUR traffic, not generic corpora.
    PLAN:
    a. [CHEAP, 0 GPU-h] Fetch QEngine model_mtp_ft.safetensors; inspect tensor
       names/shapes against our checkpoint's mtp.*; if the layout matches,
       clone the checkpoint with the swapped head and A/B acceptance on GPU2
       during an idle window. License: engine is PolyForm NC — fine for
       personal use, do not ship.
    b. [REAL] speculators convert -> train (on our own captured traffic
       corpus, Bifrost/router logs as source) -> stitch -> A/B on GPU2.
       Target: 1.9 -> 2.1-2.3 tok/step (+12-20% effective decode), the only
       double-digit decode lever this hardware offers. Rollback = original
       checkpoint unchanged (train writes a copy).
    PHASE B QWOPUS EXTENSION (2026-11-01): run the b. pipeline against the
    ABLITERATED Qwopus Coder-Flash build (item 22), not Leminkozey, once it
    is the daily driver — the head must draft for the serving distribution,
    and Leminkozey may be retired by then. Verified tooling: speculators
    ships a native MTP algorithm doc (MTPConverter -> train with frozen
    embed/lm_head + normalized exp-decay step weights, beta 0.6 default ->
    `stitch` CLI merges the head back), plus generate_offline_data /
    regenerate_responses CLIs for self-distilling OUR prompt corpus through
    OUR model (the exact fix for the distribution-mismatch failure that
    killed QEngine's head and Holodeck-Lounge's fine-tune). Red Hat
    published the FastMTP-heads serving recipe (developers.redhat.com,
    2026-09). Head-only training (~15 tensors, ~850M params, 1 decoder
    layer + projections) fits one 3090 in days. ADOPTION GATE: seeded
    conc_ladder (N=1 x3, salted, 4096-ctx) must beat 2.77 tok/step — the
    number both heads scored in tonight's protocol A/B. Sequence:
    item 22 done -> daily traffic accumulates -> convert/distill/train ->
    stitch -> gate -> swap.
    Evidence base: bench/spec counters before/after on the same seeded
    battery; keep the A/B on a dedicated boot (club-3090 methodology).
    PHASE A EXECUTED 2026-11-01 (pre-A/B state):
    - Layout audit PASSED: QEngine's model_mtp_ft.safetensors carries the
      identical 15-tensor mtp.* set (same names, dtypes BF16, shapes) as our
      checkpoint's model_extra_tensors.safetensors. 1 of 15 tensors is
      byte-identical (mtp.pre_fc_norm_embedding.weight); 14 differ — a real
      fine-tune, not a re-upload.
    - A/B checkpoint built:
      /mnt/500gb-ssd/ai/models/leminkozey-Qwen3.8-27B-Uncensored-W4A16-AutoRound-mtpft/
      — hardlinks of all original shards/config + model_extra_tensors.safetensors
      replaced with the QEngine head verbatim (sha256 56cbd7d30e07ed7c...,
      size 849,400,392 B = QEngine release v1.3.3.1 asset). Metadata block
      NOT grafted (header lengths differ; engines read mtp.* by name).
      Original checkpoint untouched.
    - A/B arm: MODEL=/app/models/leminkozey-...-mtpft on the gpu2 compose
      (MODELS_DIR mount covers it), served under the same qwen3.8-27b name.
    - A/B NOT YET RUN — needs an idle window on GPU2 + the seeded battery
      (bench counters before/after, dedicated boot). Baseline for comparison:
      tok/step 1.89 / accept 63.1% / per-pos 0.80/0.62/0.47 (live 2026-11-01).
    A/B EXECUTED 2026-11-01 (gpu2, idle window, mtpft checkpoint booted at
    GPU_UTIL 0.94 on hyperqwen-pthfa:032-b; boot clean, KV 203,831 tokens,
    greedy first-request sanity clean, no preemptions, salted prompts = 0
    cache hits):
    - run_benchmarks.sh single (real-prompt cohorts): tok/step 2.56-2.74
      across C1..C8, best 2.74 greedy C1; e2e 72-79 tok/s at C1 — but these
      are NOT comparable to the lifetime-average counters (different prompt
      mix than the engine's accumulated history).
    - conc_ladder seeded A/B (the citable comparison; same protocol as the
      2026-10-31 unlock rebaseline: N=1, 3 reps, 4096-ctx, out=256, salted):
        mtpft: per-stream 80.1 tok/s (78.1/80.2/82.1) | ms/pass 34.6 | tok/step 2.77
        stock: per-stream 83.0 tok/s             | ms/pass 33.2 | tok/step 2.77
        delta: -3.4% per-stream, +4.4% ms/pass, tok/step FLAT
    - Full-engine counters over the whole arm: accept 56.0% / 1.68 tok/step
      vs lifetime 63.1% / 1.89 (mixed warmup + battery traffic).
    VERDICT: REJECT the QEngine head for our lanes. Equal acceptance on the
    seeded battery (2.77 = 2.77), slightly slower per pass, -3.4% per-stream;
    the ~8-9% upstream claim does NOT transfer to our traffic/checkpoint.
    Their head was trained on QEngine's own answers (llama.cpp-style serving,
    RedHat INT4); ours is vLLM + Leminkozey W4A16 — distribution mismatch
    confirmed empirically, matching the Holodeck-Lounge negative result.
    Phase A closed 2026-11-01. Phase B (train OUR OWN head on OUR traffic via
    speculators/FastMTP) remains the live path; the A/B harness and the
    mtpft checkpoint dir are kept for exactly that comparison. GPU2 restored
    to stock after the test.

22. [ ] Abliterate Qwopus3.8-27B-Coder-Flash with Heretic v2 + requant to
    fleet format. First-mover: NO abliterated variant of this 2026-10-31
    release exists (verified against the full HF model tree 2026-11-01 —
    only older Qwopus 3.5/3.6 generations and Qwen3.8-Flash-Coder have
    abliteration/heretic derivatives).
    MODEL: Jackrong/Qwopus3.8-27B-Coder-Flash (unsloth/Qwen3.8-27B fine-tune,
    qwen3_5 arch, agentic-coding focused: ~20K agent/tool-use examples
    distilled from Qwen3.8-Flash-Max + DeepSeek-V4-Flash teachers + Nemotron
    reasoning data; multimodal; Apache-2.0). Downloaded 52GB/18 shards to
    /mnt/500gb-ssd/ai/models/Qwopus3.8-27B-Coder-Flash.
    RUN (launched 2026-11-01): heretic-llm 2.0.0 venv at
    /mnt/500gb-ssd/ai/heretic-run (torch 2.11.0+cu128, transformers 5.22.1),
    config.toml: bf16, quantization=none, device_map auto,
    max_memory={'0':'22GB','1':'22GB','cpu':'26GB'} (both 3090s; heretic CUDA
    enumeration ≠ nvidia-smi order — its CUDA2 is the 3060 display card,
    excluded; first-launch OOM gotcha recorded), offload_outputs_to_cpu=true,
    n_trials=100, ARA modifier (lora_rank 50, row-magnitude preservation),
    KeywordRate+KLDivergence scorers, checkpoints/ for study resume.
    DECISION: pure abliteration (refusals only) first — noslop/nosyco are
    separate runs (one keyword-list objective per run; combined config
    possible via scorer instance_name but doubles trial eval cost).
    Quantization accuracy note: bnb_4bit mode verified safe from source —
    trials search on nf4-dequantized-to-fp32 weights; final merge RELOADS
    bf16 from disk on CPU, so output is always original-precision + delta.
    We run quantization=none (both 3090s available) so the question is moot.
    QUANT PLAN (llm-compressor, the vllm-project-maintained path):
    a. MTP survival check on heretic's export (heretic touches only
       attn.o_proj/mlp.down_proj; re-attach pristine mtp.* from the
       untouched download if the re-shard drops them).
    b. Body: llm-compressor oneshot with AutoRound modifier, W4A16 g128
       symmetric compressed-tensors — same algorithm family as the
       Leminkozey recipe (Intel AutoRound standalone), implemented and
       CI-tested inside vllm-project; official Qwen3.8-27B 4-bit key-model
       example is the reference. Keep vision tower + in_proj bf16 per
       recipe. (NVFP4/MXFP4 rejected: Blackwell formats, no sm86 tensor
       cores, measured W4A16-parity on 3090.)
    c. Heads: repo prepare/ requant after the body pass — quant_lm_head.py
       (BITS=8 GPTQ-recalibrated), quant_embed.py (int8), quant_heads_stream
       (HEAD_BITS=8, MTP_BITS=8) with the repo 40,960-id draft vocab list;
       rebuild the draft-vocab list from Qwopus-Coder's own token
       distribution if MTP acceptance reads depressed (#139 Swift precedent:
       25.9k distinct ids, 99.8% coverage, deficit gone).
    d. Validate: boot CTX=long + SPEC=mtp, #404 garble check, quality
       battery (PPL+GSM8K vs repo numbers), seeded conc_ladder A/B vs
       Leminkozey baseline (83.0 tok/s, 2.77 tok/step, protocol of
       2026-10-31/11-01). Honest comparison: both models through the same
       quant pipeline.
    NOTE: while this run is live, both vLLM lanes are DOWN (approved).
    Restore = qwen38-super-pthfa-on gpu0 + gpu2 after export.
 - 0.32-rebase candidates recorded 2026-10-16 (#59973 quantized draft lm_head,
  #46963 NVFP4 paged KV, #57881 Mamba ops out of CUDA graphs, #61422/#59735
  free riders): several landed with the 0.32 series; re-diff the remaining
  ones against the served image before scheduling any.

### 2026-11-02 third-party scan: bradj00/sparsekv-hyperqwen (reviewed, not adopted)

Query-aware sparse KV decode + int4 pinned-host-RAM KV tier for HyperQwen
vLLM 0.31 + pthfa. Headline: 8 concurrent 250K agents on a 3090 Ti + 3090
(Qwen3.8-27B W4A16, MTP-7). Reviewed 2026-11-02 against our rig (README +
eval/kld methodology + their rig specs). Status: WATCH-LIST DESIGN REFERENCE
for a future multi-stream deep-context tier; NOT adoptable as-is.

21. [x] sparsekv review recorded. Key facts (all from the repo):
    - Mechanism: 16 of 64 layers are full attention (48 GDN linear).
      Per decode step, a frozen rank-64 int4 key sketch (basis27.pt =
      eigenvectors of real post-RoPE keys, prefix-cache compatible) scores
      all tokens; exact pthfa int8 attention runs on top-2048 + 64 sinks +
      last 1024. Full KV lives in pinned host RAM at int4 (4.4 GB/250K
      agent); GPU keeps sketch + sinks + tail + 2.5% CLOCK hot buffer;
      UVA zero-copy host reads ~10 GB/s (x16) / 6.2 GB/s (x4), 14-16% miss.
    - Quality (teacher-forced KLD vs dense, same card, 60 samples, temp 0,
      MTP OFF): sparse GPU-only 0.0188 KLD / 96.4% top-1 — BETTER than the
      dense-cross-card noise floor (0.0200 / 96.2%), a well-designed
      control. Production int4 tier 0.0269 / 95.6%; recall tasks 99.6%
      top-1; divergence grows 0.024@32K -> 0.033@192K; prose 2x code.
    - Their own honest notes: missed their 300 tok/s/card target
      (268-285); greedy free-running output drifts after first divergent
      token; tier hooks "monkeypatch vLLM scheduler internals and are
      fragile across vLLM upgrades"; no 5th agent/card at MTP-7.
    FIT BLOCKERS for us (each verified against our rig 2026-11-02):
    1. vLLM base: they built on 0.31 @ ce9615c; we run 0.32.0 + the
       65-patch black-monolith series. Their fragility warning applies to
       exactly this jump; port = rebase their scheduler patches + pthfa
       sparse hook onto our series, with a standing upgrade tax.
    2. Host RAM: they have 96 GB (78 GB in VM); we have 45 GB total,
       ~7 GB free (ComfyUI resident 16.2 GB). Their single-server config
       reserves 17.8 GB pinned; the 8x250K headline needs ~35 GB. Their
       configs do not fit this box.
    3. MTP: all their quality numbers ran MTP OFF (custom logits
       processors rejected under spec decode). Our production runs MTP k3;
       the fidelity numbers do not transfer to our serving state.
    4. Basis provenance: basis27.pt is fitted on STOCK Qwen3.8-27B keys;
       qwopus-heretic is a fine-tune with shifted key distributions ->
       refit required (their fit_basis27.py is included) + revalidation.
    THE INTERESTING ANGLE: it attacks exactly our tier's constraint
    (MAX_SEQS=1 = one 200K stream per card). Demonstrates 2-4 concurrent
    200K+ streams per 24 GB card at ~95.6% top-1 fidelity; single-user
    A/B at ~113K showed +15-36% decode vs dense on the SAME card. Our x4
    (GPU0) / x16 (GPU2) link split matches their rig's, reversed.
    RE-ENGAGEMENT TRIGGERS (any one): (a) Chris wants concurrent 200K+
    Hermes agents on one card (the only feature we currently lack);
    (b) a 0.32+ rebase of the sparse patches appears (theirs or fork);
    (c) host RAM upgrade to 96+ GB. PORT CHECKLIST if triggered: rebase 3
    patches + pthfa hook onto our series -> refit basis on qwopus-heretic
    keys -> KLD-validate WITH MTP k3 ON -> RAM budget (needs ~20 GB free;
    ComfyUI would have to move). Complementary repo: sparsekv-lab (same
    author; research notes behind the release).

## Conventions

- `black-monolith` is THE deployment branch: all custom work (router, launchers,
  configs) plus merged upstream PRs. This is the checked-out branch that builds
  the serving image and backs the local AI tray's Qwen tier.
- `main` tracks `origin/main` (this fork's copy of upstream). Upstream sync via
  the `upstream` remote lands on `main` first, then merges into `black-monolith`.
- Upstream PRs are merged into `black-monolith` after validation.
- Completed roadmap items move to ROADMAP-COMPLETED-<date>.md with their
  evidence (Chris, 2026-10-31).
- Never commit bench data (`bench-quality-data/`, `quantize.log`) or any
  credential material. Tuning-only `.env` files are allowed.
