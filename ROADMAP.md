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

## Watch list (upstream, no action yet)

- #424 retain SWA history at sparse Mamba offload boundaries: relevant only
  if a CPU KV-offload tier returns; fixes a DFlash2 offload-lookup zero-hit
  confirmed on f382342; unmerged, author-validated on H200/A4000.
- #425 interior GDN checkpoint export on 0.32: author holds it open on an
  unmet acceptance criterion (only parity observed, default-off); experimental.
- #423 ready-made Cyrillic draft vocab: RU-heavy traffic only; full head
  measured ~2% slower on English. Adopt only if RU workload appears.
- 0.32-rebase candidates recorded 2026-10-16 (#59973 quantized draft lm_head,
  #46963 NVFP4 paged KV, #57881 Mamba ops out of CUDA graphs, #61422/#59735
  free riders): several landed with the 0.32 series; re-diff the remaining
  ones against the served image before scheduling any.

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
