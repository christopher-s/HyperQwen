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
2. [ ] At the next vLLM pin bump, re-evaluate every `Retires-when:` line in
    patches/ (grep "Retires-when", diff each against the actual version). The
    trigger event ("next pin bump") fired with the 2026-10-31 0.32 adoption;
    56 patch files carry the marker. [CHEAP] — mechanical grep + diff review;
    retire-and-rebuild only where the premise no longer holds.
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

9. [ ] One-lane admission control: router admission reflects MAX_SEQS=1 — at
   most one forwarded inference request per GPU lane; occupied lane returns a
   retryable busy response immediately (lane + retry guidance) instead of
   holding a silent socket; concurrency tests per lane including busy-lane +
   healthy-sibling. Keep Bifrost's 300 s stream-idle timeout unchanged during
   this work. [CHEAP implementation, cheap test] — a small router change;
   testing is two simultaneous curls, not a soak. The payoff (no silent 5-minute
   stalls) is directly felt in normal use.

Acceptance: no same-lane request waits silently into the stream-idle timeout;
sibling-lane traffic independent.

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

12. [ ] Loop-defense injection: the engine natively supports
    `repetition_detection` per request (vLLM 0.32 SamplingParams:
    RepetitionDetectionParams{max_pattern_size, min_pattern_size, min_count};
    ends generation early on a repeated N-gram). First arm:
    {"max_pattern_size": 32, "min_pattern_size": 1, "min_count": 8}.
    Decide the injection point (router default vs launcher --repetition-detection
    vs leave per-request) and whether triggers need the full diagnostic log
    (item 13) or just a lane+request-ID line. CHEAP: it is a parameter pass,
    a few minutes to wire, minutes to verify with a deliberately looped prompt.
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
