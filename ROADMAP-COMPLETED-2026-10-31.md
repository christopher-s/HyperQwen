# Black Monolith — completed roadmap archive

Moved from ROADMAP.md 2026-10-31 (Chris: completed items move to a separate file).
Each entry preserves its completion evidence. Numbering follows the original
ROADMAP numbering at the time of completion.

## 2026-10-31 Plan A: 0.31 cherry-picks (superseded same day by Plan B)

29. [x] Snapshot the dirty working tree before branch work.
    (Working-tree changes left uncommitted by agreement; carried intact through
    the branch work instead of stashed.)

30. [x] After the gpu unlock completed, re-baseline: one conc_ladder pass per
    lane on the stock image as the A/B control.
    (Unlock confirmed 2026-10-31: both 3090s 250 W cap, max_limit 375 W,
    max SM 2100 MHz. Baselines in bench-results/unlock-rebaseline-20261031/.)

31. [x] Cherry-pick the 0.31 cuts of #404 (KV scale-view page preservation) and
    #396 (gdn-ba-pad-small-m) onto the pthfa base; rebuild; deploy one lane.
    (Done as branch unlock-upgrade-a: #332 seat-budget dependency added, then
    #396 and #404; image hyperqwen-pthfa:unlock-a, in-container gate passed
    fuzz 0. Deployed gpu0 first, then gpu2 after gates.)

32. [x] A/B upgraded lane vs post-unlock control.
    (ADOPTED 2026-10-31, with a corrected band: measured ms/pass -1.5 to -2.4%
    at our 4K-ctx profile — upstream's -4.8% was short-prompt; decode +3.6-9.3%;
    all gates clean: first-request "68" both lanes, KV pool unchanged 203,831,
    no loops, QUARTZ-77 exact at 186,195 prompt tokens, 0 preemptions at N=4.
    Full record: bench-results/unlock-rebaseline-20261031/RESULTS.md. pr326
    retained as rollback. Probe artifacts logged, both pre-existing xhigh
    behaviors: content=null when max_tokens is exhausted by reasoning, and
    essay re-drafting inside reasoning inflating naive n-gram counts.)

## 2026-10-31 Plan B: 0.32 bump (both lanes, adopted live)

33. [x] Revalidate the anti-loop stack on 0.32 before touching profiles.
    (Verified against upstream/main single-user/start_qwen.sh: ASYNC_SCHED=0
    remains the shipped default on long-spec paths and the launcher still maps
    it to --no-async-scheduling; PIECEWISE remains the mandated CG mode with
    the gotcha-37 warning that FULL corrupts one prompt in 128, MTP included.
    vLLM #51599 exists per third-party source but is moot while upstream keeps
    sync scheduling. The effort guard and repetition_penalty live in
    hyperqwen_router.py, which the merge carried untouched. The xhigh
    template-guard layer itself was retired the same day — see the
    reasoning-effort section below.)

34. [x] Take upstream main (#415 + re-cut series) onto black-monolith; enable
    PTH_FA2_SLAB_PREFILL=1 with the #422 takeover fix; re-run verification;
    confirm the pool fits.
    (DONE as branch unlock-upgrade-b, merge c73dd35 of upstream/main f382342 —
    56-entry 0.32 series supersedes the Plan A 0.31 picks; our router/.env/
    ROADMAP carried through; serve-404-served-names dropped per upstream's 0.32
    re-export. Image hyperqwen-pthfa:032-b, in-container gates all PASS incl.
    verify 0 failures. SIZING CORRECTION vs the plan's guess: at GPU_UTIL=0.93
    the slab workspace leaves only 197,472 tokens and the fixed 200,000 MAX_LEN
    refuses to boot (engine ValueError). GPU_UTIL=0.94 boots cleanly: 203,831
    -token pool with slab reserved. Gotcha-4's failure band is 0.95+; 0.94
    keeps margin and long-run validation is still owed at depth. Adopted on
    BOTH lanes at 0.94 + slab. Measured: slab cold prefill 186K prompt 205 s
    gpu0 / 202 s gpu2 (~908/920 tok/s) vs 242-243 s no-slab (~769 tok/s),
    +16-18%; a second seed-777 192K sample ran 894 tok/s. Slab decode within
    noise (33.9-34.7 ms/pass; no regression). Gates: first-request 68 clean on
    both lanes, QUARTZ-77 exact at 186,195 and again on both slab lanes,
    OBSIDIAN-42 exact at 192,376, vision ZENITH-77 exact in 1.0 s, N=4 zero
    preemptions, router :8020 serving both lanes. Pool at 0.32 without slab:
    205,363 (+1,532 vs 0.31 same UTIL). Remaining debt moved to open roadmap
    item 31-open: full 195,670+6,000 capacity stress and long-soak at 0.94.
    0.31 fallbacks retained: unlock-a, pr326.)

35. [x] Update router/Bifrost metadata and tray labels only after both lanes
    pass; retain the 0.31 image and launchers for rollback.
    (Router :8020 verified serving qwen-gpu0/qwen-gpu2 after the swap; both
    advertise 200000 via the engines, no metadata edit needed — Bifrost rows
    already read context_length from the engines' /v1/models at the Oct-30
    fix. Tray labels unchanged by design. NOTE: launcher env vars renamed
    QWEN38_SUPER_* -> QWEN38_PTHFA_* (Chris: SUPER is a specific finetune,
    not useful in code); tests updated, 5 passed + 2 subtests; local-ai-control
    main pushed 68c4dcf. GPU_UTIL env-overridable (QWEN38_PTHFA_GPU_UTIL,
    default 0.93; lanes run 0.94 with slab). Rollback set: unlock-a, pr326,
    fp8 200K launchers.)

## 2026-10-31 reasoning-effort policy + max_tokens (items 2, 15, 36-39)

36. [x] Document the effort policy and cap decision.
    (ROADMAP section "2026-10-31 research: reasoning-effort policy"; key
    findings: the served Leminkozey template lacks the SuperQwen template's
    xhigh anti-repeat/512-bound text, so the old forcing bought nothing;
    vLLM 0.32 accepts the OpenAI effort vocabulary at the protocol layer;
    thinking_token_budget is a typed, flowing request parameter; the router's
    effort overwrite dead-lettered Hermes's configured medium.)

37. [x] Implement in the router behind an env kill-switch.
    (Commit e908d0f on black-monolith, pushed. QWEN_ROUTER_EFFORT_POLICY=v1
    default: caller effort validated + nearest-level mapped
    [none->none; minimal,low->low; medium->medium; high,xhigh,max->xhigh;
    absent->medium]; unknown values map to the default instead of reaching the
    template raise; =legacy restores force-xhigh as rollback.
    QWEN_ROUTER_MAX_TOKENS configurable injected default. router-on passes
    both env vars to the systemd unit. 13/13 mapping unit cases pass; legacy
    rollback verified live.)

38. [x] Run the A/B (gpu2 lane, 2026-10-31 12:40-12:54, 4 arms x 3 reps x 2
    probe shapes; bench/effort_ab_20261031.py, raw record
    bench-results/unlock-rebaseline-20261031/effort-ab-20261031.txt).
    RESULTS — (a) legacy xhigh+8192: essay 3/3 finish=length, content_chars=0,
    26.8-28.4K reasoning chars, 199-714 repeated 8-grams, ~89 s wasted per
    request. (b) medium+16384: 3/3 finish=stop, 5.0-5.6K content chars,
    ~1.1-1.2K reasoning chars, loops 0, ~22 s. (c) medium+32768: 3/3
    finish=stop, 5.2-5.6K content, loops ~0, ~22 s — indistinguishable from
    16K at medium. (d) xhigh+32768: 3/3 finish=stop but 10.9-17.2K completion
    tokens with 33-47K reasoning chars vs ~6K content, loops8 1067-2282,
    112-171 s — 5x wall time for +0.5K content chars. Tool calls: clean JSON
    args, zero malformed, all arms. ADOPTED: v1 policy (default medium) +
    16384 injected default. Resolves open items 2 (reasoning-policy A/B) and
    15 (output ceiling: medium default, 16,384 ceiling, per-request override
    honored; 32K available to explicit callers and matches Bifrost's
    advertised max_output_tokens).

39. [x] Post-adoption sync.
    (Hermes config already set reasoning_effort medium for qwen-gpu0|gpu2 —
    that setting is now live instead of dead-lettered; verified both lanes
    answer through the restarted router with the v1 policy active. Bifrost
    max_output_tokens row reads 32768 for both lanes: fine as a caller-side
    ceiling, no change needed. Ops skill
    winston-skills local-ai/control/local-ai-operations/
    references/qwen38-super-pthfa-fast-tier.md updated: Router request policy
    section added, xhigh template-guard layer moved to historical, rollback
    table corrected. Winston-skills commit 7d6b3dd pushed.)
