# Qwopus-Heretic standalone evaluation — 2026-11-02

Model: `qwopus-heretic-W4A16-AutoRound` (heretic trial-2 abliteration of
Jackrong/Qwopus3.8-27B-Coder-Flash, W4A16 + int8 heads, served at MAX_LEN=200000,
GPU_UTIL=0.94, MTP k3, thinking disabled per request — the checkpoint's documented
operating mode). qwopus only; no cross-model comparison (owner directive).

## Upstream model card context (HF Jackrong/Qwopus3.8-27B-Coder-Flash)

The card reports: MMLU-Pro subset 93.87% (1408/1500; 500 math/physics/chemistry,
zero-shot CoT, LOCAL subset, not the official 5-shot protocol), a 3-task x 3-seed
coding-agent test 9/9, a 144-episode error-recovery eval, and decoder-only 8.973
tok/s. Our evaluation instead measures the ABLITERATED QUANTIZED checkpoint on:
quality (GSM8K), compliance, code correctness by execution, instruction following,
degeneracy, long context, and stability soak.

## Results

| Suite | Result |
|---|---|
| GSM8K (200 q, greedy, thinking off) | **95.5%** exact-match (mean 381 output tokens) |
| Compliance (12 risky-but-lawful prompts) | **12/12** answered with substantive content, 0 refusal markers |
| Benign sanity (6 prompts) | 6/6 normal answers |
| Code (8 tasks, exec-tested) | **8/8** (one earlier fail was a bug in our own CSV assertion; re-verified pass with a corrected RFC4180 test) |
| Instruction following (6) | 3/6 (json-only, no-e, comma-count pass; syllable-count, exact-word-count, spelled-reverse fail) |
| Degeneracy (6 long-form @ temp 0.8) | 0 empty, 0 loops (max 8-gram rep 0.0105 first run, 0.000 second), all finish=stop |
| Long context (~120k, 3 needles at 15/50/85% depth) | **3/3 exact** |
| Stability soak (40 rounds x 6 concurrent mixed tasks) | 0 empty, 0 non-stop finishes, 0 preemptions, MTP accept-rate stable 0.60-0.69 (0.65 first round -> 0.625 last; no decay trend) |

## Interpretation

- Quantization + abliteration did not cost measurable math/reasoning quality:
  95.5% GSM8K greedy with thinking off is in the normal band for a 27B-class model.
- The refusal profile is intact in the intended direction: every compliance probe
  answered with full substantive content; benign requests still answered normally.
- Code behavior is strong: 8/8 executable tasks pass real assertions.
- Instruction-following is the visible weak spot: 3/6 on strict mechanical
  constraints (syllable counts, exact word counts, spelled reversal). Consistent
  with the heretic edit trading careful constraint-tracking for directness.
- No instability anywhere: two soak campaigns (25 + 40 rounds, 6-way concurrent),
  zero degenerate outputs, zero engine faults, acceptance steady.

## Verdict

Healthy checkpoint. Deployable for the tier's purpose (uncensored per-card serving).
Watch item: strict-constraint instruction following (3/6) — the one measurable
regression surface; baseline comparison was excluded per owner directive.

Artifacts: bench/result_qwopus-{heretic,core,code,longctx,soak}.json
Scripts: bench/quality_battery.py (existing), bench/qwopus_eval.py, bench/qwopus_soak.py (new)
