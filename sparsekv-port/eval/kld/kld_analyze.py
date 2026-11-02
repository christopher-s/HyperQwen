"""Per-token KL(ref || run) over decode steps, from the recorder's .npz files.

KL is computed on the reference's top-K support (K=1024) plus one tail bucket holding the remaining mass:
  KL = sum_{i in topK(P)} p_i (log p_i - log q_i) + p_tail log(p_tail / q_tail)
This is a lower bound on the full-vocabulary KL (merging outcomes never increases KL); with K=1024 the reference
tail mass is reported and is tiny. When P is another forced run (same-card comparisons), P's log-probs are
the ones it recorded on the same reference top-K ids, so the same partition (top-K ids + tail) is used. Step 0 (the first generated token) comes from the dense prefill in every
config and is excluded; steps 1..n-1 are decode steps.
Usage: python kld_analyze.py DATA OUT_ROOT [--md results.md] [--json results.json]
  OUT_ROOT/ref/<sid>.ref.npz, OUT_ROOT/<cfg>/<sid>.<mode>.npz for each run below.
"""
import argparse
import json
import os

import numpy as np

# Runs: GPU1 (RTX 3090) = the dense reference server (ref) and the int4-tier server (tier4);
# GPU0 (RTX 3090 Ti) = one SPARSE_KV server that decodes each sample twice, dense and sparse (toggle/dense,
# toggle/sparse), and the int8-tier server (tier8).
# (label, P = (subdir, mode) or None for the reference, Q = (subdir, mode), expect_sparse)
RUNS = [
    ("noise floor: dense vs dense, other card + other server", None, ("toggle", "dense"), False),
    ("noise floor: dense vs dense, same server repeat", ("toggle", "dense"), ("toggle", "dense2"), False),
    ("sparse GPU-only vs dense, same card + same server", ("toggle", "dense"), ("toggle", "sparse"), True),
    ("sparse + RAM tier int8 vs dense, same card", ("toggle", "dense"), ("tier8", "tier8"), True),
    ("sparse + RAM tier int4 (production) vs dense, same card", None, ("tier4", "tier4"), True),
    ("sparse GPU-only vs dense, other card", None, ("toggle", "sparse"), True),
    ("sparse + RAM tier int8 vs dense, other card", None, ("tier8", "tier8"), True),
    ("sparse + RAM tier int4 (production) vs dense, other card", ("toggle", "dense"), ("tier4", "tier4"), True),
]


def kl_steps(ref, run, pref=None):
    """KL(P || Q) per decode step on the reference's top-K support + tail bucket. P = the reference itself,
    or (pref) another forced run's log-probs on the same support; Q = run."""
    n = min(len(ref["token"]), len(run["token"]), len(pref["token"]) if pref is not None else 10 ** 9)
    t = np.arange(1, n)
    p_lp = (ref["top_lp"] if pref is None else pref["q_ref"])[t].astype(np.float64)
    q_lp = run["q_ref"][t].astype(np.float64)
    p = np.exp(p_lp)
    ptail = np.clip(1.0 - p.sum(1), 0.0, 1.0)
    qtail = np.clip(1.0 - np.exp(q_lp).sum(1), 1e-12, 1.0)
    kl = (p * (p_lp - q_lp)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = np.where(ptail > 1e-12, ptail * np.log(ptail / qtail), 0.0)
    kl = np.maximum(kl + tail, 0.0)
    p_top = ref["token"][t] if pref is None else pref["top_id"][t, 0]
    top1 = run["top_id"][t, 0] == p_top
    forced_ok = run["inp"][t] == ref["token"][t - 1]
    return kl, top1, forced_ok, run["sparse"][t], ptail


def stats(kl, top1):
    return {"n": int(len(kl)), "mean": float(kl.mean()), "median": float(np.median(kl)),
            "p95": float(np.percentile(kl, 95)), "p99": float(np.percentile(kl, 99)), "max": float(kl.max()),
            "top1": float(top1.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("root")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    samples = json.load(open(os.path.join(a.data, "samples.json")))["samples"]
    res, md = {}, []
    groups = {}
    problems = []
    for label, P, (sub, mode), want_sparse in RUNS:
        for s in samples:
            rp = os.path.join(a.root, "ref", f"{s['sid']}.ref.npz")
            fp = os.path.join(a.root, sub, f"{s['sid']}.{mode}.npz")
            pp = os.path.join(a.root, P[0], f"{s['sid']}.{P[1]}.npz") if P else None
            if not (os.path.exists(rp) and os.path.exists(fp) and (pp is None or os.path.exists(pp))):
                continue
            try:
                zr, zf, zp = np.load(rp), np.load(fp), (np.load(pp) if pp else None)
            except Exception:  # missing/placeholder file: sample not run for this config
                continue
            kl, top1, fok, sp, ptail = kl_steps(zr, zf, zp)
            if not fok.all():
                problems.append(f"{label} {s['sid']}: forced path mismatch at {int((~fok).sum())} steps")
            if want_sparse and not sp.all():
                problems.append(f"{label} {s['sid']}: {int((~sp).sum())}/{len(sp)} decode steps not sparse")
            if not want_sparse and sp.any():
                problems.append(f"{label} {s['sid']}: {int(sp.sum())} dense steps ran sparse")
            for key in [(label, s["L"]), (label, "all"), (label, "task:" + s["task"]),
                        (label, "doc:" + s["doc"].split("_")[0])]:
                g = groups.setdefault(key, {"kl": [], "top1": [], "smean": [], "ptail": []})
                g["kl"].append(kl); g["top1"].append(top1); g["smean"].append(kl.mean()); g["ptail"].append(ptail)
    for (label, k), g in groups.items():
        st = stats(np.concatenate(g["kl"]), np.concatenate(g["top1"]))
        st["samples"] = len(g["smean"])
        st["sample_mean_min"], st["sample_mean_max"] = float(min(g["smean"])), float(max(g["smean"]))
        st["ref_tail_mass_mean"] = float(np.concatenate(g["ptail"]).mean())
        res.setdefault(label, {})[str(k)] = st
    lengths = sorted({s["L"] for s in samples})
    md.append("| config | context | samples | tokens | mean KLD | median | p95 | p99 | max | top-1 agree |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, *_ in RUNS:
        for k in [*lengths, "all"]:
            st = res.get(label, {}).get(str(k))
            if not st:
                continue
            ctx = f"{k / 1024:.0f}K" if k != "all" else "**all**"
            st["ref_tail"] = st["ref_tail_mass_mean"]
            md.append(f"| {label} | {ctx} | {st['samples']} | {st['n']} | {st['mean']:.5f} | {st['median']:.6f} | "
                      f"{st['p95']:.4f} | {st['p99']:.4f} | {st['max']:.3f} | {100 * st['top1']:.2f}% |")
    md.append("")
    md.append("| config | slice | samples | mean KLD | p99 | top-1 agree |")
    md.append("|---|---|---|---|---|---|")
    for label, *_ in RUNS:
        for k in sorted(x for x in res.get(label, {}) if ":" in x):
            st = res[label][k]
            md.append(f"| {label} | {k} | {st['samples']} | {st['mean']:.5f} | {st['p99']:.4f} | {100 * st['top1']:.2f}% |")
    md.append("")
    md.append("Compact: mean KLD (nats/token) and top-1 agreement per context length")
    md.append("")
    md.append("| comparison | " + " | ".join(f"{L / 1024:.0f}K" for L in lengths) + " | all |")
    md.append("|---|" + "---|" * (len(lengths) + 1))
    for label, *_ in RUNS:
        cells = []
        for k in [*lengths, "all"]:
            st = res.get(label, {}).get(str(k))
            cells.append(f"{st['mean']:.4f} / {100 * st['top1']:.1f}%" if st else "n/a")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    out = "\n".join(md)
    print(out)
    print("\nchecks:", "OK (all forced paths matched the reference; sparse/dense flags as configured)"
          if not problems else "\n  " + "\n  ".join(problems))
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump({"results": res, "problems": problems}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
