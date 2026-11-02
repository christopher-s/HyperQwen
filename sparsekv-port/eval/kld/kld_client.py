"""Drives one server through the KLD sample set.

  role ref     free greedy generation (temperature 0) on a dense server: writes <out>/<sid>.ref.npz
  role toggle  forced along the reference on a SPARSE_KV=1 server started with KLD_TOGGLE=1:
               for each sample a dense-switch run, then a sparse-switch run (same server, same prefill cache)
  role forced  forced along the reference, one run per sample (tier servers), mode label = --mode

Usage: python kld_client.py --role ref --port 8097 --data DATA --out OUT [--ref-dir REFDIR] [--only SID,..]
API key: $KLD_API_KEY_FILE (read, never printed).
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np


def post(port, body, key, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_file(path, timeout):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.5)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ref", "toggle", "forced"])
    ap.add_argument("--mode", default="sparse")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir")
    ap.add_argument("--only")
    ap.add_argument("--switch", help="forced role on a toggle server: dense or sparse")
    ap.add_argument("--model", default="qwen3.8-27b")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.ref_dir = os.path.abspath(a.ref_dir) if a.ref_dir else None
    key = open(os.environ["KLD_API_KEY_FILE"]).read().strip()
    d = json.load(open(os.path.join(a.data, "samples.json")))
    meta, samples = d["meta"], d["samples"]
    if a.only:
        keep = set(a.only.split(","))
        samples = [s for s in samples if s["sid"] in keep or s["doc"] in keep or str(s["L"]) in keep]
    docs = {}
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f"client.{a.role}.{a.mode}.jsonl"), "a")
    for s in samples:
        if s["doc"] not in docs:
            docs[s["doc"]] = np.load(os.path.join(a.data, s["doc"] + ".npy"))
        prompt = meta["H"] + docs[s["doc"]][:s["ndoc"]].tolist() + s["suffix"]
        assert len(prompt) == s["plen"]
        runs = []
        if a.role == "ref":
            runs = [("ref", None)]
        else:
            ref = os.path.join(a.ref_dir, f"{s['sid']}.ref.npz")
            if not wait_file(ref, 7200):
                print("no reference for", s["sid"], flush=True)
                continue
            runs = [("dense", "dense"), ("sparse", "sparse")] if a.role == "toggle" else [(a.mode, a.switch)]
        for mode, switch in runs:
            outp = os.path.join(a.out, f"{s['sid']}.{mode}.npz")
            if os.path.exists(outp):
                continue
            xa = {"kld_id": s["sid"], "kld_mode": mode, "kld_plen": s["plen"], "kld_out": outp,
                  "kld_eos": meta["eos"]}
            body = {"model": a.model, "prompt": prompt, "temperature": 0.0, "max_tokens": s["max_tokens"],
                    "skip_special_tokens": False, "vllm_xargs": xa}
            if mode != "ref":
                n = len(np.load(ref)["token"])
                xa["kld_ref"] = ref
                body["max_tokens"] = n
                body["ignore_eos"] = True
            if switch:
                xa["kld_switch"] = switch
            t0 = time.time()
            try:
                r = post(a.port, body, key)
            except Exception as e:
                print(s["sid"], mode, "ERROR", repr(e)[:300], flush=True)
                log.write(json.dumps({"sid": s["sid"], "mode": mode, "error": repr(e)[:500]}) + "\n"); log.flush()
                continue
            dt = time.time() - t0
            ok = wait_file(outp, 60)
            u = r.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            rec = {"sid": s["sid"], "mode": mode, "secs": round(dt, 1), "prompt_tokens": u.get("prompt_tokens"),
                   "cached": cached, "completion_tokens": u.get("completion_tokens"), "npz": ok,
                   "text": r["choices"][0]["text"][:400]}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"{s['sid']:32s} {mode:7s} {dt:7.1f}s prompt={u.get('prompt_tokens')} cached={cached} "
                  f"gen={u.get('completion_tokens')} npz={ok}", flush=True)


if __name__ == "__main__":
    main()
