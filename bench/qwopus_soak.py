#!/usr/bin/env python3
"""Stability soak for qwopus-heretic: N rounds of concurrent mixed work; watch for
output degradation, engine death, spec-acceptance decay, preemptions.
Usage: qwopus_soak.py [api] [key] [rounds]
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18020"
KEY = sys.argv[2] if len(sys.argv) > 2 else "sk-local"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 25

def post(payload, timeout=600):
    req = urllib.request.Request(API.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ask(q, max_tokens=420, temp=0.5):
    d = post({"model":"qwen3.8-27b","messages":[{"role":"user","content":q}],
              "max_tokens":max_tokens,"temperature":temp,
              "chat_template_kwargs":{"enable_thinking":False}})
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "finish": d["choices"][0].get("finish_reason")}

TASKS = [
    "Summarize the causes of World War 1 in 4 bullet points.",
    "Write a Python function `primes_upto(n)` returning all primes below n using the sieve. No tests.",
    "Translate to French: 'The engine burns fuel to produce thrust, which pushes the aircraft forward.'",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Show work.",
    "Write the opening paragraph of a noir detective story set on Mars.",
    "From this text list every date mentioned: 'On 3 March 1887, Tesla arrived in New York. By July 1888 he had patented the induction motor. He died on 7 January 1943.'",
]

def ngram_rep(t, n=8):
    w = t.split()
    if len(w) < 2*n: return 0.0
    g = [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    return 1 - len(set(g))/len(g)

def read_counter(name):
    try:
        with urllib.request.urlopen(API.rstrip("/") + "/metrics", timeout=5) as r:
            body = r.read().decode()
        m = re.search(rf"^{name}(?:\{{[^}}]*\}})? (\S+)", body, re.M)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0

rows = []
m0 = read_counter("vllm:spec_decode_num_accepted_tokens_total")
d0 = read_counter("vllm:spec_decode_num_draft_tokens_total")
t0 = time.time()
for rnd in range(1, ROUNDS+1):
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(ask, TASKS))
    m = read_counter("vllm:spec_decode_num_accepted_tokens_total")
    dd = read_counter("vllm:spec_decode_num_draft_tokens_total")
    da = m - m0
    ddd = dd - d0
    accept = da/ddd if ddd else 0.0
    row = {"round": rnd, "elapsed_s": round(time.time()-t0),
           "empty": sum(1 for r in results if not r["content"].strip()),
           "nonstop": sum(1 for r in results if r["finish"] != "stop"),
           "max_rep": round(max(ngram_rep(r["content"]) for r in results), 4),
           "accept_rate": round(accept, 3)}
    rows.append(row)
    print(json.dumps(row), flush=True)
    m0, d0 = m, dd

preempt = read_counter("vllm:num_preemptions_total")
final = {"rounds": ROUNDS, "wall_s": round(time.time()-t0),
         "total_empty": sum(r["empty"] for r in rows),
         "total_nonstop": sum(r["nonstop"] for r in rows),
         "max_rep_any": max(r["max_rep"] for r in rows),
         "first_accept": rows[0]["accept_rate"], "last_accept": rows[-1]["accept_rate"],
         "preemptions": preempt}
print(json.dumps({"summary": final}, indent=1))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_qwopus-soak.json")
with open(out, "w") as f:
    json.dump({"rows": rows, "summary": final}, f, indent=1)
print("wrote", out)
