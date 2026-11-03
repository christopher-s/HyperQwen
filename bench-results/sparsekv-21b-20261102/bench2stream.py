import json, sys, threading, time, urllib.request
BASE = "/home/chris/tools/hyperqwen/bench-results/sparsekv-21b-20261102"
docs = {
    "A": (open(f"{BASE}/{sys.argv[1]}").read(), sys.argv[3].split("|")[0], sys.argv[3].split("|")[1]),
    "B": (open(f"{BASE}/{sys.argv[2]}").read(), sys.argv[4].split("|")[0], sys.argv[4].split("|")[1]),
}
results = {}
def stream(tag, doc, comb, cust):
    def body(q, mt, think=True):
        return {"model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {q}"}],
                "max_tokens": mt, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": think, "reasoning_effort": "medium"}}
    def post(b, timeout=2400):
        req = urllib.request.Request("http://127.0.0.1:18025/v1/chat/completions",
                                     data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read()), time.time()-t0
    res = {}
    d, dt = post(body("What is the combination and who is the custodian? Answer using only the document.", 2000))
    c = d["choices"][0]["message"]["content"] or ""
    res["cold"] = {"wall_s": round(dt,1), "prompt_tokens": d["usage"]["prompt_tokens"],
                   "completion_tokens": d["usage"]["completion_tokens"],
                   "finish": d["choices"][0]["finish_reason"],
                   "pass": bool(comb in c and cust.upper() in c.upper())}
    d2, dt2 = post(body("Quote the combination only. One line.", 200), timeout=600)
    res["warm"] = {"wall_s": round(dt2,1), "cached": d2["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0),
                   "pass": comb in (d2["choices"][0]["message"]["content"] or "")}
    b3 = body("Now write exactly 300 tokens summarizing the document's structure. No thinking.", 320, think=False)
    d3, dt3 = post(b3, timeout=600)
    ct = d3["usage"]["completion_tokens"]
    res["decode"] = {"wall_s": round(dt3,1), "completion_tokens": ct,
                     "tok_s": round(ct/dt3,1) if dt3 else None,
                     "cached": d3["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0)}
    results[tag] = res
    json.dump(results, open(f"{BASE}/{sys.argv[5]}", "w"), indent=1)
threads = [threading.Thread(target=stream, args=(t, *docs[t])) for t in ("A","B")]
t0 = time.time()
for th in threads: th.start()
for th in threads: th.join()
results["_total_wall_s"] = round(time.time()-t0,1)
json.dump(results, open(f"{BASE}/{sys.argv[5]}", "w"), indent=1)
print(json.dumps(results, indent=1))
