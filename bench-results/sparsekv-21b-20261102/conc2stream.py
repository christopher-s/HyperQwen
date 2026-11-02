import json, threading, time, urllib.request

BASE = "/home/chris/tools/hyperqwen/bench-results/sparsekv-21b-20261102"
docs = {
    "A": (open(f"{BASE}/conc_docA.txt").read(), "4471-908-263", "DELPHINE"),
    "B": (open(f"{BASE}/conc_docB.txt").read(), "5560-114-782", "TORSTEN"),
}
results = {}

def stream(tag, doc, comb, cust):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{
            "role": "user",
            "content": (f"Document:\n{doc}\n\nQuestion: What is the vault combination and "
                        "who is the custodian? Answer using only the document."),
        }],
        "max_tokens": 2000,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:18025/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            d = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        results[tag] = {"error": repr(e), "wall_s": round(time.time() - t0, 1)}
        return
    m = d["choices"][0]
    c = m["message"]["content"] or ""
    results[tag] = {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "finish": m["finish_reason"],
        "needle_comb": comb in c,
        "needle_cust": cust.upper() in c.upper(),
        "content_head": c[:140],
    }

t0 = time.time()
threads = [threading.Thread(target=stream, args=(t, *docs[t])) for t in ("A", "B")]
for th in threads:
    th.start()
for th in threads:
    th.join()
total = round(time.time() - t0, 1)
results["_total_wall_s"] = total
json.dump(results, open(f"{BASE}/conc2stream.json", "w"), indent=1)
print(json.dumps(results, indent=1))
