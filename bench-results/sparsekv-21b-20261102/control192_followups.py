import json, sys, time, urllib.request

BASE = "/home/chris/tools/hyperqwen/bench-results/sparsekv-21b-20261102"
doc = open(f"{BASE}/needle_doc_190k.txt").read()

def ask(question, out, max_tokens=800):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {question}"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:18025/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    m = d["choices"][0]
    content = m["message"]["content"] or ""
    res = {
        "q": question[:60],
        "wall_s": round(dt, 1),
        "ttft_est": None,
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "cached_tokens": d["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "finish": m["finish_reason"],
        "content_head": content[:160],
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return content

q1 = ask("What is the vault combination and who is the custodian? Answer using only the document.",
         f"{BASE}/control192_q1.json", max_tokens=2000)
print("Q1 needle:", "8314-772-906" in q1 and "MARGARET" in q1.upper())

q2 = ask("Name the function used to serialize the session cache, quoting its exact name from the document. One line.",
         f"{BASE}/control192_q2.json", max_tokens=600)
print("Q2 done; cached:", json.load(open(f'{BASE}/control192_q2.json'))["cached_tokens"])

q3 = ask("What is the vault combination? One line only.",
         f"{BASE}/control192_q3.json", max_tokens=400)
print("Q3 needle:", "8314-772-906" in q3)
