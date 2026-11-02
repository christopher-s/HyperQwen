import json, sys, time, urllib.request

doc = open("/home/chris/tools/hyperqwen/bench-results/sparsekv-21b-20261102/needle_doc_190k.txt").read()
assert "PASSPORT-ID-ZK42" in doc
out_path = sys.argv[1] if len(sys.argv) > 1 else "diag192.json"
body = {
    "model": "qwen3.8-27b",
    "messages": [{
        "role": "user",
        "content": (
            f"Document:\n{doc}\n\nQuestion: What is the vault combination and who is the "
            "custodian? Answer using only the document."
        ),
    }],
    "max_tokens": 3000,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
}
t0 = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:18025/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=1500) as r:
        d = json.loads(r.read())
except Exception as e:  # noqa: BLE001
    json.dump({"error": repr(e), "wall_s": round(time.time() - t0, 1)}, open(out_path, "w"))
    raise
m = d["choices"][0]
content = m["message"]["content"] or ""
out = {
    "wall_s": round(time.time() - t0, 1),
    "prompt_tokens": d["usage"]["prompt_tokens"],
    "completion_tokens": d["usage"]["completion_tokens"],
    "finish": m["finish_reason"],
    "needle_retrieved": bool("8314-772-906" in content and "MARGARET" in content.upper()),
    "content_head": content[:200],
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
