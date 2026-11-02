import json

rows = [json.loads(l) for l in open('/home/chris/tools/hyperqwen/sparsekv-port/fitdata/niah_32k_samples.jsonl')]
assert len(rows) >= 5, f"need >=5 samples, got {len(rows)}"
assert "body" in rows[4], "S[4] missing body"
assert "body" in rows[1], "S[1] missing body"
print("corpus OK: S[4] body", len(rows[4]["body"]), "chars; S[1] body", len(rows[1]["body"]), "chars")
