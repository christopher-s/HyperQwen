"""Builds the KLD test set: 5 long documents (3 code, 2 prose) x context lengths x 3 tasks.

Documents (all public):
  code_vllm   vLLM's v1 engine sources (from the installed vllm package)
  code_torch  PyTorch's torch/_inductor sources (installed torch package)
  code_hf     Hugging Face transformers generation + integrations sources (installed package)
  prose_wp    Tolstoy, War and Peace (Project Gutenberg #2600)
  prose_mc    Dumas, The Count of Monte Cristo (Project Gutenberg #1184)

Prompts are token sequences. For a document D and length L (a multiple of the attention block size, 864):
  continue  H + D[:n] + "continue the document from where it stops (next 60 lines)" + assistant start
  explain   H + D[:n] + "explain the final part of the document" + assistant start
  recall    H + D[:n] + "quote the 30 lines after <a unique line at ~35% depth>" + assistant start
H opens a user turn; len(H) + n = L + 16 with L a multiple of the 864-token block, thinking is off. The prompts
of one document share the H + D[:n] prefix, and longer lengths extend shorter ones, so prefix caching reuses
the prefill (prefill is dense in every config, so this does not touch what is measured).
Usage: python build_samples.py OUT_DIR MODEL_DIR [--lengths 32832,65664,...]
"""
import argparse
import glob
import json
import os
import re
import urllib.request

import numpy as np
from transformers import AutoTokenizer

BLOCK = 864
EXTRA = 16  # prompt = k*BLOCK + EXTRA tokens (so a repeated prompt hits all k blocks, and <256 generated
            # tokens can never complete a new cacheable block)


def py_corpus(roots, max_chars):
    parts, n = [], 0
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
            if "__pycache__" in f:
                continue
            try:
                s = open(f, encoding="utf-8").read()
            except Exception:
                continue
            if len(s) < 200:
                continue
            rel = f.split("site-packages/")[-1]
            parts.append(f"# ===== file: {rel} =====\n{s}\n")
            n += len(parts[-1])
            if n >= max_chars:
                return "".join(parts)
    return "".join(parts)


def gutenberg(book_id, cache):
    path = os.path.join(cache, f"pg{book_id}.txt")
    if not os.path.exists(path):
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        with urllib.request.urlopen(url, timeout=60) as r:
            open(path, "wb").write(r.read())
    s = open(path, encoding="utf-8-sig").read()
    a = re.search(r"\*\*\* START OF.*?\*\*\*", s)
    b = re.search(r"\*\*\* END OF", s)
    s = s[a.end() if a else 0: b.start() if b else len(s)]
    return s.replace("\r\n", "\n").strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("model")
    ap.add_argument("--lengths", default="32832,65664,131328,196128")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    import vllm, torch, transformers
    sp = lambda m: os.path.dirname(m.__file__)
    docs = {
        "code_vllm": py_corpus([os.path.join(sp(vllm), "v1")], 1_400_000),
        "code_torch": py_corpus([os.path.join(sp(torch), "_inductor")], 1_400_000),
        "code_hf": py_corpus([os.path.join(sp(transformers), d) for d in ("generation", "integrations", "pipelines")], 1_400_000),
        "prose_wp": gutenberg(2600, a.out),
        "prose_mc": gutenberg(1184, a.out),
    }
    lengths = [int(x) for x in a.lengths.split(",")]
    tmpl = tok.apply_chat_template([{"role": "user", "content": "\x00"}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    pre, post = tmpl.split("\x00")
    H = tok.encode(pre + "Below is a long document. Read it carefully; questions about it follow at the end.\n\n",
                   add_special_tokens=False)
    eos = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")})
    samples = []
    for name, text in docs.items():
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.array(enc["input_ids"], np.int32)
        offs = enc["offset_mapping"]
        np.save(os.path.join(a.out, f"{name}.npy"), ids)
        need = max(lengths) + EXTRA - len(H)
        assert len(ids) >= need, (name, len(ids), need)
        for L in lengths:
            n = L + EXTRA - len(H)
            end = offs[n - 1][1]
            seen = text[:end]
            # unique, non-trivial line at ~35% depth for the recall task
            pos = int(end * 0.35)
            line = None
            for m in re.finditer(r"[^\n]+", seen[pos:]):
                c = m.group(0).strip()
                if 30 <= len(c) <= 110 and seen.count(c) == 1 and not c.startswith("#"):
                    line = c
                    break
            assert line, (name, L)
            q_explain = ("\n\n[End of document]\n\nExplain in detail what the final part of the document above "
                         "(the last file, or the last few paragraphs) is about.")
            q_recall = (f"\n\n[End of document]\n\nFind this exact line in the document above:\n\"{line}\"\n"
                        "Quote, verbatim, the 30 lines that come right after it in the document.")
            q_cont = ("\n\n[End of document]\n\nContinue the document from exactly where it stops: write the next "
                      "60 lines in the same style. Output only the continuation.")
            for task, q in (("continue", q_cont), ("explain", q_explain), ("recall", q_recall)):
                suffix = tok.encode(q + post, add_special_tokens=False)
                samples.append({"sid": f"{name}.{L // 1024}k.{task}", "doc": name, "L": L, "task": task,
                                "ndoc": n, "suffix": suffix, "plen": len(H) + n + len(suffix),
                                "max_tokens": a.max_tokens, "recall_line": line if task == "recall" else None})
    meta = {"H": H, "eos": eos, "block": BLOCK, "extra": EXTRA, "lengths": lengths,
            "doc_tokens": {k: int(len(np.load(os.path.join(a.out, f"{k}.npy")))) for k in docs}}
    json.dump({"meta": meta, "samples": samples}, open(os.path.join(a.out, "samples.json"), "w"))
    print(json.dumps(meta["doc_tokens"]), len(samples), "samples; H", len(H), "eos", eos)


if __name__ == "__main__":
    main()
