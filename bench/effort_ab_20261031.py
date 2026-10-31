#!/usr/bin/env python3
"""Roadmap item 38: reasoning-effort + max_tokens A/B battery.

Arms (each reboots the router with its policy env):
  a-xhigh-8k  : legacy policy (force xhigh)  + injected 8192   (pre-change control)
  b-med-16k   : v1 policy (default medium)   + injected 16384
  c-med-32k   : v1 policy                    + injected 32768
  d-xhigh-32k : v1 policy, requests pin xhigh + injected 32768

Per arm x 3 reps x 2 probe shapes (tool-call, long-form essay), on the gpu2 lane.
Writes one line per probe; summary printed at the end.
"""
import json, os, random, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (router scripts live there)
OUT = os.path.join(HERE, "bench-results", "unlock-rebaseline-20261031", "effort-ab-20261031.txt")
REPS = int(os.environ.get("AB_REPS", "3"))

ARMS = [
    ("a-xhigh-8k",  "legacy", 8192),
    ("b-med-16k",   "v1",     16384),
    ("c-med-32k",   "v1",     32768),
    ("d-xhigh-32k", "v1pin-xhigh", 32768),
]

def loopcount(text, n=8):
    w = text.split()
    grams = set(); dup = 0
    for i in range(len(w) - n):
        g = " ".join(w[i:i+n])
        if g in grams:
            dup += 1
        grams.add(g)
    return dup

def toolprompt():
    return {"model": "qwen-gpu2", "messages": [
        {"role": "user", "content": "List the /tmp directory contents using the list_dir tool, then summarize in one sentence."}],
        "tools": [{"type": "function", "function": {"name": "list_dir", "description": "List directory contents",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
        "tool_choice": "auto", "temperature": 0}

def essayprobe(seed):
    random.seed(seed)
    topics = ["the history of lighthouses", "deep-sea exploration", "the invention of printing", "alpine railways"]
    return {"model": "qwen-gpu2", "messages": [{"role": "user",
        "content": f"Write a detailed 900-word essay about {random.choice(topics)}. Vary your vocabulary and structure."}],
        "temperature": 0.7}

def post(payload, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def set_router(policy, maxtok):
    subprocess.run([os.path.join(HERE, "router-off")], capture_output=True)
    env = dict(os.environ, QWEN_ROUTER_EFFORT_POLICY=policy, QWEN_ROUTER_MAX_TOKENS=str(maxtok))
    subprocess.run([os.path.join(HERE, "router-on")], capture_output=True, text=True, env=env)
    time.sleep(1.5)

def malformed_args(args):
    """Heuristic: leaked template XML or non-JSON in tool arguments."""
    if args is None:
        return False
    s = args if isinstance(args, str) else json.dumps(args)
    return bool(re.search(r"</?(parameter|function|tool_call)>", s))

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "a")
    def emit(line):
        print(line, flush=True)
        f.write(line + "\n"); f.flush()

    emit(f"=== A/B battery start {time.strftime('%F %T')} reps={REPS} ===")
    for name, policy, maxtok in ARMS:
        set_router(policy, maxtok)
        emit(f"--- arm {name} (policy={policy} max_tokens={maxtok}) router rearmed ---")
        for rep in range(1, REPS + 1):
            for label, payload in (("toolcall", toolprompt()), ("essay", essayprobe(rep * 10 + ARMS.index((name, policy, maxtok))))):
                if policy == "legacy" or name == "a-xhigh-8k":
                    pass  # control arm: router forces xhigh itself
                elif name == "d-xhigh-32k":
                    payload = dict(payload); payload["reasoning_effort"] = "xhigh"
                # else: v1 default (medium) applies via the router
                t0 = time.time()
                try:
                    d = post(payload)
                    dt = time.time() - t0
                    m = d["choices"][0]["message"]; u = d.get("usage", {})
                    content = m.get("content") or ""
                    reason = m.get("reasoning") or ""
                    tc = m.get("tool_calls")
                    args = None
                    if tc:
                        fn = tc[0].get("function", {})
                        args = fn.get("arguments")
                    emit(f"{name} rep{rep} {label}: finish={d['choices'][0]['finish_reason']} "
                         f"completion_tokens={u.get('completion_tokens')} content_chars={len(content)} "
                         f"reason_chars={len(reason)} loops8={loopcount(content + ' ' + reason)} "
                         f"tool_calls={'yes' if tc else 'no'} args={json.dumps(args)[:110] if args is not None else 'none'} "
                         f"malformed_args={malformed_args(args)} wall={dt:.0f}s")
                except Exception as e:
                    emit(f"{name} rep{rep} {label}: ERROR {type(e).__name__} {str(e)[:100]}")
                time.sleep(1)
    # leave the fleet on the candidate default arm
    set_router("v1", 16384)
    emit(f"=== A/B complete {time.strftime('%F %T')}; router restored to v1/16384 ===")

if __name__ == "__main__":
    main()
