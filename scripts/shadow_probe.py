#!/usr/bin/env python3
"""ShadowPrefill probe (Phase 30): what a warm reasoning model saves on an escalation, and what warming it costs the quick
model's draft when both share four cores.

For each round a fresh prompt (a unique first line, so no cache from an earlier round applies) is asked three ways,
interleaved:
  cold      the quick model drafts; then the reasoning model reads the whole prompt (time to its first token).
  arrival   the reasoning model starts reading the same prompt (`shadow`) when the draft starts, on the same cores.
  idle      the reasoning model read the prompt beforehand, while nothing else ran (the between-turns case).
Reported per arm: the draft's wall time, the reasoning model's time to first token after the draft, and their sum.

Usage: python3 scripts/shadow_probe.py [--rounds 5] [--chars 3600] [--s1 http://127.0.0.1:8201] [--s2 http://127.0.0.1:8202]
       [--shadow-field]   send {"shadow": true} (a prefill-only, lowest-priority request) instead of max_tokens 1
"""
import json, os, statistics, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
ROUNDS, CHARS = int(arg("--rounds", 5)), int(arg("--chars", 3600))
S1, S2 = arg("--s1", "http://127.0.0.1:8201"), arg("--s2", "http://127.0.0.1:8202")
SHADOW_FIELD = "--shadow-field" in args

src = open(os.path.join(ROOT, "docs/report/sections/knowledge.tex")).read() + open(os.path.join(ROOT, "docs/report/sections/inference.tex")).read()
src = " ".join(src.split())


def post(base, body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return time.time() - t, out


def messages(r):
    start = (r * 997) % max(1, len(src) - CHARS)
    ctx = src[start:start + CHARS]
    return [
        {"role": "system", "content": f"Session {r}-{time.time_ns()}. Answer from the sources below.\n\n<sources>\n{ctx}\n</sources>"},
        {"role": "user", "content": "In two sentences, what does the passage say about speed?"},
    ]


def draft(msgs):
    return post(S1, {"messages": msgs, "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})


def s2_first_token(msgs):
    return post(S2, {"messages": msgs, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}})


def shadow(msgs):
    body = {"messages": msgs, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
    if SHADOW_FIELD:
        body["shadow"] = True
    return post(S2, body)


rows = {"cold": [], "arrival": [], "idle": []}
for r in range(ROUNDS):
    for arm in ("cold", "arrival", "idle"):
        msgs = messages(r * 3 + ["cold", "arrival", "idle"].index(arm))
        if arm == "idle":
            shadow(msgs)
        sh = None
        if arm == "arrival":
            sh = threading.Thread(target=shadow, args=(msgs,))
            sh.start()
        td, d = draft(msgs)
        if sh is not None:
            sh.join()  # what an escalation would find: the shadow done or (here) waited for
        tf, f = s2_first_token(msgs)
        u = f.get("usage", {})
        rows[arm].append({"draft_s": td, "s2_first_token_s": tf, "total_s": td + tf,
                          "prompt_tokens": u.get("prompt_tokens"), "timings": f.get("timings")})
        print(f"round {r} {arm:<8} draft {td:6.1f}s  s2 first token {tf:6.1f}s  total {td + tf:6.1f}s  prompt {u.get('prompt_tokens')}", flush=True)

summary = {a: {k: round(statistics.median(x[k] for x in v), 2) for k in ("draft_s", "s2_first_token_s", "total_s")} for a, v in rows.items()}
print(json.dumps(summary, indent=1))
out = arg("--out", "")
if out:
    json.dump({"rounds": ROUNDS, "chars": CHARS, "shadow_field": SHADOW_FIELD, "load": os.getloadavg(), "rows": rows, "median": summary},
              open(os.path.join(ROOT, out), "w"), indent=1)
