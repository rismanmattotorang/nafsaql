#!/usr/bin/env python3
"""ShadowPrefill session study at the model servers (Phase 30): a conversation of quick turns answered by the quick
model, then a switch to the reasoning model, measured as the reasoning model's time to read the conversation (a request
of one token). Arms, interleaved by round:

  cold            nothing warms the reasoning model (today)
  between         after each quick answer, while the person reads, the reasoning model reads the conversation so far
                  (a shadow ending with the assistant's turn, at full speed)
Layouts: usual (what changes with each question - the domain instruction and a note - is in the system prompt, as
the gateway does today) and stable (the system prompt never changes; that text goes before the question).

Usage: python3 scripts/shadow_session_server.py [--rounds 3] [--turns 4] [--think 20] [--out benchmarks/engine_phase30/session_server.json]
"""
import json, os, statistics, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
ROUNDS, TURNS, THINK = int(arg("--rounds", 3)), int(arg("--turns", 4)), float(arg("--think", 20))
S1, S2 = "http://127.0.0.1:8201", "http://127.0.0.1:8202"
SYSTEM = "You are Nafs-Aql, a careful assistant running on one computer. Be accurate and concise."
QUESTIONS = [
    "I am planning a small home server for a family photo library. What should I consider first? About 100 words.",
    "Which file system would you pick for it, and why? About 100 words.",
    "How should I back it up so one disk failure loses nothing? About 100 words.",
    "What would you monitor on it each week? About 100 words.",
    "Putting all of that together, what is the single biggest risk?",
]
NOTES = ["Domain: open question. Answer helpfully and concisely.", "Domain: factual. Say clearly if you are not sure.",
         "Domain: open question. Prefer plain words.", "Domain: factual. Give one recommendation.", "Domain: open question."]


def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return time.time() - t, out


def messages(history, i, stable, tag):
    q = (f"(Session {tag}) " if i == 0 else "") + QUESTIONS[i]
    if stable:
        sys_msg, user = SYSTEM, f"{NOTES[i]}\n\nQuestion: {q}"
    else:
        sys_msg, user = f"{SYSTEM}\n\n{NOTES[i]}", q
    return [{"role": "system", "content": sys_msg}] + history + [{"role": "user", "content": user}], q


rows = {}
for r in range(ROUNDS):
    for stable in (False, True):
        for shadow in (False, True):
            arm = f"{'stable' if stable else 'usual'}/{'between' if shadow else 'cold'}"
            tag = f"{r}-{arm}-{time.time_ns()}"
            history, quick = [], []
            for i in range(TURNS - 1):
                msgs, q = messages(history, i, stable, tag)
                t, out = post(S1, {"messages": msgs, "max_tokens": 160, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                ans = out["choices"][0]["message"]["content"]
                tm = out.get("timings", {})
                quick.append({"seconds": round(t, 2), "read": tm.get("prompt_n"), "cached": tm.get("cache_n"), "read_ms": tm.get("prompt_ms")})
                # The next turn's history holds the question as the person wrote it.
                history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
                if shadow:
                    # While the person reads: the reasoning model reads the conversation so far.
                    convo = [{"role": "system", "content": msgs[0]["content"] if not stable else SYSTEM}] + history
                    post(S2, {"messages": convo, "max_tokens": 1, "temperature": 0, "shadow": {"duty": 1.0},
                              "chat_template_kwargs": {"enable_thinking": True}})
                time.sleep(THINK)
            msgs, _ = messages(history, TURNS - 1, stable, tag)
            t, out = post(S2, {"messages": msgs, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}})
            tm = out.get("timings", {})
            rows.setdefault(arm, []).append({"switch_first_token_s": round(t, 2), "read": tm.get("prompt_n"), "cached": tm.get("cache_n"),
                                             "prompt_tokens": out.get("usage", {}).get("prompt_tokens"), "quick": quick})
            print(f"round {r} {arm:<16} switch: first token {t:6.1f}s  read {tm.get('prompt_n')} cached {tm.get('cache_n')}  "
                  f"quick turns read/cached: {[(x['read'], x['cached']) for x in quick]}", flush=True)

summary = {arm: {"switch_first_token_s": round(statistics.median(x["switch_first_token_s"] for x in v), 2),
                 "switch_read_tokens": statistics.median(x["read"] or 0 for x in v),
                 "prompt_tokens": statistics.median(x["prompt_tokens"] or 0 for x in v),
                 "quick_read_tokens": statistics.median(q["read"] or 0 for x in v for q in x["quick"][1:]),
                 "quick_seconds": round(statistics.median(q["seconds"] for x in v for q in x["quick"][1:]), 2)}
           for arm, v in rows.items()}
print(json.dumps(summary, indent=1))
out = arg("--out", "")
if out:
    json.dump({"rounds": ROUNDS, "turns": TURNS, "think_s": THINK, "load_end": os.getloadavg(), "rows": rows, "median": summary},
              open(os.path.join(ROOT, out), "w"), indent=1)
