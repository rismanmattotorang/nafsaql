#!/usr/bin/env python3
"""ShadowPrefill session study (Phase 30): a conversation of several quick turns, then a turn that goes to the reasoning
model, asked through the gateway with each layout and shadow setting, interleaved by round.

Arms: baseline (shadow off, prompt as before), stable (stable prefix, no shadow), between (shadow between turns, prompt as
before), stable+between (both). Each round starts fresh conversations (a unique opening), waits THINK seconds between
turns (the person reading the answer), and records each turn's wall time and the reasoning turn's prompt reading.

Usage: python3 scripts/shadow_session.py [--rounds 3] [--think 30] [--budget 32] [--out benchmarks/engine_phase30/session.json]
"""
import json, os, statistics, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
ROUNDS, THINK, BUDGET = int(arg("--rounds", 3)), float(arg("--think", 30)), int(arg("--budget", 32))
GW = arg("--gw", "http://127.0.0.1:8080")
ARMS = {"baseline": ("off", False), "stable": ("off", True), "between": ("between", False), "stable+between": ("between", True)}
TURNS = [
    "I am planning a small home server for a family photo library. What should I consider first? Answer in about 120 words.",
    "Which file system would you pick for it, and why? About 120 words.",
    "How should I back it up so one disk failure loses nothing? About 120 words.",
    "Putting all of that together, write a short checklist I can follow this weekend.",
]


def chat(body):
    req = urllib.request.Request(GW + "/v1/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return time.time() - t, out


rows = {a: [] for a in ARMS}
for r in range(ROUNDS):
    for arm, (shadow, stable) in ARMS.items():
        conv, turns = None, []
        for i, q in enumerate(TURNS):
            last = i == len(TURNS) - 1
            text = (f"(Session {r}-{arm}-{time.time_ns()}) " if i == 0 else "") + q
            body = {"messages": [{"role": "user", "content": text}], "force_mode": "slow" if last else "fast",
                    "shadow": shadow, "stable_prefix": stable}
            if conv:
                body["conversation_id"] = conv
            if last:
                body["thinking_budget"] = BUDGET
            wall, out = chat(body)
            conv = out["conversation_id"]
            u = out.get("usage", {})
            stages = u.get("stages") or []
            # Prompt reading per model: what was read and what came from the cache.
            # Every model call of the turn, in order (a reasoning turn may be two: the first pass, then its continuation).
            read = {f"{k}:{st['stage']}": {"prefill_ms": st.get("prefill_ms"), "prompt_n": st.get("prompt_n"), "cache_n": st.get("cache_n")}
                    for k, st in enumerate(stages) if st.get("prompt_n") is not None}
            turns.append({"turn": i + 1, "mode": out.get("reasoning_mode"), "wall_s": round(wall, 2), "ttft_ms": u.get("ttft_ms"),
                          "prompt_tokens": u.get("prompt_tokens"), "read": read})
            brief = " ".join(f"{k}: read {v['prompt_n']} cached {v['cache_n']} in {round((v['prefill_ms'] or 0) / 1000, 1)}s" for k, v in read.items())
            print(f"round {r} {arm:<15} turn {i + 1} {out.get('reasoning_mode'):<5} {wall:6.1f}s  first token {(u.get('ttft_ms') or 0) / 1000:5.1f}s  {brief}", flush=True)
            if not last:
                time.sleep(THINK)
        rows[arm].append(turns)
        time.sleep(THINK)

summary = {}
for arm, sessions in rows.items():
    summary[arm] = {
        "reasoning_turn_s": round(statistics.median(s[-1]["wall_s"] for s in sessions), 1),
        # The reasoning model's first token: its reading of the conversation, plus the first chunk of reasoning.
        "reasoning_first_token_s": round(statistics.median((s[-1]["ttft_ms"] or 0) for s in sessions) / 1000, 1),
        "quick_first_token_s": round(statistics.median((t["ttft_ms"] or 0) for s in sessions for t in s[1:-1]) / 1000, 1),
        "reasoning_prefill_s": round(statistics.median(sum((v["prefill_ms"] or 0) for v in s[-1]["read"].values()) / 1000 for s in sessions), 1),
        "reasoning_read_tokens": statistics.median(sum((v["prompt_n"] or 0) for v in s[-1]["read"].values()) for s in sessions),
        "quick_prefill_s": round(statistics.median(sum((v["prefill_ms"] or 0) for v in t["read"].values()) / 1000 for s in sessions for t in s[1:-1]), 1),
        "quick_turns_s": round(statistics.median(t["wall_s"] for s in sessions for t in s[1:-1]), 1),
    }
print(json.dumps(summary, indent=1))
out = arg("--out", "")
if out:
    json.dump({"rounds": ROUNDS, "think_s": THINK, "budget": BUDGET, "load_end": os.getloadavg(), "rows": rows, "median": summary},
              open(os.path.join(ROOT, out), "w"), indent=1)
