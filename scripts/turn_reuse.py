#!/usr/bin/env python3
"""Phase 34: what the model that answered reads again on the next turn, and what reading it between turns saves.

A turn's cache holds the question with its context and the reply after a reasoning block; the next turn replays the
question alone and the reply without that block, so the answering model's cache stops matching at the question and it
reads the question and its own reply again. This asks conversations of several turns through the gateway with each
between-turns setting, interleaved by round:

  other     only the model that did not answer reads the conversation between turns (as before Phase 34)
  between   the model that answered reads it too, first

and records, for every turn after the first, the first-token time and what each model call read and found cached.
`--mode fast` keeps every turn on the quick model, `--mode slow` on the reasoning model (with a small thinking budget).

Usage: python3 -I scripts/turn_reuse.py [--mode fast] [--rounds 3] [--think 20] [--out benchmarks/engine_phase34/turn_reuse_fast.json]
"""
import json, os, statistics, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
ROUNDS, THINK, BUDGET = int(arg("--rounds", 3)), float(arg("--think", 20)), int(arg("--budget", 32))
MODE, GW = arg("--mode", "fast"), arg("--gw", "http://127.0.0.1:8080")
ARMS = ["other", "between"]
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


def delete(conv):
    try:
        urllib.request.urlopen(urllib.request.Request(f"{GW}/v1/conversations/{conv}", method="DELETE"), timeout=30).read()
    except Exception as e:  # the study's conversations should not stay in the owner's list
        print(f"could not delete {conv}: {e}", file=sys.stderr)


rows = {a: [] for a in ARMS}
for r in range(ROUNDS):
    for k in range(len(ARMS)):
        arm = ARMS[(k + r) % len(ARMS)]
        conv, turns = None, []
        for i, q in enumerate(TURNS):
            text = (f"(Session {r}-{arm}-{time.time_ns()}) " if i == 0 else "") + q
            body = {"messages": [{"role": "user", "content": text}], "force_mode": MODE, "shadow": arm, "stable_prefix": True}
            if conv:
                body["conversation_id"] = conv
            if MODE == "slow":
                body["thinking_budget"] = BUDGET
            wall, out = chat(body)
            conv = out["conversation_id"]
            u = out.get("usage", {})
            stages = [st for st in (u.get("stages") or []) if st.get("prompt_n") is not None]
            read = sum(st.get("prompt_n") or 0 for st in stages)
            cached = max((st.get("cache_n") or 0 for st in stages), default=0)
            prefill = sum(st.get("prefill_ms") or 0 for st in stages)
            turns.append({"turn": i + 1, "mode": out.get("reasoning_mode"), "wall_s": round(wall, 2), "ttft_ms": u.get("ttft_ms"),
                          "prompt_tokens": u.get("prompt_tokens"), "read_tokens": read, "cached_tokens": cached,
                          "prefill_ms": round(prefill, 1), "calls": len(stages)})
            print(f"round {r} {arm:<8} turn {i + 1} {out.get('reasoning_mode'):<5} {wall:6.1f}s  first token {(u.get('ttft_ms') or 0) / 1000:5.2f}s"
                  f"  read {read} cached {cached} in {prefill / 1000:.2f}s over {len(stages)} call(s)", flush=True)
            time.sleep(THINK)
        rows[arm].append(turns)
        delete(conv)

summary = {}
for arm, sessions in rows.items():
    later = [t for s in sessions for t in s[1:]]
    summary[arm] = {
        "turns": len(later),
        "first_token_s": round(statistics.median((t["ttft_ms"] or 0) for t in later) / 1000, 2),
        "read_tokens": statistics.median(t["read_tokens"] for t in later),
        "prefill_s": round(statistics.median(t["prefill_ms"] for t in later) / 1000, 2),
        "turn_s": round(statistics.median(t["wall_s"] for t in later), 1),
    }
print(json.dumps(summary, indent=1))
out = arg("--out", "")
if out:
    json.dump({"mode": MODE, "rounds": ROUNDS, "think_s": THINK, "budget": BUDGET, "load_end": os.getloadavg(),
               "note": "medians over turns 2.. of every session; arms interleaved by round", "rows": rows, "median": summary},
              open(os.path.join(ROOT, out), "w"), indent=1)
