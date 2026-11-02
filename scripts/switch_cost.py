"""What it costs an engine to switch between kinds of request (Phase 17).

The gateway sends one model prompts of different kinds: a chat turn (the assistant's system prompt plus the domain's), a code-agent
step (a long system prompt and the task), a quick draft and its repair. An engine that keeps one cached prefix recomputes the prompt
whenever the kind changes; one that keeps several (llama.cpp's slots, a prefix store) does not. This sends the same sequence of
requests to one model server, alternating kinds A and B, and records per request how many prompt tokens were prefilled and how
many came from the cache, the server's prefill time and the client's time to the first token.

    python3 -I scripts/switch_cost.py --port 8202 --label smiley-system2 [--rounds 4]
    python3 -I scripts/switch_cost.py --report

Servers: Smiley Core 8201 (System 1) / 8202 (System 2); llama.cpp 8001 / 8002 (scripts/serve_llama.sh).
"""
import argparse, json, os, re, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/switch_cost.json")


def agent_system():
    src = open(os.path.join(ROOT, "rust/crates/nafs-harness/src/agent.rs")).read()
    m = re.search(r'pub const SYSTEM_PROMPT: &str = "((?:[^"\\]|\\.)*)";', src, re.S)
    return m.group(1).replace("\\\n", "").replace("\\n", "\n").replace('\\"', '"')


KINDS = {
    "chat": lambda: ("You are Nafs-Aql, a careful assistant. Answer directly and say when you are unsure.\n\nDomain: open question. Keep answers short and plain.",
                     "Why do cats purr?"),
    "agent": lambda: (agent_system(),
                      "Task:\nsecond_largest(xs) returns the second largest distinct value of the list, or None when there is none. Fix solution.py.\n\n"
                      "Workspace files:\n--- solution.py\ndef second_largest(xs):\n    xs = sorted(xs)\n    return xs[-2]\n\n\nReply with one action."),
    "math": lambda: ("You are Nafs-Aql, a careful assistant. Show the steps, then give the final number.\n\nDomain: arithmetic and word problems.",
                     "A train leaves at 3:40 pm and arrives at 6:05 pm. How many minutes is the trip?"),
}
SEQUENCES = {
    "same": ["agent"] * 4,
    "alternate_2": ["agent", "chat"] * 3,
    "rotate_3": ["agent", "chat", "math"] * 2,
}


def call(port, model, kind, think, max_tokens=8):
    system, user = KINDS[kind]()
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.0, "seed": 1,
            "stream": True, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, timings, text = None, {}, ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            v = json.loads(line[5:])
            if "timings" in v:
                timings = v["timings"]
            text += "".join((c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning_content") or "") for c in v.get("choices", []))
            if first is None and any((c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")) for c in v.get("choices", [])):
                first = time.time() - t0
    return {"prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"), "prompt_ms": timings.get("prompt_ms"), "ttft_s": first, "text": text}


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    think = "system2" in a.label
    model = "system2" if think else "system1"
    for rnd in range(a.rounds):
        for seq, kinds in SEQUENCES.items():
            # Start every sequence from a different prompt than it begins with, so its first request is cold for all engines.
            call(a.port, model, "math" if kinds[0] != "math" else "chat", think)
            rows = []
            for i, k in enumerate(kinds):
                r = call(a.port, model, k, think)
                rows.append({"i": i, "kind": k, **r})
            doc["runs"].append({"label": a.label, "round": rnd, "sequence": seq, "rows": rows, "load": round(os.getloadavg()[0], 1), "at": time.strftime("%H:%M:%S")})
            json.dump(doc, open(OUT, "w"), indent=1)
            print(f"r{rnd} {a.label} {seq:12s} " + " ".join(f"{x['kind'][0]}:{x['prompt_n']}+{x['cache_n']}/{(x['prompt_ms'] or 0)/1e3:.1f}s" for x in rows), flush=True)
    print("done")


def verify(a):
    """A restored cache must give the same tokens as a cold prefill: the same greedy answer to the same prompt, cold and after other kinds."""
    think = "system2" in a.label
    model = "system2" if think else "system1"
    ok = True
    for kind in ("agent", "chat", "math"):
        cold = call(a.port, model, kind, think, 48)
        for other in ("chat", "math", "agent"):
            if other != kind:
                call(a.port, model, other, think, 4)
        again = call(a.port, model, kind, think, 48)
        same = cold["text"] == again["text"]
        ok &= same
        import hashlib
        digest = hashlib.sha256(again["text"].encode()).hexdigest()[:12]
        print(f"{kind:6s} cold {cold['prompt_n']}+{cold['cache_n']} tokens -> after other kinds {again['prompt_n']}+{again['cache_n']}: {'IDENTICAL' if same else 'DIFFERENT'}  sha {digest}  {again['text'][:50]!r}")
    print("verify:", "ok" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


def report(a):
    doc = json.load(open(OUT))
    print(f"{'engine':16s} {'sequence':12s} {'prefilled tokens/request':>26s} {'prefill s/request':>18s} {'ttft s (median)':>16s}")
    for label in sorted({r["label"] for r in doc["runs"]}):
        for seq in SEQUENCES:
            rows = [x for r in doc["runs"] if r["label"] == label and r["sequence"] == seq for x in r["rows"] if x["i"] > 0]
            if not rows:
                continue
            print(f"{label:16s} {seq:12s} {statistics.mean(x['prompt_n'] or 0 for x in rows):26.0f} {statistics.mean((x['prompt_ms'] or 0) / 1e3 for x in rows):18.2f} {statistics.median(x['ttft_s'] or 0 for x in rows):16.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8202)
    ap.add_argument("--label", default="smiley-system2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check that a restored cache gives the same greedy answer as a cold prefill")
    a = ap.parse_args()
    verify(a) if a.verify else report(a) if a.report else run(a)
