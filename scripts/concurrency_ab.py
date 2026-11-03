"""Many users at once against one model server: Smiley Core's batching engine and llama.cpp's slots (Phase 18).

Each arm gets the same N distinct questions, sent at the same moment, and is measured on what a user actually waits for:
the time to the first token, the tokens per second that user sees, and the wall time until the last of them is done.
The answers are compared against the same questions asked one at a time on the same server, so a throughput gain that
changed what anyone was told would show up as a mismatch rather than as a win.

    python3 -I scripts/concurrency_ab.py --port 8251 --label smiley --users 1,2,4 [--rounds 2]
    python3 -I scripts/concurrency_ab.py --report

Servers: Smiley Core `scripts/serve_smiley.sh` (8201/8202), llama.cpp `scripts/serve_llama.sh` (8001/8002).
"""
import argparse, json, os, statistics, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/concurrency_ab.json")
# Two regimes, because the answer differs between them: with a short reply a request is mostly prefill, which the
# engine runs one at a time, so batching has little to work with; with a long reply it is mostly decode, which is
# where several users share one read of the weights. Nafs-Aql's thorough path writes hundreds of tokens, so "long"
# is the shape that matters for it.
SETS = {"short": ("You are a careful assistant. Answer directly and briefly.", [
    "What is the capital of Indonesia, and on which island is it?",
    "Name three prime numbers between 10 and 30.",
    "What does the acronym CPU stand for, and what does it do?",
    "Give one reason to write tests before fixing a bug.",
    "What is 17 times 23?",
    "In one sentence, what is a hash table?",
    "Name two uses for a linked list.",
    "What is the boiling point of water at sea level, in Celsius?",
]), "long": ("You are a careful assistant. Explain thoroughly, with examples, in several paragraphs.", [
    "Explain how a hash table works, and when it degrades.",
    "Explain the difference between a process and a thread, with examples.",
    "Explain what a database index is and when it does not help.",
    "Explain how garbage collection works, and its trade-offs.",
    "Explain what TCP does that UDP does not, and why it matters.",
    "Explain how public-key cryptography lets strangers agree on a secret.",
    "Explain what a cache line is and why false sharing is slow.",
    "Explain how a compiler turns a loop into machine code.",
])}


def ask(port, system, question, max_tokens, think):
    """One streamed request; returns what this user waited for."""
    body = {"model": "x", "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 1, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, text, n = None, "", 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                v = json.loads(line[5:])
            except ValueError:
                continue
            for c in v.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if first is None:
                        first = time.time() - t0
                    text += piece
                    n += 1
    wall = time.time() - t0
    return {"ttft_s": first, "wall_s": wall, "tokens": n, "tok_s": n / wall if wall else 0, "text": text}


def burst(port, system, questions, n, max_tokens, think):
    """`n` users sending at the same moment."""
    out = [None] * n
    start = threading.Barrier(n)

    def one(i):
        start.wait()
        out[i] = ask(port, system, questions[i % len(questions)], max_tokens, think)

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, time.time() - t0


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    users = [int(x) for x in a.users.split(",")]
    think = a.think
    system, questions = SETS[a.set]
    # What each question answers when it has the server to itself: the yardstick for "nobody was served worse".
    alone = {}
    for i in range(max(users)):
        q = questions[i % len(questions)]
        alone[q] = ask(a.port, system, q, a.max_tokens, think)["text"]
    for rnd in range(a.rounds):
        for n in users:
            rows, wall = burst(a.port, system, questions, n, a.max_tokens, think)
            same = sum(r["text"] == alone[questions[i % len(questions)]] for i, r in enumerate(rows))
            total = sum(r["tokens"] for r in rows)
            rec = {"label": a.label, "set": a.set, "round": rnd, "users": n, "wall_s": round(wall, 2),
                   "throughput_tok_s": round(total / wall, 2) if wall else 0,
                   "ttft_s": [round(r["ttft_s"] or 0, 2) for r in rows],
                   "per_user_tok_s": [round(r["tok_s"], 2) for r in rows],
                   "identical_to_solo": f"{same}/{n}", "tokens": total,
                   "load": round(os.getloadavg()[0], 1), "at": time.strftime("%H:%M:%S")}
            doc["runs"].append(rec)
            json.dump(doc, open(OUT, "w"), indent=1)
            print(f"{a.label:8s} r{rnd} {n} user(s): {rec['throughput_tok_s']:6.2f} tok/s total, "
                  f"{statistics.median(rec['per_user_tok_s']):5.2f} per user, ttft med {statistics.median(rec['ttft_s']):4.2f}s, "
                  f"wall {wall:5.1f}s, same as alone {rec['identical_to_solo']}, load {rec['load']}", flush=True)
    print("done")


def report(a):
    doc = json.load(open(OUT))
    print(f"{'engine':14s} {'users':>5s} {'total tok/s':>12s} {'per user':>9s} {'ttft med':>9s} {'wall s':>7s} {'vs 1 user':>10s} {'same as alone':>14s}")
    for label in sorted({(r["label"], r.get("set", "?")) for r in doc["runs"]}):
        base = None
        for n in sorted({r["users"] for r in doc["runs"] if (r["label"], r.get("set", "?")) == label}):
            rs = [r for r in doc["runs"] if (r["label"], r.get("set", "?")) == label and r["users"] == n]
            tp = statistics.median(r["throughput_tok_s"] for r in rs)
            pu = statistics.median(x for r in rs for x in r["per_user_tok_s"])
            tt = statistics.median(x for r in rs for x in r["ttft_s"])
            wall = statistics.median(r["wall_s"] for r in rs)
            base = base or tp
            same = "/".join({r["identical_to_solo"] for r in rs})
            print(f"{label[0] + '/' + label[1]:14s} {n:5d} {tp:12.2f} {pu:9.2f} {tt:9.2f} {wall:7.1f} {tp / base:9.2f}x {same:>14s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8251)
    ap.add_argument("--label", default="smiley")
    ap.add_argument("--users", default="1,2,4")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--set", default="long", choices=sorted(SETS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
