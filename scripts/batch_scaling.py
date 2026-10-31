#!/usr/bin/env python3
"""Batch scaling of decode on this CPU: aggregate tokens/s when N streams decode at once on one llama-server.

Why: every parallel scheme in Nafs-Aql (N concurrent candidates, a draft and its agreement samples, two users)
rests on one number, the cost factor c(N) = wall time of N batched streams / wall time of one stream. If decode is
memory-bound, c(N) stays near 1 and parallel sampling is almost free; if it is compute-bound, c(N) approaches N.

    python3 -I scripts/batch_scaling.py --port 8001 --label system1 [--max-n 4] [--tokens 96] [--reps 3]

Each request is a raw /completion of `tokens` tokens (ignore_eos, greedy) after a short prompt, sent at the same
instant by N threads; the cache is disabled so every rep does the same work. Interleaved reps (N = 1..max, repeated),
median over reps. Writes benchmarks/batch_scaling_<label>.json.
"""
import argparse, json, os, statistics, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEPTH_CACHE = False
PROMPT = ("Write a Python function that merges two sorted lists into one sorted list without using sort(), "
          "then explain its time complexity step by step. ") * 2


def one(port, tokens, seed, out, i):
    body = {"prompt": PROMPT + f" (variant {seed})", "n_predict": tokens, "temperature": 0.0, "ignore_eos": True,
            "cache_prompt": bool(DEPTH_CACHE), "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    out[i] = (time.time() - t, d.get("timings", {}), d.get("tokens_predicted", tokens))


def run(port, n, tokens, seed):
    out = [None] * n
    ths = [threading.Thread(target=one, args=(port, tokens, seed * 10 + i, out, i)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    total = sum(o[2] for o in out)
    dec = [o[1].get("predicted_per_second", 0) for o in out]
    pre = [o[1].get("prompt_ms", 0) for o in out]
    return {"n": n, "wall_s": wall, "tokens": total, "agg_tok_s": total / wall, "decode_agg_tok_s": sum(dec),
            "per_stream_tok_s": statistics.mean(dec), "prefill_ms": statistics.mean(pre)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depth", type=int, default=0, help="prompt words of shared context (about 1.3 tokens each): decode at depth")
    a = ap.parse_args()
    global PROMPT
    global DEPTH_CACHE
    if a.depth:
        DEPTH_CACHE = True
        PROMPT = ("Consider this function and explain every step of how it merges two sorted lists into one sorted list. " * (a.depth // 16 + 1))
    run(a.port, 1, 8, 0)  # warm-up
    rows = {n: [] for n in range(1, a.max_n + 1)}
    for rep in range(a.reps):
        for n in range(1, a.max_n + 1):
            r = run(a.port, n, a.tokens, rep + 1)
            rows[n].append(r)
            print(f"rep {rep} N={n}: agg {r['agg_tok_s']:.2f} tok/s (decode-only {r['decode_agg_tok_s']:.2f}), per stream {r['per_stream_tok_s']:.2f}, wall {r['wall_s']:.1f}s", flush=True)
    summary = []
    base = statistics.median(r["wall_s"] for r in rows[1])
    for n, rs in rows.items():
        w = statistics.median(r["wall_s"] for r in rs)
        summary.append({"n": n, "median_wall_s": round(w, 2), "agg_tok_s": round(statistics.median(r["agg_tok_s"] for r in rs), 2),
                        "decode_agg_tok_s": round(statistics.median(r["decode_agg_tok_s"] for r in rs), 2),
                        "per_stream_tok_s": round(statistics.median(r["per_stream_tok_s"] for r in rs), 2),
                        "cost_factor_c": round(w / base, 3), "speedup_vs_serial": round(n * base / w, 3)})
    print(json.dumps(summary, indent=1))
    json.dump({"label": a.label, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "tokens": a.tokens, "reps": a.reps,
               "summary": summary, "raw": rows}, open(os.path.join(ROOT, f"benchmarks/batch_scaling_{a.label}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
