#!/usr/bin/env python3
"""Self-speculative (n-gram) decoding on this CPU: llama-server with and without --spec-type ngram-*.

N-gram speculation drafts tokens by matching the recent output against the prompt and earlier output, so it needs no
draft model and no memory. It pays when output copies earlier text (code repair, JSON keys); it costs a wasted verify
when a draft is rejected. Verification of k tokens costs about as much as k decode steps on this CPU (decision 004),
so the break-even acceptance is high; this script measures it instead of assuming.

Per config a temporary server is started on :8013, four prompt kinds are run greedily, the server is stopped; the
configs are cycled `--reps` times (A B C A B C), so slow drift of the host hits every config. Reports decode tokens/s
(the server's own timing, prompt excluded) and the drafted / accepted token counts per kind.

    python3 -I scripts/ngram_ab.py --model models/Qwen3-1.7B-Q4_K_M.gguf --label s1 [--think 0] [--reps 2]
"""
import argparse, json, os, statistics, subprocess, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.join(ROOT, "third_party/llama.cpp/build/bin/llama-server")
PORT = 8013

CONFIGS = {
    "none": [],
    "ngram-simple": ["--spec-type", "ngram-simple"],
    "ngram-map-k": ["--spec-type", "ngram-map-k"],
    "ngram-mod": ["--spec-type", "ngram-mod"],
}

BUGGY = '''def merge_intervals(intervals):
    """Merge overlapping intervals; each interval is a (start, end) tuple. Return a sorted list of merged tuples."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start < last_end:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged
'''
PROMPTS = {
    "code_repair": ("This function fails `assert merge_intervals([(1, 5), (2, 3)]) == [(1, 5)]`. Fix it and return the complete corrected function in one ```python block.\n```python\n" + BUGGY + "```"),
    "code_write": "Write a Python class LRUCache with get and put, using an OrderedDict, with a docstring and type hints. Return one ```python block.",
    "json": "Return a JSON array of 8 objects with keys id, name, email, city and active (boolean) describing fictional customers. JSON only, in a ```json block.",
    "math": "A shop sells pens at 3 dollars each and notebooks at 5 dollars each. Maria buys 4 pens and 3 notebooks and pays with a 50 dollar bill. How much change does she get? Think step by step, then give 'Answer: <number>'.",
}


def start(model, extra, ctx=8192):
    p = subprocess.Popen([BIN, "-m", os.path.join(ROOT, model), "-t", "4", "-c", str(ctx), "-np", "1", "--host", "127.0.0.1", "--port", str(PORT),
                          "--jinja", "--reasoning-format", "deepseek", "--cache-reuse", "256"] + extra,
                         stdout=subprocess.DEVNULL, stderr=open(os.path.join(ROOT, "logs/ngram_ab_server.log"), "w"))
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server did not start")


def ask(prompt, think, budget, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
            "cache_prompt": False}
    if think:
        body["thinking_budget_tokens"] = budget
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    tm = d.get("timings", {})
    return {"wall_s": time.time() - t, "tok_s": tm.get("predicted_per_second"), "n": tm.get("predicted_n"), "draft_n": tm.get("draft_n", 0),
            "draft_ok": tm.get("draft_n_accepted", 0), "text": d["choices"][0]["message"].get("content", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    names = a.configs.split(",")
    res = {c: {k: [] for k in PROMPTS} for c in names}
    texts = {}
    for rep in range(a.reps):
        for c in names:
            srv = start(a.model, CONFIGS[c])
            try:
                ask("Say hi.", a.think, 16, 8)  # warm-up
                for k, pr in PROMPTS.items():
                    r = ask(pr, a.think, a.budget, a.max_tokens)
                    texts.setdefault(k, {}).setdefault(c, r.pop("text"))
                    res[c][k].append(r)
                    print(f"rep {rep} {c:13s} {k:12s} {r['tok_s']:.2f} tok/s  n={r['n']}  drafted={r['draft_n']} accepted={r['draft_ok']}", flush=True)
            finally:
                srv.terminate(); srv.wait(timeout=30)
    summary = {}
    for c in names:
        summary[c] = {}
        for k in PROMPTS:
            rs = res[c][k]
            base = statistics.median(x["tok_s"] for x in res["none"][k]) if "none" in res else None
            m = statistics.median(x["tok_s"] for x in rs)
            summary[c][k] = {"tok_s": round(m, 2), "vs_none": round(m / base, 3) if base else None,
                             "drafted": int(statistics.median(x["draft_n"] for x in rs)), "accepted": int(statistics.median(x["draft_ok"] for x in rs)),
                             "same_text_as_none": texts[k][c] == texts[k].get("none", texts[k][c])}
    print(json.dumps(summary, indent=1))
    json.dump({"model": a.model, "think": a.think, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "reps": a.reps, "summary": summary, "raw": res},
              open(os.path.join(ROOT, f"benchmarks/ngram_ab_{a.label}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
