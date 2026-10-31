#!/usr/bin/env python3
"""Does one request with n = 4 choices prefill the prompt once, against four separate concurrent requests?

A repair prompt on System 2 is 1,000+ tokens (the earlier candidate and the verifier's evidence) and prefill runs at about
31 tokens/s with no batching gain, so a round of four candidates that each prefill their own slot pays the prompt four
times. llama-server's `n` (n_cmpl) is meant to share the prompt. Measured here with a long prompt and a short answer, so
prefill dominates; the cache is off so every arm does the same work. Arms (interleaved, median of reps):
  one       one request
  four_sep  four concurrent requests
  n4        one request with n = 4

    python3 -I scripts/prefill_share.py --port 8002 [--prompt-words 600] [--reps 3]
"""
import argparse, json, statistics, threading, time, urllib.request

WORDS = ("The quick brown fox jumps over the lazy dog while a function merges intervals and returns the sorted list of tuples. ")


def req(port, n, seed, prompt, out, i, tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt + f" Variant {seed}. Reply with one short sentence."}], "temperature": 0.7,
            "max_tokens": tokens, "stream": False, "cache_prompt": False, "n": n, "thinking_budget_tokens": 0}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=900).read())
    out[i] = (time.time() - t, d.get("usage", {}), len(d.get("choices", [])), d.get("timings", {}))


def run(port, arm, prompt, tokens):
    out = [None] * (4 if arm == "four_sep" else 1)
    n = 4 if arm == "n4" else 1
    ths = [threading.Thread(target=req, args=(port, n, 1 + i, prompt, out, i, tokens)) for i in range(len(out))]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    return time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--prompt-words", type=int, default=600)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    prompt = (WORDS * (a.prompt_words // 20 + 1))
    run(a.port, "one", "Say hi", 4)
    res = {k: [] for k in ("one", "four_sep", "n4")}
    for rep in range(a.reps):
        for arm in res:
            wall, out = run(a.port, arm, prompt, a.tokens)
            u = out[0][1]
            choices = sum(o[2] for o in out)
            res[arm].append({"wall_s": round(wall, 1), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "choices": choices})
            print(rep, arm, res[arm][-1], flush=True)
    for arm, rs in res.items():
        print(arm, "median wall", statistics.median(r["wall_s"] for r in rs), "s")
    json.dump(res, open("benchmarks/prefill_share.json", "w"), indent=1)


if __name__ == "__main__":
    main()
