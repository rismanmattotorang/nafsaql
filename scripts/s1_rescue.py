#!/usr/bin/env python3
"""How often can cheap concurrent System 1 samples rescue a failed draft, before the thorough model is used?

A failed draft on a checkable task costs an escalation to System 2 (about 300 s on this CPU). System 1 batches well at
short context (four streams cost 1.7 times one, `batch_scaling_system1.json`) and one request with n choices prefills the
prompt once, so k extra samples are cheap. If the exact verifier passes any of them, System 2 is not needed. This script
measures, on MBPP+ items, pass@1 of the draft (production settings), pass@(1+k), and the rescue rate: among drafts that
fail, how many of the k samples' sets contain a passing one. Verdicts come from the gateway's own sandbox (`/v1/verify`).

    python3 -I scripts/s1_rescue.py [--n 60] [--k 4] [--temp 0.7] [--seed 20261031]
Writes benchmarks/s1_rescue.json (resumable).
"""
import argparse, json, os, random, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def post(url, body, timeout=600):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def gen(port, prompt, n, temp, top_p, max_tokens):
    t = time.time()
    d = post(f"http://127.0.0.1:{port}/v1/chat/completions", {"model": "system1", "messages": [{"role": "system", "content": "You are a helpful assistant. Return one complete, self-contained Python solution in a single ```python code block, then at most two sentences of explanation."}, {"role": "user", "content": prompt}],
                                                               "temperature": temp, "top_p": top_p, "max_tokens": max_tokens, "n": n, "stream": False, "chat_template_kwargs": {"enable_thinking": False}})
    return [c["message"]["content"] for c in d["choices"]], time.time() - t


def verify(base, answer, tests):
    return post(base + "/v1/verify", {"answer": answer, "verification": {"tests": tests}})["passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/s1_rescue.json"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/bench/mbpp_plus.jsonl"))]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"k": a.k, "temp": a.temp, "items": []}
    done = {r["id"] for r in out["items"]}
    for it in rows:
        if it["id"] in done:
            continue
        tests = it["verification"]["tests"]
        draft, t_d = gen(a.port, it["prompt"], 1, 0.2, 0.8, 512)
        samples, t_s = gen(a.port, it["prompt"], a.k, a.temp, 0.8, 512)
        rec = {"id": it["id"], "draft_pass": verify(a.base, draft[0], tests), "sample_pass": [verify(a.base, x, tests) for x in samples],
               "draft_s": round(t_d, 1), "samples_s": round(t_s, 1)}
        out["items"].append(rec)
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps(rec), flush=True)
    it = out["items"]
    fail = [r for r in it if not r["draft_pass"]]
    p1 = sum(r["draft_pass"] for r in it) / len(it)
    pk = sum(r["draft_pass"] or any(r["sample_pass"]) for r in it) / len(it)
    resc = [r for r in fail if any(r["sample_pass"])]
    print(f"items {len(it)}  draft pass@1 {p1:.3f}  pass@(1+{a.k}) {pk:.3f}  failed drafts {len(fail)}  rescued by a sample {len(resc)} ({len(resc)/max(1,len(fail)):.0%})")
    print("median draft s", statistics.median(r["draft_s"] for r in it), " median n-sample s", statistics.median(r["samples_s"] for r in it))
    print("mean per-sample pass rate", round(statistics.mean(statistics.mean(r["sample_pass"]) for r in it), 3))


if __name__ == "__main__":
    main()
