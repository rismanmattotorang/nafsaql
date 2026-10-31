#!/usr/bin/env python3
"""Interleaved A/B of fusing the agreement samples with the draft (FR-095) on unverified maths.

For an unverified maths draft the gate always draws k = 2 cheap samples (A-060). Sequentially they run after the
draft; fused they share the draft's batch (four streams cost 1.7 times one on this CPU, `batch_scaling_system1.json`).
Each request is streamed and closed at the gate's decision event, so the measured time is exactly draft + samples +
gate and no System 2 run is started. Arms alternate per prompt and per repetition.

    python3 -I scripts/fuse_ab.py [--n 14] [--reps 2] [--label "IDLE HOST"]
Writes benchmarks/fuse_ab.json (resumable).
"""
import argparse, json, os, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def decision_time(base, prompt, fuse, lam):
    body = {"messages": [{"role": "user", "content": prompt}], "ephemeral": True, "domain_hint": "math", "fuse_samples": fuse, "lambda": lam}
    req = urllib.request.Request(base + "/v1/chat/stream", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ev, first = None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                if ev == "token" and first is None:
                    first = time.time() - t0
                if ev == "decision":
                    d = json.loads(line[5:])["data"]
                    return {"s": round(time.time() - t0, 2), "first_token_s": round(first or 0, 2), "escalate": d.get("escalate"), "rule": d.get("rule"),
                            "k": d["features"].get("k"), "agreement": d["features"].get("agreement"), "draft_tokens": d["features"].get("draft_tokens")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/fuse_ab.json"))
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] if "prompt" in json.loads(l) else json.loads(l)["question"] for l in open(os.path.join(ROOT, "data/bench/gsm8k.jsonl"))][: a.n]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"label": a.label, "runs": []}
    done = {(r["i"], r["arm"], r["rep"]) for r in out["runs"]}
    for rep in range(a.reps):
        for i, p in enumerate(prompts):
            order = ["fused", "seq"] if (i + rep) % 2 == 0 else ["seq", "fused"]
            for arm in order:
                if (i, arm, rep) in done:
                    continue
                r = decision_time(a.base, p, arm == "fused", a.lam)
                rec = {"i": i, "arm": arm, "rep": rep, **(r or {"error": "no decision"})}
                out["runs"].append(rec)
                json.dump(out, open(a.out, "w"), indent=1)
                print(json.dumps(rec), flush=True)
    for arm in ("seq", "fused"):
        xs = [r["s"] for r in out["runs"] if r["arm"] == arm and "s" in r]
        print(arm, "n", len(xs), "median", statistics.median(xs), "mean", round(statistics.mean(xs), 2))


if __name__ == "__main__":
    main()
