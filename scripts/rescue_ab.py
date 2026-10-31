#!/usr/bin/env python3
"""Interleaved live A/B of the System 1 rescue (FR-096) on checkable code tasks.

Same MBPP+ item, same request, rescue off against rescue on (k samples); arms alternate per item. Each request is streamed
and closed at the gate's decision event, so the time is exactly draft + (rescue) + verification + gate and no System 2
run is started; `escalate` says whether System 2 would have been used. Items are the shuffled MBPP+ list of
scripts/s1_rescue.py (same seed), so the two studies describe the same tasks.

The saving of an avoided escalation is not measured here (that would run System 2 on every failing item): it comes from the
100-prompt run, where the 8 code items that reached System 2 took a median of 289 s and a mean of 580 s (`perf_100.json`).

    python3 -I scripts/rescue_ab.py [--n 40] [--k 4] [--label "IDLE HOST"]
Writes benchmarks/rescue_ab.json (resumable).
"""
import argparse, json, os, random, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def decision(base, it, rescue):
    body = {"messages": [{"role": "user", "content": it["prompt"]}], "verification": it["verification"], "ephemeral": True, "rescue": rescue}
    req = urllib.request.Request(base + "/v1/chat/stream", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ev, verdicts, phases = None, [], []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])["data"]
                if ev == "verification":
                    verdicts.append(bool(d["result"]["passed"]))
                elif ev == "phase":
                    phases.append(d.get("label", ""))
                elif ev == "decision":
                    return {"s": round(time.time() - t0, 2), "escalate": d.get("escalate"), "rule": d.get("rule"), "verdicts": verdicts,
                            "rescued": any("drawing" in p for p in phases) and bool(verdicts) and verdicts[-1] and not d.get("escalate"),
                            "rescue_run": any("drawing" in p for p in phases)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/rescue_ab.json"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/bench/mbpp_plus.jsonl"))]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"label": a.label, "k": a.k, "runs": []}
    done = {(r["id"], r["arm"]) for r in out["runs"]}
    for n, it in enumerate(rows):
        for arm in (["off", "on"] if n % 2 == 0 else ["on", "off"]):
            if (it["id"], arm) in done:
                continue
            r = decision(a.base, it, a.k if arm == "on" else 0)
            rec = {"id": it["id"], "arm": arm, **(r or {"error": "no decision"})}
            out["runs"].append(rec)
            json.dump(out, open(a.out, "w"), indent=1)
            print(json.dumps(rec), flush=True)
    by = {}
    for r in out["runs"]:
        by.setdefault(r["id"], {})[r["arm"]] = r
    pairs = [(v["off"], v["on"]) for v in by.values() if "off" in v and "on" in v and "s" in v["off"] and "s" in v["on"]]
    esc_off = sum(1 for o, n in pairs if o["escalate"]); esc_on = sum(1 for o, n in pairs if n["escalate"])
    print(f"pairs {len(pairs)}  escalations off {esc_off} on {esc_on}  mean s off {statistics.mean(o['s'] for o,n in pairs):.1f} on {statistics.mean(n['s'] for o,n in pairs):.1f}")


if __name__ == "__main__":
    main()
