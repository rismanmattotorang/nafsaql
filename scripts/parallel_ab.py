#!/usr/bin/env python3
"""Interleaved live A/B of concurrent hedged rounds on System 2 (FR-094) against the sequential loop.

Items: the 8 MBPP+ code tasks that went to System 2 in the 100-prompt run (benchmarks/perf_100.json): four were
solved by System 2's first attempt, four failed all three attempts. They are the population the slow path exists for.
Each request is forced onto System 2 (`force_mode: slow`), ephemeral, so only the slow path is measured.

Arms (same item, same model, same budget; order rotated per item):
  seq      parallel 1, repair_rounds 2   the loop of Phase 12: one attempt, then up to two serial repairs
  par4     parallel 4, repair_rounds 0   one round of four concurrent fresh candidates, no repairs
  par4r1   parallel 4, repair_rounds 1   four concurrent candidates, then one round of four repairs of the best
  hedge    parallel 4, repair_rounds 1, hedge_after_failure   the first attempt alone, then (only if it failed) one round of four repairs

Per run: verified or not, wall time, model calls, tokens, and the per-candidate notes. Results are appended to
benchmarks/parallel_ab.json after every run and runs already in it are skipped, so the script can be stopped and
resumed (check nafs_backend_inflight before resuming). Holds `caffeinate -i` so the Mac does not sleep.

    python3 -I scripts/parallel_ab.py [--arms seq,par4] [--items hard|all] [--label "IDLE HOST"]
"""
import argparse, json, os, subprocess, time, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARMS = {
    "seq": {"parallel": 1, "repair_rounds": 2},
    "seq_b": {"parallel": 1, "repair_rounds": 2},   # a second replicate of seq, interleaved with the other arms
    "par4": {"parallel": 4, "repair_rounds": 0},
    "par4r1": {"parallel": 4, "repair_rounds": 1},
    "par2": {"parallel": 2, "repair_rounds": 1},
    "hedge": {"parallel": 4, "repair_rounds": 1, "hedge_after_failure": True},   # first attempt alone, then one round of four
}
HARD = ["mbpp_plus-239", "mbpp_plus-300", "mbpp_plus-610", "mbpp_plus-91", "mbpp_plus-759", "mbpp_plus-274", "mbpp_plus-576", "mbpp_plus-311"]


def load_items(ids):
    rows = {}
    for line in open(os.path.join(ROOT, "data/bench/mbpp_plus.jsonl")):
        r = json.loads(line)
        rows[r["id"]] = r
    return [rows[i] for i in ids if i in rows]


def post(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001 - the record says what failed
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--arms", default="seq,par4")
    ap.add_argument("--items", default="hard")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/parallel_ab.json"))
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    arms = a.arms.split(",")
    items = load_items(HARD if a.items == "hard" else [i for i in a.items.split(",")])
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"label": a.label, "runs": []}
    done = {(r["item"], r["arm"]) for r in out["runs"]}
    caff = subprocess.Popen(["caffeinate", "-i"])
    try:
        for n, it in enumerate(items):
            order = arms[n % len(arms):] + arms[:n % len(arms)]
            for arm in order:
                if (it["id"], arm) in done:
                    continue
                body = {"messages": [{"role": "user", "content": it["prompt"]}], "force_mode": "slow", "ephemeral": True, "lab": False,
                        "verification": it["verification"], **ARMS[arm]}
                t = time.time()
                r = post(a.base, body)
                rec = {"item": it["id"], "arm": arm, "order": n, "client_s": round(time.time() - t, 1)}
                if "error" in r:
                    rec["error"] = r["error"]
                else:
                    u = r["usage"]
                    rec.update({"verified": r["assurance"] == "verified", "assurance": r["assurance"], "wall_s": round(u["wall_ms"] / 1000, 1),
                                "model_calls": u["model_calls"], "completion_tokens": u["completion_tokens"], "reasoning_tokens": u["reasoning_tokens"],
                                "rounds": [{"round": x["round"], "passed": x["passed"]} for x in r["rounds"]],
                                "notes": [s.get("note") for s in u["stages"] if (s.get("note") or "").startswith("candidate")]})
                out["runs"].append(rec)
                json.dump(out, open(a.out, "w"), indent=1)
                print(json.dumps({k: rec.get(k) for k in ("item", "arm", "verified", "wall_s", "model_calls", "error")}), flush=True)
    finally:
        caff.terminate()


if __name__ == "__main__":
    main()
