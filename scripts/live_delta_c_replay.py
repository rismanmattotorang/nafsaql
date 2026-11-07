#!/usr/bin/env python3
"""Paired evaluation of the live latency cost ([gate] live_delta_c) by replay of one evaluation run.

Evaluation runs never use the live latency cost (lab requests keep delta_c static, so runs stay reproducible).
The calibrated gate is a deterministic function of (e1_hat, e2, lambda, delta_c), and a run with `fast_only`,
`slow_only` and `nafs_blind` holds both outcomes of every item. So both arms can be scored on the SAME items
and the SAME model outputs, which is a tighter pairing than two runs at different moments of the owner's load:

  off: escalate iff L (e1 - e2) > lambda * delta_c / cost_unit_s            (delta_c static, as today)
  on:  the same with delta_c x clamp(ref_tok_s / median System 2 decode tok/s)

The median is the gateway's estimator replayed causally: a sliding window over the `slow_only` items that ran
before each `nafs_blind` item, each timed as completion tokens over wall time (approximate: wall time also holds
prefill and verification, so it reads a little slower than server decode time). An escalated item scores as
`slow_only` on that task; a kept item as the draft's exact label. Run the evaluation while the host is under
load, or the two arms barely differ.

Usage: python3.11 -I scripts/live_delta_c_replay.py RUN_ID [--base http://127.0.0.1:8080] [--config config/nafs-aql.toml]
           [--ref 6.9] [--min-scale 0.5] [--max-scale 3.0] [--window 16] [--min-samples 4] [--out FILE]
"""
import json, os, random, statistics, sys, tomllib, urllib.request
from collections import deque
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    run_id, base = sys.argv[1], arg("--base", "http://127.0.0.1:8080")
    cfg = tomllib.load(open(os.path.join(ROOT, arg("--config", "config/nafs-aql.toml")), "rb"))["gate"]
    ref = float(arg("--ref", cfg.get("live_delta_c_ref_tok_s", 6.9)))
    lo = float(arg("--min-scale", cfg.get("live_delta_c_min_scale", 0.5)))
    hi = float(arg("--max-scale", cfg.get("live_delta_c_max_scale", 3.0)))
    window = int(arg("--window", cfg.get("live_delta_c_window", 16)))
    min_samples = int(arg("--min-samples", cfg.get("live_delta_c_min_samples", 4)))
    run, gate = get(base, f"/v1/eval/runs/{run_id}"), get(base, "/v1/gate")
    items = [i for i in run["items"] if not i.get("error")]
    params = gate["params"]
    lam_run = run["request"].get("lambda")
    fast = {i["task_id"]: i for i in items if i["system"] == "fast_only"}
    slow = {i["task_id"]: i for i in items if i["system"] == "slow_only"}

    speeds, rows = deque(maxlen=window), []
    for it in items:  # run order: the estimator only sees System 2 timings from before each decision
        if it["system"] == "slow_only" and it["completion_tokens"] >= 32 and it["wall_ms"] > 0:
            speeds.append(it["completion_tokens"] * 1000.0 / it["wall_ms"])
            continue
        sysname = it["system"]
        if not sysname.startswith("nafs_blind"):
            continue
        lam = int(sysname.rsplit("_l", 1)[1]) / 100 if "_l" in sysname else (lam_run or cfg["lambda_default"])
        s, f = slow.get(it["task_id"]), fast.get(it["task_id"])
        draft_ok = it.get("draft_correct")
        if draft_ok is None and f is not None:
            draft_ok = f.get("correct")
        if s is None or s.get("correct") is None or draft_ok is None:
            continue
        p = params[it["domain"]]
        median = statistics.median(speeds) if len(speeds) >= min_samples else None
        k = min(max(ref / median, lo), hi) if median else 1.0
        gain = cfg["loss_wrong"] * (it["e1_hat"] - p["e2_hat"])
        esc = {arm: gain > lam * p["delta_c_s"] * m / cfg["cost_unit_s"] for arm, m in (("off", 1.0), ("on", k))}
        fw = f["wall_ms"] / 1000 if f else 0.0
        rows.append({"task_id": it["task_id"], "domain": it["domain"], "median_tok_s": median, "scale": k,
                     "draft_correct": bool(draft_ok), "slow_correct": bool(s["correct"]),
                     **{f"escalate_{a}": e for a, e in esc.items()},
                     **{f"correct_{a}": bool(s["correct"]) if e else bool(draft_ok) for a, e in esc.items()},
                     **{f"wall_s_{a}": fw + s["wall_ms"] / 1000 if e else fw for a, e in esc.items()}})
    if not rows:
        sys.exit("no scorable nafs_blind items: the run needs fast_only, slow_only and nafs_blind on the same tasks")

    n = len(rows)
    def arm(a):
        esc = [r for r in rows if r[f"escalate_{a}"]]
        return {"escalation_rate": len(esc) / n, "accuracy": sum(r[f"correct_{a}"] for r in rows) / n,
                "needless_escalations": sum(r["draft_correct"] for r in esc),
                "missed_escalations": sum((not r["draft_correct"]) and r["slow_correct"] and not r[f"escalate_{a}"] for r in rows),
                "mean_wall_s_est": statistics.fmean(r[f"wall_s_{a}"] for r in rows)}
    diffs = [r["correct_on"] - r["correct_off"] for r in rows]
    rng = random.Random(0)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(2000))
    ci = [boots[49], boots[1949]]
    b = sum(d > 0 for d in diffs); c = sum(d < 0 for d in diffs)
    scales = [r["scale"] for r in rows]
    out = {"run_id": run_id, "n": n, "promoted": gate["promoted"], "effective_rule": gate["effective_rule"],
           "estimator": {"ref_tok_s": ref, "clamp": [lo, hi], "window": window, "min_samples": min_samples,
                         "scale_min": min(scales), "scale_median": statistics.median(scales), "scale_max": max(scales),
                         "items_before_min_samples": sum(r["median_tok_s"] is None for r in rows)},
           "off": arm("off"), "on": arm("on"),
           "paired": {"delta_accuracy_on_minus_off": statistics.fmean(diffs), "ci95": ci,
                      "discordant": {"only_on_correct": b, "only_off_correct": c}, "mcnemar_p": mcnemar_exact(b, c),
                      "decisions_changed": sum(r["escalate_on"] != r["escalate_off"] for r in rows),
                      "non_inferior_3pt": ci[0] > -0.03},
           "adopt_if": "non_inferior_3pt and on.needless_escalations < off.needless_escalations, on a run under load (label load1)",
           "note": ("A calibration is promoted: the replay matches what the live gate would do." if gate["promoted"] else
                    "No promoted calibration: the live gate uses the fixed rule, which ignores delta_c, so the switch "
                    "changes nothing today. These numbers show what the calibrated rule would do with this e1_hat."),
           "items": rows}
    path = arg("--out", os.path.join(ROOT, f"benchmarks/live_delta_c_replay_{run_id[:8]}.json"))
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
