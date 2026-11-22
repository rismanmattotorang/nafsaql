#!/usr/bin/env python3
"""Phase 34: what a request pays to take over a kept conversation, before and after the zero-copy prefix store.

Runs `handoff_bench` of two builds in turn (the arms interleaved, so a drift in the machine's load reaches each), and
prints the median of every column per arm, model and depth.

    python3 -I scripts/handoff_ab.py --old <old handoff_bench> [--rounds 3] [--out benchmarks/engine_phase34/handoff_ab.json]

The old build is the engine before Phase 34 (a worktree of the previous commit with this example copied in). Arms:
  old      copies the cache in and out, last prompt token in a pass of its own
  move     the cache is moved, not copied (HANDOFF_FUSE=0)
  new      moved, and the first token is sampled from the last prompt chunk's pass
"""
import argparse, json, os, statistics, subprocess, sys

MODELS = {"system1": "models/Qwen3-1.7B-Q4_K_M.gguf", "system2": "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf"}
COLS = ["admit_ms", "first_token_ms", "decode_step_ms", "finish_step_ms"]


def run(binary, model, threads, depths, turns, env=None):
    out = subprocess.run([binary, model, str(threads), depths, str(turns), "8", "6"], capture_output=True, text=True,
                         env={**os.environ, **(env or {})}, check=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 7 and p[0].isdigit():
            rows.append({"depth": int(p[0]), "turn": int(p[1]), **{c: float(v) for c, v in zip(COLS, p[2:6])}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default="rust/target/release/examples/handoff_bench")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--depths", default="512,2048")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--models", default="system1,system2")
    ap.add_argument("--out", default="benchmarks/engine_phase34/handoff_ab.json")
    a = ap.parse_args()
    arms = [("old", a.old, None), ("move", a.new, {"HANDOFF_FUSE": "0"}), ("new", a.new, None)]
    data = {}
    for name in a.models.split(","):
        for r in range(a.rounds):
            for k in range(len(arms)):
                arm, binary, env = arms[(k + r) % len(arms)]
                for row in run(binary, MODELS[name], a.threads, a.depths, a.turns, env):
                    if row["turn"] == 1:
                        continue  # the first turn follows the long first read; the later ones are the steady state
                    data.setdefault((name, row["depth"], arm), []).append(row)
                print(f"{name} round {r + 1} {arm} done", file=sys.stderr, flush=True)
    summary = []
    for (name, depth, arm), rows in sorted(data.items()):
        med = {c: round(statistics.median(x[c] for x in rows), 1) for c in COLS}
        # What the request paid beyond its own work: taking the cache, and giving it back (the last step against an ordinary one).
        med["handoff_ms"] = round(med["admit_ms"] + max(0.0, med["finish_step_ms"] - med["decode_step_ms"]), 1)
        summary.append({"model": name, "depth": depth, "arm": arm, "n": len(rows), **med})
    print(f"{'model':8} {'depth':>5} {'arm':5} {'n':>3} " + " ".join(f"{c:>15}" for c in COLS + ["handoff_ms"]))
    for s in summary:
        print(f"{s['model']:8} {s['depth']:5} {s['arm']:5} {s['n']:3} " + " ".join(f"{s[c]:15.1f}" for c in COLS + ["handoff_ms"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"threads": a.threads, "rounds": a.rounds, "turns": a.turns, "suffix_tokens": 8, "new_tokens": 6,
               "note": "medians over turns 2.. of every round; arms interleaved", "rows": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
