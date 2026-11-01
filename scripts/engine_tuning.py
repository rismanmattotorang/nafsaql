"""llama.cpp server-flag matrix on this CPU: prefill and decode of both models under each flag, against the baseline.

Phase 13 measured batching and speculation; this measures the engine's own knobs that the serving script never
varied: flash attention, micro-batch size, thread count, the busy-wait level (`--poll`), memory locking. Each
configuration is run with `llama-bench` on the production GGUF files: prompt processing of 512 tokens (`pp512`),
generation of 64 tokens at depth 0 (`tg64`) and at depth 1024 (`tg64@d1024`, the depth of a repair prompt).

Method: configurations are cycled in rounds (base, fa_off, ... then again), one repetition each, and the report takes
the median over rounds, so a burst of load from another program hits every arm alike instead of one arm's block
(the first, blocked version of this script, `benchmarks/engine_tuning_blocked.json`, was unusable for that reason:
load 7-11 on a 4-core host turned single blocks by 0.4x). The host load before every run is recorded.
Writes benchmarks/engine_tuning.json; `--report` prints the table from it.

    python3 -I scripts/engine_tuning.py [--models system1,system2] [--rounds 3] [--only base,fa_off,...] [--tests pp512@d0,tg64@d1024]
    python3 -I scripts/engine_tuning.py --report
"""
import argparse, json, os, statistics, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "third_party/llama.cpp/build/bin/llama-bench")
OUT = os.path.join(ROOT, "benchmarks/engine_tuning.json")
MODELS = {"system1": "models/Qwen3-1.7B-Q4_K_M.gguf", "system2": "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf"}
# name -> extra llama-bench flags (the baseline is what scripts/serve_llama.sh runs today: -t 4, fa auto, ub 512, poll 50, prio 0)
CONFIGS = {
    "base": [],
    "fa_off": ["-fa", "off"],
    "fa_on": ["-fa", "on"],
    "ub256": ["-ub", "256"],
    "ub1024": ["-ub", "1024"],
    "t3": ["-t", "3"],
    "poll0": ["--poll", "0"],
    "mlock": ["-lm", "mlock"],
}
# --prio is refused on macOS (SCHED_FIFO needs privileges); -ub 128 was 0.6x on both models in the blocked run.
TESTS = {"pp512@d0": (0, 512, 0), "tg64@d0": (0, 0, 64), "tg64@d1024": (1024, 0, 64)}


def bench(model, extra, reps, depth, pp, tg):
    cmd = [BENCH, "-m", model, "-t", "4", "-r", str(reps), "-p", str(pp), "-n", str(tg), "-d", str(depth), "-o", "json"] + extra
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if out.returncode != 0:
        return [{"error": out.stderr[-400:]}]
    rows = json.loads(out.stdout)
    return [{"test": f"{'pp' + str(r['n_prompt']) if r['n_prompt'] else 'tg' + str(r['n_gen'])}@d{r['n_depth']}",
             "avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"], "samples_ns": r.get("samples_ns")} for r in rows]


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    names = a.only.split(",") if a.only else list(CONFIGS)
    models = a.models.split(",")
    tests = a.tests.split(",")
    for rnd in range(a.rounds):
        for model in models:
            path = os.path.join(ROOT, MODELS[model])
            for name in names:
                for test in tests:
                    if any(r["model"] == model and r["config"] == name and r.get("test") == test and r.get("round") == rnd for r in doc["runs"]):
                        continue
                    depth, pp, tg = TESTS[test]
                    load = os.getloadavg()[0]
                    t0 = time.time()
                    rows = bench(path, CONFIGS[name], 1, depth, pp, tg)
                    for r in rows:
                        if r.get("test") and r["test"] != test:
                            continue  # llama-bench also reports pp at depth when -p and -d are both set
                        r.update({"model": model, "config": name, "flags": " ".join(CONFIGS[name]), "round": rnd, "load_before": round(load, 2),
                                  "wall_s": round(time.time() - t0, 1), "at": time.strftime("%H:%M:%S")})
                        doc["runs"].append(r)
                        print(f"r{rnd} {model} {name:8s} {r.get('test', 'ERR'):12s} {r.get('avg_ts', 0):7.2f} tok/s  load {load:.1f}", flush=True)
                    json.dump(doc, open(OUT, "w"), indent=1)
    print("done")


def report(a):
    doc = json.load(open(OUT))
    tests = [t for t in TESTS if any(r.get("test") == t for r in doc["runs"])]
    for model in sorted({r["model"] for r in doc["runs"]}):
        print(f"\n{model}: median tokens/s over rounds per flag (baseline = 1.00); n = rounds")
        med = lambda name, t: (lambda xs: (statistics.median(xs), len(xs)) if xs else (None, 0))([r["avg_ts"] for r in doc["runs"] if r["model"] == model and r["config"] == name and r.get("test") == t and "avg_ts" in r])
        base = {t: med("base", t)[0] for t in tests}
        print(f"{'config':8s} {'flags':14s} " + " ".join(f"{t:>20s}" for t in tests))
        for name in CONFIGS:
            cells = []
            for t in tests:
                v, n = med(name, t)
                cells.append(f"{'–':>20s}" if v is None or not base[t] else f"{v:7.2f} ({v / base[t]:.2f}x) n{n}")
            if any("(" in c for c in cells):
                print(f"{name:8s} {' '.join(CONFIGS[name]):14s} " + " ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="system1,system2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--tests", default="pp512@d0,tg64@d1024")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
