"""Where Smiley Core's decode time goes, and how it scales with threads (Phase 17).

Starts temporary nafs-smiley-server processes (not the live ones) with 1..4 threads, sends the same greedy request, and records
the server's own timings: decode tokens/s, the bytes moved per token (and so the memory bandwidth reached), and the engine's phase
split (matmul, output head, attention, other, sampling, drafting). Rounds interleave the thread counts so a drift in the shared host's
load does not favour one. Writes benchmarks/engine_profile.json; `--report` prints the table.

    python3 -I scripts/engine_profile.py [--rounds 3] [--model system1|system2|both] [--threads 1,2,3,4]
    python3 -I scripts/engine_profile.py --report
"""
import argparse, json, os, statistics, subprocess, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/engine_profile.json")
BIN = os.path.join(ROOT, "rust/target/release/nafs-smiley-server")
MODELS = {"system1": "models/Qwen3-1.7B-Q4_K_M.gguf", "system2": "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf"}
PORT = 8231
PROMPT = "Explain, in plain words, why the sky is blue and why sunsets are red. Be thorough."
NAMES = ["matmul", "head", "attention", "other", "sampling", "drafting"]


def post(body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def serve(model, threads, extra, env=None):
    p = subprocess.Popen([BIN, "-m", os.path.join(ROOT, MODELS[model]), "--alias", model, "-t", str(threads), "-c", "4096", "--port", str(PORT), *extra],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.1)
    p.kill()
    raise RuntimeError("server did not start")


def measure(model, threads, extra=(), n=2, env=None):
    p = serve(model, threads, list(extra), env)
    try:
        rows = []
        for i in range(n + 1):  # the first request warms the caches and is dropped
            body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 96, "temperature": 0.0, "seed": 1,
                    "chat_template_kwargs": {"enable_thinking": model == "system2"}}
            load = os.getloadavg()[0]
            r = post(body)
            if i == 0:
                continue
            t = r["timings"]
            s = t["smiley"]
            dec = s["phases_ms"]["decode"]
            tot = sum(dec) or 1.0
            rows.append({"tok_s": t["predicted_per_second"], "bytes_per_token": s["bytes_per_token"], "gbs": s["bytes_per_token"] * t["predicted_per_second"] / 1e9,
                         "predicted_n": t["predicted_n"], "phase_share": {k: v / tot for k, v in zip(NAMES, dec)}, "phase_ms": dict(zip(NAMES, dec)), "load": round(load, 1)})
        return rows
    finally:
        p.terminate()
        p.wait()


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    models = ["system1", "system2"] if a.model == "both" else [a.model]
    threads = [int(x) for x in a.threads.split(",")]
    env = dict(kv.split("=", 1) for kv in a.env)
    for rnd in range(a.start_round, a.start_round + a.rounds):
        for model in models:
            order = threads[rnd % len(threads):] + threads[:rnd % len(threads)]
            for th in order:
                if any(r["round"] == rnd and r["model"] == model and r["threads"] == th and r.get("variant", "") == a.variant for r in doc["runs"]):
                    continue
                rows = measure(model, th, a.extra.split() if a.extra else [], env=env)
                doc["runs"].append({"round": rnd, "model": model, "threads": th, "variant": a.variant, "rows": rows, "at": time.strftime("%H:%M:%S")})
                json.dump(doc, open(OUT, "w"), indent=1)
                print(f"r{rnd} {model} t={th} {a.variant or 'base':10s} tok/s={statistics.median(r['tok_s'] for r in rows):.2f} GB/s={statistics.median(r['gbs'] for r in rows):.1f} load={rows[0]['load']}", flush=True)
    print("done")


def report(a):
    doc = json.load(open(OUT))
    print(f"{'model':8s} {'variant':10s} {'threads':>7s} {'tok/s':>7s} {'GB/s':>6s} " + " ".join(f"{n:>9s}" for n in NAMES))
    for model in ("system1", "system2"):
        for variant in sorted({r.get("variant", "") for r in doc["runs"]}):
            for th in (1, 2, 3, 4):
                rs = [x for r in doc["runs"] if r["model"] == model and r["threads"] == th and r.get("variant", "") == variant for x in r["rows"]]
                if not rs:
                    continue
                share = [statistics.median(x["phase_share"][n] for x in rs) * 100 for n in NAMES]
                print(f"{model:8s} {variant or 'base':10s} {th:7d} {statistics.median(x['tok_s'] for x in rs):7.2f} {statistics.median(x['gbs'] for x in rs):6.1f} " + " ".join(f"{v:8.1f}%" for v in share))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="both", choices=["system1", "system2", "both"])
    ap.add_argument("--threads", default="1,2,3,4")
    ap.add_argument("--variant", default="")
    ap.add_argument("--extra", default="", help="extra nafs-smiley-server flags, e.g. '--kv i8'")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server process (repeatable), e.g. SMILEY_CORE_PF=2048")
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
