"""Co-scheduling of the two model servers on four cores: how much System 2 slows when System 1 decodes at the same
time, under each server-flag variant, and the reverse.

Both llama-servers run `-t 4` on a 4-core CPU. When one is decoding alone it owns the cores; when both decode (a user's
draft while another's thorough answer is being written, or the rescue samples of one request beside the thinking of
another) eight busy threads share four cores, and llama.cpp's threads busy-wait for each other (`--poll 50`), so the
loss can exceed the arithmetic 2x. This script starts a temporary pair of servers per variant (ports 8011/8012, the
production servers stay up and idle), measures three things with a cached ~800-token context, and stops them:

  s2_alone     System 2, one stream of `--tokens` tokens: tokens/s
  s2_with_s1   the same, while System 1 decodes one stream after another until System 2 ends: both rates
  s1_alone     System 1, one stream: tokens/s

Variants (name -> flags for S1 | S2): base, poll0, s1t2, s1t3, poll0_s1t2 (`--prio` is refused on macOS: SCHED_FIFO needs privileges). Reps are interleaved across
variants. Writes benchmarks/cosched_ab.json; `--report` prints the table.

    python3 -I scripts/cosched_ab.py [--reps 3] [--tokens 96] [--only base,poll0]
"""
import argparse, json, os, statistics, subprocess, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.join(ROOT, "third_party/llama.cpp/build/bin/llama-server")
OUT = os.path.join(ROOT, "benchmarks/cosched_ab.json")
S1 = os.path.join(ROOT, "models/Qwen3-1.7B-Q4_K_M.gguf")
S2 = os.path.join(ROOT, "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf")
P1, P2 = 8011, 8012
COMMON = ["--host", "127.0.0.1", "--jinja", "--reasoning-format", "deepseek", "--cache-reuse", "256"]
VARIANTS = {
    "base": (["-t", "4"], ["-t", "4"]),
    "poll0": (["-t", "4", "--poll", "0"], ["-t", "4", "--poll", "0"]),
    "s1t2": (["-t", "2"], ["-t", "4"]),
    "s1t3": (["-t", "3"], ["-t", "4"]),
    "poll0_s1t2": (["-t", "2", "--poll", "0"], ["-t", "4", "--poll", "0"]),
}
CONTEXT = ("Consider this function and explain every step of how it merges two sorted lists into one sorted list, "
           "then discuss its complexity and an alternative using a heap. ") * 40  # ~800 tokens


def serve(model, port, flags, ctx, slots):
    args = [BIN, "-m", model, "-c", str(ctx), "-np", str(slots), "--port", str(port)] + COMMON + flags
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(os.path.join(ROOT, f"logs/cosched_{port}.log"), "w"))
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit(f"server on {port} did not start")


def gen(port, tokens, variant_seed, cache=True):
    body = {"prompt": CONTEXT + f" Variant {variant_seed}. Explanation:", "n_predict": tokens, "temperature": 0.0, "ignore_eos": True,
            "cache_prompt": cache, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    tm = d.get("timings", {})
    return {"wall_s": time.time() - t, "tok_s": tm.get("predicted_per_second"), "n": tm.get("predicted_n"), "prefill_ms": tm.get("prompt_ms")}


def warm(port):
    gen(port, 4, 0)


def measure(tokens):
    warm(P1); warm(P2)
    s2_alone = gen(P2, tokens, 1)
    s1_alone = gen(P1, tokens, 1)
    # S2 with S1 decoding beside it until S2 ends.
    stop = threading.Event()
    s1_runs = []

    def s1_load():
        while not stop.is_set():
            s1_runs.append(gen(P1, 64, 2))

    th = threading.Thread(target=s1_load, daemon=True)
    th.start()
    time.sleep(0.5)
    s2_with = gen(P2, tokens, 1)
    stop.set()
    th.join(timeout=120)
    return {"s2_alone_tok_s": s2_alone["tok_s"], "s1_alone_tok_s": s1_alone["tok_s"], "s2_with_s1_tok_s": s2_with["tok_s"],
            "s1_with_s2_tok_s": statistics.mean(r["tok_s"] for r in s1_runs) if s1_runs else None, "s1_runs_during_s2": len(s1_runs),
            "s2_alone_wall_s": s2_alone["wall_s"], "s2_with_s1_wall_s": s2_with["wall_s"]}


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    names = a.only.split(",") if a.only else list(VARIANTS)
    for rep in range(a.reps):
        order = names if rep % 2 == 0 else names[::-1]
        for name in order:
            f1, f2 = VARIANTS[name]
            p1 = serve(S1, P1, f1, 4096, 2)
            p2 = serve(S2, P2, f2, 4096, 2)
            try:
                load = os.getloadavg()[0]
                m = measure(a.tokens)
            finally:
                p1.terminate(); p2.terminate(); p1.wait(); p2.wait()
            m.update({"variant": name, "rep": rep, "flags_s1": " ".join(f1), "flags_s2": " ".join(f2), "load_before": round(load, 2), "at": time.strftime("%H:%M:%S")})
            doc["runs"].append(m)
            json.dump(doc, open(OUT, "w"), indent=1)
            print(f"{name:11s} rep{rep} S2 alone {m['s2_alone_tok_s']:.2f} with S1 {m['s2_with_s1_tok_s']:.2f} tok/s | S1 alone {m['s1_alone_tok_s']:.2f} with S2 {m['s1_with_s2_tok_s'] or 0:.2f} | load {load:.1f}", flush=True)
    print("done")


def report(a):
    doc = json.load(open(OUT))
    print(f"{'variant':11s} {'S2 alone':>9s} {'S2+S1':>8s} {'S2 keeps':>9s} {'S1 alone':>9s} {'S1+S2':>8s} {'S1 keeps':>9s} {'sum tok/s':>10s}")
    for name in VARIANTS:
        rs = [r for r in doc["runs"] if r["variant"] == name and r.get("s1_with_s2_tok_s")]
        if not rs:
            continue
        med = lambda k: statistics.median(r[k] for r in rs)
        s2a, s2w, s1a, s1w = med("s2_alone_tok_s"), med("s2_with_s1_tok_s"), med("s1_alone_tok_s"), med("s1_with_s2_tok_s")
        print(f"{name:11s} {s2a:9.2f} {s2w:8.2f} {s2w / s2a:8.0%} {s1a:9.2f} {s1w:8.2f} {s1w / s1a:8.0%} {s2w + s1w:10.2f}   (n={len(rs)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--only", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
