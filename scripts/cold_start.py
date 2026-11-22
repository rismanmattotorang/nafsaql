#!/usr/bin/env python3
"""Phase 34: the first request of a freshly started engine, with and without reading the weights ahead.

A started server has the model file mapped and none of it in memory. This makes a copy of the model that the system
has not cached (written with F_NOCACHE), starts `nafs-smiley-server` on it, sends one chat request as soon as the server
is healthy (or `--wait` seconds later) and times it; then a second request of the same size, which finds the weights in
memory. Arms, interleaved by round: `off` (--no-read-ahead) and `on` (the default).

Usage: python3 -I scripts/cold_start.py [--model models/Qwen3-1.7B-Q4_K_M.gguf] [--rounds 3] [--wait 0]
                                        [--out benchmarks/engine_phase34/cold_start.json]
"""
import fcntl, json, os, statistics, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
MODEL = os.path.join(ROOT, arg("--model", "models/Qwen3-1.7B-Q4_K_M.gguf"))
ROUNDS, WAIT, PORT = int(arg("--rounds", 3)), float(arg("--wait", 0)), int(arg("--port", 8291))
BIN = os.path.join(ROOT, "rust/target/release/nafs-smiley-server")
F_NOCACHE = 48  # macOS: writes through this descriptor do not enter the unified buffer cache


def cold_copy(dst):
    """Copy the model so that the copy's pages are not in memory."""
    with open(MODEL, "rb") as src, open(dst, "wb") as out:
        if sys.platform == "darwin":
            fcntl.fcntl(out.fileno(), F_NOCACHE, 1)
        while True:
            block = src.read(8 << 20)
            if not block:
                break
            out.write(block)
        out.flush()
        os.fsync(out.fileno())


def ask(text):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return time.time() - t, out["timings"]


def prompt(seed):
    # About 200 tokens that no earlier request shares, so nothing comes from the prefix store.
    words = ["river", "stone", "paper", "garden", "window", "letter", "morning", "copper", "silver", "market"]
    return f"Note {seed}: " + " ".join(words[(seed * 7 + i * 3) % 10] for i in range(150)) + ". Reply with one word."


def healthy():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).read()
        return True
    except Exception:
        return False


rows = {"off": [], "on": []}
tmp = tempfile.mkdtemp(prefix="nafs-cold-")
try:
    for r in range(ROUNDS):
        for k in range(2):
            arm = ["off", "on"][(k + r) % 2]
            copy = os.path.join(tmp, f"cold-{r}-{arm}.gguf")
            cold_copy(copy)
            cmd = [BIN, "-m", copy, "--alias", "cold", "--port", str(PORT), "-t", "3", "--prefill-threads", "4",
                   "-c", "2048", "--slots", "1"] + (["--no-read-ahead"] if arm == "off" else [])
            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                while not healthy():
                    if proc.poll() is not None:
                        raise SystemExit("the server did not start")
                    time.sleep(0.02)
                start_s = time.time() - t0
                time.sleep(WAIT)
                first_s, t1 = ask(prompt(r * 10 + k))
                second_s, t2 = ask(prompt(r * 10 + k + 5))
            finally:
                proc.terminate()
                proc.wait()
                os.remove(copy)
            rows[arm].append({"start_s": round(start_s, 2), "first_request_s": round(first_s, 2),
                              "second_request_s": round(second_s, 2), "first_prompt_ms": round(t1["prompt_ms"]),
                              "second_prompt_ms": round(t2["prompt_ms"]), "prompt_n": t1["prompt_n"]})
            print(f"round {r} {arm:3}  start {start_s:5.2f}s  first request {first_s:6.2f}s  second {second_s:5.2f}s"
                  f"  ({t1['prompt_n']} prompt tokens)", flush=True)
finally:
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)

summary = {arm: {k: round(statistics.median(x[k] for x in v), 2) for k in ("start_s", "first_request_s", "second_request_s")}
           for arm, v in rows.items() if v}
print(json.dumps(summary, indent=1))
out = arg("--out", "")
if out:
    json.dump({"model": os.path.basename(MODEL), "rounds": ROUNDS, "wait_s": WAIT, "load_end": os.getloadavg(),
               "note": "each run starts a server on a copy of the model the system has not cached; arms interleaved",
               "rows": rows, "median": summary}, open(os.path.join(ROOT, out), "w"), indent=1)
