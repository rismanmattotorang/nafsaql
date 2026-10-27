"""Performance test of the running service with 100 prompts of mixed kinds, all in Auto mode.

The set is built deterministically (seed 20261027) from the shipped question sets plus hand-written chat,
trick, factual and Indonesian prompts: 40 maths (30 GSM8K, 10 short word problems), 20 code (MBPP+ with their
tests), 12 structured (JSON format), 28 factual and open. Maths, code and structured prompts carry an exact check,
so the run measures accuracy as well as time. Requests are ephemeral (no chat history is written) and opt out of
the audit stream, so nothing but the prompt itself loads the models.

    python3 -I scripts/perf_100.py run    [--base http://127.0.0.1:8080] [--users 2] [--label "..."]
    python3 -I scripts/perf_100.py report [--out benchmarks/perf_100.json]

`run` is resumable: finished prompts are kept in the output file and skipped. While it runs a sampler records
the machine load, the models' in-flight requests and the CPU share of the model servers every 10 s.

On macOS `run` holds `caffeinate -i` for its own lifetime: a Mac that falls asleep mid-run stops the models, and
the wall clock then counts the sleep (the first full run, 2026-10-27, lost 6,383 s of 11,442 s to sleep cycles after
its keep-awake ended). `report` finds any sleep as gaps of more than 25 s between sampler ticks and reports the
active time, throughput and per-request client time without it.
"""
import argparse, json, os, random, statistics, subprocess, sys, threading, time, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks", "perf_100.json")

TRICK = [("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Answer with a number.", "5"),
         ("A lily pad patch doubles in size every day. It takes 48 days to cover the whole lake. How many days does it take to cover half the lake?", "47"),
         ("How many months of the year have 28 days?", "12"),
         ("I have 3 apples. I eat 2 and then buy 5 more. How many apples do I have?", "6"),
         ("What is 17 × 24?", "408"),
         ("A rectangle has length 12 and width 7. What is its perimeter?", "38"),
         ("What is the sum of the first 20 positive integers?", "210"),
         ("Tom is twice as old as Ann. In 6 years the sum of their ages will be 54. How old is Tom now?", "28"),
         ("A shirt costs $40 after a 20% discount. What was the original price in dollars?", "50"),
         ("What is the remainder when 2^10 is divided by 7?", "2")]
INDONESIAN = ["Jelaskan apa itu inflasi dalam tiga kalimat.", "Apa ibu kota Jepang dan terkenal karena apa? Jawab dalam dua kalimat.",
              "Berikan tiga tips belajar bahasa pemrograman bagi pemula.", "Terjemahkan ke bahasa Inggris: 'Saya suka membaca buku di pagi hari.'",
              "Jelaskan perbedaan antara RAM dan penyimpanan (storage) dengan singkat.", "Tulis salam pembuka formal untuk email kepada dosen dalam bahasa Indonesia.",
              "Mengapa langit berwarna biru? Jelaskan dengan sederhana.", "Apa manfaat olahraga teratur bagi kesehatan? Sebutkan empat."]
SHORT = ["Hello! What can you help me with?", "Name a colour.", "What is the chemical symbol for gold?", "How many days are in a leap year?",
         "What is the largest ocean on Earth?", "Who painted the Mona Lisa?", "How do you say thank you in French?", "What does CPU stand for?"]


def get(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build(base):
    rng = random.Random(20261027)
    ds = lambda name, n: get(base, f"/v1/eval/datasets/{name}?limit={n}")
    items = []
    for t in rng.sample(ds("gsm8k", 1319), 30):
        items.append(("math-gsm8k", t["id"], t["prompt"], t["verification"]))
    for i, (q, ref) in enumerate(TRICK):
        items.append(("math-short", f"trick-{i:02d}", q, {"reference_answer": ref}))
    for t in rng.sample(ds("mbpp_plus", 378), 20):
        items.append(("code", t["id"], t["prompt"], t["verification"]))
    for t in ds("schema_tasks", 12):
        items.append(("structured", t["id"], t["prompt"], t["verification"]))
    for t in ds("open_prompts", 12):
        items.append((t["domain"] + "-en", t["id"], t["prompt"], None))
    for i, q in enumerate(INDONESIAN):
        items.append(("open-id", f"id-{i:02d}", q, None))
    for i, q in enumerate(SHORT):
        items.append(("short-chat", f"short-{i:02d}", q, None))
    assert len(items) == 100, len(items)
    rng.shuffle(items)  # so heavy and light prompts are spread over the run
    return [{"n": i + 1, "kind": k, "id": id_, "prompt": p, "verification": v} for i, (k, id_, p, v) in enumerate(items)]


def sampler(base, stop, samples):
    def cpu():
        out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True).stdout
        return round(sum(float(l.split(None, 1)[0]) for l in out.splitlines()[1:] if "llama-server" in l), 1)
    while not stop.is_set():
        inflight = {}
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
                for l in r.read().decode().splitlines():
                    if l.startswith("nafs_backend_inflight"):
                        inflight[l.split('"')[1]] = float(l.split()[-1])
        except Exception:
            pass
        samples.append({"t": round(time.time() - T0, 1), "load1": round(os.getloadavg()[0], 2), "llama_cpu_pct": cpu(), "inflight": inflight})
        stop.wait(10)


def run(args):
    global T0
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    doc = json.load(open(args.out)) if os.path.exists(args.out) else {}
    prompts = build(args.base)
    doc.setdefault("runs", [])
    doc.update({"label": args.label, "base": args.base, "users": args.users, "n_prompts": len(prompts), "host": {"cpus": os.cpu_count(), "load_at_start": round(os.getloadavg()[0], 2)}})
    doc["health"] = get(args.base, "/healthz")
    doc["config"] = {k: v for k, v in get(args.base, "/v1/system").get("gate", {}).items() if k in ("k_samples", "lambda_default", "audit_fraction")}
    done = {r["n"] for r in doc["runs"]}
    todo = [p for p in prompts if p["n"] not in done]
    lock = threading.Lock()
    samples = doc.setdefault("samples", [])
    T0 = time.time() - (samples[-1]["t"] if samples else 0)
    started = doc.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
    stop = threading.Event()
    threading.Thread(target=sampler, args=(args.base, stop, samples), daemon=True).start()
    t_start = time.time()
    counter = [len(done)]

    def one(p):
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "ephemeral": True, "audit": False, "retrieve": False}
        if p["verification"]:
            body["verification"] = p["verification"]
        t0 = time.time()
        rec = {"n": p["n"], "kind": p["kind"], "id": p["id"], "prompt": p["prompt"][:120], "started_at": round(t0 - t_start, 1)}
        try:
            r = post(args.base, "/v1/chat", body, 2400)
            u = r["usage"]
            rec.update({"ok": True, "client_s": round(time.time() - t0, 2), "mode": r["reasoning_mode"], "assurance": r["assurance"], "domain": r["domain"],
                        "wall_ms": u["wall_ms"], "ttft_ms": u.get("ttft_ms"), "model_calls": u["model_calls"], "verifier_calls": u["verifier_calls"],
                        "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "reasoning_tokens": u["reasoning_tokens"],
                        "stages": u.get("stages", []), "rule": (r.get("decision") or {}).get("rule"), "e1_hat": (r.get("decision") or {}).get("e1_hat"),
                        "verified": r["assurance"] == "verified", "checkable": bool(p["verification"]), "answer": r["message"]["content"][:200]})
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            rec.update({"ok": False, "client_s": round(time.time() - t0, 2), "error": str(e)[:200]})
        with lock:
            doc["runs"].append(rec)
            counter[0] += 1
            doc["wall_s"] = round(doc.get("wall_prev_s", 0) + time.time() - t_start, 1)
            json.dump(doc, open(args.out, "w"))
            print(f"[{counter[0]:3d}/100] {rec['kind']:12s} {rec.get('mode','-'):4s} {rec.get('assurance','-'):10s} {rec['client_s']:7.1f}s "
                  f"ttft={rec.get('ttft_ms')} calls={rec.get('model_calls')} {'' if rec['ok'] else 'ERROR ' + rec['error']}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.users) as ex:
        list(ex.map(one, todo))
    stop.set()
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["wall_s"] = round(time.time() - t_start + doc.get("wall_prev_s", 0), 1)
    doc["host"]["load_at_end"] = round(os.getloadavg()[0], 2)
    json.dump(doc, open(args.out, "w"))
    print("done in", doc["wall_s"], "s")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summary(xs):
    return {"n": len(xs), "mean": statistics.fmean(xs) if xs else None, "p50": q(xs, .5), "p95": q(xs, .95), "p99": q(xs, .99), "max": max(xs) if xs else None}


def report(args):
    doc = json.load(open(args.out))
    R = [r for r in doc["runs"] if r["ok"]]
    bad = [r for r in doc["runs"] if not r["ok"]]
    s = lambda ms: None if ms is None else ms / 1000
    sm = doc.get("samples", [])
    gaps = [(a["t"], b["t"]) for a, b in zip(sm, sm[1:]) if b["t"] - a["t"] > 25]
    asleep = sum(g1 - g0 for g0, g1 in gaps)
    active = doc["wall_s"] - asleep
    out = {"requests": len(doc["runs"]), "failed": len(bad), "wall_s": doc["wall_s"], "asleep_s": asleep, "active_s": active, "users": doc["users"]}
    out["throughput_per_min"] = len(R) / active * 60
    for r in R:
        r["active_s"] = r["client_s"] - sum(max(0, min(r["started_at"] + r["client_s"], g1) - max(r["started_at"], g0)) for g0, g1 in gaps)
    out["active_client_s"] = summary([r["active_s"] for r in R])
    out["latency_s"] = summary([s(r["wall_ms"]) for r in R])
    out["queue_included_s"] = summary([r["client_s"] for r in R])
    out["ttft_s"] = summary([s(r["ttft_ms"]) for r in R if r["ttft_ms"] is not None])
    for m in ("fast", "slow"):
        rs = [r for r in R if r["mode"] == m]
        out[m] = {"latency_s": summary([s(r["wall_ms"]) for r in rs]), "ttft_s": summary([s(r["ttft_ms"]) for r in rs if r["ttft_ms"] is not None]),
                  "model_calls_mean": statistics.fmean([r["model_calls"] for r in rs]) if rs else None, "reasoning_tokens_mean": statistics.fmean([r["reasoning_tokens"] for r in rs]) if rs else None,
                  "completion_tokens_mean": statistics.fmean([r["completion_tokens"] for r in rs]) if rs else None}
    out["escalation_rate"] = sum(r["mode"] == "slow" for r in R) / len(R)
    out["assurance"] = {a: sum(r["assurance"] == a for r in R) for a in sorted({r["assurance"] for r in R})}
    chk = [r for r in R if r["checkable"]]
    out["checkable"] = {"n": len(chk), "verified": sum(r["verified"] for r in chk), "rate": sum(r["verified"] for r in chk) / max(1, len(chk))}
    out["by_kind"] = {}
    for k in sorted({r["kind"] for r in R}):
        rs = [r for r in R if r["kind"] == k]
        c = [r for r in rs if r["checkable"]]
        out["by_kind"][k] = {"n": len(rs), "escalated": sum(r["mode"] == "slow" for r in rs), "latency_s": summary([s(r["wall_ms"]) for r in rs]),
                             "verified": sum(r["verified"] for r in c), "checkable": len(c)}
    # where the time goes, from the stage records
    stage = {}
    for r in R:
        for st in r["stages"]:
            d = stage.setdefault(st["stage"], {"calls": 0, "ms": 0.0, "tokens": 0, "prefill_ms": 0.0, "decode_ms": 0.0, "prompt_n": 0, "cache_n": 0, "queue_ms": 0.0})
            d["calls"] += 1; d["ms"] += st["ms"]; d["tokens"] += st.get("tokens") or 0; d["queue_ms"] += st.get("queue_ms") or 0
            for k in ("prefill_ms", "decode_ms"):
                d[k] += st.get(k) or 0
            for k in ("prompt_n", "cache_n"):
                d[k] += st.get(k) or 0
    tot = sum(d["ms"] for d in stage.values()) or 1
    out["stages"] = {k: {**d, "share": d["ms"] / tot, "prefill_tok_s": d["prompt_n"] / d["prefill_ms"] * 1000 if d["prefill_ms"] else None,
                         "decode_tok_s": d["tokens"] / d["decode_ms"] * 1000 if d["decode_ms"] else None,
                         "cache_reuse": d["cache_n"] / max(1, d["cache_n"] + d["prompt_n"])} for k, d in sorted(stage.items(), key=lambda kv: -kv[1]["ms"])}
    out["tokens"] = {"prompt": sum(r["prompt_tokens"] for r in R), "completion": sum(r["completion_tokens"] for r in R), "reasoning": sum(r["reasoning_tokens"] for r in R)}
    out["tokens_per_s"] = out["tokens"]["completion"] / active
    if sm:
        out["machine"] = {"load1_mean": statistics.fmean(x["load1"] for x in sm), "load1_max": max(x["load1"] for x in sm), "llama_cpu_pct_mean": statistics.fmean(x["llama_cpu_pct"] for x in sm),
                          "llama_cpu_pct_max": max(x["llama_cpu_pct"] for x in sm), "s2_inflight_max": max((x["inflight"].get("system2", 0) for x in sm), default=0),
                          "s1_inflight_max": max((x["inflight"].get("system1", 0) for x in sm), default=0), "samples": len(sm)}
    out["slowest"] = [{"n": r["n"], "kind": r["kind"], "mode": r["mode"], "s": s(r["wall_ms"]), "prompt": r["prompt"][:70]} for r in sorted(R, key=lambda r: -r["wall_ms"])[:5]]
    out["failures"] = [{"n": r["n"], "kind": r["kind"], "error": r["error"]} for r in bad]
    doc["report"] = out
    json.dump(doc, open(args.out, "w"))
    print(json.dumps(out, indent=1, default=lambda x: round(x, 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--users", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
