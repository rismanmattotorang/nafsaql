"""Smiley Core against llama.cpp on the same models, same prompts, same machine (Phase 15).

Both engines are driven through their own OpenAI-compatible chat endpoint with identical requests (greedy, thinking off for
System 1, thinking on with a 128-token budget for System 2), interleaved per prompt (A, B, B, A, ...), several rounds. It
records the server's own timings (prompt tokens and ms, predicted tokens and ms, tokens/s), the client-side time to the first
token, whether the answer is the same, and, from Smiley Core, the bytes moved per token and the achieved bandwidth.
Writes benchmarks/smiley_ab.json; `--report` prints the table.

    python3 -I scripts/smiley_ab.py [--rounds 3] [--only system1|system2]
    python3 -I scripts/smiley_ab.py --report

Ports: llama.cpp 8001/8002 (scripts/serve_llama.sh), Smiley Core 8201/8202 (scripts/serve_smiley.sh).
"""
import argparse, json, os, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/smiley_ab.json")
ENGINES = {"llama": {"system1": 8001, "system2": 8002}, "smiley": {"system1": 8201, "system2": 8202}}
PROMPTS = [
    ("math", "A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip? End with 'Answer: <number>'."),
    ("code", "Write a Python function run_length_encode(s) that compresses consecutive repeats, e.g. 'aaabcc' -> 'a3b1c2'. Only the code."),
    ("open", "Give three practical tips for writing clear technical documentation, two sentences each."),
    ("fact", "What is the capital of Indonesia, and on which island is it located? One sentence."),
]
SYSTEM = "You are a careful assistant. Answer directly."


def call(port, model, prompt, thinking, max_tokens):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 1, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if thinking:
        body["thinking_budget_tokens"] = 128
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    content, reasoning, timings, usage = "", "", {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                v = json.loads(data)
            except ValueError:
                continue
            if "timings" in v:
                timings = v["timings"]
            if v.get("usage"):
                usage = v["usage"]
            for c in v.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if first is None:
                        first = time.time() - t0
                    content += d.get("content") or ""
                    reasoning += d.get("reasoning_content") or ""
    return {"wall_s": round(time.time() - t0, 2), "ttft_s": first and round(first, 3), "content": content[:400], "reasoning_tokens_est": len(reasoning) // 4,
            "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"), "prompt_ms": timings.get("prompt_ms"), "predicted_n": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"), "tok_s": timings.get("predicted_per_second"), "smiley": timings.get("smiley"), "usage": usage}


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}
    doc["label"] = a.label
    models = [a.only] if a.only else ["system1", "system2"]
    for rnd in range(a.rounds):
        for model in models:
            thinking = model == "system2"
            max_tokens = 320 if thinking else 160
            for i, (kind, prompt) in enumerate(PROMPTS):
                order = ["llama", "smiley"] if (i + rnd) % 2 == 0 else ["smiley", "llama"]
                for eng in order:
                    if any(r["round"] == rnd and r["model"] == model and r["engine"] == eng and r["kind"] == kind for r in doc["runs"]):
                        continue
                    load = os.getloadavg()[0]
                    try:
                        r = call(ENGINES[eng][model], model, prompt, thinking, max_tokens)
                    except Exception as e:
                        r = {"error": str(e)[:200]}
                    r.update({"round": rnd, "model": model, "engine": eng, "kind": kind, "load_before": round(load, 2), "at": time.strftime("%H:%M:%S")})
                    doc["runs"].append(r)
                    json.dump(doc, open(OUT, "w"), indent=1)
                    print(f"r{rnd} {model} {kind:5s} {eng:6s} tok/s={r.get('tok_s') and round(r['tok_s'], 2)} ttft={r.get('ttft_s')} prompt={r.get('prompt_n')}+{r.get('cache_n')} pred={r.get('predicted_n')} load={load:.1f} {r.get('error', '')}", flush=True)
    print("done")


def report(a):
    doc = json.load(open(OUT))
    R = [r for r in doc["runs"] if "error" not in r]
    med = lambda xs: statistics.median(xs) if xs else None
    print(f"{'model':8s} {'engine':7s} {'n':>3s} {'decode tok/s':>13s} {'prefill tok/s':>14s} {'ttft s':>7s} {'wall s':>7s} {'bytes/token':>12s} {'same answer':>12s}")
    for model in ("system1", "system2"):
        for eng in ("llama", "smiley"):
            rs = [r for r in R if r["model"] == model and r["engine"] == eng and r.get("tok_s")]
            if not rs:
                continue
            pf = [r["prompt_n"] / r["prompt_ms"] * 1e3 for r in rs if r.get("prompt_ms") and r.get("prompt_n")]
            bpt = [r["smiley"]["bytes_per_token"] for r in rs if r.get("smiley")]
            same = []
            for r in rs:
                other = next((o for o in R if o["model"] == model and o["engine"] != eng and o["kind"] == r["kind"] and o["round"] == r["round"]), None)
                if other:
                    same.append(r["content"].strip()[:60] == other["content"].strip()[:60])
            print(f"{model:8s} {eng:7s} {len(rs):3d} {med([r['tok_s'] for r in rs]):13.2f} {med(pf) or 0:14.1f} {med([r['ttft_s'] for r in rs if r['ttft_s']]) or 0:7.2f} {med([r['wall_s'] for r in rs]):7.1f} {('%.0f' % (med(bpt) / 1e6)) + ' MB' if bpt else '–':>12s} {('%d/%d' % (sum(same), len(same))) if same else '–':>12s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
