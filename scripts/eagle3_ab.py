#!/usr/bin/env python3
"""EAGLE-3 speculative decoding A/B on the reference CPU (P7-T04, FR-081, NFR-031).

Starts one plain llama-server and one or more EAGLE-3 servers for the same target model on spare ports, then
sends the same prompts to each in alternating order for several rounds, so host noise hits every variant alike.
Reports decode tokens/s from the server's own timings, the acceptance statistics, and whether greedy outputs are
identical to the plain server's (FR-081 requires identical greedy output and a >= 1.2x decode speed-up).

Usage: python3 scripts/eagle3_ab.py --system 1|2 [--rounds 2] [--n-predict 160] [--slots 4]
Writes benchmarks/eagle3_ab_system{N}.json. Production servers on :8001/:8002 are left alone.
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "third_party/llama.cpp/build/bin/llama-server")
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
SYSTEM = int(arg("--system", "1"))
ROUNDS = int(arg("--rounds", "2"))
NPRED = int(arg("--n-predict", "160"))
SLOTS = int(arg("--slots", "4"))
MODEL = {1: "Qwen3-1.7B-Q4_K_M", 2: "Qwen3-4B-Thinking-2507-Q4_K_M"}[SYSTEM]
HEAD = {1: "Qwen3-1.7B-eagle3-Q8_0", 2: "Qwen3-4B-eagle3-Q8_0"}[SYSTEM]
THINK = SYSTEM == 2

PROMPTS = [
    ("math", "A shop sells pens at 3 for $2. How much do 18 pens cost? Solve step by step and end with 'Answer: <number>'."),
    ("code", "Write a Python function is_palindrome(s) that ignores case and non-letters, then explain it in two sentences."),
    ("factual", "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?"),
    ("open", "Give three practical tips for staying focused while studying, with one sentence of reasoning each."),
    ("math", "If a train travels 180 km in 2.5 hours, what is its average speed in km/h? End with 'Answer: <number>'."),
    ("open", "Explain in a short paragraph why the sky looks blue."),
]

VARIANTS = [("plain", []),
            ("eagle3_n4", ["-md", f"models/{HEAD}.gguf", "--spec-type", "draft-eagle3", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.5"]),
            ("eagle3_n8", ["-md", f"models/{HEAD}.gguf", "--spec-type", "draft-eagle3", "--spec-draft-n-max", "8", "--spec-draft-p-min", "0.5"])]


def start(port, extra):
    cmd = [BIN, "-m", f"models/{MODEL}.gguf", "-t", "4", "-c", str(4096 * SLOTS), "-np", str(SLOTS), "--host", "127.0.0.1",
           "--port", str(port), "--jinja", "--reasoning-format", "deepseek", "--cache-reuse", "256", "--metrics"] + extra
    log = open(os.path.join(ROOT, "logs", f"eagle3_ab_{port}.log"), "w")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok":
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise SystemExit(f"server on {port} exited; see logs/eagle3_ab_{port}.log")
        time.sleep(0.5)
    raise SystemExit(f"server on {port} did not become healthy")


def ask(port, prompt, temperature):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": NPRED, "temperature": temperature, "top_p": 0.95 if temperature else 1.0,
            "seed": 1, "chat_template_kwargs": {"enable_thinking": THINK}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        v = json.load(r)
    m = v["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
    tm = v.get("timings", {})
    return {"text": text, "wall_s": time.time() - t, "tok": tm.get("predicted_n"), "tok_s": tm.get("predicted_per_second"),
            "prefill_tok_s": tm.get("prompt_per_second"), "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted")}


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    procs, ports = [], {}
    try:
        for i, (name, extra) in enumerate(VARIANTS):
            ports[name] = 8011 + i
            procs.append(start(ports[name], extra))
            print(f"started {name} on :{ports[name]}", flush=True)
        results = {name: [] for name, _ in VARIANTS}
        for temp_label, temp in (("greedy", 0.0), ("production", 0.2 if SYSTEM == 1 else 0.6)):
            for rnd in range(ROUNDS):
                for dom, prompt in PROMPTS:
                    order = VARIANTS if (rnd % 2 == 0) else VARIANTS[::-1]
                    for name, _ in order:
                        r = ask(ports[name], prompt, temp)
                        r.update({"domain": dom, "round": rnd, "temperature": temp_label})
                        results[name].append(r)
                        print(f"{temp_label:10s} r{rnd} {dom:8s} {name:10s} {r['tok_s'] or 0:6.2f} tok/s  {r['tok']} tok  accept {r['draft_accepted']}/{r['draft_n']}", flush=True)
        summary = {}
        plain = results["plain"]
        for name in results:
            rows = results[name]
            for temp_label in ("greedy", "production"):
                rs = [r for r in rows if r["temperature"] == temp_label and r["tok_s"]]
                if not rs:
                    continue
                tok_s = sum(r["tok_s"] for r in rs) / len(rs)
                acc = [(r["draft_accepted"], r["draft_n"]) for r in rs if r.get("draft_n")]
                same = None
                if temp_label == "greedy" and name != "plain":
                    pairs = [(a["text"], b["text"]) for a in plain for b in rs if a["temperature"] == "greedy" and a["domain"] == b["domain"] and a["round"] == b["round"] and a is not b]
                    same = sum(1 for a, b in pairs if a == b) / max(1, len(pairs))
                summary[f"{name}/{temp_label}"] = {"decode_tok_s": tok_s, "n": len(rs),
                                                    "acceptance": (sum(a for a, _ in acc) / max(1, sum(n for _, n in acc))) if acc else None,
                                                    "greedy_identical_share": same}
        base = {t: summary[f"plain/{t}"]["decode_tok_s"] for t in ("greedy", "production") if f"plain/{t}" in summary}
        for k, v in summary.items():
            t = k.split("/")[1]
            v["speedup"] = v["decode_tok_s"] / base[t] if t in base else None
        out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "system": SYSTEM, "model": MODEL, "head": HEAD, "slots": SLOTS,
               "n_predict": NPRED, "rounds": ROUNDS, "load_average": os.getloadavg(), "summary": summary, "results": results,
               "rule": "FR-081: adopt only if decode speed-up >= 1.2x at production temperature and greedy outputs are identical"}
        path = os.path.join(ROOT, "benchmarks", f"eagle3_ab_system{SYSTEM}.json")
        json.dump(out, open(path, "w"), indent=1)
        print("\nload average", out["load_average"])
        for k, v in summary.items():
            print(f"{k:22s} {v['decode_tok_s']:6.2f} tok/s  speedup {v['speedup'] or 0:.2f}x  acceptance {v['acceptance'] if v['acceptance'] is None else round(v['acceptance'], 2)}  identical {v['greedy_identical_share']}")
        print("wrote", os.path.relpath(path, ROOT))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(15)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
