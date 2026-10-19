#!/usr/bin/env python3
"""Profile-C evaluation (P5-T04, MR-005): System 1 candidate Qwen3.5-2B vs the current Qwen3-1.7B on the same tasks,
with the production prompt and settings, alternating order per task, scored by the gateway's exact verifiers.
Records benchmarks/profile_c_decision.json (read by gate G5).
Usage: python3 -I scripts/profile_c_compare.py [--n-math 40 --n-code 20]"""
import json, os, random, subprocess, sys, time, urllib.error, urllib.request
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
N_MATH, N_CODE, N_SCHEMA = arg("--n-math", 16), arg("--n-code", 8), arg("--n-schema", 6)
SYSTEM = "You are Nafs-Aql, a careful assistant that pairs a fast first answer with deliberate reasoning when a question needs it. Be accurate and concise. If you are unsure, say so."
INSTR = {"math": "Solve step by step, briefly. End with a final line of the form 'Answer: <number>'.",
         "code": "Return one complete, self-contained Python solution in a single ```python code block, then at most two sentences of explanation.",
         "structured": "Reply with a single valid JSON value only, in a ```json code block, with no extra commentary."}
def post(url, body, timeout=600, retries=30):
    for i in range(retries):  # survive a gateway restart (the verifier is behind it)
        try:
            r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except urllib.error.URLError:
            if i == retries - 1: raise
            time.sleep(5)
def load(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "data/bench", name + ".jsonl"))]
rng = random.Random(55)
tasks = rng.sample(load("gsm8k"), N_MATH) + rng.sample(load("mbpp_plus"), N_CODE) + rng.sample(load("schema_tasks"), N_SCHEMA)
cand_port = 8011
srv = subprocess.Popen([os.path.join(ROOT, "third_party/llama.cpp/build/bin/llama-server"), "-m", os.path.join(ROOT, "models/Qwen3.5-2B-Q4_K_M.gguf"), "--alias", "candidate",
                        "-t", "4", "-c", "8192", "-np", "2", "--host", "127.0.0.1", "--port", str(cand_port), "--jinja", "--reasoning-format", "deepseek", "--cache-reuse", "256"],
                       stdout=subprocess.DEVNULL, stderr=open(os.path.join(ROOT, "logs/profile_c_server.log"), "w"))
try:
    for _ in range(180):
        try: urllib.request.urlopen(f"http://127.0.0.1:{cand_port}/health", timeout=2); break
        except Exception: time.sleep(1)
    models = {"Qwen3-1.7B (Profile A)": ("http://127.0.0.1:8001/v1/chat/completions", "system1"), "Qwen3.5-2B (Profile C)": (f"http://127.0.0.1:{cand_port}/v1/chat/completions", "candidate")}
    rows = []
    for i, t in enumerate(tasks):
        order = list(models.items()) if i % 2 == 0 else list(reversed(list(models.items())))
        row = {"task": t["id"], "domain": t["domain"]}
        for name, (url, alias) in order:
            body = {"model": alias, "messages": [{"role": "system", "content": SYSTEM + "\n\n" + INSTR[t["domain"]]}, {"role": "user", "content": t["prompt"]}],
                    "max_tokens": 512, "temperature": 0.2, "top_p": 0.8, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.time()
            try:
                r = post(url, body); ans = r["choices"][0]["message"].get("content") or ""; toks = r["usage"]["completion_tokens"]
            except Exception as e:
                ans, toks = "", 0; print("error", name, e, file=sys.stderr)
            wall = time.time() - t0
            v = post("http://127.0.0.1:8080/v1/verify", {"answer": ans, "verification": t["verification"]})
            row[name] = {"correct": bool(v.get("passed")), "wall_s": round(wall, 2), "tokens": toks}
        rows.append(row)
        print(i + 1, t["id"], {k: (v["correct"], v["wall_s"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)
finally:
    srv.terminate(); srv.wait()
names = list(models)
def acc(n, dom=None):
    rs = [r for r in rows if dom is None or r["domain"] == dom]; return sum(r[n]["correct"] for r in rs) / max(1, len(rs))
a, c = names
diffs = [int(r[c]["correct"]) - int(r[a]["correct"]) for r in rows]
def resample(seed):
    g = random.Random(seed); return sum(g.choice(diffs) for _ in diffs) / len(diffs)
boots = sorted(resample(i) for i in range(2000))
lo, hi = boots[50], boots[1949]
lat = {n: sum(r[n]["wall_s"] for r in rows) / len(rows) for n in names}
tps = {n: sum(r[n]["tokens"] for r in rows) / max(1e-9, sum(r[n]["wall_s"] for r in rows)) for n in names}
delta = sum(diffs) / len(diffs)
adopt = lo > -0.03 and delta >= 0 and lat[c] <= 1.3 * lat[a]
out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hypothesis_H7": "pass (79-80% decode speed, benchmarks/reference-machine/h7_result.json)",
       "n_tasks": len(rows), "accuracy": {n: {"all": acc(n), **{d: acc(n, d) for d in ("math", "code", "structured")}} for n in names},
       "paired_delta_accuracy": delta, "ci95": [lo, hi], "mean_wall_s": lat, "tokens_per_s": tps,
       "rule": "Adopt Profile C for System 1 only if the paired accuracy difference has a lower 95% bound above -3 points, a non-negative point estimate, and mean latency at most 1.3x Profile A (MR-005 requires H7 first, which passes).",
       "decision": ("Adopt Qwen3.5-2B as System 1 (Profile C)" if adopt else "Keep Profile A (Qwen3-1.7B) as System 1 default") + f": accuracy {acc(c):.0%} vs {acc(a):.0%} (delta {delta:+.0%}, 95% CI [{lo:+.0%}, {hi:+.0%}]), mean latency {lat[c]:.1f} s vs {lat[a]:.1f} s on {len(rows)} tasks (shared machine). System 2 (Qwen3.5-4B) not evaluated at this scale; GPU scale-out (P5-T03) deferred by the user.",
       "adopt": adopt, "rows": rows}
json.dump(out, open(os.path.join(ROOT, "benchmarks/profile_c_decision.json"), "w"), indent=2)
print(out["decision"])
