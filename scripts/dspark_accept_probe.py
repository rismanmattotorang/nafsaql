"""Acceptance of the DSpark drafter on Qwen3-4B-Thinking-2507, one drafter server (:8041) against production System 2 (:8002) as the plain baseline.
Greedy, same prompts, alternating order, pausing whenever the gateway has work. Acceptance does not depend on host speed; tok/s does."""
import json, os, subprocess, sys, time, urllib.request
A = "/Users/rismanadnan/Downloads/nafs-aql"; BIN = f"{A}/third_party/llama.cpp/build/bin/llama-server"
PROMPTS = [("math", "A shop sells pens at 3 for $2. How much do 18 pens cost? Solve step by step and end with 'Answer: <number>'."),
 ("code", "Write a Python function is_palindrome(s) that ignores case and non-letters, then explain it in two sentences."),
 ("factual", "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?"),
 ("open", "Give three practical tips for staying focused while studying, with one sentence of reasoning each."),
 ("math", "If a train travels 180 km in 2.5 hours, what is its average speed in km/h? End with 'Answer: <number>'."),
 ("open", "Explain in a short paragraph why the sky looks blue.")]
get = lambda u: json.load(urllib.request.urlopen(u, timeout=5))
def busy():
    try:
        if get("http://127.0.0.1:8080/v1/system").get("inflight"): return True
        return any(r.get("status") == "running" for r in get("http://127.0.0.1:8080/v1/eval/runs"))
    except Exception: return True
def ask(port, prompt, n=160):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0, "seed": 1, "chat_template_kwargs": {"enable_thinking": True}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    v = json.load(urllib.request.urlopen(r, timeout=900)); t = v.get("timings", {}); m = v["choices"][0]["message"]
    return {"tok": t.get("predicted_n"), "tok_s": t.get("predicted_per_second"), "draft_n": t.get("draft_n"), "acc": t.get("draft_n_accepted"), "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")}
cmd = [BIN, "-m", f"{A}/models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf", "-md", f"{A}/models/Qwen3-4B-dspark-block7-Q8_0.gguf", "--spec-type", "draft-dspark", "--spec-draft-n-max", "7", "--spec-draft-p-min", "0",
       "-t", "4", "-c", "2048", "-np", "1", "--host", "127.0.0.1", "--port", "8041", "--jinja", "--reasoning-format", "deepseek", "--metrics"]
p = subprocess.Popen(cmd, stdout=open("/tmp/dspark_accept_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            if get("http://127.0.0.1:8041/health").get("status") == "ok": break
        except Exception: pass
        if p.poll() is not None: sys.exit("server exited: " + open("/tmp/dspark_accept_server.log").read()[-400:])
        time.sleep(1)
    rows = []; load0 = os.getloadavg()
    for dom, pr in PROMPTS:
        while busy(): time.sleep(20)
        a = ask(8041, pr); b = ask(8002, pr); rows.append((dom, a, b))
        print(f"{dom:8s} dspark {a['tok_s'] or 0:5.2f} tok/s accept {a['acc']}/{a['draft_n']}   plain(S2) {b['tok_s'] or 0:5.2f} tok/s   identical={a['text'] == b['text']}", flush=True)
    acc = sum(r[1]['acc'] or 0 for r in rows); dn = sum(r[1]['draft_n'] or 0 for r in rows)
    out = {"load_before": load0, "load_after": os.getloadavg(), "rows": [{"domain": d, "dspark": {k: v for k, v in a.items() if k != 'text'}, "plain": {k: v for k, v in b.items() if k != 'text'}, "identical": a['text'] == b['text']} for d, a, b in rows],
           "accepted": acc, "drafted": dn, "acceptance": acc / dn if dn else None,
           "mean_dspark_tok_s": sum(r[1]['tok_s'] or 0 for r in rows) / len(rows), "mean_plain_tok_s": sum(r[2]['tok_s'] or 0 for r in rows) / len(rows)}
    json.dump(out, open("benchmarks/dspark_probe/acceptance_probe.json", "w"), indent=1)
    print("acceptance %.3f (%d of %d drafted tokens); mean tok/s dspark %.2f vs plain %.2f; load %s -> %s" % (out["acceptance"] or 0, acc, dn, out["mean_dspark_tok_s"], out["mean_plain_tok_s"], load0, os.getloadavg()))
finally:
    p.terminate()
