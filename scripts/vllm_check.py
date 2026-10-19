#!/usr/bin/env python3
"""vLLM backend check (P7-T10): drive the gateway against the vLLM backend set and compare with llama.cpp.

Switches the gateway to the `vllm` backend (refused if its server is down), then runs the request paths the
orchestrator depends on: a verified fast reply with log-probs for the gate, a forced deliberate reply with a thinking
budget (reasoning parsed), an SSE stream, a judge-scored open answer, the stage timings, and the same prompts on the
`llama` backend for a timing comparison. Switches back to the original backend at the end.

Usage: python3 scripts/vllm_check.py [--base http://127.0.0.1:8080]   -> benchmarks/vllm_check.json
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8080"
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} {str(detail)[:110]}", flush=True)


def http(method, path, body=None, timeout=1800):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def switch(name):
    return http("POST", "/v1/system/backend", {"name": name})


def chat(body):
    t = time.time()
    r = http("POST", "/v1/chat", dict(body, ephemeral=True))
    r["_elapsed_s"] = round(time.time() - t, 1)
    return r


def main():
    sysinfo = http("GET", "/v1/system")
    original = sysinfo["active_backend"]
    vb = next(b for b in sysinfo["backends"] if b["name"] == "vllm")
    check("vllm backend reachable (health)", vb["system1"]["ok"] and vb["system2"]["ok"], f"{vb['system1']['url']} served {vb['system1']['served_models']}")
    timings = {}
    try:
        switch("vllm")
        check("switched to vllm", http("GET", "/v1/system")["active_backend"] == "vllm")
        r = chat({"messages": [{"role": "user", "content": "What is 9*8? Solve briefly and end with Answer: <number>."}], "verification": {"reference_answer": "72"}})
        f = r["decision"]["features"]
        check("vllm: fast verified reply", r["assurance"] == "verified" and r["reasoning_mode"] == "fast", f"{r['assurance']} {r['_elapsed_s']}s mean_logprob {f.get('mean_logprob')}")
        check("vllm: log-probs reach the gate", f.get("mean_logprob") is not None, f.get("mean_logprob"))
        timings["vllm_fast_s"] = r["_elapsed_s"]
        r = chat({"messages": [{"role": "user", "content": "A box holds 6 red and 4 blue balls. What fraction is red? End with Answer: <fraction>."}],
                  "verification": {"reference_answer": "3/5"}, "force_mode": "slow", "thinking_budget": 256, "show_reasoning": True})
        u = r["usage"]
        check("vllm: deliberate reply with thinking budget", r["reasoning_mode"] == "slow" and u["reasoning_tokens"] > 0,
              f"{r['assurance']} reasoning {u['reasoning_tokens']}/{u['thinking_budget']} wall {u['wall_ms']} ms")
        check("vllm: reasoning parsed separately from the answer", bool(r.get("reasoning")) and "<think>" not in r["message"]["content"], r["message"]["content"][-60:])
        check("vllm: stage timings recorded", any(s["stage"] == "deliberate" for s in u.get("stages", [])), [s["stage"] for s in u.get("stages", [])])
        timings["vllm_slow_s"] = r["_elapsed_s"]
        req = urllib.request.Request(BASE + "/v1/chat/stream", data=json.dumps({"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "ephemeral": True, "force_mode": "fast"}).encode(), headers={"content-type": "application/json"})
        events = []
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("event:"):
                    events.append(line[6:].strip())
        order = [e for e in events if e in ("start", "decision", "token", "done")]
        check("vllm: SSE stream order", order and order[0] == "start" and order[-1] == "done" and "token" in order, ",".join(dict.fromkeys(events)))
        r = chat({"messages": [{"role": "user", "content": "Why do leaves change colour in autumn? Two sentences."}], "force_mode": "slow", "thinking_budget": 128})
        v = r.get("verifier") or {}
        check("vllm: judge scores an open answer", v.get("verifier") == "judge" and v.get("details") is not None, f"{r['assurance']} judge passed {v.get('passed')} {r['_elapsed_s']}s")
        timings["vllm_open_s"] = r["_elapsed_s"]
    finally:
        switch(original if original != "vllm" else "llama")
    if original == "llama":
        r = chat({"messages": [{"role": "user", "content": "What is 9*8? Solve briefly and end with Answer: <number>."}], "verification": {"reference_answer": "72"}})
        timings["llama_fast_s"] = r["_elapsed_s"]
        r = chat({"messages": [{"role": "user", "content": "A box holds 6 red and 4 blue balls. What fraction is red? End with Answer: <fraction>."}],
                  "verification": {"reference_answer": "3/5"}, "force_mode": "slow", "thinking_budget": 256})
        timings["llama_slow_s"] = r["_elapsed_s"]
    check("switched back", http("GET", "/v1/system")["active_backend"] == original, original)
    failed = [x for x in results if not x["pass"]]
    out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_average": os.getloadavg(), "timings_s": timings,
           "passed": len(results) - len(failed), "failed": len(failed), "results": results}
    json.dump(out, open(os.path.join(ROOT, "benchmarks", "vllm_check.json"), "w"), indent=1)
    print(json.dumps(timings), f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
