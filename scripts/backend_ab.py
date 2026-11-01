"""The whole service on one backend against the other: llama.cpp and Smiley Core through the gateway, interleaved (Phase 15).

Raw engine speed (scripts/smiley_ab.py) is one number; what the owner gets is the time to a checked answer through the
gateway, where llama.cpp also brings what the engine affords (four slots, the n-choice rescue, the harness grammar, the
raw-continuation exit) and Smiley Core does not. This script switches the gateway between the two backends per prompt
(`POST /v1/system/backend`, warm-first) in alternating order and sends the same checkable prompts in Auto mode, recording
whether the answer was checked, the mode, the wall time, the time to the first token and the tokens. A code-agent task is
run on each backend at the end. The gateway is left on llama.cpp.

    python3 -I scripts/backend_ab.py [--rounds 1] [--base http://127.0.0.1:8080]
    python3 -I scripts/backend_ab.py --report
"""
import argparse, json, os, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/backend_ab.json")
PROMPTS = [
    ("math", "A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip?", {"reference_answer": "145"}),
    ("math", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", {"reference_answer": "72"}),
    ("math", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in dollars?", {"reference_answer": "0.05"}),
    ("math", "Tom is twice as old as Ann. In 6 years the sum of their ages will be 54. How old is Tom now?", {"reference_answer": "28"}),
    ("code", "Write a Python function is_palindrome(s) that ignores case and every character that is not a letter or digit.",
     {"tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')", "assert is_palindrome('')"]}),
    ("code", "Write a Python function chunks(xs, n) that splits a list into chunks of size n.",
     {"tests": ["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunks([], 3) == []", "assert chunks([1, 2], 5) == [[1, 2]]"]}),
    ("code", "Write a Python function second_largest(xs) that returns the second largest distinct value of a list, or None when there is none.",
     {"tests": ["assert second_largest([3, 1, 2]) == 2", "assert second_largest([5, 5, 4]) == 4", "assert second_largest([7]) is None"]}),
    ("structured", "Create a JSON object for a library book with fields title (string), author (string), year (integer between 1450 and 2026) and available (boolean).",
     {"json_schema": {"type": "object", "required": ["title", "author", "year", "available"], "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer", "minimum": 1450, "maximum": 2026}, "available": {"type": "boolean"}}, "additionalProperties": False}}),
    ("structured", "Create a JSON object describing a person with name (string) and age (integer, 0 to 150).",
     {"json_schema": {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150}}, "additionalProperties": False}}),
    ("factual", "What is the capital of Indonesia, and on which island is it located? One sentence.", None),
    ("open", "Give three practical tips for writing clear technical documentation.", None),
    ("open", "Explain the difference between a process and a thread in two short paragraphs.", None),
]
AGENT = {"task": "second_largest(xs) returns the second largest distinct value of the list, or None when there is none. Fix solution.py.",
         "files": {"solution.py": "def second_largest(xs):\n    xs = sorted(xs)\n    return xs[-2]\n"},
         "verification": {"tests": ["assert second_largest([3, 1, 2]) == 2", "assert second_largest([5, 5, 4]) == 4", "assert second_largest([7]) is None", "assert second_largest([2, 2]) is None"]}}


def post(base, path, body, timeout=2400):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def switch(base, name):
    t = time.time()
    r = post(base, "/v1/system/backend", {"name": name}, 600)
    return {"warm_ms": r.get("warm_ms"), "warmed": r.get("warmed_prefixes"), "switch_s": round(time.time() - t, 1)}


def chat(base, prompt, verification):
    body = {"messages": [{"role": "user", "content": prompt}], "ephemeral": True, "audit": False, "retrieve": False}
    if verification:
        body["verification"] = verification
    t = time.time()
    try:
        r = post(base, "/v1/chat", body)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "client_s": round(time.time() - t, 1)}
    u = r["usage"]
    return {"ok": True, "client_s": round(time.time() - t, 1), "wall_s": u["wall_ms"] / 1000, "ttft_s": (u.get("ttft_ms") or 0) / 1000, "mode": r["reasoning_mode"],
            "assurance": r["assurance"], "verified": r["assurance"] == "verified", "model_calls": u["model_calls"], "completion_tokens": u["completion_tokens"],
            "reasoning_tokens": u["reasoning_tokens"], "stages": [{k: s.get(k) for k in ("stage", "ms", "prefill_ms", "prompt_n", "cache_n", "tokens")} for s in u.get("stages", [])],
            "answer": r["message"]["content"][:160]}


def run(a):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": [], "switches": []}
    try:
        for rnd in range(a.rounds):
            for i, (kind, prompt, ver) in enumerate(PROMPTS):
                order = ["llama", "smiley"] if (i + rnd) % 2 == 0 else ["smiley", "llama"]
                for eng in order:
                    if any(r["round"] == rnd and r["engine"] == eng and r["prompt"] == prompt[:60] for r in doc["runs"]):
                        continue
                    sw = switch(a.base, eng)
                    doc["switches"].append({"to": eng, **sw})
                    load = os.getloadavg()[0]
                    r = chat(a.base, prompt, ver)
                    r.update({"round": rnd, "engine": eng, "kind": kind, "prompt": prompt[:60], "checkable": ver is not None, "load_before": round(load, 2), "at": time.strftime("%H:%M:%S")})
                    doc["runs"].append(r)
                    json.dump(doc, open(OUT, "w"), indent=1)
                    print(f"r{rnd} {kind:10s} {eng:6s} {r.get('mode', '-'):4s} {r.get('assurance', r.get('error', ''))[:10]:10s} wall={r.get('wall_s')} ttft={r.get('ttft_s')} calls={r.get('model_calls')} switch={sw['switch_s']}s load={load:.1f}", flush=True)
        for eng in ["llama", "smiley"]:
            if any(r.get("agent") and r["engine"] == eng for r in doc["runs"]):
                continue
            sw = switch(a.base, eng)
            t = time.time()
            try:
                r = post(a.base, "/v1/harness/code", AGENT)
                rec = {"ok": True, "agent": True, "engine": eng, "status": r["status"], "verified": r["passed"], "wall_s": r["wall_ms"] / 1000, "model_calls": r["model_calls"],
                       "steps": [(s["model"], s["action"], s.get("samples"), s.get("passed")) for s in r["steps"]], "constrained": r["constrained"], "cascade": r["cascade"]}
            except Exception as e:
                rec = {"ok": False, "agent": True, "engine": eng, "error": str(e)[:200], "wall_s": round(time.time() - t, 1)}
            rec.update({"round": 0, "kind": "agent", "prompt": "agent", "checkable": True, "at": time.strftime("%H:%M:%S")})
            doc["runs"].append(rec)
            json.dump(doc, open(OUT, "w"), indent=1)
            print(f"agent {eng:6s} {rec}", flush=True)
    finally:
        switch(a.base, "llama")
    print("done")


def report(a):
    doc = json.load(open(OUT))
    med = lambda xs: statistics.median(xs) if xs else None
    print(f"{'engine':7s} {'n':>3s} {'checked':>8s} {'thorough':>8s} {'wall med':>9s} {'wall sum':>9s} {'ttft med':>9s} {'tokens':>7s}")
    for eng in ("llama", "smiley"):
        rs = [r for r in doc["runs"] if r["engine"] == eng and r.get("ok") and not r.get("agent")]
        chk = [r for r in rs if r["checkable"]]
        print(f"{eng:7s} {len(rs):3d} {sum(r['verified'] for r in chk):3d}/{len(chk):<4d} {sum(r['mode'] == 'slow' for r in rs):8d} {med([r['wall_s'] for r in rs]) or 0:9.1f} {sum(r['wall_s'] for r in rs):9.0f} {med([r['ttft_s'] for r in rs]) or 0:9.2f} {sum(r['completion_tokens'] + r['reasoning_tokens'] for r in rs):7d}")
    pairs = {}
    for r in doc["runs"]:
        if r.get("ok") and not r.get("agent"):
            pairs.setdefault((r["round"], r["prompt"]), {})[r["engine"]] = r
    both = [p for p in pairs.values() if len(p) == 2]
    faster = sum(p["smiley"]["wall_s"] < p["llama"]["wall_s"] for p in both)
    same_mode = sum(p["smiley"]["mode"] == p["llama"]["mode"] for p in both)
    print(f"pairs {len(both)}: Smiley faster in {faster}, same mode in {same_mode}")
    for r in doc["runs"]:
        if r.get("agent"):
            print("agent", r["engine"], {k: r.get(k) for k in ("status", "verified", "wall_s", "model_calls", "steps", "constrained", "error")})
    sw = doc.get("switches", [])
    if sw:
        print("switch (warm-first): median", med([s["switch_s"] for s in sw]), "s, warm_ms median", med([s["warm_ms"] for s in sw if s.get("warm_ms") is not None]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)
