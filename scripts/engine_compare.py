"""Smiley Core (ParagonCorp) against Peregrine (RantAI) on the same models, prompts and machine (Phase 39).

Two levels, both interleaved so that a load change on this shared host hits every arm alike:

  engine  the gateway's built-in comparison (POST /v1/bench/engines): the same prompts straight to System 1 and System 2
          of each engine, greedy, engines taking turns inside each round and swapping order between rounds. Time to first
          token, reading (prefill) and writing (decode) speed, wall time, whether the answer holds the expected value.
  e2e     the whole service: the gateway is switched (POST /v1/system/backend) between the selections smiley, peregrine,
          peregrine/smiley (System 1 on Peregrine, System 2 on Smiley Core) and smiley/peregrine, per prompt and in rotating
          order, and the same checkable questions are asked in Auto mode. Time to a checked answer is the criterion.
  conc    several people at once: N simultaneous quick-model requests on each engine (Peregrine batches them in one
          weight pass; Smiley Core serves one stream per model).

Results are appended to benchmarks/engines_phase39/<level>.json after every request, so a run that stops (power cut,
sleep) resumes where it was with --resume. `report` prints the tables.

    python3 -I scripts/engine_compare.py engine [--rounds 3]
    python3 -I scripts/engine_compare.py e2e [--rounds 2] [--resume]
    python3 -I scripts/engine_compare.py conc [--users 4]
    python3 -I scripts/engine_compare.py all
    python3 -I scripts/engine_compare.py report

Needs the gateway (:8080), Smiley Core (scripts/serve_smiley.sh start) and Peregrine (scripts/serve_peregrine.sh start).
The gateway is left on the selection it had when the run started.
"""
import argparse, json, os, statistics, subprocess, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/engines_phase39")
SELECTIONS = ["smiley", "peregrine", "peregrine/smiley", "smiley/peregrine"]
PROMPTS = [
    ("math", "A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip?", {"reference_answer": "145"}),
    ("math", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", {"reference_answer": "72"}),
    ("math", "Tom is twice as old as Ann. In 6 years the sum of their ages will be 54. How old is Tom now?", {"reference_answer": "28"}),
    ("math", "A rectangle has perimeter 50 cm and length 15 cm. What is its area in square centimetres?", {"reference_answer": "150"}),
    ("code", "Write a Python function is_palindrome(s) that ignores case and every character that is not a letter or digit.",
     {"tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')", "assert is_palindrome('')"]}),
    ("code", "Write a Python function chunks(xs, n) that splits a list into chunks of size n.",
     {"tests": ["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunks([], 3) == []", "assert chunks([1, 2], 5) == [[1, 2]]"]}),
    ("structured", "Create a JSON object describing a person with name (string) and age (integer, 0 to 150).",
     {"json_schema": {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150}}, "additionalProperties": False}}),
    ("factual", "What is the capital of Indonesia, and on which island is it located? One sentence.", None),
]
ENGINE_PORTS = {"smiley": 8201, "peregrine": 8211}


def post(base, path, body, timeout=2400):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def load1():
    try:
        return float(subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip("{} \n").split()[0])
    except Exception:
        return os.getloadavg()[0]


def path(level):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"{level}.json")


def save(level, doc):
    tmp = path(level) + ".tmp"
    json.dump(doc, open(tmp, "w"), indent=1)
    os.replace(tmp, path(level))


def engine_level(a):
    r = post(a.base, "/v1/bench/engines", {"engines": ["smiley", "peregrine"], "rounds": a.rounds})
    rid = r["run_id"]
    print("engine comparison", rid, flush=True)
    while True:
        time.sleep(10)
        v = get(a.base, f"/v1/bench/runs/{rid}")
        print(f"  {len(v.get('samples', []))} answers, load {load1():.1f}", flush=True)
        if v.get("status") != "running":
            break
    v["host"] = {"cpus": os.cpu_count(), "note": "Intel i3-8100, 4 cores, no GPU (reference machine)"}
    save("engine", v)
    report_engine(v)


def switch(base, sel):
    t = time.time()
    r = post(base, "/v1/system/backend", {"name": sel}, 900)
    return round(time.time() - t, 1), r.get("active_backend")


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
    return {"ok": True, "client_s": round(time.time() - t, 1), "wall_s": u["wall_ms"] / 1000, "ttft_s": (u.get("ttft_ms") or 0) / 1000,
            "mode": r["reasoning_mode"], "assurance": r["assurance"], "verified": r["assurance"] == "verified", "model_calls": u["model_calls"],
            "completion_tokens": u["completion_tokens"], "reasoning_tokens": u["reasoning_tokens"], "answer": r["message"]["content"][:160]}


def e2e(a):
    doc = json.load(open(path("e2e"))) if a.resume and os.path.exists(path("e2e")) else {"runs": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    done = {(x["round"], x["i"], x["selection"]) for x in doc["runs"] if x.get("ok")}
    before = get(a.base, "/v1/system")["active_backend"]
    sels = a.selections.split(",")
    try:
        for rnd in range(a.rounds):
            for i, (kind, prompt, ver) in enumerate(PROMPTS):
                k = rnd * len(PROMPTS) + i
                order = sels[k % len(sels):] + sels[:k % len(sels)]  # rotate: every selection goes first equally often
                for sel in order:
                    if (rnd, i, sel) in done:
                        continue
                    sw, active = switch(a.base, sel)
                    ld = load1()
                    r = chat(a.base, prompt, ver)
                    r.update({"round": rnd, "i": i, "kind": kind, "selection": sel, "active": active, "switch_s": sw, "load1": ld})
                    doc["runs"] = [x for x in doc["runs"] if not (x["round"] == rnd and x["i"] == i and x["selection"] == sel)] + [r]
                    save("e2e", doc)
                    print(f"r{rnd} {kind:10s} {sel:17s} {r.get('mode', '-'):4s} {str(r.get('assurance', r.get('error', '')))[:10]:10s} wall={r.get('wall_s')} ttft={r.get('ttft_s')} load={ld:.1f}", flush=True)
    finally:
        switch(a.base, before)
    report_e2e(doc)


def conc(a):
    """N simultaneous quick-model requests per engine, straight to the engine's System 1 server."""
    doc = {"users": a.users, "rows": []}
    body = {"model": "system1", "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
            "max_tokens": 96, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    for rnd in range(a.rounds):
        for eng in (["smiley", "peregrine"] if rnd % 2 == 0 else ["peregrine", "smiley"]):
            port = ENGINE_PORTS[eng]
            res = [None] * a.users
            def one(j):
                t = time.time()
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
                    v = json.load(urllib.request.urlopen(req, timeout=1800))
                    res[j] = {"s": time.time() - t, "tokens": v["usage"]["completion_tokens"]}
                except Exception as e:
                    res[j] = {"s": time.time() - t, "error": str(e)[:120]}
            ld = load1()
            t0 = time.time()
            th = [threading.Thread(target=one, args=(j,)) for j in range(a.users)]
            [x.start() for x in th]
            [x.join() for x in th]
            wall = time.time() - t0
            toks = sum(r.get("tokens", 0) for r in res)
            row = {"round": rnd, "engine": eng, "users": a.users, "wall_s": round(wall, 2), "tokens": toks, "agg_tok_s": round(toks / wall, 2),
                   "p50_s": round(statistics.median(r["s"] for r in res), 2), "max_s": round(max(r["s"] for r in res), 2),
                   "errors": sum("error" in r for r in res), "load1": ld}
            doc["rows"].append(row)
            print(row, flush=True)
            save("conc", doc)
    report_conc(doc)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def report_engine(v):
    print("\nEngine level (straight to each engine's servers, greedy)")
    print(f"{'role':8s} {'engine':10s} {'ok':>5s} {'right':>5s} {'ttft ms':>8s} {'read t/s':>9s} {'write t/s':>9s} {'wall ms':>8s}")
    for x in v["results"]["per_engine"]:
        print(f"{x['role']:8s} {x['engine']:10s} {x['ok']:>2d}/{x['n']:<2d} {x['correct']:>5d} {fmt(x['ttft_ms_p50'],0):>8s} {fmt(x['prefill_tok_s_p50'],0):>9s} {fmt(x['decode_tok_s_p50']):>9s} {fmt(x['wall_ms_p50'],0):>8s}")
    for p in v["results"]["paired"]:
        print(f"{p['role']}: {p['engine']} faster than {p['baseline']} in {p['faster']}/{p['pairs']} pairs, median wall ratio {fmt(p['wall_ratio_median'],2)}, sign test p={fmt(p['sign_test_p'],3)}")
    hl = v["results"]["host_load"]
    print(f"host load {fmt(hl['min'])}..{fmt(hl['max'])}")


def report_e2e(doc):
    print("\nEnd to end through the gateway (Auto mode)")
    rs = [r for r in doc["runs"] if r.get("ok")]
    print(f"{'selection':17s} {'n':>3s} {'checked':>8s} {'thorough':>8s} {'wall p50':>9s} {'wall sum':>9s} {'ttft p50':>9s}")
    for sel in SELECTIONS:
        x = [r for r in rs if r["selection"] == sel]
        if not x:
            continue
        chk = [r for r in x if r["kind"] in ("math", "code", "structured")]
        print(f"{sel:17s} {len(x):3d} {sum(r['verified'] for r in chk):3d}/{len(chk):<4d} {sum(r['mode'] == 'slow' for r in x):8d} {fmt(med([r['wall_s'] for r in x])):>9s} {sum(r['wall_s'] for r in x):9.0f} {fmt(med([r['ttft_s'] for r in x]),2):>9s}")
    base = {(r["round"], r["i"]): r for r in rs if r["selection"] == "smiley"}
    for sel in SELECTIONS[1:]:
        pairs = [(base[(r["round"], r["i"])], r) for r in rs if r["selection"] == sel and (r["round"], r["i"]) in base]
        if pairs:
            faster = sum(b["wall_s"] < s["wall_s"] for s, b in pairs)
            print(f"{sel} vs smiley: faster in {faster}/{len(pairs)} pairs, median ratio {fmt(med([b['wall_s'] / s['wall_s'] for s, b in pairs if s['wall_s']]),2)}")
    lds = [r["load1"] for r in doc["runs"] if "load1" in r]
    if lds:
        print(f"host load {min(lds):.1f}..{max(lds):.1f}")


def report_conc(doc):
    print(f"\n{doc['users']} simultaneous quick-model requests")
    for eng in ("smiley", "peregrine"):
        x = [r for r in doc["rows"] if r["engine"] == eng]
        if x:
            print(f"{eng:10s} aggregate {fmt(med([r['agg_tok_s'] for r in x]))} tok/s, wall {fmt(med([r['wall_s'] for r in x]))} s, slowest user {fmt(med([r['max_s'] for r in x]))} s, errors {sum(r['errors'] for r in x)}")


def report(_a):
    for level, f in (("engine", report_engine), ("e2e", report_e2e), ("conc", report_conc)):
        if os.path.exists(path(level)):
            f(json.load(open(path(level))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("level", choices=["engine", "e2e", "conc", "all", "report"])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--selections", default=",".join(SELECTIONS))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.level == "report":
        report(a)
    else:
        asked = a.rounds
        for lv in (["engine", "conc", "e2e"] if a.level == "all" else [a.level]):
            a.rounds = asked or {"engine": 3, "e2e": 2, "conc": 3}[lv]
            {"engine": engine_level, "e2e": e2e, "conc": conc}[lv](a)
