#!/usr/bin/env python3
"""MVP acceptance tests MVP-A1..A9 (roadmap Section 10) against a running gateway with the llama backend.
Usage: python3 scripts/mvp_acceptance.py [--base http://127.0.0.1:8080] [--skip-degraded]
Writes benchmarks/mvp_acceptance.json and prints a table. A9 stops and restarts the System 2 llama-server."""
import json, os, subprocess, sys, time, urllib.request
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8080"
results = []
def post(path, body, timeout=1200):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def get(path): return json.load(urllib.request.urlopen(BASE + path, timeout=60))
def stream(body):
    r = urllib.request.Request(BASE + "/v1/chat/stream", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    evs, ev = [], None
    for raw in urllib.request.urlopen(r, timeout=1800):
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"): ev = line[6:].strip()
        elif line.startswith("data:"): evs.append((ev, json.loads(line[5:])["data"]))
    return evs
def record(tid, name, ok, detail, t0):
    results.append({"id": tid, "name": name, "pass": bool(ok), "detail": detail, "seconds": round(time.time() - t0, 1)})
    print(f"{'PASS' if ok else 'FAIL'}  {tid:7s} {name:28s} {time.time()-t0:7.1f}s  {detail}", flush=True)
def run(tid, name, fn):
    t0 = time.time()
    try: ok, detail = fn(); record(tid, name, ok, detail, t0)
    except Exception as e: record(tid, name, False, f"exception: {e}", t0)
cfg = get("/v1/system"); k = cfg["gate"]["k_samples"]
# Effective caps: measured (Phase-0 derived) where present, else the configured placeholders.
caps = {d: v["caps"] for d, v in cfg["gate"]["state"]["params"].items()}

def a1():
    r = post("/v1/chat", {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "ephemeral": True})
    ok = r["reasoning_mode"] == "fast" and r["assurance"] in ("unverified", "consensus") and (r["usage"]["ttft_ms"] or 1e9) <= 2000
    return ok, f"mode={r['reasoning_mode']} assurance={r['assurance']} ttft={r['usage']['ttft_ms']}ms rule={r['decision']['rule']}"
def a2():
    v = {"tests": ['assert is_even(4) is True', 'assert is_even(7) is False', 'assert is_even(0) is True']}
    r = post("/v1/chat", {"messages": [{"role": "user", "content": "Write a Python function is_even(n) that returns True for even integers."}], "verification": v, "ephemeral": True})
    # second half: a draft that cannot pass is escalated by rule forced_failed_verification
    bad = post("/v1/chat", {"messages": [{"role": "user", "content": "Write a Python function one() that returns 1."}], "verification": {"tests": ["assert one() == 2"]}, "thinking_budget": 64, "ephemeral": True})
    ok = r["assurance"] == "verified" and bad["decision"]["rule"] == "forced_failed_verification" and bad["decision"]["escalate"]
    return ok, f"verified={r['assurance']} via {r['reasoning_mode']}; impossible test -> rule={bad['decision']['rule']} escalate={bad['decision']['escalate']} final={bad['assurance']}"
def a3():
    task = json.loads(open(f"{ROOT}/data/bench/gsm8k.jsonl").readline())
    r = post("/v1/chat", {"messages": [{"role": "user", "content": task["prompt"]}], "verification": task["verification"], "force_mode": "slow", "thinking_budget": 1024, "ephemeral": True})
    u = r["usage"]; ok = r["reasoning_mode"] == "slow" and u["reasoning_tokens"] <= (u["thinking_budget"] or 1024) * 1.05 + 8 and u["wall_ms"] <= 240_000
    return ok, f"mode={r['reasoning_mode']} reasoning={u['reasoning_tokens']}/{u['thinking_budget']} wall={u['wall_ms']/1000:.0f}s assurance={r['assurance']}"
def a4():
    r = post("/v1/chat", {"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}], "verification": {"reference_answer": "-999999"}, "force_mode": "slow", "thinking_budget": 64, "ephemeral": True})
    ok = r["assurance"] == "abstained" and bool(r.get("uncertainty_notice")) and r["trace_state"] == "X"
    return ok, f"assurance={r['assurance']} state={r['trace_state']} notice={'yes' if r.get('uncertainty_notice') else 'no'} answer_nonempty={bool(r['message']['content'])}"
def a5():
    r = post("/v1/chat", {"messages": [{"role": "user", "content": "Return the number 5."}], "verification": {"reference_answer": "-123456"}, "thinking_budget": 64, "ephemeral": True})
    c = caps["math"]; bound = 1 + k + c["N"] * (c["R"] + 1)
    return r["usage"]["model_calls"] <= bound, f"model_calls={r['usage']['model_calls']} <= 1 + k + N(R+1) = {bound} (k={k}, N={c['N']}, R={c['R']})"
def a6():
    evs = stream({"messages": [{"role": "user", "content": "Name a colour."}], "force_mode": "fast", "ephemeral": True, "max_tokens": 16})
    names = [e for e, _ in evs]; ok = names[0] == "start" and "decision" in names and "token" in names and names[-1] == "done" and names.index("token") < names.index("done")
    return ok, "order: " + " ".join(dict.fromkeys(names))
def a7():
    room = post("/v1/rooms", {"name": f"acceptance-{int(time.time())}"})
    js = f"""
const base='{BASE.replace('http', 'ws')}/ws/rooms/{room['room_id']}';
const a=new WebSocket(base+'?user=alice'), b=new WebSocket(base+'?user=bob'); let seenUser=false, seenAssistant=false, presence=0;
b.onmessage=(e)=>{{const f=JSON.parse(e.data); if(f.type==='presence') presence=Math.max(presence,f.payload.members); if(f.type==='user_message'&&f.user==='alice') seenUser=true;
  if(f.type==='assistant_event'&&f.payload.event.event==='done'){{seenAssistant=true; console.log(JSON.stringify({{seenUser,seenAssistant,presence}})); process.exit(0);}}}};
let open=0; const go=()=>{{ if(++open===2) setTimeout(()=>a.send(JSON.stringify({{type:'user_message',payload:{{content:'Say hi to everyone.',force_mode:'fast'}}}})),300); }};
a.onopen=go; b.onopen=go; setTimeout(()=>{{console.log(JSON.stringify({{seenUser,seenAssistant,presence,timeout:true}})); process.exit(1);}}, 300000);"""
    try:
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=320).stdout.strip().splitlines()[-1]
    finally:
        try:
            req = urllib.request.Request(f"{BASE}/v1/rooms/{room['room_id']}", method="DELETE")
            urllib.request.urlopen(req, timeout=30).read()  # the acceptance room does not outlive the test
        except Exception:
            pass
    d = json.loads(out); return d["seenUser"] and d["seenAssistant"] and d["presence"] >= 2, f"bob saw alice's message={d['seenUser']}, saw streamed answer={d['seenAssistant']}, members={d['presence']}"
def a8():
    r = post("/v1/chat", {"messages": [{"role": "user", "content": "Reply with OK."}], "force_mode": "fast", "max_tokens": 8})
    d = get(f"/v1/decisions/{r['decision_id']}"); conv = get(f"/v1/conversations/{r['conversation_id']}")
    rc = subprocess.run([sys.executable, "-I", f"{ROOT}/scripts/contract_test.py"], capture_output=True, text=True)
    ok = d["decision_id"] == r["decision_id"] and len(conv["messages"]) >= 2 and rc.returncode == 0
    return ok, f"decision retrievable={d['decision_id'] == r['decision_id']}, conversation messages={len(conv['messages'])}, schemas: {rc.stdout.strip().splitlines()[-1]}"
def a9():
    subprocess.run([f"{ROOT}/scripts/serve_llama.sh", "stop-one", "system2"], capture_output=True)
    time.sleep(1)
    try:
        evs = stream({"messages": [{"role": "user", "content": "What is 2 + 2?"}], "verification": {"reference_answer": "5"}, "ephemeral": True})
        names = [e for e, _ in evs]; done = [d for e, d in evs if e == "done"]
        ok = "degraded" in names and done and done[0]["degraded"] and done[0]["reasoning_mode"] == "fast"
        detail = f"events={' '.join(dict.fromkeys(names))}; reply degraded={done[0]['degraded'] if done else None}"
    finally:
        subprocess.run([f"{ROOT}/scripts/serve_llama.sh", "start"], capture_output=True)
    # Recovery is part of the test: System 2 must be healthy again (NFR-010).
    back = False
    for _ in range(120):
        try: urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=2); back = True; break
        except Exception: time.sleep(1)
    return ok and back, detail + f"; System 2 back after restart={back}"

for tid, name, fn in [("MVP-A1", "Fast reply", a1), ("MVP-A2", "Verified code + forced escalation", a2), ("MVP-A3", "Slow path within budget", a3), ("MVP-A4", "Abstention with notice", a4),
                      ("MVP-A5", "Bounded loop (Lemma 1)", a5), ("MVP-A6", "SSE event order", a6), ("MVP-A7", "Rooms over WebSocket", a7), ("MVP-A8", "Records and schemas", a8)] + ([] if "--skip-degraded" in sys.argv else [("MVP-A9", "Degraded mode", a9)]):
    if "--only" in sys.argv and tid not in sys.argv[sys.argv.index("--only") + 1].split(","): continue
    run(tid, name, fn)
prev = {r["id"]: r for r in (json.load(open(f"{ROOT}/benchmarks/mvp_acceptance.json"))["results"] if os.path.exists(f"{ROOT}/benchmarks/mvp_acceptance.json") else [])}
prev.update({r["id"]: r for r in results}); results = [prev[k] for k in sorted(prev)]
json.dump({"base": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results}, open(f"{ROOT}/benchmarks/mvp_acceptance.json", "w"), indent=2)
print(f"\n{sum(r['pass'] for r in results)}/{len(results)} acceptance tests passed")
