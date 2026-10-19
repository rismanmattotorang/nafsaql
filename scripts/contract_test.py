#!/usr/bin/env python3
"""Contract tests (CT-003, CT-005, CT-008): validate real gateway output against api/schemas.
Checks stored ChatReply/DecisionRecord rows, TraceRecord lines, and (optionally) a live SSE stream.
Usage: python3 scripts/contract_test.py [--live http://127.0.0.1:8080]"""
import json, sqlite3, sys, os, urllib.request, urllib.error
from jsonschema import Draft202012Validator
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V = {n: Draft202012Validator(json.load(open(f"{ROOT}/api/schemas/{n}.schema.json"))) for n in ["ChatReply", "DecisionRecord", "StreamEvent", "TraceRecord", "SelfModel", "ChatRequest"]}
fails = 0
def check(name, obj, where):
    global fails
    errs = sorted(V[name].iter_errors(obj), key=lambda e: list(e.path))
    if errs:
        fails += 1; print(f"FAIL {name} {where}: {errs[0].message} at {list(errs[0].path)}")
    return not errs
db = sqlite3.connect(f"{ROOT}/data/runtime/nafs.sqlite")
replies = [json.loads(r[0]) for r in db.execute("SELECT json FROM replies ORDER BY created_at DESC LIMIT 50")]
decs = [json.loads(r[0]) for r in db.execute("SELECT json FROM decisions ORDER BY created_at DESC LIMIT 50")]
n1 = sum(check("ChatReply", r, r["reply_id"]) for r in replies); n2 = sum(check("DecisionRecord", d, d["decision_id"]) for d in decs)
traces = [json.loads(l) for l in open(f"{ROOT}/data/runtime/traces/trace.jsonl")][-50:]
n3 = sum(check("TraceRecord", t, t["request_id"]) for t in traces)
print(f"stored: ChatReply {n1}/{len(replies)}, DecisionRecord {n2}/{len(decs)}, TraceRecord {n3}/{len(traces)} valid")
if "--live" in sys.argv:
    base = sys.argv[sys.argv.index("--live") + 1]
    body = json.dumps({"messages": [{"role": "user", "content": "Reply with the word OK."}], "force_mode": "fast", "ephemeral": True, "max_tokens": 8}).encode()
    req = urllib.request.Request(base + "/v1/chat/stream", data=body, headers={"Content-Type": "application/json"})
    names, n, ev = [], 0, None
    for raw in urllib.request.urlopen(req, timeout=600):
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"): ev = line[6:].strip()
        elif line.startswith("data:"):
            obj = json.loads(line[5:]); names.append(ev); n += check("StreamEvent", obj, ev)
            if obj["event"] != ev: fails += 1; print(f"FAIL event name {ev} != payload {obj['event']}")
    order_ok = names[0] == "start" and names[-1] in ("done", "error") and names.index("decision") < names.index("done")
    if not order_ok: fails += 1
    print(f"live SSE: {n}/{len(names)} events valid; order {'ok' if order_ok else 'WRONG'}: {' '.join(dict.fromkeys(names))}")
    sm = json.load(urllib.request.urlopen(base + "/v1/self-model", timeout=30))
    print(f"live self-model: {'valid' if check('SelfModel', sm, 'self-model') else 'INVALID'} ({len(sm['domains'])} domains)")
    # Requests the schema rejects must also be rejected by the server with 400 before streaming (M14).
    for bad in [{"messages": [{"role": "user", "content": "x"}], "lambda": 5}, {"messages": [{"role": "user", "content": "x"}], "thinking_budget": 0}, {"messages": []}]:
        schema_rejects = not V["ChatRequest"].is_valid(bad)
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/stream", data=json.dumps(bad).encode(), headers={"Content-Type": "application/json"}), timeout=30); code = 200
        except urllib.error.HTTPError as e: code = e.code
        if not (schema_rejects and code == 400): fails += 1; print(f"FAIL bounds: {bad} schema_rejects={schema_rejects} http={code}")
    print("live bounds: schema and server agree on 3 out-of-range requests" if fails == 0 else "live bounds: mismatch")
print("CONTRACT TESTS", "PASSED" if fails == 0 else f"FAILED ({fails})"); sys.exit(1 if fails else 0)
