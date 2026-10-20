#!/usr/bin/env python3
"""End-to-end sweep (P7-T07): every documented read endpoint, the main request paths, and every UI page.

1. API: GET every OpenAPI path without path parameters (200 and JSON or text as declared).
2. Requests: a verified chat reply (stage timings present), an SSE stream (event order), /v1/verify, the
   latency breakdown. Requests are ephemeral, so nothing is added to the conversation history.
3. UI: load each page in a headless Chrome (throw-away profile) and fail on any JavaScript exception or
   console error.

Usage: python3 scripts/e2e_sweep.py [--base http://127.0.0.1:8080] [--no-ui]
Writes benchmarks/e2e_sweep.json and exits non-zero on any failure.
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
BASE = args[args.index("--base") + 1] if "--base" in args else "http://127.0.0.1:8080"
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {str(detail)[:90]}", flush=True)


def http(method, path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("content-type", ""), r.read()


# 1. API sweep
spec = json.loads(http("GET", "/openapi.json")[2])
skip = {"/v1/chat/stream", "/v1/rooms/{room_id}/ws"}
for path, ops in sorted(spec["paths"].items()):
    if "get" not in ops or "{" in path or path in skip:
        continue
    if any(p.get("required") for p in ops["get"].get("parameters", []) if p.get("in") == "query"):
        continue
    try:
        st, ct, data = http("GET", path, timeout=120)
        ok = st == 200 and (("json" not in ct) or json.loads(data) is not None)
        check(f"GET {path}", ok, f"{st} {ct.split(';')[0]} {len(data)} B")
    except Exception as e:
        check(f"GET {path}", False, e)

# 2. Request paths
try:
    t = time.time()
    st, _, data = http("POST", "/v1/chat", {"messages": [{"role": "user", "content": "What is 9*9?"}], "ephemeral": True,
                                          "verification": {"reference_answer": "81"}})
    r = json.loads(data)
    stages = [s["stage"] for s in r["usage"].get("stages", [])]
    check("chat: verified math reply", r["assurance"] == "verified", f"{r['reasoning_mode']} {r['assurance']} {time.time() - t:.1f}s")
    check("chat: stage timings recorded (NFR-030)", "draft" in stages, stages)
except Exception as e:
    check("chat: verified math reply", False, e)
try:
    # Unverified math: the gate draws its agreement samples after the draft (Phase 8), so the waterfall must count
    # them: the stages that add to the wall time may not exceed it, and "Gateway and other" may not be inflated by them.
    st, _, data = http("POST", "/v1/chat", {"messages": [{"role": "user", "content": "What is 12 * 12 + 6?"}], "ephemeral": True})
    r = json.loads(data)
    sts = r["usage"].get("stages", [])
    counted = sum(x["ms"] for x in sts if x["stage"] != "sample")
    after = [x for x in sts if x["stage"] == "sample_after"]
    check("chat: unverified math stages add up to the wall time", counted <= r["usage"]["wall_ms"] + 100 and (not after or r["usage"]["wall_ms"] - counted < 2000),
          f"counted {counted} ms of {r['usage']['wall_ms']} ms; {len(after)} sample_after stage(s)")
except Exception as e:
    check("chat: unverified math stages add up to the wall time", False, e)
def rooms_call(method, path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")
try:
    # Rooms lifecycle (the live WebSocket behaviour is covered by scripts/rooms_test.py).
    tag = f"sweep-{int(time.time())}"
    st, rm = rooms_call("POST", "/v1/rooms", {"name": f"  {tag}   room "})
    check("rooms: create normalises the name (201)", st == 201 and rm["name"] == f"{tag} room" and rm["questions"] == 0, f"{st}")
    try:
        st, e = rooms_call("POST", "/v1/rooms", {"name": f"{tag.upper()} ROOM"})
        check("rooms: a duplicate name (any case) is a 409 conflict", st == 409 and e["code"] == "E_CONFLICT", f"{st}")
        st, r2 = rooms_call("PATCH", f"/v1/rooms/{rm['room_id']}", {"name": f"{tag} renamed"})
        check("rooms: rename (200) is visible in the list", st == 200 and any(x["name"] == f"{tag} renamed" for x in rooms_call("GET", "/v1/rooms")[1]), f"{st}")
    finally:
        st, rep = rooms_call("DELETE", f"/v1/rooms/{rm['room_id']}")
    check("rooms: delete (200) reports what was removed", st == 200 and rep["deleted"] and "removed" in rep, f"{st}")
    check("rooms: a deleted room is a 404, and deleting it again is a 404", rooms_call("GET", f"/v1/rooms/{rm['room_id']}")[0] == 404 and rooms_call("DELETE", f"/v1/rooms/{rm['room_id']}")[0] == 404)
    st, e = rooms_call("GET", "/v1/rooms/not-a-uuid")
    check("rooms: a malformed id is a 400 with the JSON envelope", st == 400 and e["code"] == "E_BAD_REQUEST", f"{st}")
except Exception as e:
    check("rooms: lifecycle", False, e)
for name, path, body, raw in [("malformed JSON", "/v1/eval/runs", b"not json", True),
                              ("wrong field type", "/v1/eval/runs", json.dumps({"benchmark": "smoke_mixed", "max_items": -3, "systems": ["fast_only"]}).encode(), True),
                              ("oversized message", "/v1/chat", json.dumps({"messages": [{"role": "user", "content": "word " * 9000}], "ephemeral": True}).encode(), True)]:
    try:
        req = urllib.request.Request(BASE + path, data=body, method="POST", headers={"content-type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60)
            check(f"errors: {name} is refused with the JSON envelope", False, "accepted")
        except urllib.error.HTTPError as e:
            b = json.loads(e.read())
            check(f"errors: {name} is refused with the JSON envelope", e.code in (400, 422) and b.get("code") == "E_BAD_REQUEST", f"{e.code} {b.get('code')}: {str(b.get('message'))[:60]}")
    except Exception as e:
        check(f"errors: {name} is refused with the JSON envelope", False, e)
try:
    req = urllib.request.Request(BASE + "/v1/chat/stream", data=json.dumps({"messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                                 "ephemeral": True, "force_mode": "fast"}).encode(), headers={"content-type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("event:"):
                events.append(line[6:].strip())
    order = [e for e in events if e in ("start", "decision", "token", "done")]
    ok = order and order[0] == "start" and order[-1] == "done" and "decision" in order and "token" in order
    check("stream: SSE start, decision, token, done", ok, ",".join(dict.fromkeys(events)))
except Exception as e:
    check("stream: SSE start, decision, token, done", False, e)
try:
    st, _, data = http("POST", "/v1/verify", {"answer": "```python\ndef add(a, b):\n    return a + b\n```",
                                             "verification": {"tests": ["assert add(2, 3) == 5"]}})
    v = json.loads(data)
    check("verify: code tests in the sandbox", v["passed"], v["verifier"])
except Exception as e:
    check("verify: code tests in the sandbox", False, e)
try:
    st, _, data = http("GET", "/v1/perf/stages?hours=24&lab=true")
    v = json.loads(data)
    check("perf: latency breakdown (P7-T01)", v["replies"] > 0 and v["paths"], f"{v['replies']} replies")
except Exception as e:
    check("perf: latency breakdown (P7-T01)", False, e)

try:
    sysinfo = json.loads(http("GET", "/v1/system")[2])
    down = [b["name"] for b in sysinfo["backends"] if not (b["system1"]["ok"] and b["system2"]["ok"])]
    if down:
        try:
            http("POST", "/v1/system/backend", {"name": down[0]})
            check("system: switching to a stopped backend is refused", False, "accepted")
        except urllib.error.HTTPError as e:
            after = json.loads(http("GET", "/v1/system")[2])["active_backend"]
            check("system: switching to a stopped backend is refused", e.code == 503 and after == sysinfo["active_backend"], f"{e.code}, active stays {after}")
except Exception as e:
    check("system: switching to a stopped backend is refused", False, e)

# 3. UI pages
if "--no-ui" not in args:
    import websocket
    CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    port, prof = 9341, tempfile.mkdtemp(prefix="nafs-e2e-")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                               "--no-first-run", "--window-size=1440,900", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                page = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")) if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        n = [0]
        errors = []

        def call(method, **params):
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
            while True:
                m = json.loads(ws.recv())
                if m.get("method") == "Runtime.exceptionThrown":
                    errors.append(m["params"]["exceptionDetails"].get("exception", {}).get("description", "exception")[:200])
                elif m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] == "error":
                    errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:200])
                elif m.get("method") == "Log.entryAdded" and m["params"]["entry"]["level"] == "error":
                    errors.append(m["params"]["entry"]["text"][:200])
                if m.get("id") == n[0]:
                    return m.get("result", {})

        call("Runtime.enable"); call("Log.enable"); call("Page.enable")
        for route in ["chat", "eval", "gate", "tazkiya", "rooms", "bench", "system"]:
            errors.clear()
            call("Page.navigate", url=f"{BASE}/?e2e={route}#/{route}")
            t = time.time()
            while time.time() - t < 6:  # drain events while the page loads its data
                call("Runtime.evaluate", expression="1")
                time.sleep(0.3)
            title = call("Runtime.evaluate", expression="document.title", returnByValue=True)["result"].get("value", "")
            check(f"ui: #{route} renders without errors", not errors and "Nafs-Aql" in title, errors[:2] or title)
        ws.close()
    finally:
        chrome.terminate()
        shutil.rmtree(prof, ignore_errors=True)

failed = [r for r in results if not r["pass"]]
json.dump({"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base": BASE, "passed": len(results) - len(failed),
           "failed": len(failed), "results": results}, open(os.path.join(ROOT, "benchmarks", "e2e_sweep.json"), "w"), indent=1)
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
