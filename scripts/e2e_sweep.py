#!/usr/bin/env python3
"""End-to-end sweep (P7-T07): every documented read endpoint, the main request paths, and every UI page.

1. API: GET every OpenAPI path without path parameters (200 and JSON or text as declared).
2. Requests: a verified chat reply (stage timings present), an SSE stream (event order), /v1/verify, the
   latency breakdown. Requests are ephemeral, so nothing is added to the conversation history.
3. UI: load each page in a headless Chrome (throw-away profile) and fail on any JavaScript exception or
   console error, on sideways scrolling at phone width, or on internal vocabulary in the visible text.

Usage: python3 scripts/e2e_sweep.py [--base http://127.0.0.1:8080] [--no-ui | --ui-only]
Writes benchmarks/e2e_sweep.json and exits non-zero on any failure.
"""
import re, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

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


if "--ui-only" not in args:
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
        # A loaded host can take more than ten seconds to bring Chrome's debugger up; give it 60 s and
        # fail with a clear message instead of a NameError when it never appears.
        page = None
        for _ in range(300):
            try:
                page = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")) if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        if page is None:
            raise RuntimeError("headless Chrome did not expose a debugger page within 60 s (host overloaded?)")
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
        # Production copy: the visible text never shows the research vocabulary (requirement ids, theorem
        # references, model symbols, "System 1/2"). Chat and Rooms are checked as a first-time visitor sees
        # them; every page again with Advanced mode on (the Insights pages and answer details).
        JARGON = re.compile(r"\b(?:FR|NFR|ER|TR|OR|SR)-\d+|\bP\d-T\d+|\bProp\. ?\d|\bTheorem \d|\bThm \d|ê[₁₂]|λ|Δc|\bSystem [12]\b|\bS[12]\b"
                            r"|\bescalat|\bdeliberat(?!ely)|\bverifier|\bAql Gate\b|\bledger\b|\bTTFT\b|\bp(?:50|95|99)\b|\bprefill|\bplaceholder")
        def settle(url, secs):
            call("Page.navigate", url=url)
            t = time.time()
            while time.time() - t < secs:  # drain events while the page loads its data
                call("Runtime.evaluate", expression="1")
                time.sleep(0.3)
        # Stored questions and answers (the Learning page's review queue, the Experiments page's item answers) are
        # user and model text, not the product's wording: they may contain any word, so they are hidden while the page
        # text is read. Everything else, including the "Thinking it through" copy that sits in an `.answer` element in
        # Chat, is still checked.
        CONTENT = (".review .q,.review .answer,#eItems .answer,#eItems .codebox")
        READ = ("(()=>{const s=document.createElement('style');s.textContent='" + CONTENT + "{display:none!important}';"
                "document.head.appendChild(s);const t=document.querySelector('#app').innerText;s.remove();return t})()")
        def jargon():
            text = call("Runtime.evaluate", expression=READ, returnByValue=True)["result"].get("value", "")
            return sorted({m.group(0) for m in JARGON.finditer(text)})
        for route in ["chat", "rooms", "knowledge", "code", "files"]:
            settle(f"{BASE}/?e2e=p{route}#/{route}", 4)
            if route == "chat":
                call("Runtime.evaluate", expression="localStorage.setItem('nafs.advanced', 'false')")
                settle(f"{BASE}/?e2e=p{route}2#/{route}", 4)
            hits = jargon()
            check(f"ui: #{route} uses plain language", not hits, hits[:6])
        call("Runtime.evaluate", expression="localStorage.setItem('nafs.advanced', 'true')")
        for route in ["chat", "eval", "gate", "tazkiya", "rooms", "bench", "system", "knowledge", "code", "files"]:
            errors.clear()
            settle(f"{BASE}/?e2e={route}#/{route}", 6)
            title = call("Runtime.evaluate", expression="document.title", returnByValue=True)["result"].get("value", "")
            check(f"ui: #{route} renders without errors", not errors and "Nafs-Aql" in title, errors[:2] or title)
            hits = jargon()
            check(f"ui: #{route} (Advanced) uses plain language", not hits, hits[:6])
        # Phase 35: the controls themselves. Every control a person can reach has a name a screen reader can say; the
        # segmented controls say which button is chosen; forms say what is wrong beside the field; the answer
        # renderer's tables sort and its code wraps.
        ev = lambda expr: call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)["result"].get("value")
        NAMELESS = ("(() => { const name = (e) => (e.getAttribute('aria-label') || '').trim() || (e.getAttribute('aria-labelledby') || '').split(' ').map(i => document.getElementById(i)?.textContent || '').join(' ').trim()"
                    " || (e.id && document.querySelector('label[for=\"' + e.id + '\"]')?.textContent.trim()) || e.closest('label')?.textContent.trim() || (e.getAttribute('title') || '').trim()"
                    " || (e.tagName === 'BUTTON' || e.tagName === 'A' || e.tagName === 'SUMMARY' || e.getAttribute('role') === 'button' ? e.textContent.trim() : '') || (e.getAttribute('placeholder') || '').trim();"
                    " return [...document.querySelectorAll('#app button, #app input:not([type=hidden]), #app select, #app textarea, #app a[href], #app [role=button], dialog[open] button, dialog[open] input, dialog[open] select')]"
                    ".filter(e => e.offsetParent !== null && !e.closest('[hidden]')).filter(e => !name(e)).map(e => e.outerHTML.slice(0, 90)); })()")
        for route in ["chat", "rooms", "knowledge", "code", "files", "system"]:
            settle(f"{BASE}/?e2e=n{route}#/{route}", 5)
            bad = ev(NAMELESS) or []
            check(f"ui: every control on #{route} has a name", not bad, bad[:3])
        settle(f"{BASE}/?e2e=set#/settings", 4)
        st = ev("(() => ({ open: document.querySelector('#modal').open, rows: document.querySelectorAll('#modalContent .set-row').length,"
                " hints: [...document.querySelectorAll('#modalContent .set-row')].filter(r => !r.querySelector('.hint, p')).length,"
                " form: !!document.querySelector('#modal form[method=dialog]'), reset: !!document.querySelector('#stResetAll') }))()") or {}
        check("ui: Settings opens from its link with every setting explained", st.get("open") and st.get("rows", 0) >= 10 and st.get("hints") == 0 and st.get("reset"), st)
        check("ui: the dialog is not a form (Enter in a text box does not close it)", st.get("form") is False, st)
        bad = ev(NAMELESS) or []
        check("ui: every control in Settings has a name", not bad, bad[:3])
        found = ev("(() => { const f = document.querySelector('#stFind'); f.value = 'thinking'; f.dispatchEvent(new Event('input')); const shown = [...document.querySelectorAll('#modalContent .set-row')].filter(r => !r.hidden).length;"
                   " f.value = 'zzzz'; f.dispatchEvent(new Event('input')); const none = !document.querySelector('#stNone').hidden; f.value = ''; f.dispatchEvent(new Event('input')); return [shown, none]; })()") or [0, False]
        check("ui: finding a setting narrows the list and says when nothing matches", 1 <= found[0] <= 4 and found[1], found)
        ev("document.querySelector('#modal').close()")
        segs = ev("(() => [...document.querySelectorAll('.seg > button')].filter(b => !['true','false'].includes(b.getAttribute('aria-pressed'))).length)()")
        check("ui: segmented controls say which button is chosen", segs == 0, segs)
        settle(f"{BASE}/?e2e=code#/code", 4)
        errs = ev("(async () => { document.querySelector('#cgForm').requestSubmit(); await new Promise(r => setTimeout(r, 150)); const a = [document.querySelector('#cgTaskErr').textContent, document.querySelector('#cgTestsErr').textContent];"
                  " document.querySelector('#cgTask').value = 'Write a function double(x) that returns twice x.'; document.querySelector('#cgTests').value = 'double(2) == 4'; document.querySelector('#cgTask').dispatchEvent(new Event('input')); document.querySelector('#cgTests').dispatchEvent(new Event('input'));"
                  " await new Promise(r => setTimeout(r, 100)); return a.concat([document.querySelector('#cgTaskErr').textContent, document.querySelector('#cgTestsErr').textContent, document.activeElement?.id]); })()") or []
        check("ui: the Code form says what is missing beside each field and puts the cursor there",
              len(errs) == 5 and errs[0] and errs[1] and errs[2] == "" and "assert" in errs[3] and errs[4] in ("cgTask", "cgTests"), errs)
        settle(f"{BASE}/?e2e=render#/chat", 4)
        rd = ev("(async () => { const m = await import('/render.js'); const host = document.querySelector('#inner'); host.innerHTML = '<div class=\"msg assistant\"><div class=\"answer\" id=\"e2eA\"></div></div>';"
                " document.querySelector('#e2eA').innerHTML = m.markdown('| n | v |\\n|---|--:|\\n| a | 1,200 |\\n| b | 95 |\\n| c | 800 |\\n\\n```py\\n' + 'x = 1\\n'.repeat(7) + '```\\n'); await new Promise(r => setTimeout(r, 300));"
                " const th = document.querySelectorAll('#e2eA thead th')[1]; th.click(); const asc = [...document.querySelectorAll('#e2eA tbody tr')].map(r => r.cells[0].textContent).join('');"
                " const pre = document.querySelector('#e2eA pre.code'); pre.querySelector('[data-act=wrap]').click();"
                " return { asc, sort: th.getAttribute('aria-sort'), tab: th.tabIndex, wrapped: pre.classList.contains('wrapped'), gutter: !!pre.querySelector('.gutter'),"
                " bar: [...document.querySelectorAll('#e2eA .tbl-bar button')].every(b => getComputedStyle(b).opacity === '1') }; })()") or {}
        check("ui: a table in an answer sorts by a column, numbers as numbers, from the keyboard too", rd.get("asc") == "bca" and rd.get("sort") == "ascending" and rd.get("tab") == 0, rd)
        check("ui: code wraps and is numbered; the table's actions are always visible", rd.get("wrapped") and rd.get("gutter") and rd.get("bar"), rd)
        # The same pages on a phone-sized screen: nothing may force the page to scroll sideways.
        call("Emulation.setDeviceMetricsOverride", width=375, height=812, deviceScaleFactor=2, mobile=True)
        for route in ["chat", "eval", "gate", "tazkiya", "rooms", "bench", "system", "knowledge", "code", "files"]:
            call("Page.navigate", url=f"{BASE}/?e2e=m{route}#/{route}")
            t = time.time()
            while time.time() - t < 4:
                call("Runtime.evaluate", expression="1")
                time.sleep(0.3)
            over = call("Runtime.evaluate", expression="(() => { const v = document.querySelector('.view'); return v ? v.scrollWidth - v.clientWidth : -1; })()", returnByValue=True)["result"].get("value")
            check(f"ui: #{route} fits a 375 px screen (no sideways scroll)", over is not None and 0 <= over <= 1, f"{over} px over")
        ws.close()
    finally:
        chrome.terminate()
        shutil.rmtree(prof, ignore_errors=True)

failed = [r for r in results if not r["pass"]]
json.dump({"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base": BASE, "passed": len(results) - len(failed),
           "failed": len(failed), "results": results}, open(os.path.join(ROOT, "benchmarks", "e2e_sweep_ui.json" if "--ui-only" in args else "e2e_sweep.json"), "w"), indent=1)
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
