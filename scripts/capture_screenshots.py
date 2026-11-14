#!/usr/bin/env python3
"""Capture Web UI screenshots for the report and slides (P7-T07).

Drives a headless Chrome with a throw-away profile over the DevTools protocol against the running gateway
(http://127.0.0.1:8080 by default) and writes PNGs to docs/shared/screenshots/. Nothing is submitted: the script
only navigates, clicks a conversation and the Inspect button, and scrolls.

Usage: python3 scripts/capture_screenshots.py [--base http://127.0.0.1:8080] [--only chat,lab] [--kb ID] [--kb-graph ID]
(--kb and --kb-graph are knowledge-base ids for the knowledge, knowledge_graph and knowledge_chat shots.)
Needs Google Chrome and the websocket-client package.
"""
import base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "shared", "screenshots")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9339
W, H, SCALE = 1440, 900, 2


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def wait_for(self, expr, timeout=30):
        t = time.time()
        while time.time() - t < timeout:
            if self.js(expr):
                return True
            time.sleep(0.4)
        return False

    def shot(self, name, height=None):
        h = height or H
        self.call("Emulation.setDeviceMetricsOverride", width=W, height=h, deviceScaleFactor=SCALE, mobile=False)
        time.sleep(0.8)
        data = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)["data"]
        path = os.path.join(OUT, name + ".png")
        open(path, "wb").write(base64.b64decode(data))
        print("wrote", os.path.relpath(path, ROOT))
        self.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=SCALE, mobile=False)


def api(base, method, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"null")


def ask(ws, text, timeout=240):
    """Send one question to a room as this member and wait for the assistant's answer to finish."""
    ws.send(json.dumps({"type": "user_message", "payload": {"content": text, "force_mode": "fast"}}))
    end = time.time() + timeout
    ws.settimeout(5)
    while time.time() < end:
        try:
            f = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        if f.get("type") == "assistant_event" and f["payload"]["event"]["event"] == "done":
            return True
    return False


def scroll_to_heading(tab, text):
    tab.js(f"""(() => {{ const h = [...document.querySelectorAll('h2,h3')].find(x => x.textContent.startsWith({json.dumps(text)}));
      if (!h) return false; const sc = h.closest('main, .page, .content') || document.scrollingElement; h.scrollIntoView({{block: 'start'}}); return true; }})()""")
    time.sleep(0.6)


def main():
    args = sys.argv[1:]
    base = args[args.index("--base") + 1] if "--base" in args else "http://127.0.0.1:8080"
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    os.makedirs(OUT, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="nafs-shots-")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                               "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", f"--window-size={W},{H}",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        tab = Tab(page["webSocketDebuggerUrl"])
        tab.call("Page.enable")
        tab.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=SCALE, mobile=False)
        tab.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])

        def go(route):
            tab.call("Page.navigate", url=f"{base}/#/{route}")
            tab.wait_for("document.readyState === 'complete'")
            time.sleep(2.5)

        want = lambda k: only is None or k in only
        if want("chat"):
            go("chat")
            tab.wait_for("document.querySelectorAll('.conv').length > 0")
            # A conversation with a verified answer, then open its inspector.
            tab.js("""(() => { const c = [...document.querySelectorAll('.conv')].find(x => /train leaves|python function|run_length/i.test(x.textContent)) || document.querySelector('.conv'); c && c.click(); return !!c; })()""")
            tab.wait_for("document.querySelectorAll('[data-a=\"inspect\"]').length > 0", 20)
            time.sleep(1.5)
            tab.shot("chat")
        # Phase 31: how a rich answer is drawn (table, formula, diagram, code). The text is a fixed sample run through the
        # real renderer inside the real message markup; nothing is sent to a model.
        if want("rich"):
            sample = open(os.path.join(ROOT, "scripts", "rich_sample.md")).read()
            go("chat")
            tab.wait_for("!!document.getElementById('inner')", 30)
            tab.js("""(async () => { const { markdown } = await import('/render.js?v=' + Date.now()); const md = %s;
              document.getElementById('inner').innerHTML = '<div class="msg bot"><div class="body"><div class="answer">' + markdown(md, new Set([1])) + '</div></div></div>'; return true; })()""" % json.dumps(sample))
            time.sleep(1.5)
            tab.shot("rich_answer", 1700)
        if want("lab"):
            go("eval")
            tab.wait_for("!!document.querySelector('table.policy-map')", 30)
            scroll_to_heading(tab, "Roadmap gates")
            tab.shot("lab_gates")
            scroll_to_heading(tab, "Insights")
            tab.shot("lab_insights")
        if want("muhasaba"):
            go("gate")
            time.sleep(2)
            tab.shot("muhasaba")
            scroll_to_heading(tab, "Drift monitor")
            tab.shot("muhasaba_drift_alerts")
        if want("performance"):
            go("bench")
            tab.wait_for("!!document.querySelector('#bStages .stagebar') || !!document.querySelector('#bStages .empty')", 30)
            scroll_to_heading(tab, "Load-test protocol")
            tab.shot("performance")
            scroll_to_heading(tab, "Where the time goes")
            tab.shot("performance_stages")
        if want("tazkiya"):
            go("tazkiya")
            tab.shot("tazkiya")
        if want("system"):
            go("system")
            tab.shot("system")
        # Phase 20/21: the Knowledge page, its graph, and a chat answer with numbered sources. They need a base: pass
        # --kb <id> (a base with passages) and, for the graph shot, --kb-graph <id> (a base whose graph is built).
        kb = args[args.index("--kb") + 1] if "--kb" in args else ""
        kb_graph = args[args.index("--kb-graph") + 1] if "--kb-graph" in args else kb
        def pick_base(base_id):
            # localStorage belongs to the page's origin, so the app must be loaded once before it can be written.
            go("chat")
            tab.js(f"localStorage.setItem('nafs.kbCurrent', JSON.stringify({json.dumps(base_id)})); localStorage.setItem('nafs.advanced', 'true')")
        if want("knowledge") and kb:
            pick_base(kb)
            go("knowledge")
            tab.wait_for("document.querySelectorAll('#kbDocs table tr, #kbDocs .kb-doc').length > 2", 40)
            time.sleep(1.5)
            tab.shot("knowledge")
        if want("knowledge_graph") and kb_graph:
            pick_base(kb_graph)
            go("knowledge")
            tab.wait_for("!!document.querySelector('#kbGraph svg')", 40)
            scroll_to_heading(tab, "Knowledge graph")
            time.sleep(2.5)
            tab.shot("knowledge_graph")
        if want("knowledge_chat") and kb:
            # The base with a graph, in "Passages and graph" mode, so the answer shows both the relations it used
            # (with their verified quotes) and the numbered passages.
            pick_base(kb_graph)
            tab.js(f"localStorage.setItem('nafs.chatSettings', JSON.stringify({{mode: 'fast', kb: {json.dumps(kb_graph)}, kbMode: 'graph'}}))")
            # The chat page reads its settings once at load, and a hash navigation does not reload it.
            tab.js("location.reload()")
            time.sleep(3)
            tab.wait_for("!!document.getElementById('input') && document.getElementById('kbSel')?.value", 30)
            tab.js("""(() => { const i = document.getElementById('input'); i.value = 'Which revision of Smiley Core did Phase 19 merge, and what slowdown did it expose?';
              i.dispatchEvent(new Event('input', {bubbles: true})); document.getElementById('sendBtn').click(); return true; })()""")
            tab.wait_for("document.querySelectorAll('.kb-sources').length > 0", 240)
            time.sleep(2)
            tab.shot("knowledge_chat")
        # Phase 29: the Files page, at a document's overview, at the memory with its change log, and a search the
        # assistant planned. --kb <id> names the base to open; --doc <id> the document in it.
        doc = args[args.index("--doc") + 1] if "--doc" in args else ""
        def files_at(uri):
            go("chat")
            tab.js(f"localStorage.setItem('nafs.fsUri', JSON.stringify({json.dumps(uri)}))")
            go("files")
            tab.wait_for("document.querySelectorAll('#fsList [data-u]').length > 0 && !document.querySelector('#fsMain .spin')", 40)
            time.sleep(1.5)
        if want("files") and kb and doc:
            files_at(f"nafs://kb/{kb}")
            tab.js(f"""(() => {{ const b = document.querySelector('#fsList [data-u="nafs://kb/{kb}/{doc}"]'); b && b.click(); return !!b; }})()""")
            time.sleep(1.5)
            tab.wait_for("!!document.querySelector('#fsMain .fs-body h1, #fsMain .fs-body h2')", 30)
            time.sleep(1)
            tab.shot("files")
        if want("files_memory"):
            files_at("nafs://memory")
            tab.wait_for("!!document.querySelector('#fsExtra .card')", 30)
            time.sleep(1)
            tab.shot("files_memory")
        if want("files_search"):
            files_at("nafs://")
            tab.js("""(() => { document.getElementById('fsQ').value = 'How should I write the release notes, and in which language?';
              document.querySelector('#fsHow [data-h="search"]').click(); document.getElementById('fsGo').click(); return true; })()""")
            tab.wait_for("!!document.querySelector('#fsMain .fs-hits, #fsMain .card-h h3') && !document.querySelector('#fsMain .spin')", 60)
            time.sleep(1.5)
            tab.shot("files_search")
        # Phase 30: Settings, at the Speed section (ShadowPrefill and the stable prompt beginning).
        if want("settings_speed"):
            go("chat")
            tab.js("document.getElementById('settingsBtn').click(); true")
            tab.wait_for("!!document.getElementById('stShadow')", 20)
            tab.js("document.getElementById('stShadow').closest('section').scrollIntoView({block: 'start'}); true")
            time.sleep(1)
            tab.shot("settings_speed")
            tab.js("document.getElementById('modal').close(); true")
        if want("rooms"):
            # A throw-away room with three people and two real answers, photographed, then deleted.
            ws_base = base.replace("http", "ws", 1)
            room = api(base, "POST", "/v1/rooms", {"name": "Study group"})
            members = []
            try:
                for who in ("Aisha", "Bilal"):
                    members.append(websocket.create_connection(f"{ws_base}/ws/rooms/{room['room_id']}?user={who}", timeout=10))
                    members[-1].recv()  # welcome
                ask(members[0], "Give me two tips for studying in a group.")
                ask(members[1], "Which of those is easier to start with?")
                go("rooms")
                tab.js("localStorage.setItem('nafs.user', JSON.stringify('Maya')); localStorage.setItem('nafs.roomMode', JSON.stringify('fast'))")
                go(f"rooms/{room['room_id']}")
                tab.wait_for("document.querySelectorAll('#rinner .msg.bot').length >= 2 && /Connected/.test(document.querySelector('#rstate')?.innerText || '')", 40)
                time.sleep(1.5)
                tab.shot("rooms")
                tab.js("document.getElementById('delBtn').click()")
                tab.wait_for("document.getElementById('modal').open", 5)
                time.sleep(0.6)
                tab.shot("rooms_delete")
                tab.js("document.getElementById('cdNo').click()")
            finally:
                for m in members:
                    try:
                        m.close()
                    except Exception:
                        pass
                try:
                    api(base, "DELETE", f"/v1/rooms/{room['room_id']}")
                except urllib.error.HTTPError:
                    pass
    finally:
        chrome.terminate()
        try:
            chrome.wait(10)
        except Exception:
            chrome.kill()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
