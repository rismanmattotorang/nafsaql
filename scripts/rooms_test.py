#!/usr/bin/env python3
"""Live tests of Rooms (IF-004) against a running gateway: the REST surface, the WebSocket protocol with several
simulated members, and deletion while an answer is being generated. Every room the script creates is deleted
again, whatever happens.

Usage: python3 scripts/rooms_test.py [--base http://127.0.0.1:8080] [--slow]
--slow adds the checks that take longer: the 25 s heartbeat, a full room (65 members), and the queue limit.
Needs the websocket-client package. Writes benchmarks/rooms_test.json and exits non-zero on any failure.
"""
import json, os, secrets, sqlite3, sys, time, urllib.error, urllib.request

import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
BASE = args[args.index("--base") + 1] if "--base" in args else "http://127.0.0.1:8080"
SLOW = "--slow" in args
WS = BASE.replace("http", "ws", 1)
DB = os.path.join(ROOT, "data", "runtime", "nafs.sqlite")
results, created = [], []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
    print(f"{'PASS' if ok else 'FAIL'}  {name:70s} {str(detail)[:80]}", flush=True)


def http(method, path, body=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def make_room(name):
    st, r = http("POST", "/v1/rooms", {"name": name})
    if st == 201:
        created.append(r["room_id"])
    return st, r


class Member:
    """One WebSocket connection with a frame log."""

    def __init__(self, room_id, user):
        self.ws = websocket.create_connection(f"{WS}/ws/rooms/{room_id}?user={user}", timeout=5)
        self.frames, self.closed = [], None

    def pump(self, timeout):
        """Read frames for up to `timeout` seconds; stops early on close. Returns the frames read."""
        out, end = [], time.time() + timeout
        while time.time() < end and self.closed is None:
            self.ws.settimeout(max(0.05, end - time.time()))
            try:
                op, fr = self.ws.recv_data_frame(True)
            except websocket.WebSocketTimeoutException:
                break
            except (websocket.WebSocketConnectionClosedException, ConnectionError, OSError):
                self.closed = self.closed or -1
                break
            if op == websocket.ABNF.OPCODE_TEXT:
                f = json.loads(fr.data)
                self.frames.append(f)
                out.append(f)
            elif op == websocket.ABNF.OPCODE_CLOSE:
                self.closed = int.from_bytes(fr.data[:2], "big") if len(fr.data) >= 2 else 0
        return out

    def until(self, pred, timeout=30):
        """Pump until a frame matches `pred`; returns it (or None on timeout)."""
        for f in self.frames:
            if pred(f):
                self.frames.remove(f)
                return f
        end = time.time() + timeout
        while time.time() < end and self.closed is None:
            for f in self.pump(0.5):
                if pred(f):
                    self.frames.remove(f)
                    return f
        return None

    def send(self, **frame):
        self.ws.send(json.dumps(frame))

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def presence(user, joined):
    return lambda f: f.get("type") == "presence" and f.get("user") == user and f["payload"]["joined"] is joined


def idle(f):
    p = f.get("payload", {})
    return f.get("type") == "queue" and p.get("busy") is False and p.get("queued") == 0


def of_type(t, **kw):
    return lambda f: f.get("type") == t and all(f.get("payload", {}).get(k) == v or f.get(k) == v for k, v in kw.items())


def main():
    tag = secrets.token_hex(3)
    # ------------------------------------------------------------------ REST: create, list, get
    st, a = make_room(f"  T-{tag}   Alpha ")
    check("create: name is trimmed and collapsed, 201", st == 201 and a["name"] == f"T-{tag} Alpha", f"{st} {a.get('name') if a else a}")
    check("create: the room starts empty", a["questions"] == 0 and a["members"] == 0 and a["users"] == [] and a["busy"] is False and a["last_activity"], a)
    for dup in (f"t-{tag} alpha", f"T-{tag}    ALPHA", f" T-{tag} Alpha "):
        st, e = make_room(dup)
        check(f"create: duplicate '{dup}' is a 409 conflict", st == 409 and e["code"] == "E_CONFLICT", f"{st} {e}")
    for bad, why in (("", "empty"), ("   \n\t ", "blank"), ("\u0000\u0007", "control characters only")):
        st, e = make_room(bad)
        check(f"create: {why} name is a 400", st == 400 and e["code"] == "E_BAD_REQUEST", f"{st} {e}")
    st, e = http("POST", "/v1/rooms", {"name": 5})
    check("create: wrong type is refused with the JSON envelope", st in (400, 422) and e and e["code"] == "E_BAD_REQUEST", f"{st} {e}")
    st, r = make_room("x" * 300)
    check("create: a 300-character name is cut to 80", st == 201 and len(r["name"]) == 80, f"{st} {len(r['name']) if r else '-'}")
    st, h = make_room(f"T-{tag} <img src=x onerror=alert(1)>")
    check("create: markup is stored as plain text", st == 201 and "<img" in h["name"], h and h["name"])
    st, lst = http("GET", "/v1/rooms")
    ids = [r["room_id"] for r in lst]
    check("list: every created room is listed, newest activity first", a["room_id"] in ids and h["room_id"] in ids and ids.index(h["room_id"]) < ids.index(a["room_id"]), f"{len(lst)} rooms")
    st, g = http("GET", f"/v1/rooms/{a['room_id']}")
    check("get: one room by id", st == 200 and g["room_id"] == a["room_id"] and g["name"] == a["name"], st)
    st, e = http("GET", "/v1/rooms/00000000-0000-0000-0000-000000000000")
    check("get: unknown room is a 404", st == 404 and e["code"] == "E_NOT_FOUND", f"{st} {e}")
    st, e = http("GET", "/v1/rooms/not-a-uuid")
    check("get: a malformed id is a 400 with the JSON envelope", st == 400 and e and e["code"] == "E_BAD_REQUEST", f"{st} {e}")
    # ------------------------------------------------------------------ REST: rename
    st, b = make_room(f"T-{tag} Beta")
    st1, r = http("PATCH", f"/v1/rooms/{b['room_id']}", {"name": f"T-{tag}  Beta  Two"})
    check("rename: normalised, 200, listed under the new name", st1 == 200 and r["name"] == f"T-{tag} Beta Two" and http("GET", f"/v1/rooms/{b['room_id']}")[1]["name"] == r["name"], st1)
    st1, e = http("PATCH", f"/v1/rooms/{b['room_id']}", {"name": f"t-{tag} alpha"})
    check("rename: onto another room's name (any case) is a 409", st1 == 409 and e["code"] == "E_CONFLICT", f"{st1} {e}")
    st1, r = http("PATCH", f"/v1/rooms/{b['room_id']}", {"name": f"T-{tag} BETA TWO"})
    check("rename: changing only the case of its own name is allowed", st1 == 200 and r["name"] == f"T-{tag} BETA TWO", st1)
    st1, e = http("PATCH", f"/v1/rooms/{b['room_id']}", {"name": "  "})
    check("rename: a blank name is a 400", st1 == 400, f"{st1} {e}")
    st1, e = http("PATCH", "/v1/rooms/00000000-0000-0000-0000-000000000000", {"name": "x"})
    check("rename: unknown room is a 404", st1 == 404, st1)
    # ------------------------------------------------------------------ WebSocket: welcome, presence, names
    ann = Member(a["room_id"], "Ann")
    first = ann.pump(1.5)
    check("ws: the first frame is the welcome, addressed to the new member", first and first[0]["type"] == "welcome", [f["type"] for f in first])
    w = first[0]["payload"]
    check("ws: welcome carries the name, the room, the members and the limits", w["you"] == "Ann" and w["users"] == ["Ann"] and w["room"]["room_id"] == a["room_id"]
          and w["assistant"] == {"busy": False, "working_for": None, "queued": 0} and w["limits"]["message_chars"] == 8000, w)
    check("ws: the member then sees its own presence frame", any(f["type"] == "presence" and f["payload"]["members"] == 1 for f in first), [f["type"] for f in first])
    bob = Member(a["room_id"], "ann")  # same name apart from case
    bw = bob.pump(1.5)
    check("ws: a clashing name gets a number, so every member is unique", bw[0]["payload"]["you"] == "ann 2" and bw[0]["payload"]["users"] == ["Ann", "ann 2"], bw[0]["payload"]["you"])
    pj = ann.until(presence("ann 2", True), 3)
    check("ws: others see the join, with the member list and count", pj and pj["payload"]["members"] == 2 and pj["payload"]["users"] == ["Ann", "ann 2"], pj)
    st, g = http("GET", f"/v1/rooms/{a['room_id']}")
    check("rest: the room list reports who is online", g["members"] == 2 and g["users"] == ["Ann", "ann 2"], g["users"])
    imp = Member(a["room_id"], "Assistant")
    check("ws: nobody can take the name Assistant", imp.pump(1.5)[0]["payload"]["you"] == "Assistant 2")
    imp.close()
    ann.until(presence("Assistant 2", False), 5)
    # ------------------------------------------------------------------ WebSocket: ping, typing, limits
    ann.send(type="ping")
    check("ws: ping is answered with pong", ann.until(lambda f: f["type"] == "pong", 3) is not None)
    ann.send(type="typing"); ann.send(type="typing"); ann.send(type="typing")
    typing = [f for f in bob.pump(1.5) if f["type"] == "typing"]
    check("ws: typing reaches the others, throttled to one frame per 2 s", len(typing) == 1 and typing[0]["user"] == "Ann", len(typing))
    ann.send(type="user_message", payload={"content": "x" * 8001})
    e = ann.until(of_type("error"), 3)
    check("ws: an over-long message is refused, privately, not truncated", e and e["payload"]["code"] == "E_TOO_LONG" and "8001" in e["payload"]["message"], e and e["payload"])
    check("ws: ... and the others never see it", not [f for f in bob.pump(0.8) if f["type"] == "user_message"])
    ann.send(type="user_message", payload={"content": "   "}); ann.send(type="nonsense"); ann.ws.send("not json")
    check("ws: blank messages, unknown frames and garbage are ignored", not ann.pump(0.8))
    # ------------------------------------------------------------------ WebSocket: a message and the assistant's answer
    ann.send(type="user_message", payload={"content": "Reply with the single word OK.", "force_mode": "fast"})
    um = bob.until(of_type("user_message"), 10)
    check("ws: the others receive the message with sender, id and timestamp", um and um["user"] == "Ann" and um["payload"]["id"] and um["payload"]["ts"], um and um["user"])
    qb = bob.until(of_type("queue", busy=True), 20)
    check("ws: the queue frame says the assistant is answering for Ann", qb and qb["payload"]["working_for"] == "Ann", qb and qb["payload"])
    done = bob.until(lambda f: f["type"] == "assistant_event" and f["payload"]["event"]["event"] == "done", 120)
    check("ws: the others see the assistant's answer stream to done", done is not None and done["payload"]["reply_to"] == um["payload"]["id"])
    qi = bob.until(idle, 20)
    check("ws: the queue frame returns to idle (nothing running, nothing queued)", qi is not None, qi and qi["payload"])
    reply_decision = done["payload"]["event"]["data"]["decision_id"] if done and "decision_id" in done["payload"]["event"]["data"] else None
    st, g = http("GET", f"/v1/rooms/{a['room_id']}")
    check("rest: the room now counts the question and its activity", g["questions"] == 1 and g["last_activity"] > a["last_activity"], f"questions={g['questions']}")
    st, conv = http("GET", f"/v1/conversations/{g['conversation_id']}")
    check("rest: the transcript has the question (with the sender) and the answer", st == 200 and len(conv["messages"]) == 2 and conv["messages"][0]["content"] == "[Ann] Reply with the single word OK.", st)
    check("rest: room conversations are not in the chat history list", g["conversation_id"] not in [c["conversation_id"] for c in http("GET", "/v1/conversations")[1]])
    late = Member(a["room_id"], "Cy")
    lw = late.pump(1.5)
    check("ws: welcome hands a late joiner the history cursor (2 stored messages) and nothing pending",
          lw[0]["payload"]["cursor"] == 2 and lw[0]["payload"]["pending"] == [], (lw[0]["payload"]["cursor"], lw[0]["payload"]["pending"]))
    check("ws: a late joiner's welcome shows the assistant idle", lw[0]["payload"]["assistant"]["busy"] is False and lw[0]["payload"]["users"] == ["Ann", "ann 2", "Cy"], lw[0]["payload"]["users"])
    late.close()
    ann.until(presence("Cy", False), 5)
    # ------------------------------------------------------------------ joining while an answer is being written
    ann.send(type="user_message", payload={"content": "Name three primary colours.", "force_mode": "fast"})
    ann.until(of_type("queue", busy=True), 30)
    mid = Member(a["room_id"], "Mid")
    mw = mid.pump(1.0)[0]["payload"]
    check("ws: joining mid-answer, the welcome lists the running question as pending, with the cursor before it",
          mw["assistant"]["busy"] is True and [p["content"] for p in mw["pending"]] == ["Name three primary colours."] and mw["pending"][0]["user"] == "Ann" and mw["cursor"] == 2,
          (mw["assistant"], mw["cursor"], [p["content"] for p in mw["pending"]]))
    d2 = mid.until(lambda f: f["type"] == "assistant_event" and f["payload"]["event"]["event"] == "done", 120)
    check("ws: ... and the same member receives that answer's done frame, closing the pending turn", d2 is not None and d2["payload"]["reply_to"] == mw["pending"][0]["id"])
    mid.until(idle, 20)
    mid2 = Member(a["room_id"], "Mid")
    m2 = mid2.pump(1.0)[0]["payload"]
    check("ws: after the answer, a joiner finds the turn in the stored messages (cursor 4) and nothing pending", m2["cursor"] == 4 and m2["pending"] == [], (m2["cursor"], m2["pending"]))
    mid.close(); mid2.close()
    # ------------------------------------------------------------------ rename is broadcast
    st1, r = http("PATCH", f"/v1/rooms/{a['room_id']}", {"name": f"T-{tag} Alpha Renamed"})
    rn = ann.until(of_type("room_renamed"), 5)
    check("ws: a rename reaches every member", rn and rn["payload"]["name"] == f"T-{tag} Alpha Renamed" and bob.until(of_type("room_renamed"), 5) is not None, rn)
    # ------------------------------------------------------------------ departure
    bob.close()
    pl = ann.until(presence("ann 2", False), 5)
    check("ws: a departure is announced with the remaining members", pl and pl["payload"]["users"] == ["Ann"] and pl["payload"]["members"] == 1, pl and pl["payload"])
    # ------------------------------------------------------------------ delete with members connected
    cy = Member(a["room_id"], "Cy"); cy.pump(1.0)
    t0 = time.time()
    st, rep = http("DELETE", f"/v1/rooms/{a['room_id']}")
    check("delete: 200 with a report of what was removed", st == 200 and rep["deleted"] and rep["disconnected"] == 2 and rep["removed"]["messages"] == 4 and rep["removed"]["replies"] >= 2, rep)
    created.remove(a["room_id"])
    check("delete: members are told first, then disconnected with code 4004",
          ann.until(of_type("room_deleted"), 5) is not None and cy.until(of_type("room_deleted"), 5) is not None
          and (ann.pump(2), cy.pump(2)) and ann.closed == 4004 and cy.closed == 4004, (ann.closed, cy.closed))
    st, e = http("GET", f"/v1/rooms/{a['room_id']}")
    check("delete: the room is gone (404)", st == 404, st)
    check("delete: its transcript is gone", http("GET", f"/v1/conversations/{g['conversation_id']}")[0] == 404)
    check("delete: its decision record is gone", reply_decision is None or http("GET", f"/v1/decisions/{reply_decision}")[0] == 404, reply_decision)
    st, e = http("DELETE", f"/v1/rooms/{a['room_id']}")
    check("delete: deleting again is a 404", st == 404, st)
    try:
        Member(a["room_id"], "Dee"); check("delete: connecting to a deleted room is refused", False, "connected")
    except websocket.WebSocketBadStatusException as ex:
        check("delete: connecting to a deleted room is refused", ex.status_code == 404, ex.status_code)
    st, lst = http("GET", "/v1/rooms")
    check("delete: the list no longer holds it", a["room_id"] not in [r["room_id"] for r in lst])
    # ------------------------------------------------------------------ delete while the assistant is answering
    st, c = make_room(f"T-{tag} Busy")
    m = Member(c["room_id"], "Eve"); m.pump(1.0)
    m.send(type="user_message", payload={"content": "Write a detailed essay about the history of the printing press.", "force_mode": "fast"})
    busy = m.until(of_type("queue", busy=True), 30)
    time.sleep(1.5)
    t0 = time.time()
    st, rep = http("DELETE", f"/v1/rooms/{c['room_id']}")
    took = time.time() - t0
    created.remove(c["room_id"])
    check("delete while answering: succeeds promptly (the turn is cancelled)", st == 200 and took < 15, f"{st} in {took:.1f} s")
    check("delete while answering: the member is told and disconnected", m.until(of_type("room_deleted"), 5) is not None and (m.pump(2) or True) and m.closed == 4004, m.closed)
    time.sleep(1.0)
    orphans = None
    if os.path.exists(DB):
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
        orphans = (db.execute("SELECT COUNT(*) FROM replies WHERE conversation_id=?", (c["conversation_id"],)).fetchone()[0]
                   + db.execute("SELECT COUNT(*) FROM conversations WHERE id=?", (c["conversation_id"],)).fetchone()[0]
                   + db.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (c["conversation_id"],)).fetchone()[0])
        db.close()
    check("delete while answering: nothing of the room is left in the database", orphans in (0, None), f"orphan rows: {orphans}")
    st, g = http("GET", "/v1/conversations")
    check("delete while answering: no stray conversation appeared in the chat list", c["conversation_id"] not in [x["conversation_id"] for x in g])

    if SLOW:
        # -------------------------------------------------------------- heartbeat, capacity, queue
        st, s = make_room(f"T-{tag} Slow")
        a1 = Member(s["room_id"], "Hb"); a1.pump(1.0)
        hb = a1.until(lambda f: f["type"] == "heartbeat", 30)
        check("slow ws: a heartbeat arrives within 30 s", hb is not None)
        a1.close()
        st, f = make_room(f"T-{tag} Full")
        crowd = []
        try:
            for i in range(64):
                crowd.append(Member(f["room_id"], f"m{i}"))
            check("slow ws: 64 members fit", True)
            try:
                over = Member(f["room_id"], "extra")
                check("slow ws: the 65th member is refused", False, "connected")
            except websocket.WebSocketBadStatusException as ex:
                check("slow ws: the 65th member is refused (400)", ex.status_code == 400, ex.status_code)
        finally:
            for mm in crowd:
                mm.close()
        st, q = make_room(f"T-{tag} Queue")
        qm = Member(q["room_id"], "Flo"); qm.pump(1.0)
        for i in range(6):
            qm.send(type="user_message", payload={"content": f"Count to {i + 3}.", "force_mode": "fast"})
        errs = []
        end = time.time() + 6
        while time.time() < end:
            errs += [f for f in qm.pump(0.5) if f["type"] == "error"]
        codes = sorted({e["payload"]["code"] for e in errs})
        check("slow ws: beyond four quick messages the sender is told the queue or rate limit is full", codes and set(codes) <= {"E_QUEUE_FULL", "E_RATE_LIMITED"}, codes)
        qm.close()


try:
    main()
finally:
    for rid in list(created):
        http("DELETE", f"/v1/rooms/{rid}")
    left = [r for r in http("GET", "/v1/rooms")[1] if r["room_id"] in created]
    check("cleanup: every room this script created was deleted", not left, f"{len(created)} to clean, {len(left)} left")
    failed = [r for r in results if not r["pass"]]
    json.dump({"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base": BASE, "slow": SLOW,
               "passed": len(results) - len(failed), "failed": len(failed), "results": results},
              open(os.path.join(ROOT, "benchmarks", "rooms_test.json"), "w"), indent=1)
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)
