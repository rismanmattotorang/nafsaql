#!/usr/bin/env python3
"""Live check of the memory of earlier conversations (Phase 26) against a running gateway.

Tells the assistant two facts in two conversations, then asks for them from other conversations, in the same words and
in other words; checks that an impersonal question, a private (ephemeral) request and the conversation that holds the
fact do not read the memory; forgets a note and checks that it is gone; and deletes everything it created.

Usage: python3 scripts/recall_check.py [--base http://127.0.0.1:8080] [--out benchmarks/kb_phase26/recall_check.json]
"""
import argparse, json, os, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def call(base, method, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def ask(base, text, conv=None, **extra):
    body = {"messages": [{"role": "user", "content": text}], "force_mode": "fast", **extra}
    if conv:
        body["conversation_id"] = conv
    t = time.time()
    r = call(base, "POST", "/v1/chat", body)
    src = r.get("sources") or []
    recall = next((s["recall"] for s in src if "recall" in s), [])
    stage = next((s.get("ms") for s in (r.get("usage") or {}).get("stages") or [] if s.get("stage") == "recall"), None)
    return {"conversation": r.get("conversation_id"), "answer": r.get("message", {}).get("content", ""),
            "recall": recall, "notes": [s["note"] for s in src if "note" in s], "recall_ms": stage,
            "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="benchmarks/kb_phase26/recall_check.json")
    a = ap.parse_args()
    base, checks, convs = a.base, [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:140]}]" if detail else ""), flush=True)

    before = call(base, "GET", "/v1/memory?limit=1")
    check("the memory is on", before.get("enabled"), f"{before.get('count')} notes held before the check")

    # Two facts, two conversations.
    r1 = ask(base, "Remember that the staging server of project Halcyon listens on port 8443.")
    convs.append(r1["conversation"])
    check("a 'Remember that' message is saved as a note", any("Saved to memory" in n for n in r1["notes"]), r1["answer"][:120])
    r2 = ask(base, "My sister's dog is called Biscuit and he is a beagle. Which three breeds are similar to a beagle?")
    convs.append(r2["conversation"])
    check("a turn that states a fact reads no memory when none matches", not any("Biscuit" in n.get("text", "") for n in r2["recall"]))

    # The notes get their vectors after the reply.
    deadline = time.time() + 180
    while time.time() < deadline:
        m = call(base, "GET", "/v1/memory?limit=20")
        if m["count"] >= before["count"] + 2:
            break
        time.sleep(2)
    check("both facts are held as notes", m["count"] >= before["count"] + 2, f"{m['count']} notes")
    time.sleep(8)

    # Asked first, while the only note about the dog is the one of this very conversation.
    r8 = ask(base, "What did I say my sister's dog is called?", conv=r2["conversation"])
    check("the conversation that holds the fact answers from its own history, not from the memory",
          not any("Biscuit" in n.get("text", "") for n in r8["recall"]) and "biscuit" in r8["answer"].lower(), r8["answer"][:100])

    r3 = ask(base, "Which port does my staging server for Halcyon listen on?")
    convs.append(r3["conversation"])
    check("a fact told in one conversation is recalled in another", any("8443" in n.get("text", "") for n in r3["recall"]),
          f"recall {r3['recall_ms']} ms")
    check("and the answer uses it", "8443" in r3["answer"], r3["answer"][:120])

    r4 = ask(base, "What is my sister's dog called?")
    convs.append(r4["conversation"])
    check("a fact stated inside a turn is recalled", any("Biscuit" in n.get("text", "") for n in r4["recall"]), f"recall {r4['recall_ms']} ms")
    check("and the answer uses it", "biscuit" in r4["answer"].lower(), r4["answer"][:120])

    r5 = ask(base, "Remind me: which pet does my sibling own, and what is its name?")
    convs.append(r5["conversation"])
    check("a question in other words finds the note (meaning arm)", any("Biscuit" in n.get("text", "") for n in r5["recall"]),
          f"recall {r5['recall_ms']} ms; answer: {r5['answer'][:80]}")

    r6 = ask(base, "What is the capital of France?")
    convs.append(r6["conversation"])
    check("an impersonal question does not read the memory", not r6["recall"] and r6["recall_ms"] is None)

    r7 = ask(base, "What is my sister's dog called?", ephemeral=True)
    check("a private request does not read the memory", not r7["recall"])

    # Forget the port note, then ask again.
    items = call(base, "GET", "/v1/memory?limit=200")["items"]
    mine = [x for x in items if "8443" in x["text"] or "Biscuit" in x["text"] or "Halcyon" in x["text"]]
    for x in [x for x in mine if x["kind"] == "note" and "8443" in x["text"]]:
        call(base, "DELETE", f"/v1/memory/{x['id']}")
    # The conversation that asked about the port holds the answer as a note of its own: it goes with the conversation.
    call(base, "DELETE", f"/v1/conversations/{r3['conversation']}")
    convs.remove(r3["conversation"])
    r9 = ask(base, "Which port does my staging server for Halcyon listen on?")
    convs.append(r9["conversation"])
    check("a forgotten note, and the notes of a deleted conversation, are not recalled",
          not any("8443" in n.get("text", "") for n in r9["recall"]), r9["answer"][:100])

    # Clean up: the conversations (their notes go with them) and what is left of the test's notes.
    for c in [c for c in convs if c]:
        try:
            call(base, "DELETE", f"/v1/conversations/{c}")
        except Exception as e:  # noqa: BLE001
            print("could not delete", c, e)
    for x in call(base, "GET", "/v1/memory?limit=200")["items"]:
        if any(w in x["text"] for w in ("8443", "Biscuit", "Halcyon")):
            call(base, "DELETE", f"/v1/memory/{x['id']}")
    after = call(base, "GET", "/v1/memory?limit=1")
    check("deleting the conversations removes their notes", after["count"] == before["count"], f"{after['count']} notes after, {before['count']} before")

    load = os.getloadavg()
    out = {"what": "Live check of the memory of earlier conversations (Phase 26)", "date": time.strftime("%Y-%m-%d %H:%M"),
           "host_load_average": [round(x, 1) for x in load], "note": "SHARED MACHINE" if load[0] > 4 else "quiet host",
           "passed": sum(c["ok"] for c in checks), "checks": checks,
           "recall_ms": [r["recall_ms"] for r in (r3, r4, r5, r9) if r["recall_ms"] is not None]}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"{out['passed']}/{len(checks)} passed; wrote {a.out}")
    return 0 if out["passed"] == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
