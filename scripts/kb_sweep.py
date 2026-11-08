#!/usr/bin/env python3
"""Live sweep of the Knowledge Base API and the chat knowledge modes (Phase 20, FR-K*).

Covers what the other live suites do not: the `/v1/kb/*` routes, ingestion end to end on a scratch knowledge base,
the three search modes, the graph routes, the Memorilla training routes, and a chat answer in every knowledge mode.
Mutating checks run on a scratch base the sweep creates and deletes; the chat and graph checks read an existing base,
which must have passages (and, for the graph checks, a built graph).

Usage: python3 -I scripts/kb_sweep.py [--base http://127.0.0.1:8080] [--kb <id>] [--no-chat] [--out logs/kb_sweep.json]

Exit code 0 when every check passes. With no --kb the sweep uses the first base the server lists, and skips the chat
and graph checks if there is none.
"""
import argparse, json, os, random, string, sys, time, urllib.error, urllib.parse, urllib.request

PASS, FAIL, SKIP = "pass", "FAIL", "skip"
results = []


def call(base, method, path, body=None, raw=None, ctype=None, timeout=1200):
    """Returns (status, parsed-or-text). Never raises for an HTTP error status."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"Content-Type": ctype or "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(text) if text.strip().startswith(("{", "[")) else text)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        return e.code, (json.loads(text) if text.strip().startswith(("{", "[")) else text)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, str(e)


def check(name, ok, detail=""):
    results.append({"check": name, "result": PASS if ok else FAIL, "detail": str(detail)[:300]})
    print(f"{PASS if ok else FAIL:5} {name}" + (f"  — {str(detail)[:140]}" if detail else ""))
    return bool(ok)


def skip(name, why):
    results.append({"check": name, "result": SKIP, "detail": why})
    print(f"{SKIP:5} {name}  — {why}")


def multipart(fields):
    """A multipart/form-data body for (filename, bytes) pairs under the field name `files`."""
    b = "----nafs" + "".join(random.choices(string.ascii_letters, k=16))
    out = b""
    for filename, content in fields:
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\n".encode()
        out += b"Content-Type: application/octet-stream\r\n\r\n" + content + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


DOC = b"""# Sweep fixture

The sweep marker word is zarafshan. The fixture server listens on port 65201.

## Second heading

A passage about the marker word zarafshan and nothing else of interest.
"""


def status_checks(base):
    s, st = call(base, "GET", "/v1/kb/status")
    if not check("status: 200 and knowledge enabled", s == 200 and st.get("enabled"), st if s != 200 else ""):
        return None
    check("status: the store is reachable", st.get("reachable"), st.get("error"))
    check("status: upload limits are reported",
          isinstance(st.get("limits", {}).get("max_upload_mb"), int), st.get("limits"))
    check("status: the file kinds are reported", "pdf" in (st.get("kinds") or []), st.get("kinds"))
    check("status: memory is a boolean", isinstance(st.get("memory"), bool), st.get("memory"))
    return st


def base_crud(base):
    """Create, read, rename and list a scratch base. Returns its id, or None."""
    s, r = call(base, "POST", "/v1/kb", {"name": "Sweep scratch", "description": "temporary"})
    if not check("create: 201 with an id", s in (200, 201) and isinstance(r, dict) and r.get("id"), r):
        return None
    kb = r["id"]
    s, r = call(base, "GET", f"/v1/kb/{kb}")
    check("read: the base comes back with its name", s == 200 and r.get("name") == "Sweep scratch", r)
    s, r = call(base, "PATCH", f"/v1/kb/{kb}", {"name": "Sweep renamed"})
    check("rename: 200", s == 200, r)
    s, r = call(base, "GET", "/v1/kb")
    bases = {b["id"]: b for b in (r.get("bases") or [])} if s == 200 else {}
    check("list: the renamed base is listed", bases.get(kb, {}).get("name") == "Sweep renamed", list(bases))
    check("list: every base says whether it has a memory module",
          all(isinstance(b.get("memory"), bool) for b in bases.values()), list(bases.values())[:1])
    s, r = call(base, "GET", "/v1/kb/this-id-does-not-exist")
    check("read: an unknown base is 404", s == 404, (s, r))
    return kb


def ingest(base, kb):
    """Upload the fixture, wait for it to be ready, and check what was stored. Returns the document, or None."""
    body, ctype = multipart([("sweep_fixture.md", DOC)])
    s, r = call(base, "POST", f"/v1/kb/{kb}/documents", raw=body, ctype=ctype)
    if not check("upload: accepted, with nothing refused",
                 s in (200, 201, 202) and (r.get("documents") or []) and not (r.get("refused") or []), r):
        return None
    doc, waited = None, 0
    while waited < 300:
        s, r = call(base, "GET", f"/v1/kb/{kb}/documents")
        docs = r.get("documents") or [] if s == 200 else []
        doc = docs[0] if docs else None
        if doc and doc.get("status") in ("ready", "failed"):
            break
        time.sleep(3)
        waited += 3
    if not check("ingest: the document reaches ready", doc and doc["status"] == "ready",
                 (doc or {}).get("error") or (doc or {}).get("status")):
        return None
    check("ingest: it was split into passages", doc.get("n_chunks", 0) >= 1, doc.get("n_chunks"))
    check("ingest: the title came from the heading", doc.get("title") == "Sweep fixture", doc.get("title"))
    s, r = call(base, "GET", f"/v1/kb/{kb}/documents/{doc['id']}/chunks")
    chunks = r.get("chunks") or [] if s == 200 else []
    check("passages: the text is stored", any("zarafshan" in c.get("text", "") for c in chunks), len(chunks))
    # A second upload of the same bytes is a duplicate, not a second copy.
    body, ctype = multipart([("sweep_fixture.md", DOC)])
    s, r = call(base, "POST", f"/v1/kb/{kb}/documents", raw=body, ctype=ctype)
    dup = (r.get("documents") or [{}])[0] if isinstance(r, dict) else {}
    check("upload: the same file again is reported as a duplicate", s in (200, 202) and dup.get("duplicate"), dup)
    return doc


def search(base, kb, embedder):
    for mode in ("lexical", "vector", "hybrid"):
        if mode != "lexical" and not embedder:
            skip(f"search ({mode})", "embeddings are off on this server")
            continue
        s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "What is the marker word?", "k": 5, "mode": mode})
        hits = r.get("results") or [] if s == 200 else []
        check(f"search ({mode}): the fixture passage is found",
              any("zarafshan" in h.get("text", "") for h in hits), (s, len(hits)))
    s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "zarafshan", "k": 0})
    check("search: k below the range is brought into it", s == 200 and len(r.get("results") or []) >= 1, (s, r))
    s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "zarafshan", "k": 100000})
    check("search: k above the range is capped", s == 200 and len(r.get("results") or []) <= 100, (s, len(r.get("results") or [])))
    s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "x", "mode": "nonsense"})
    check("search: an unknown mode is refused", s == 400, (s, r))
    s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "x" * 2001})
    check("search: an over-long query is refused", s == 400, (s, str(r)[:120]))


def graph_checks(base, kb):
    s, r = call(base, "GET", f"/v1/kb/{kb}/graph?limit=20")
    if s != 200 or not isinstance(r, dict):
        return skip("graph", f"GET /graph returned {s}")
    nodes = r.get("nodes") or []
    n = (r.get("stats") or {}).get("entities") or 0
    if not n:
        return skip("graph", "this base has no built graph")
    check("graph: the view carries nodes, edges and counts", nodes and "edges" in r and "progress" in r,
          {"nodes": len(nodes), "stats": r.get("stats")})
    name = nodes[0]["name"].split()[-1]
    s, r2 = call(base, "GET", f"/v1/kb/{kb}/graph/entities?q={urllib.parse.quote(name)}")
    ents = r2.get("entities") or [] if s == 200 else []
    check("graph: an entity can be looked up by name", s == 200 and len(ents) > 0, (s, name, len(ents)))
    s, r3 = call(base, "GET", f"/v1/kb/{kb}/graph/entities/{nodes[0]['id']}")
    check("graph: one entity's relations come with the passages that state them",
          s == 200 and "facts" in r3 and "passages" in r3, (s, list(r3) if isinstance(r3, dict) else r3))
    s, r4 = call(base, "GET", f"/v1/kb/{kb}/graph/entities/deadbeefdeadbeefdeadbeefd")
    check("graph: an unknown entity is 404 or empty",
          s == 404 or (s == 200 and not (r4.get("facts") or [])), (s, str(r4)[:120]))


def memory_runs(base, kb):
    s, r = call(base, "GET", f"/v1/kb/{kb}/memory/runs")
    if not check("memory runs: the list is served", s == 200 and "runs" in r and "trainer" in r, (s, r)):
        return
    check("memory runs: the list says whether the trainer is built", isinstance(r.get("trainer"), bool), r.get("trainer"))
    s, r = call(base, "POST", f"/v1/kb/{kb}/memory/runs", {"kind": "nonsense"})
    check("memory runs: an unknown kind is refused", s == 400, (s, r))
    s, r = call(base, "POST", f"/v1/kb/{kb}/memory/runs/no-such-run/stop")
    check("memory runs: stopping an unknown run is 404", s == 404, (s, r))
    s, r = call(base, "POST", f"/v1/kb/{kb}/memory/runs/no-such-run/activate")
    check("memory runs: activating a run with no checkpoint is refused", s == 400, (s, r))
    s, r = call(base, "GET", "/v1/kb/this-id-does-not-exist/memory/runs")
    check("memory runs: an unknown base is 404", s == 404, (s, r))


def chat_modes(base, kb, st):
    """One chat answer per knowledge mode, with the sources the reply must carry."""
    memory_ok = bool(st.get("memory"))
    modes = ["off", "rag", "graph", "auto"] + (["memory", "hybrid"] if memory_ok else [])
    for mode in modes:
        body = {"messages": [{"role": "user", "content": "What does this knowledge base cover?"}],
                "ephemeral": True, "retrieve": False, "force_mode": "fast",
                "knowledge": {"kb": kb, "mode": mode}}
        s, r = call(base, "POST", "/v1/chat", body)
        if not check(f"chat ({mode}): 200 with an answer",
                     s == 200 and (r.get("message") or {}).get("content"), (s, str(r)[:200])):
            continue
        src = r.get("sources") or []
        notes = [x["note"] for x in src if "note" in x]
        if mode == "off":
            check("chat (off): no sources are claimed", not src, src)
        elif mode in ("rag", "graph", "auto"):
            check(f"chat ({mode}): passages or a reason are reported",
                  any("chunk_id" in x for x in src) or notes, src[:1])
        else:
            mem = next((x["memory"] for x in src if "memory" in x), None)
            check(f"chat ({mode}): memory tokens were read, or it fell back and said so",
                  (mem and mem.get("tokens")) or notes, mem or notes)
            if mode == "hybrid" and mem:
                check("chat (hybrid): passages come with the memory tokens",
                      any("chunk_id" in x for x in src), src)
        check(f"chat ({mode}): the quick path answered",
              r.get("reasoning_mode") in ("fast", "slow"), r.get("reasoning_mode"))
    s, r = call(base, "POST", "/v1/chat", {"messages": [{"role": "user", "content": "hello"}], "ephemeral": True,
                                           "knowledge": {"kb": "this-id-does-not-exist", "mode": "rag"}})
    note = [x.get("note") for x in (r.get("sources") or [])] if s == 200 else []
    check("chat: an unknown base is reported in the sources, not a 500", s == 200 and note, (s, note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--kb", default="", help="an existing base for the chat and graph checks")
    ap.add_argument("--no-chat", action="store_true", help="skip the chat checks (they take minutes per mode)")
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch base")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    st = status_checks(a.base)
    if st is None:
        print("knowledge is off on this server; nothing to sweep")
        return 1
    embedder = bool((st.get("embedder") or {}).get("reachable"))

    scratch = base_crud(a.base)
    if scratch:
        doc = ingest(a.base, scratch)
        if doc:
            search(a.base, scratch, embedder)
        memory_runs(a.base, scratch)
        if not a.keep:
            s, r = call(a.base, "DELETE", f"/v1/kb/{scratch}")
            check("delete: the scratch base is removed with its documents and passages",
                  s in (200, 204) and (r.get("deleted") if isinstance(r, dict) else True), (s, r))
            s, _ = call(a.base, "GET", f"/v1/kb/{scratch}")
            check("delete: it is gone afterwards", s == 404, s)

    kb = a.kb
    if not kb:
        s, r = call(a.base, "GET", "/v1/kb")
        kb = next((b["id"] for b in (r.get("bases") or []) if b.get("chunks")), "") if s == 200 else ""
    if kb:
        graph_checks(a.base, kb)
        if a.no_chat:
            skip("chat", "--no-chat")
        else:
            chat_modes(a.base, kb, st)
    else:
        skip("graph", "no base with passages")
        skip("chat", "no base with passages")

    passed = sum(1 for r in results if r["result"] == PASS)
    failed = sum(1 for r in results if r["result"] == FAIL)
    skipped = sum(1 for r in results if r["result"] == SKIP)
    print(f"\nkb sweep: {passed} passed, {failed} failed, {skipped} skipped")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "base": a.base, "kb": kb,
                   "host_load_average": [round(x, 1) for x in os.getloadavg()],
                   "passed": passed, "failed": failed, "skipped": skipped, "checks": results},
                  open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
