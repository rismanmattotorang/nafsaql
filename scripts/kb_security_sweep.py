#!/usr/bin/env python3
"""Live security sweep of the Knowledge Base uploads and deletion (Phase 8, SR-K01..SR-K04, SR-K11..SR-K13, NFR-K07).

Run it against a gateway started with small limits so every refusal can be provoked cheaply (scripts/kb_security_gateway.sh
starts one on :8093 with its own SurrealDB database and ledger). The limits it expects are the ones that script sets:
`max_upload_mb = 2`, `max_request_mb = 3`, `max_documents_per_base = 5`, `max_chunks_per_document = 60`,
`concurrent_uploads = 1`, `parse_timeout_s = 120`.

What it checks:
  - a hostile file name never leaves the base's folder, and the stored name is ours;
  - the content must match the extension;
  - an over-large file is refused without the whole of it being held, and a request over the request cap is refused;
  - a zip bomb, an XML entity bomb, a scripted web page and a damaged PDF end as a document error or as plain text,
    never as a crash, and the server keeps answering;
  - a document that makes too many passages is refused with a request to split it;
  - the number of documents in a base is capped;
  - a second upload while one is being received is told to try again (429), not queued without bound;
  - deleting a document, and then the base, removes the stored files, the passages, and the passages copied into the
    ledger's stored replies; nothing of the base is left on disk.

Usage: python3 -I scripts/kb_security_sweep.py [--base http://127.0.0.1:8093] [--ledger run/kb-security/nafs.sqlite]
       [--files run/kb-security/files] [--chat]

`--chat` adds the ledger checks, which need one real chat answer from the quick model (about 10 s to a few minutes).
Exit code 0 when every check passes.
"""
import argparse, io, json, os, socket, sqlite3, sys, threading, time, urllib.parse, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_sweep import call, check, multipart, results, skip  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def docx(parts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in parts:
            z.writestr(name, text)
    return buf.getvalue()


def wait_state(base, kb, doc, limit=180):
    """The document once it is ready or failed (or the last state seen)."""
    last, t = None, 0
    while t < limit:
        s, r = call(base, "GET", f"/v1/kb/{kb}/documents/{doc}")
        last = r if s == 200 else last
        if isinstance(last, dict) and last.get("status") in ("ready", "failed"):
            return last
        time.sleep(1)
        t += 1
    return last


def upload(base, kb, name, content):
    body, ctype = multipart([(name, content)])
    return call(base, "POST", f"/v1/kb/{kb}/documents", raw=body, ctype=ctype)


def slow_upload(host, port, path, boundary, hold, started):
    """Send the start of a multipart request and keep it open: a request that is still being received."""
    s = socket.create_connection((host, port), timeout=60)
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"slow.md\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n# Slow\n\nA slow upload.\n").encode()
    total = len(head) + 200
    s.sendall((f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\n"
               f"Content-Length: {total + 2000}\r\nConnection: close\r\n\r\n").encode() + head)
    started.set()
    time.sleep(hold)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8093")
    ap.add_argument("--ledger", default=os.path.join(ROOT, "run/kb-security/nafs.sqlite"))
    ap.add_argument("--files", default=os.path.join(ROOT, "run/kb-security/files"))
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = a.base
    u = urllib.parse.urlparse(base)

    s, st = call(base, "GET", "/v1/kb/status")
    if s != 200 or not st.get("enabled"):
        print("knowledge is off on this server; nothing to sweep")
        return 1
    lim = st.get("limits", {})
    made = []

    def new_kb(label):
        """A scratch base per section, so the cap on documents in one section cannot refuse another's uploads."""
        s, r = call(base, "POST", "/v1/kb", {"name": f"security sweep {label} {int(time.time())}", "description": "scratch"})
        made.append(r["id"])
        return r["id"]

    kb = new_kb("names")
    kbdir = os.path.join(a.files, kb)

    # ---- file names and content
    s, r = upload(base, kb, "../../../etc/passwd-notes.md", b"# Notes\n\nSimple text about trains.\n")
    d = (r.get("documents") or [{}])[0] if isinstance(r, dict) else {}
    check("name: a path in the file name is reduced to its last part", d.get("filename") == "passwd-notes.md", d.get("filename"))
    stored = os.listdir(kbdir) if os.path.isdir(kbdir) else []
    check("name: the stored file is named by its hash, inside the base's folder",
          stored and all(len(f.split(".")[0]) == 32 and ".." not in f for f in stored), stored)
    for evil in ("..\\..\\windows\\system32\\x.md", "a\x00b.md", "x" * 400 + ".md", ".hidden.md"):
        s, r = upload(base, kb, evil, b"# T\n\nsome words " + evil[:8].encode("ascii", "ignore") + b"\n")
        ok = s in (202, 400, 422) and isinstance(r, dict)
        check(f"name: {evil[:24]!r} is handled without an error", ok, (s, str(r)[:100]))
    check("name: nothing was written outside the base's folder",
          sorted(os.listdir(a.files)) == sorted(set(os.listdir(a.files))) and not any(
              f.endswith(".md") for f in os.listdir(a.files) if os.path.isfile(os.path.join(a.files, f))), os.listdir(a.files))
    s, r = upload(base, kb, "fake.pdf", b"This is plain text, not a PDF.\n")
    check("type: text named .pdf is refused", s == 422 and "not a PDF" in json.dumps(r), (s, str(r)[:120]))
    s, r = upload(base, kb, "fake.md", docx([("word/document.xml", "<x/>")]))
    check("type: a zip named .md is refused", s == 422 and "does not match" in json.dumps(r), (s, str(r)[:120]))
    s, r = upload(base, kb, "fake.docx", b"just text")
    check("type: text named .docx is refused", s == 422, (s, str(r)[:120]))
    s, r = upload(base, kb, "run.exe", b"MZ\x90\x00")
    check("type: an unsupported extension is refused", s == 422, (s, str(r)[:120]))
    s, r = upload(base, kb, "bin.txt", b"abc\x00def" * 100)
    check("type: a binary file named .txt is refused", s == 422 and "binary" in json.dumps(r), (s, str(r)[:120]))

    # ---- sizes
    max_mb = lim.get("max_upload_mb", 2)
    big = b"# Big\n\n" + b"word " * (max_mb * 1024 * 1024 // 5 + 1000)
    t0 = time.time()
    s, r = upload(base, kb, "big.md", big)
    check(f"size: a file over {max_mb} MB is refused", s == 422 and "larger than the limit" in json.dumps(r), (s, str(r)[:140]))
    check("size: ... quickly, and the server still answers", time.time() - t0 < 30 and call(base, "GET", "/v1/kb/status")[0] == 200,
          f"{time.time() - t0:.1f}s")
    req_mb = lim.get("max_request_mb", 3)
    kb_size = new_kb("request")
    parts = [(f"part{i}.md", b"# P\n\n" + b"abc " * (700 * 1024 // 4)) for i in range(req_mb * 2 + 2)]
    body, ctype = multipart(parts)
    s, r = call(base, "POST", f"/v1/kb/{kb_size}/documents", raw=body, ctype=ctype)
    # The server may answer 413 or close the connection while the client is still sending; either is a refusal.
    check(f"size: a request over {req_mb} MB in all is refused as a whole",
          s == 413 or (s == 0 and "reset" in str(r).lower() or "broken pipe" in str(r).lower()), (s, str(r)[:120]))
    s, docs = call(base, "GET", f"/v1/kb/{kb_size}/documents")
    check("size: ... and none of its parts was stored", s == 200 and not docs.get("documents"), len(docs.get("documents", [])))

    kb = new_kb("content")

    # ---- hostile content
    bomb = docx([("word/document.xml", "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body><w:p><w:r><w:t>" + "a" * (6 * 1024 * 1024) + "</w:t></w:r></w:p></w:body></w:document>")])
    s, r = upload(base, kb, "bomb.docx", bomb)
    if s == 202:
        d = wait_state(base, kb, r["documents"][0]["id"])
        check("content: a zip bomb ends as a document error", d and d["status"] == "failed" and "zip bomb" in str(d.get("error")), (d or {}).get("error"))
    else:
        check("content: a zip bomb is refused", s in (413, 422), (s, str(r)[:100]))
    laughs = ("<?xml version=\"1.0\"?><!DOCTYPE d [<!ENTITY a \"aaaaaaaaaa\"><!ENTITY b \"&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;\">"
              "<!ENTITY c \"&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;\"><!ENTITY d \"&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;\">]>"
              "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body><w:p><w:r><w:t>start &d; end</w:t></w:r></w:p></w:body></w:document>")
    s, r = upload(base, kb, "laughs.docx", docx([("word/document.xml", laughs)]))
    d = wait_state(base, kb, r["documents"][0]["id"]) if s == 202 else None
    check("content: an XML entity bomb is not expanded", d and d["status"] == "ready" and d.get("n_chars", 0) < 500, (d or {}).get("n_chars"))
    html = b"<html><head><title>T</title><script>alert('SCRIPT-MARK')</script></head><body onload=\"x()\"><p>Plain words here <a href=\"javascript:evil('HREF-MARK')\">link text</a></p><iframe src=\"http://evil.example\"></iframe></body></html>"
    s, r = upload(base, kb, "page.html", html)
    d = wait_state(base, kb, r["documents"][0]["id"]) if s == 202 else None
    chunks = call(base, "GET", f"/v1/kb/{kb}/documents/{d['id']}/chunks")[1].get("chunks", []) if d else []
    text = " ".join(c.get("text", "") for c in chunks)
    check("content: a scripted web page keeps its words and loses its script and script links",
          d and "Plain words here" in text and "link text" in text and not any(m in text for m in ("SCRIPT-MARK", "HREF-MARK", "javascript:")), text[:160])
    real_pdf = os.path.join(ROOT, "docs/publication/nafs_aql_lab_results.pdf")
    if os.path.exists(real_pdf):
        data = open(real_pdf, "rb").read()
        s, r = upload(base, kb, "cut.pdf", data[: len(data) // 2])
        d = wait_state(base, kb, r["documents"][0]["id"]) if s == 202 else None
        check("content: a truncated PDF ends cleanly (ready or a document error) and the server answers",
              (d is None or d["status"] in ("ready", "failed")) and call(base, "GET", "/v1/kb/status")[0] == 200, (d or {}).get("status"))
    many = ("# Long\n\n" + "\n\n".join(f"Paragraph {i}. " + "filler words " * 80 for i in range(400))).encode()
    s, r = upload(base, kb, "long.md", many)
    d = wait_state(base, kb, r["documents"][0]["id"]) if s == 202 else None
    max_chunks = lim.get("max_chunks_per_document", 60)
    check(f"limit: a file making more than {max_chunks} passages is refused with a request to split it",
          d and d["status"] == "failed" and "split" in str(d.get("error")), (d or {}).get("error"))

    # ---- the number of documents
    kb = new_kb("cap")
    s, docs = call(base, "GET", f"/v1/kb/{kb}/documents")
    n = len(docs.get("documents", [])) if s == 200 else 0
    cap = lim.get("max_documents_per_base", 5)
    refused = None
    for i in range(cap + 2 - min(n, cap)):
        s, r = upload(base, kb, f"fill{i}.md", f"# Fill {i}\n\nUnique text number {i} {time.time()}.\n".encode())
        if s == 422 and "already holds" in json.dumps(r):
            refused = r
            break
    check(f"limit: a base holds at most {cap} documents; the next is refused", refused is not None, str(refused)[:140])

    # ---- one upload at a time is received; another is told to try again
    started = threading.Event()
    th = threading.Thread(target=slow_upload, args=(u.hostname, u.port, f"/v1/kb/{kb}/documents", "----slowboundary", 6, started))
    th.start()
    started.wait(5)
    time.sleep(1)
    s, r = upload(base, kb, "second.md", b"# Second\n\nText.\n")
    check("busy: an upload while another is being received is told to try again (429)", s == 429, (s, str(r)[:120]))
    th.join()

    # ---- deletion (on a base with a document of its own)
    kb = new_kb("delete")
    kbdir = os.path.join(a.files, kb)
    s, r = upload(base, kb, "notes.md", b"# Delete me\n\nThe quiet harbour holds seventeen small boats.\n")
    if s == 202:
        wait_state(base, kb, r["documents"][0]["id"])
    upload(base, kb, "other.md", b"# Other\n\nA second document that stays until the base goes.\n")
    s, docs = call(base, "GET", f"/v1/kb/{kb}/documents")
    victim = next((d for d in docs.get("documents", []) if d["status"] == "ready" and d["filename"] == "notes.md"), None)
    reply_id = None
    if a.chat and victim:
        s, r = call(base, "POST", "/v1/chat", {"messages": [{"role": "user", "content": "How many small boats does the quiet harbour hold?"}],
                                                "force_mode": "fast", "knowledge": {"kb": kb, "mode": "rag"}}, timeout=900)
        reply_id = r.get("reply_id") if isinstance(r, dict) else None
        hit = any(x.get("doc_id") == victim["id"] for x in (r.get("sources") or []) if isinstance(x, dict)) if isinstance(r, dict) else False
        check("ledger: a chat answer stored the passages it was answered from", reply_id and os.path.exists(a.ledger) and hit, (s, reply_id))
    s, r = call(base, "DELETE", f"/v1/kb/{kb}/documents/{victim['id']}") if victim else (0, "no ready document")
    check("delete: a document is deleted", s == 200 and r.get("deleted"), (s, r))
    if victim:
        files_before = len(os.listdir(kbdir))
        check("delete: its stored original is gone", not any(f.startswith(victim["sha256"][:32]) for f in os.listdir(kbdir)), files_before)
        s, _ = call(base, "GET", f"/v1/kb/{kb}/documents/{victim['id']}")
        check("delete: ... and it can no longer be fetched", s == 404, s)
        if reply_id and os.path.exists(a.ledger):
            con = sqlite3.connect(a.ledger)
            js = con.execute("SELECT json FROM replies WHERE id = ?", (reply_id,)).fetchone()[0]
            con.close()
            check("delete: the document's passages are gone from the stored reply",
                  victim["id"] not in js and "kb_deleted" in js, js[:120])
    s, r = call(base, "DELETE", f"/v1/kb/{kb}")
    check("delete: the base is deleted", s == 200 and r.get("deleted"), (s, str(r)[:160]))
    check("delete: its folder, with every original, is gone from disk", not os.path.exists(kbdir), os.path.exists(kbdir))
    s, _ = call(base, "GET", f"/v1/kb/{kb}")
    check("delete: the base can no longer be fetched", s == 404, s)
    s, r = call(base, "POST", f"/v1/kb/{kb}/search", {"query": "words", "mode": "lexical"})
    check("delete: a search of it finds nothing", s == 404 or not (r.get("results") if isinstance(r, dict) else []), (s, str(r)[:100]))
    if reply_id and os.path.exists(a.ledger):
        con = sqlite3.connect(a.ledger)
        js = con.execute("SELECT json FROM replies WHERE id = ?", (reply_id,)).fetchone()[0]
        con.close()
        check("delete: no passage of the base is left in the ledger", f'"kb":"{kb}"' not in js.replace('"kb": "', '"kb":"') or "kb_deleted" in js, js[:120])
    s, r = call(base, "DELETE", f"/v1/kb/{kb}")
    check("delete: deleting it again is a 404, not a second deletion", s == 404, s)
    s, r = call(base, "DELETE", "/v1/kb/..%2F..%2Fetc")
    check("delete: a path in the id deletes nothing", s in (400, 404), s)
    check("delete: the files folder itself is intact", os.path.isdir(a.files), a.files)

    for k in made:
        call(base, "DELETE", f"/v1/kb/{k}")
    n_fail = sum(1 for x in results if x["result"] == "FAIL")
    print(f"\nkb security sweep: {sum(1 for x in results if x['result'] == 'pass')} passed, {n_fail} failed, "
          f"{sum(1 for x in results if x['result'] == 'skip')} skipped")
    if a.out:
        json.dump({"results": results}, open(a.out, "w"), indent=1)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
