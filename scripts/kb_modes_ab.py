#!/usr/bin/env python3
"""Knowledge modes in chat, paired (Phase 20, Phase 7, FR-K24): `off`, `rag`, `graph`, `memory`, `hybrid` and `auto`
answered by the same model on the same questions, interleaved so a loaded host hits every arm alike.

A question counts as answered when the expected phrase appears in the reply (the same phrases the retrieval evaluation
uses, matched on the answer instead of the passage). Arms are compared with an exact McNemar test on the paired
verdicts; latency is reported as the paired median ratio against `rag`.

Usage: python3 -I scripts/kb_modes_ab.py --base http://127.0.0.1:8080 --kb <id> [--modes off,rag,graph,memory,hybrid]
                                         [--limit N] [--out benchmarks/kb_modes_ab.json]
"""
import argparse, json, math, os, re, statistics, sys, time, urllib.error, urllib.request

# (question, phrases that mark a right answer). Held out from training: the trainer sees reconstruction and claim rows,
# never these questions. They are paraphrased, so an arm must read the base rather than repeat its words; the phrases
# are the facts themselves, so a right answer contains one however it is worded.
QUESTIONS = [
    ("How many memory tokens does the module give for each question?", ["sixteen", "16"]),
    ("What do the memory tokens replace in the prompt?", ["placeholder"]),
    ("How many cross-attention refinement blocks read the documents?", ["two", " 2 "]),
    ("Which inference engine accepts caller-supplied input vectors?", ["smiley"]),
    ("Which port does the gateway use to reach SurrealDB?", ["8300"]),
    ("How large may each uploaded file be?", ["fifty", "50"]),
    ("How many files may one upload carry?", ["twenty", "20"]),
    ("How many hops does local search walk over the entity graph?", ["one or two", "two hops", "1 or 2", "1-2"]),
    ("What is removed when a knowledge base is deleted?", ["stored files", "files"]),
    ("Which crate implements the orchestrator and the gate?", ["nafs-core"]),
    ("Which crate holds the exact verifiers?", ["nafs-verify"]),
    ("Which crate is the HTTP gateway?", ["nafs-gateway"]),
    ("Which embedding model builds the retrieval index?", ["qwen3-embedding-0.6b", "qwen3 embedding 0.6b"]),
    ("Which model does Profile A use on the fast path?", ["qwen3-1.7b", "qwen3 1.7b"]),
    ("Which model does Profile A use on the thorough path?", ["qwen3-4b", "qwen3 4b"]),
    ("What share of traffic goes into the audit stream?", ["5%", "5 %", "five per cent", "five percent"]),
    ("Which database does the conversation store use?", ["sqlite"]),
    ("Which figure of al-Ghazali's does the orchestrator play?", ["king"]),
    ("Which role does the controller play in the same picture?", ["vizier"]),
    ("Which kind of vector index does SurrealDB hold for the passages?", ["hnsw"]),
]


def http(base, path, body, timeout=1200):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def norm(s):
    """Lower case, with punctuation that carries meaning kept: a crate is `nafs-core`, a score is `84.3`, a share
    is `5%`. Everything else becomes a space, so markup around a word does not hide it."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%-]+", " ", (s or "").lower()))


def correct(answer, phrases):
    a = norm(answer)
    return any(norm(p) in a for p in phrases)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for the discordant counts b (A right, B wrong) and c (the other way)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def regrade(a):
    """Score a stored run's answers again with the current rules. The answers are kept in the file, so a correction
    to the grader does not cost another run; the summary is rebuilt from the new verdicts."""
    old = json.load(open(a.regrade))
    want = {q: p for q, p in QUESTIONS}
    changed = 0
    for row in old["rows"]:
        if row["question"] in want and not row.get("error"):
            was = row["ok"]
            row["ok"] = correct(row.get("answer", ""), want[row["question"]])
            changed += was != row["ok"]
    out = dict(old, rows=old["rows"], summary=summarise(old["rows"], old["modes"], old.get("reference", "rag")),
               regraded=time.strftime("%Y-%m-%d %H:%M"), regraded_changed=changed)
    for m, s in out["summary"].items():
        print(f"== {m:7} {s['correct']}/{s['n']}  median {s['median_seconds']}s"
              + (f"  vs {out['reference']}: {s['delta_points']:+} points, p {s['mcnemar_p']}, "
                 f"time x{s['median_time_ratio']}" if "vs_reference" in s else ""))
    print(f"{changed} verdicts changed")
    json.dump(out, open(a.out or a.regrade, "w"), indent=2)
    print("wrote", a.out or a.regrade)
    return 0


def summarise(rows, modes, reference):
    summary = {}
    by_mode = {m: {r["question"]: r for r in rows if r["mode"] == m} for m in modes}
    ref = by_mode.get(reference, {})
    for m in modes:
        d = by_mode[m]
        n = len(d)
        acc = sum(1 for r in d.values() if r["ok"])
        secs = sorted(r["seconds"] for r in d.values())
        anatomy = {}
        for r in d.values():
            f = r.get("fate") or ("answered" if r["ok"] else "not_retrieved")
            anatomy[f] = anatomy.get(f, 0) + 1
        assert sum(anatomy.values()) == n, "every question has exactly one fate"
        s = {"n": n, "correct": acc, "accuracy": round(acc / max(1, n), 3),
             "median_seconds": round(statistics.median(secs), 1) if secs else None,
             "mean_seconds": round(statistics.fmean(secs), 1) if secs else None,
             "errors": sum(1 for r in d.values() if r.get("error")), "anatomy": anatomy}
        if m != reference and ref:
            pairs = [(d[q]["ok"], ref[q]["ok"]) for q in d if q in ref]
            b = sum(1 for x, y in pairs if x and not y)
            c = sum(1 for x, y in pairs if y and not x)
            ratios = [d[q]["seconds"] / ref[q]["seconds"] for q in d if q in ref and ref[q]["seconds"] > 0]
            s.update({"vs_reference": reference, "pairs": len(pairs), "better": b, "worse": c,
                      "delta_points": round(100 * (b - c) / max(1, len(pairs)), 1),
                      "mcnemar_p": float(f"{mcnemar_exact(b, c):.3g}"),
                      "median_time_ratio": round(statistics.median(ratios), 2) if ratios else None})
        summary[m] = s
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--modes", default="off,rag,graph,memory,hybrid")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--reference", default="rag", help="arm the others are compared with")
    ap.add_argument("--force-mode", default="fast",
                    help="fast (comparable with memory, which is quick-model only), slow, auto or '' for the gate")
    ap.add_argument("--questions", default="",
                    help="a JSON file of [question, [phrases]] pairs replacing the built-in set")
    ap.add_argument("--out", default="")
    ap.add_argument("--regrade", default="",
                    help="score the answers of an earlier run again (no model calls) and write them to --out")
    a = ap.parse_args()
    if a.questions:
        QUESTIONS[:] = [(q, p) for q, p in json.load(open(a.questions))]
    if a.regrade:
        return regrade(a)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    qs = QUESTIONS[: a.limit] if a.limit else QUESTIONS
    rows = []
    for i, (q, phrases) in enumerate(qs):
        # Rotate the arm order per question so a drifting host load does not favour one arm.
        for mode in modes[i % len(modes):] + modes[: i % len(modes)]:
            body = {"messages": [{"role": "user", "content": q}], "ephemeral": True, "retrieve": False}
            if a.force_mode:
                body["force_mode"] = a.force_mode
            if mode != "off":
                # "rag@2" is the mode rag with two passages: the control that gives the passages alone as many as
                # `hybrid` gets, so whatever the memory tokens add is the difference between the two.
                name, _, k = mode.partition("@")
                body["knowledge"] = {"kb": a.kb, "mode": name, **({"k": int(k)} if k else {})}
            t = time.time()
            try:
                r = http(a.base, "/v1/chat", body)
                took, text = time.time() - t, r.get("message", {}).get("content", "")
                src = r.get("sources", []) or []
                # NFR-K11: one fate per question. answered; retrieved_not_used (the fact reached the prompt and
                # not the reply); not_retrieved (no retrieved passage states it). Closed-book has no retrieval, so
                # its only fates are answered and not_retrieved.
                ok = correct(text, phrases)
                in_prompt = any(correct(s.get("text", ""), phrases) for s in src if "chunk_id" in s)
                fate = "answered" if ok else ("retrieved_not_used" if in_prompt else "not_retrieved")
                row = {"question": q, "mode": mode, "ok": ok, "fate": fate, "seconds": round(took, 2),
                       "reasoning_mode": r.get("reasoning_mode"), "passages": sum(1 for s in src if "chunk_id" in s),
                       "memory_tokens": next((s["memory"]["tokens"] for s in src if "memory" in s), 0),
                       "resolved": next((s["memory"]["mode"] for s in src if "memory" in s),
                                        next((s.get("mode") for s in src if "chunk_id" in s), None)),
                       "note": next((s["note"] for s in src if "note" in s), None),
                       "answer": text[:400]}
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                row = {"question": q, "mode": mode, "ok": False, "seconds": round(time.time() - t, 2),
                       "error": str(e)[:200], "answer": ""}
            rows.append(row)
            print(f"{mode:7} {'OK ' if row['ok'] else '   '} {row['seconds']:7.1f}s  {q[:58]}")
    summary = summarise(rows, modes, a.reference)
    for m, st in summary.items():
        print(f"== {m:7} {st['correct']}/{st['n']}  median {st['median_seconds']}s"
              + (f"  vs {a.reference}: {st['delta_points']:+} points, p {st['mcnemar_p']}, "
                 f"time x{st['median_time_ratio']}" if "vs_reference" in st else "")
              + f"  fates {st['anatomy']}")
    load = os.getloadavg()
    out = {"what": "Knowledge modes in chat, paired on one question set (Phase 20, Phase 7)",
           "date": time.strftime("%Y-%m-%d %H:%M"), "kb": a.kb, "questions": len(qs), "modes": modes,
           "force_mode": a.force_mode or "gate", "reference": a.reference,
           "host_load_average": [round(x, 1) for x in load],
           "note": "SHARED MACHINE" if load[0] > 4 else "quiet host",
           "summary": summary, "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
