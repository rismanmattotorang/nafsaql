#!/usr/bin/env python3
"""Fewer, better-chosen sentences in a knowledge prompt: the same answers, sooner? (Phase 27.)

Starts two throw-away gateways on spare ports that differ only in `[knowledge] rag_k`, `rag_max_chars` and
`rag_compact_sources` (each arm is "k,chars,compact"; each gateway with
its own SQLite file, the label stream and audits off; the engines, the embedding server and the knowledge base are
the running ones), asks both the same questions through the quick model, alternating which goes first, and compares
right answers (the expected phrase is in the reply), prompt tokens, time to the first token and total time.

Usage: python3 scripts/rag_budget_ab.py --kb <id> [--a 4,4800,0] [--b 8,1800,1] [--questions scripts/kb_paper_questions_50.json]
                                        [--limit N] [--out benchmarks/kb_phase27/rag_budget_ab.json]
"""
import argparse, json, math, os, re, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%-]+", " ", (s or "").lower()))


def correct(answer, phrases):
    a = norm(answer)
    return any(norm(p) in a for p in phrases)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))


def config(name, port, k, chars, compact=1):
    text = open(os.path.join(ROOT, "config/nafs-aql.toml")).read()
    run = os.path.join(ROOT, "run/phase27")
    os.makedirs(run, exist_ok=True)

    def sub(pattern, repl, count=1):
        nonlocal text
        text, n = re.subn(pattern, repl, text, count=count, flags=re.M)
        assert n == count, (pattern, n)

    sub(r'^bind = "127\.0\.0\.1:8080"', f'bind = "127.0.0.1:{port}"')
    sub(r'^sqlite_path = .*$', f'sqlite_path = "run/phase27/{name}.sqlite"')
    sub(r'^trace_path = .*$', f'trace_path = "run/phase27/{name}.trace.jsonl"')
    sub(r'^cursor_file = .*$', f'cursor_file = "run/phase27/{name}.cursor.json"')
    sub(r'^rag_k = \d+', f'rag_k = {k}')
    sub(r'^rag_max_chars = \d+', f'rag_max_chars = {chars}')
    sub(r'^rag_compact_sources = \w+', f'rag_compact_sources = {"true" if compact else "false"}')
    # The label stream and the audit worker would answer practice tasks between the questions.
    sub(r'(^\[labels\][^\n]*\n)enabled = true', r'\1enabled = false')
    sub(r'(^\[audit\][^\n]*\n)enabled = true', r'\1enabled = false')
    path = os.path.join(run, f"{name}.toml")
    open(path, "w").write(text)
    return path


def ask(port, kb, q):
    body = {"messages": [{"role": "user", "content": q}], "ephemeral": True, "retrieve": False, "force_mode": "fast",
            "knowledge": {"kb": kb, "mode": "rag"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage") or {}
    src = d.get("sources") or []
    return {"answer": d.get("message", {}).get("content", ""), "seconds": round(time.time() - t, 2),
            "prompt_tokens": u.get("prompt_tokens") or u.get("input_tokens"), "ttft_ms": u.get("ttft_ms"),
            "passages": sum(1 for s in src if "chunk_id" in s),
            "support": next((s["support"]["share"] for s in src if "support" in s), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--a", default="4,4800,0")
    ap.add_argument("--b", default="8,1800,1")
    ap.add_argument("--questions", default="scripts/kb_paper_questions_50.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/kb_phase27/rag_budget_ab.json")
    a = ap.parse_args()
    # Each arm: passages searched, character budget, and (optionally) 1 or 0 for the compact source headers.
    arms = {"a": (8091, *map(int, a.a.split(","))), "b": (8092, *map(int, a.b.split(",")))}
    qs = json.load(open(os.path.join(ROOT, a.questions)))
    if a.limit:
        qs = qs[: a.limit]
    procs = []
    try:
        for name, (port, *settings) in arms.items():
            cfg = config(name, port, *settings)
            procs.append(subprocess.Popen([os.path.join(ROOT, "rust/target/release/nafs-gateway"), "--config", cfg], cwd=ROOT,
                                          stdout=open(os.path.join(ROOT, f"run/phase27/{name}.log"), "w"), stderr=subprocess.STDOUT))
        for port, *_ in arms.values():
            for _ in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
            else:
                raise SystemExit(f"gateway on :{port} did not start; see run/phase27/")
        rows = []
        for i, (q, phrases) in enumerate(qs):
            row = {"question": q}
            for place, name in enumerate(("a", "b") if i % 2 == 0 else ("b", "a")):
                try:
                    r = ask(arms[name][0], a.kb, q)
                    r["ok"] = correct(r["answer"], phrases)
                except Exception as e:  # noqa: BLE001
                    r = {"error": str(e)[:200], "ok": False}
                # The arm asked second finds the beginning it shares with the first already read by the engine,
                # so times are also reported for the questions an arm was asked first.
                r["first"] = place == 0
                row[name] = r
            rows.append(row)
            print(f"{i + 1:3} " + "  ".join(f"{n}: {'OK ' if row[n].get('ok') else '   '} {row[n].get('prompt_tokens')!s:>5} tok {row[n].get('seconds', 0):6.1f}s"
                                             for n in ("a", "b")) + f"  {q[:44]}", flush=True)
    finally:
        for p in procs:
            p.terminate()
    good = [r for r in rows if "error" not in r["a"] and "error" not in r["b"]]
    med = lambda n, k: statistics.median(r[n][k] for r in good if r[n].get(k) is not None)  # noqa: E731
    b_only = sum(1 for r in good if r["b"]["ok"] and not r["a"]["ok"])
    a_only = sum(1 for r in good if r["a"]["ok"] and not r["b"]["ok"])
    summary = {n: {"k": arms[n][1], "max_chars": arms[n][2], "compact_sources": bool(arms[n][3]) if len(arms[n]) > 3 else True, "right": sum(1 for r in good if r[n]["ok"]), "of": len(good),
                   "median_prompt_tokens": med(n, "prompt_tokens"), "median_ttft_ms": med(n, "ttft_ms"),
                   "median_seconds": med(n, "seconds"), "median_passages": med(n, "passages"),
                   "median_seconds_asked_first": statistics.median(r[n]["seconds"] for r in good if r[n].get("first")),
                   "median_ttft_ms_asked_first": statistics.median(r[n]["ttft_ms"] for r in good if r[n].get("first") and r[n].get("ttft_ms"))}
               for n in ("a", "b")}
    summary["pairs"] = {"b_right_a_wrong": b_only, "a_right_b_wrong": a_only, "mcnemar_p": round(mcnemar_exact(b_only, a_only), 3),
                        "b_faster_in": f"{sum(1 for r in good if r['b']['seconds'] < r['a']['seconds'])} of {len(good)}",
                        "median_time_ratio_b_over_a": round(statistics.median(r["b"]["seconds"] / r["a"]["seconds"] for r in good), 3)}
    load = os.getloadavg()
    out = {"what": "Knowledge prompts with fewer, better-chosen sentences: arm b against arm a (Phase 27)", "date": time.strftime("%Y-%m-%d %H:%M"),
           "kb": a.kb, "questions": len(qs), "host_load_average": [round(x, 1) for x in load],
           "note": "SHARED MACHINE" if load[0] > 4 else "quiet host", "summary": summary, "rows": rows}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
