#!/usr/bin/env python3
"""The stable-prefix prompt layout against the usual one: the same answers from documents? (Phase 30.)

With `stable_prefix` the system prompt is the assistant's own text alone and the passages, notes and domain instruction
go before the question; without it they are in the system prompt. Both arms ask the same questions of the running
gateway through the quick model with a knowledge base (`rag`), alternating which arm goes first, and compare right
answers (an expected phrase is in the reply), prompt tokens, time to the first token and total time.

Usage: python3 scripts/stable_prefix_ab.py --kb <id> [--questions scripts/kb_paper_questions_50.json] [--limit N]
                                           [--gw http://127.0.0.1:8080] [--out benchmarks/engine_phase30/stable_prefix_ab.json]
"""
import argparse, json, math, os, re, statistics, time, urllib.request

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


ap = argparse.ArgumentParser()
ap.add_argument("--kb", required=True)
ap.add_argument("--questions", default="scripts/kb_paper_questions_50.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gw", default="http://127.0.0.1:8080")
ap.add_argument("--out", default="")
a = ap.parse_args()
qs = json.load(open(os.path.join(ROOT, a.questions)))
if a.limit:
    qs = qs[: a.limit]


def ask(q, stable):
    body = {"messages": [{"role": "user", "content": q}], "force_mode": "fast", "ephemeral": True,
            "knowledge": {"kb": a.kb, "mode": "rag"}, "stable_prefix": stable, "shadow": "off"}
    req = urllib.request.Request(a.gw + "/v1/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return time.time() - t, out


rows = []
for i, item in enumerate(qs):
    q, phrases = (item[0], item[1]) if isinstance(item, list) else (item["question"], item["phrases"])
    order = [False, True] if i % 2 == 0 else [True, False]
    rec = {"question": q}
    for first, stable in zip((True, False), order):
        secs, out = ask(q, stable)
        u = out.get("usage", {})
        arm = "stable" if stable else "usual"
        rec[arm] = {"right": correct(out["message"]["content"], phrases), "seconds": round(secs, 2), "ttft_ms": u.get("ttft_ms"),
                    "prompt_tokens": u.get("prompt_tokens"), "asked_first": first}
    rows.append(rec)
    print(f"{i + 1:>3} usual {'R' if rec['usual']['right'] else '-'} {rec['usual']['seconds']:5.1f}s  "
          f"stable {'R' if rec['stable']['right'] else '-'} {rec['stable']['seconds']:5.1f}s", flush=True)


def summ(arm):
    r = [x[arm] for x in rows]
    return {"right": sum(x["right"] for x in r), "of": len(r), "median_seconds": statistics.median(x["seconds"] for x in r),
            "median_ttft_ms": statistics.median((x["ttft_ms"] or 0) for x in r),
            "median_prompt_tokens": statistics.median((x["prompt_tokens"] or 0) for x in r),
            "median_seconds_asked_first": statistics.median(x["seconds"] for x in r if x["asked_first"])}


b = sum(1 for x in rows if x["stable"]["right"] and not x["usual"]["right"])
c = sum(1 for x in rows if x["usual"]["right"] and not x["stable"]["right"])
summary = {"usual": summ("usual"), "stable": summ("stable"),
           "pairs": {"stable_right_usual_wrong": b, "usual_right_stable_wrong": c, "mcnemar_p": round(mcnemar_exact(b, c), 3)}}
print(json.dumps(summary, indent=1))
if a.out:
    json.dump({"what": "stable-prefix layout against the usual one, answers from documents (Phase 30)", "kb": a.kb,
               "date": time.strftime("%Y-%m-%d %H:%M"), "questions": len(rows), "host_load_average": os.getloadavg(),
               "note": "SHARED MACHINE", "summary": summary, "rows": rows}, open(os.path.join(ROOT, a.out), "w"), indent=1)
