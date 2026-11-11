#!/usr/bin/env python3
"""Does the support check tell an answer that is in its passages from one that is not? (Phase 26.)

For each question of a set with accepted phrases, the quick model answers twice through the running gateway: from the
passages of a knowledge base (`rag`), and with no passages (`off`). The answer from passages carries the server's own
support row. The answer without passages is then checked against the passages the first answer was given, by the same
code (`cargo run --example support_eval`): it is what an answer made up without reading looks like to the check.

Reported: the share of supported sentences by group (from passages and right, from passages and wrong, made up), the
area under the curve of "the answer is wrong" by a low share, and for each threshold how many answers from passages
fall below it and how many of those are wrong.

Usage: python3 scripts/support_study.py --kb <id> [--questions scripts/kb_paper_questions_50.json] [--limit N]
                                        [--base http://127.0.0.1:8080] [--out benchmarks/kb_phase26/support.json]
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def http(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%-]+", " ", (s or "").lower()))


def correct(answer, phrases):
    a = norm(answer)
    return any(norm(p) in a for p in phrases)


def auc(scores_pos, scores_neg):
    """P(score of a positive > score of a negative), ties half."""
    if not scores_pos or not scores_neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in scores_pos for n in scores_neg)
    return round(wins / (len(scores_pos) * len(scores_neg)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--questions", default="scripts/kb_paper_questions_50.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/kb_phase26/support.json")
    ap.add_argument("--checker", default="rust/target/release/examples/support_eval")
    ap.add_argument("--recheck", default="", help="check the stored answers of an earlier run again (no model calls)")
    a = ap.parse_args()
    qs = json.load(open(os.path.join(ROOT, a.questions)))
    if a.limit:
        qs = qs[: a.limit]
    rows, offline, answered = [], [], None
    if a.recheck:
        # The stored answers and passages of an earlier run, checked again by the current code: no model call.
        earlier = json.load(open(os.path.join(ROOT, a.recheck)))
        rows, answered = earlier["rows"], earlier.get("answered", earlier.get("date"))
        qs = qs[: len(rows)]
        for i, r in enumerate(rows):
            for mode in ("rag", "off"):
                if "answer" in r.get(mode, {}) and "passages" in r.get("rag", {}):
                    offline.append({"id": [i, mode], "question": r["question"], "answer": r[mode]["answer"],
                                    "passages": r["rag"]["passages"]})
    for i, (q, phrases) in enumerate([] if a.recheck else qs):
        row = {"question": q}
        for mode in (("rag", "off") if i % 2 == 0 else ("off", "rag")):
            body = {"messages": [{"role": "user", "content": q}], "ephemeral": True, "retrieve": False, "force_mode": "fast"}
            if mode == "rag":
                body["knowledge"] = {"kb": a.kb, "mode": "rag"}
            t = time.time()
            try:
                r = http(a.base, "/v1/chat", body)
            except Exception as e:  # noqa: BLE001
                row[mode] = {"error": str(e)[:200]}
                continue
            text = r.get("message", {}).get("content", "")
            src = r.get("sources", []) or []
            entry = {"answer": text, "ok": correct(text, phrases), "seconds": round(time.time() - t, 1)}
            if mode == "rag":
                entry["passages"] = [s.get("text", "") for s in src if "chunk_id" in s]
                entry["in_prompt"] = any(correct(p, phrases) for p in entry["passages"])
                entry["support"] = next((s["support"] for s in src if "support" in s), None)
            row[mode] = entry
        if "passages" in row.get("rag", {}) and "answer" in row.get("off", {}):
            offline.append({"id": i, "question": q, "answer": row["off"]["answer"], "passages": row["rag"]["passages"]})
        rows.append(row)
        rag, off = row.get("rag", {}), row.get("off", {})
        print(f"{i + 1:3} rag {'OK ' if rag.get('ok') else '   '} share {(rag.get('support') or {}).get('share', '-')!s:>5}"
              f" {rag.get('seconds', 0):6.1f}s | off {'OK ' if off.get('ok') else '   '} {off.get('seconds', 0):6.1f}s  {q[:50]}", flush=True)
    # The made-up answers against the same passages, by the server's own code.
    if offline:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(offline, f)
        out = subprocess.run([os.path.join(ROOT, a.checker), f.name], capture_output=True, text=True, check=True).stdout
        os.unlink(f.name)
        for line in out.splitlines():
            v = json.loads(line)
            i, mode = v["id"] if isinstance(v["id"], list) else (v["id"], "off")
            rows[i][mode]["support"] = {k: v[k] for k in ("claims", "supported", "share", "unsupported")}

    def shares(pick):
        return [r[m]["support"]["share"] for r in rows for m in ("rag", "off")
                if pick(r, m) and (r.get(m, {}).get("support") or {}).get("claims", 0) > 0]

    groups = {
        "from passages, right": shares(lambda r, m: m == "rag" and r["rag"].get("ok")),
        "from passages, wrong": shares(lambda r, m: m == "rag" and not r["rag"].get("ok")),
        "made up (no passages), right": shares(lambda r, m: m == "off" and r["off"].get("ok")),
        "made up (no passages), wrong": shares(lambda r, m: m == "off" and not r["off"].get("ok")),
    }
    summary = {g: {"n": len(v), "mean_share": round(statistics.fmean(v), 3) if v else None,
                   "median_share": round(statistics.median(v), 3) if v else None,
                   "fully_supported": sum(1 for x in v if x >= 0.999)} for g, v in groups.items()}
    right = groups["from passages, right"]
    wrong_rag = groups["from passages, wrong"]
    made_up = groups["made up (no passages), wrong"] + groups["made up (no passages), right"]
    # A low share should mark the wrong or made-up answer: positives are scored by 1 - share.
    summary["auc_wrong_among_answers_from_passages"] = auc([1 - x for x in wrong_rag], [1 - x for x in right])
    summary["auc_made_up_against_from_passages"] = auc([1 - x for x in made_up], [1 - x for x in right + wrong_rag])
    thresholds = {}
    for t in (0.34, 0.5, 0.67, 0.75, 1.0):
        below = [(r["rag"]["ok"]) for r in rows if (r.get("rag", {}).get("support") or {}).get("claims", 0) > 0
                 and r["rag"]["support"]["share"] < t]
        below_off = [1 for r in rows if (r.get("off", {}).get("support") or {}).get("claims", 0) > 0
                     and r["off"]["support"]["share"] < t]
        thresholds[str(t)] = {"answers_from_passages_below": len(below), "of_them_wrong": sum(1 for ok in below if not ok),
                              "made_up_answers_below": len(below_off)}
    summary["below_threshold"] = thresholds
    n_rag = sum(1 for r in rows if "ok" in r.get("rag", {}))
    summary["accuracy"] = {"rag": f"{sum(1 for r in rows if r.get('rag', {}).get('ok'))}/{n_rag}",
                           "off": f"{sum(1 for r in rows if r.get('off', {}).get('ok'))}/{sum(1 for r in rows if 'ok' in r.get('off', {}))}"}
    load = os.getloadavg()
    report = {"what": "The support check on answers from passages and on answers made up without them (Phase 26)",
              "date": time.strftime("%Y-%m-%d %H:%M"), "kb": a.kb, "questions": len(qs),
              "host_load_average": [round(x, 1) for x in load], "note": "SHARED MACHINE" if load[0] > 4 else "quiet host",
              "summary": summary, "rows": rows}
    if answered:
        report["answered"] = answered
        report["what"] += "; the answers and passages are those of the run of " + answered + ", checked again by the current rule"
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
