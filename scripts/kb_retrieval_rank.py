#!/usr/bin/env python3
"""Where does the passage that states the answer rank? (Phase 8 follow-up, FR-K58 groundwork.)

Retrieval and generation fail for different reasons, and generation is a hundred times dearer to measure. For every
question of a question file (default scripts/kb_paper_questions.json: [question, [accepted phrases]]) this runs the
search of one knowledge base in each mode and records the rank of the first passage that contains an accepted phrase.
It prints recall@k and the mean reciprocal rank per mode, and lists the questions whose answer is not in the top k.

    python3 -I scripts/kb_retrieval_rank.py --kb <id> [--k 20] [--modes lexical,vector,hybrid] [--out benchmarks/x.json]
"""
import argparse, json, os, re, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%/-]", " ", s.lower())).strip()


def fusion_sweep(a, qs):
    """Weighted reciprocal-rank fusion: score = 1/(60 + rank_lexical) + w / (60 + rank_vector). w = 1 is the shipped
    hybrid, w = 0 is the words alone. Both lists come from one search each, so the sweep costs two calls a question."""
    lists = []
    for q, phrases in qs:
        got = {}
        for m in ("lexical", "vector"):
            r = post(a.base, f"/v1/kb/{a.kb}/search", {"query": q, "k": a.k, "mode": m})
            got[m] = r.get("results", [])
        lists.append((phrases, got))
    def rank_of(order, phrases, texts):
        for i, cid in enumerate(order, 1):
            if any(norm(p) in norm(texts[cid]) for p in phrases):
                return i
        return None
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"{len(qs)} questions, base {a.kb}: weighted fusion, vector weight w")
    print(f"  {'w':>4} {'R@1':>6} {'R@2':>6} {'R@4':>6} {'R@8':>6} {'MRR':>7}")
    out = {}
    for w in grid:
        ranks = []
        for phrases, got in lists:
            texts = {h["chunk_id"]: h["text"] for m in got for h in got[m]}
            score = {}
            for m, wt in (("lexical", 1.0), ("vector", w)):
                for i, h in enumerate(got[m], 1):
                    score[h["chunk_id"]] = score.get(h["chunk_id"], 0.0) + wt / (60 + i)
            order = [c for c, _ in sorted(score.items(), key=lambda x: -x[1])]
            ranks.append(rank_of(order, phrases, texts))
        n = len(ranks)
        rec = lambda k: sum(1 for x in ranks if x and x <= k) / n
        mrr = sum(1 / x for x in ranks if x) / n
        out[w] = {"R@1": rec(1), "R@2": rec(2), "R@4": rec(4), "R@8": rec(8), "MRR": round(mrr, 3), "ranks": ranks}
        print(f"  {w:4.1f} {rec(1):6.2f} {rec(2):6.2f} {rec(4):6.2f} {rec(8):6.2f} {mrr:7.3f}")
    if a.out:
        json.dump({"kb": a.kb, "sweep": out}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--modes", default="lexical,vector,hybrid")
    ap.add_argument("--questions", default=os.path.join(ROOT, "scripts/kb_paper_questions.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--fusion-sweep", action="store_true",
                    help="fuse the lexical and vector lists here with the vector arm weighted w, for a grid of w")
    a = ap.parse_args()
    qs = json.load(open(a.questions))
    modes = a.modes.split(",")
    if a.fusion_sweep:
        return fusion_sweep(a, qs)
    rows, ks = [], [1, 2, 4, 8, a.k]
    for q, phrases in qs:
        row = {"question": q, "ranks": {}}
        for m in modes:
            r = post(a.base, f"/v1/kb/{a.kb}/search", {"query": q, "k": a.k, "mode": m})
            rank = None
            for i, h in enumerate(r.get("results", []), 1):
                if any(norm(p) in norm(h["text"]) for p in phrases):
                    rank = i
                    break
            row["ranks"][m] = rank
        rows.append(row)
    print(f"{len(qs)} questions, base {a.kb}")
    summary = {}
    for m in modes:
        rs = [r["ranks"][m] for r in rows]
        rec = {k: sum(1 for x in rs if x and x <= k) / len(rs) for k in ks}
        mrr = sum(1 / x for x in rs if x) / len(rs)
        summary[m] = {"recall": rec, "mrr": round(mrr, 3), "missing": sum(1 for x in rs if not x)}
        print(f"{m:8} " + "  ".join(f"R@{k} {rec[k]:.2f}" for k in ks) + f"  MRR {mrr:.3f}  not in top {a.k}: {summary[m]['missing']}")
    for r in rows:
        rk = r["ranks"]
        if any((x is None or x > 4) for x in rk.values()):
            print("  ", r["question"][:70], rk)
    if a.out:
        json.dump({"kb": a.kb, "k": a.k, "summary": summary, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
