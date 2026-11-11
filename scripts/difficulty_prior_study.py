#!/usr/bin/env python3
"""Does the question alone say whether the quick model's draft will be wrong? (Phase 26, the difficulty prior.)

Every task of a stored evaluation run that carries the exact label of its draft (`draft_correct`) is joined with its
question from `data/bench`, the question is embedded by the embedding server, and three predictors of "the draft is
wrong" are compared out of fold (5 folds, stratified, the same folds for all):

  domain      the wrong rate of the question's domain (what the gate already knows before the draft)
  knn         similarity-weighted vote of the k nearest labelled questions
  banks       the prototype-bank margin of vLLM Semantic Router: score(hard bank) - score(easy bank), each score
              0.75 * best + 0.25 * mean(top 2) cosine
  draft       logistic regression on what the draft produces (mean log-probability, lowest window, agreement)
  draft+knn   the same with the out-of-fold knn score added

Usage: python3 scripts/difficulty_prior_study.py [--embed-url http://127.0.0.1:8203] [--out benchmarks/router_phase26/difficulty_prior.json]
"""
import argparse
import glob
import json
import os
import sqlite3
import urllib.request

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def labelled_items(db, bench_glob):
    """task id -> (draft wrong, domain, features); the first record of a task wins."""
    out = {}

    def take(doc):
        for it in doc.get("items") or []:
            if it.get("draft_correct") is None or not it.get("task_id"):
                continue
            out.setdefault(it["task_id"], (not it["draft_correct"], it.get("domain") or "other", it.get("features") or {}))

    if os.path.exists(db):
        con = sqlite3.connect(db)
        for (j,) in con.execute("select json from runs order by created_at"):
            try:
                take(json.loads(j))
            except ValueError:
                pass
    for path in sorted(glob.glob(bench_glob)):
        try:
            doc = json.load(open(path))
        except ValueError:
            continue
        if isinstance(doc, dict):
            take(doc)
    return out


def prompts(bench_dir):
    out = {}
    for path in glob.glob(os.path.join(bench_dir, "*.jsonl")):
        for line in open(path):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if "id" in t and "prompt" in t:
                out.setdefault(t["id"], t["prompt"])
    return out


def embed(url, texts, batch=8):
    vecs = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"input": texts[i : i + batch]}).encode()
        req = urllib.request.Request(url + "/v1/embeddings", body, {"content-type": "application/json"})
        data = json.load(urllib.request.urlopen(req, timeout=600))["data"]
        data.sort(key=lambda d: d["index"])
        vecs.extend(d["embedding"] for d in data)
        print(f"embedded {len(vecs)}/{len(texts)}", flush=True)
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def knn_score(sim, y_train, k):
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, axis=1).clip(min=0)
    return (w * y_train[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)


def bank_margin(sim, y_train):
    def score(cols):
        if cols.sum() == 0:
            return np.zeros(sim.shape[0])
        s = -np.sort(-sim[:, cols], axis=1)[:, :2]
        return 0.75 * s[:, 0] + 0.25 * s.mean(1)

    return score(y_train == 1) - score(y_train == 0)


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-url", default="http://127.0.0.1:8203")
    ap.add_argument("--db", default="data/runtime/nafs.sqlite")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--runs", default="benchmarks/*.json")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="benchmarks/router_phase26/difficulty_prior.json")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()

    items = labelled_items(a.db, a.runs)
    text = prompts(a.bench)
    ids = sorted(i for i in items if i in text)
    y = np.array([int(items[i][0]) for i in ids])
    dom = np.array([items[i][1] for i in ids])
    print(f"{len(ids)} labelled questions, {int(y.sum())} wrong drafts; by domain:", {d: int((dom == d).sum()) for d in set(dom)})

    if a.cache and os.path.exists(a.cache):
        v = np.load(a.cache)
    else:
        v = embed(a.embed_url, [text[i] for i in ids])
        if a.cache:
            np.save(a.cache, v)

    def feat(i):
        f = items[i][2]
        return [f.get("mean_logprob") or 0.0, f.get("low_window_logprob") or 0.0,
                f.get("agreement") if f.get("agreement") is not None else 0.5, 1.0 if f.get("agreement") is None else 0.0]

    x = np.array([feat(i) for i in ids], dtype=np.float64)
    doms = sorted(set(dom))
    onehot = np.array([[1.0 if d == q else 0.0 for q in doms] for d in dom])
    s = {n: np.zeros(len(ids)) for n in ["domain", "knn", "banks", "draft", "draft+knn"]}
    strat = np.array([f"{d}{l}" for d, l in zip(dom, y)])
    # A stratum with fewer members than folds cannot be stratified; fold it into its label.
    counts = {t: int((strat == t).sum()) for t in set(strat)}
    strat = np.array([t if counts[t] >= 5 else t[-1] for t in strat])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=11).split(v, strat):
        sim = v[te] @ v[tr].T
        rate = {d: y[tr][dom[tr] == d].mean() if (dom[tr] == d).any() else y[tr].mean() for d in doms}
        s["domain"][te] = [rate[d] for d in dom[te]]
        s["knn"][te] = knn_score(sim, y[tr], a.k)
        s["banks"][te] = bank_margin(sim, y[tr])
        # The knn score of a training question, from the other training questions (no question votes for itself).
        sim_tr = v[tr] @ v[tr].T
        np.fill_diagonal(sim_tr, -1.0)
        knn_tr = knn_score(sim_tr, y[tr], a.k)
        base_tr, base_te = np.hstack([x[tr], onehot[tr]]), np.hstack([x[te], onehot[te]])
        m = LogisticRegression(max_iter=2000).fit(base_tr, y[tr])
        s["draft"][te] = m.predict_proba(base_te)[:, 1]
        m2 = LogisticRegression(max_iter=2000).fit(np.hstack([base_tr, knn_tr[:, None]]), y[tr])
        s["draft+knn"][te] = m2.predict_proba(np.hstack([base_te, s["knn"][te][:, None]]))[:, 1]

    report = {"n": len(ids), "wrong": int(y.sum()), "k": a.k, "folds": 5, "auc": {}, "skip_draft": {}}
    for scope in ["all"] + doms:
        mask = np.ones(len(ids), bool) if scope == "all" else dom == scope
        if mask.sum() < 30:
            continue
        report["auc"][scope] = {"n": int(mask.sum()), "wrong": int(y[mask].sum()), **{n: auc(y[mask], sc[mask]) for n, sc in s.items()}}
    # Skipping the draft pays only where the question alone is nearly sure of a wrong draft.
    for thr in [0.6, 0.7, 0.8, 0.9]:
        sel = s["knn"] >= thr
        report["skip_draft"][str(thr)] = {"questions": int(sel.sum()), "of_them_wrong": int(y[sel].sum())}
    print(json.dumps(report, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
