#!/usr/bin/env python3
"""Does a model-written expansion of the question find the answering passage sooner? (Phase 8 follow-up, RAG.)

The questions people ask are paraphrases of what the passages say: on the heavily paraphrased set the answering passage
is first for only 30 to 55 % of the questions. The textbook remedy (HyDE, query expansion) has the model write the
passage that would answer the question, and searches with that as well. It costs one quick-model call a question, which
on this CPU is seconds, so it is measured before it is built.

For every question the quick model (Qwen3-1.7B, no thinking, greedy) writes either a short hypothetical passage ("hyde")
or the technical terms a report would use ("terms"). The expansions are cached in `--cache`, so a second run costs only
searches. The ranks of the answering passage are then compared, with the rank fusion of FR-K63 done here:

  base      the words of the question, the meaning of the question          (what ships)
  words+    the words of question and expansion, the meaning of the question
  meaning+  the words of the question, the meaning of question and expansion
  both+     the words and the meaning of question and expansion together

    python3 -I scripts/kb_query_expansion.py --kb <id> --questions scripts/kb_paper_questions_para.json [--style hyde|terms]
"""
import argparse, json, os, re, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = {
    "hyde": ("Write two short sentences, in the style of a technical report, that would contain the answer to the question. "
             "Use the exact technical terms and numbers such a report would use. Reply with the sentences only.\n\nQuestion: {q}"),
    "terms": ("List ten technical terms, abbreviations and units that a technical report would use in a passage answering the "
              "question. Reply with the terms only, separated by spaces.\n\nQuestion: {q}"),
}


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%/-]", " ", s.lower())).strip()


def expand(q, style, llm):
    r = post(llm + "/chat/completions", {
        "model": "system1", "temperature": 0, "max_tokens": 90,
        "messages": [{"role": "user", "content": PROMPTS[style].format(q=q) + " /no_think"}]})
    t = r["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--llm", default="http://127.0.0.1:8201/v1")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--questions", default=os.path.join(ROOT, "scripts/kb_paper_questions_para.json"))
    ap.add_argument("--style", default="hyde", choices=list(PROMPTS))
    ap.add_argument("--cache", default="")
    ap.add_argument("--vector-weight", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    qs = json.load(open(a.questions))
    cache_path = a.cache or os.path.join(ROOT, f"benchmarks/kb_expansions_{a.style}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    t_llm = []
    for q, _ in qs:
        if q not in cache:
            t = time.time()
            cache[q] = expand(q, a.style, a.llm)
            t_llm.append(time.time() - t)
            json.dump(cache, open(cache_path, "w"), indent=1)
    if t_llm:
        print(f"expansions written: {len(t_llm)}, median {sorted(t_llm)[len(t_llm) // 2]:.1f} s each")

    def search(text, mode):
        r = post(f"{a.base}/v1/kb/{a.kb}/search", {"query": text, "k": a.k, "mode": mode})
        return r.get("results", [])

    def rank(order, texts, phrases):
        for i, cid in enumerate(order, 1):
            if any(norm(p) in norm(texts[cid]) for p in phrases):
                return i
        return None

    def fuse(lex, vec, w):
        score, texts = {}, {}
        for lst, wt in ((lex, 1.0), (vec, w)):
            for i, h in enumerate(lst, 1):
                score[h["chunk_id"]] = score.get(h["chunk_id"], 0.0) + wt / (60 + i)
                texts[h["chunk_id"]] = h["text"]
        return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])], texts

    variants = ["base", "words+", "meaning+", "both+"]
    ranks = {v: [] for v in variants}
    for q, phrases in qs:
        x = cache[q]
        lq, vq = search(q, "lexical"), search(q, "vector")
        lx, vx = search(q + " " + x, "lexical"), search(q + " " + x, "vector")
        for v, (lex, vec) in {"base": (lq, vq), "words+": (lx, vq), "meaning+": (lq, vx), "both+": (lx, vx)}.items():
            order, texts = fuse(lex, vec, a.vector_weight)
            ranks[v].append(rank(order, texts, phrases))
    n = len(qs)
    print(f"{n} questions, base {a.kb}, expansion style {a.style}, vector weight {a.vector_weight}")
    print(f"  {'':9} {'R@1':>6} {'R@2':>6} {'R@4':>6} {'R@8':>6} {'MRR':>7}")
    summary = {}
    for v in variants:
        rs = ranks[v]
        rec = lambda k: sum(1 for x in rs if x and x <= k) / n
        mrr = sum(1 / x for x in rs if x) / n
        summary[v] = {"R@1": rec(1), "R@2": rec(2), "R@4": rec(4), "R@8": rec(8), "MRR": round(mrr, 3), "ranks": rs}
        print(f"  {v:9} {rec(1):6.2f} {rec(2):6.2f} {rec(4):6.2f} {rec(8):6.2f} {mrr:7.3f}")
    if a.out:
        json.dump({"kb": a.kb, "style": a.style, "questions": a.questions, "summary": summary,
                   "median_llm_s": sorted(t_llm)[len(t_llm) // 2] if t_llm else None}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
