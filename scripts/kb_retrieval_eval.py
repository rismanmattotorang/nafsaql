#!/usr/bin/env python3
"""Knowledge Base retrieval quality (Phase 20, Phase 2): hit@1, recall@5 and MRR of the three search modes on a fixed
question set over the fixture knowledge base (the Memorilla and PagedAttention papers, the report's architecture
section, slides, notes). A question is answered when a retrieved passage of the expected document contains the
expected phrase. Most questions are paraphrased, so they share few words with the passage.

Usage: python3 -I scripts/kb_retrieval_eval.py --base http://127.0.0.1:8080 --kb <id> [--out benchmarks/kb_retrieval.json]
"""
import argparse, json, os, statistics, sys, time, urllib.request

# (question, words in the expected document's title or file name, phrase the passage must contain)
QUESTIONS = [
    ("How many soft tokens does the memory module hand to the frozen decoder?", "memorilla", "16 memory tokens"),
    ("What accuracy did the method reach on the knowledge-graph fact-checking benchmark, against giving the model every document?", "memorilla", "84.33"),
    ("Which model turns the documents into embeddings?", "memorilla", "qwen3-embedding-4b"),
    ("How fast is an answer with 512 stored documents compared with putting them all in the prompt?", "memorilla", "19.9"),
    ("Does training with eight times more memory slots help?", "memorilla", "k=128"),
    ("Which reinforcement learning method trained the agent's memory in the text adventure games?", "memorilla", "grpo"),
    ("Why can the system lose small factual details?", "memorilla", "fine-grained"),
    ("How many Wikipedia examples were used in the first training stage?", "memorilla", "1.98"),
    ("Which kinds of narrative questions does plain retrieval answer better?", "memorilla", "who"),
    ("Which bridging architecture from vision-language models was compared in the ablation?", "memorilla", "q-former"),
    ("Which operating-system idea inspired the way the cache memory is managed?", "efficient memory", "virtual memory"),
    ("By how much does the serving system improve throughput over earlier systems?", "efficient memory", "throughput"),
    ("How are blocks shared between sequences duplicated only when one of them writes?", "efficient memory", "copy-on-write"),
    ("How does the system serve several output samples for one prompt?", "efficient memory", "parallel sampling"),
    ("Why do existing systems waste most of the key-value cache?", "efficient memory", "fragmentation"),
    ("What does the scheduler do when the GPU has no free blocks left?", "efficient memory", "swap"),
    ("Which port does the gateway use to reach the database?", "notes", "8300"),
    ("How large may each uploaded file be?", "upload", "fifty megabytes"),
    ("Where do the memory vectors enter the inference engine?", "memorilla in nafs-aql", "placeholder"),
    ("How does local search walk the entity graph?", "knowledge base design", "one or two hops"),
]


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    modes = ["lexical", "vector", "hybrid"]
    rows, summary = [], {}
    for mode in modes:
        ranks, took, embed = [], [], []
        for q, doc, phrase in QUESTIONS:
            r = post(a.base, f"/v1/kb/{a.kb}/search", {"query": q, "k": a.k, "mode": mode})
            rank = next((i + 1 for i, h in enumerate(r["results"])
                         if doc in h["doc_title"].lower() and phrase in h["text"].lower()), None)
            # Documents whose title is the file name: match on the title the store recorded.
            ranks.append(rank)
            took.append(r["took_ms"]); embed.append(r.get("embed_ms", 0))
            rows.append({"mode": r["mode"], "asked": mode, "question": q, "rank": rank, "took_ms": r["took_ms"], "fallback": r.get("fallback")})
        n = len(QUESTIONS)
        summary[mode] = {
            "hit@1": sum(1 for x in ranks if x == 1) / n,
            f"recall@{a.k}": sum(1 for x in ranks if x) / n,
            "mrr": sum(1 / x for x in ranks if x) / n,
            "median_ms": statistics.median(took), "median_embed_ms": statistics.median(embed),
        }
        print(f"{mode:8} hit@1 {summary[mode]['hit@1']:.2f}  recall@{a.k} {summary[mode][f'recall@{a.k}']:.2f}  "
              f"MRR {summary[mode]['mrr']:.3f}  median {summary[mode]['median_ms']} ms (query embedding {summary[mode]['median_embed_ms']} ms)")
    load = os.getloadavg()
    out = {"what": "Knowledge Base retrieval, fixture knowledge base, fixed question set", "date": time.strftime("%Y-%m-%d %H:%M"),
           "questions": len(QUESTIONS), "k": a.k, "embedder": "Qwen3-Embedding-0.6B Q8_0 on Smiley Core",
           "host_load_average": [round(x, 1) for x in load],
           "note": "SHARED MACHINE" if load[0] > 4 else "quiet host", "summary": summary, "rows": rows}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
