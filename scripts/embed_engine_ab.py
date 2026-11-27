"""Knowledge-base embeddings on Smiley Core against Peregrine (Phase 39): speed and whether the vectors agree.

The same passages (pieces of the paper's LaTeX sections, about 1,000 characters, as the knowledge base cuts them) are
embedded in batches of `--batch` by both servers, alternating which goes first per batch, plus the same questions in
Qwen3-Embedding's query format. Agreement is the cosine between the two engines' vectors of the same text: a base built
on one engine can be searched with the other only if it is about 1. Writes benchmarks/engines_phase39/embed_ab.json.

    python3 -I scripts/embed_engine_ab.py [--passages 64] [--batch 8] [--rounds 2]

Servers: Smiley Core embed :8203 (scripts/serve_smiley.sh embed), Peregrine embed :8213 (scripts/serve_peregrine.sh embed).
"""
import argparse, glob, json, math, os, re, statistics, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "benchmarks/engines_phase39/embed_ab.json")
SERVERS = {"smiley": "http://127.0.0.1:8203", "peregrine": "http://127.0.0.1:8213"}
INSTR = "Given a question, retrieve passages of the knowledge base that answer it"
QUESTIONS = ["How does the gate decide when to escalate to System 2?", "What does the verifier check for code answers?",
             "Which models run as System 1 and System 2?", "How is the knowledge graph built from passages?",
             "What is the reasoning cache?", "How many CPU cores does the reference machine have?"]


def passages(n):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "docs/report/sections/*.tex"))):
        t = open(f, encoding="utf-8", errors="replace").read()
        t = re.sub(r"%.*", "", t)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
        t = re.sub(r"[{}$]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        for i in range(0, len(t) - 400, 1000):
            out.append(t[i:i + 1000])
    # Vary the start of every text: Smiley Core reuses a shared prefix's cache (Phase 25 trap), which would flatter it.
    return [f"[{i}] {p}" for i, p in enumerate(out[:n])]


def embed(url, texts):
    req = urllib.request.Request(url + "/v1/embeddings", json.dumps({"input": texts}).encode(), {"Content-Type": "application/json"})
    t = time.time()
    v = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time() - t, [d["embedding"] for d in sorted(v["data"], key=lambda d: d["index"])]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    ps = passages(a.passages)
    qs = [f"Instruct: {INSTR}\nQuery:{q}" for q in QUESTIONS]
    times = {k: [] for k in SERVERS}
    vecs = {k: {} for k in SERVERS}
    for rnd in range(a.rounds):
        for b, i in enumerate(range(0, len(ps), a.batch)):
            batch = [f"{t} (r{rnd})" if rnd else t for t in ps[i:i + a.batch]]
            order = list(SERVERS) if (b + rnd) % 2 == 0 else list(SERVERS)[::-1]
            for eng in order:
                dt, v = embed(SERVERS[eng], batch)
                times[eng].append({"s": dt, "n": len(batch), "chars": sum(map(len, batch))})
                if rnd == 0:
                    for t, x in zip(batch, v):
                        vecs[eng][t] = x
            print(f"r{rnd} batch {b}: " + "  ".join(f"{e} {times[e][-1]['s']:.2f}s" for e in SERVERS), flush=True)
    qv = {e: embed(SERVERS[e], qs)[1] for e in SERVERS}
    pc = [cos(vecs["smiley"][t], vecs["peregrine"][t]) for t in vecs["smiley"]]
    qc = [cos(x, y) for x, y in zip(qv["smiley"], qv["peregrine"])]
    # Does the ranking of passages for each question agree between engines? (top-5 overlap)
    def top(e, q):
        return [t for _, t in sorted(((cos(q, vecs[e][t]), t) for t in vecs[e]), reverse=True)[:5]]
    overlap = [len(set(top("smiley", qv["smiley"][j])) & set(top("peregrine", qv["peregrine"][j]))) / 5 for j in range(len(qs))]
    res = {e: {"batches": len(times[e]), "passages_per_s": sum(x["n"] for x in times[e]) / sum(x["s"] for x in times[e]),
               "batch_s_median": statistics.median(x["s"] for x in times[e])} for e in SERVERS}
    pairs = list(zip(times["smiley"], times["peregrine"]))
    doc = {"passages": len(ps), "batch": a.batch, "rounds": a.rounds, "engines": res,
           "peregrine_faster_batches": sum(p["s"] < s["s"] for s, p in pairs), "pairs": len(pairs),
           "speed_ratio_median": statistics.median(s["s"] / p["s"] for s, p in pairs),
           "passage_cosine": {"min": min(pc), "mean": statistics.mean(pc)}, "query_cosine": {"min": min(qc), "mean": statistics.mean(qc)},
           "top5_overlap_mean": statistics.mean(overlap), "load1": os.getloadavg()[0]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
