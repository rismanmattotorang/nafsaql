#!/usr/bin/env python3
"""The embedding server with every core against one fewer (Phase 27): question and passage embedding time.

Starts two copies of the embedding server on spare ports (`-t A` and `-t B`), embeds the same texts on both,
alternating which goes first, and reports the median time of a question (short, with the search instruction in
front, whose cache rows the server reuses) and of a passage (about 1,000 characters, no shared beginning).

Usage: python3 scripts/embed_threads_ab.py [--a 3] [--b 4] [--n 24] [--out benchmarks/engine_phase27/embed_threads.json]
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INSTRUCT = "Instruct: Given a question, retrieve passages of the knowledge base that answer it\nQuery: "
WORDS = ("gate verifier passage index memory engine prompt token cache thread model answer search graph note "
         "question budget draft reason check ledger router decode prefill vector").split()


def embed(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=json.dumps({"input": [text], "model": "embed"}).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)
    return (time.time() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=3)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B-Q8_0.gguf")
    ap.add_argument("--out", default="benchmarks/engine_phase27/embed_threads.json")
    a = ap.parse_args()
    arms = {"a": (8213, a.a), "b": (8214, a.b)}
    procs = []
    try:
        for port, t in arms.values():
            procs.append(subprocess.Popen([os.path.join(ROOT, "rust/target/release/nafs-smiley-server"), "-m", os.path.join(ROOT, a.model),
                                           "--alias", "embed", "-t", str(t), "--port", str(port), "--embedding"],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for port, _ in arms.values():
            for _ in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
        times = {"question": {"a": [], "b": []}, "passage": {"a": [], "b": []}}
        for i in range(a.n):
            w = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(140)]
            question = INSTRUCT + f"How does the {w[0]} decide when the {w[1]} needs a {w[2]} for item {i}?"
            passage = f"Section {i * 31 + 5}. " + " ".join(w)
            for kind, text in (("question", question), ("passage", passage)):
                for name in (("a", "b") if i % 2 == 0 else ("b", "a")):
                    times[kind][name].append(embed(arms[name][0], text))
    finally:
        for p in procs:
            p.terminate()
    summary = {}
    for kind, t in times.items():
        ma, mb = statistics.median(t["a"]), statistics.median(t["b"])
        summary[kind] = {f"{a.a}_threads_ms": round(ma, 1), f"{a.b}_threads_ms": round(mb, 1), "ratio": round(mb / ma, 3),
                         "b_faster_in": f"{sum(1 for x, y in zip(t['a'], t['b']) if y < x)} of {len(t['a'])}"}
    load = os.getloadavg()
    out = {"what": "Embedding server threads (Phase 27)", "date": time.strftime("%Y-%m-%d %H:%M"),
           "host_load_average": [round(x, 1) for x in load], "note": "SHARED MACHINE" if load[0] > 4 else "quiet host",
           "summary": summary, "times_ms": times}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
