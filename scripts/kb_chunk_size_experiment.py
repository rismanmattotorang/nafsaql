#!/usr/bin/env python3
"""How big should a passage be? (Phase 8 follow-up, RAG.)

The chunker cuts passages of 1,200 characters (about 300 tokens) with 150 of overlap. A smaller passage points more
exactly at a fact but may split the sentence that states it; a larger one keeps the sentence and adds neighbours. The
fair comparison gives every size the same prompt budget (`rag_max_chars`, 4,800 characters), so 600-character passages
get 8 slots where 1,200-character ones get 4.

For each size this starts a throw-away gateway (port 8095, its own SurrealDB database and ledger) with that chunk size,
uploads the Markdown and LaTeX files of the paper corpus, waits until every passage has its vector, ranks the passage
that holds each answer for the quoted and the paraphrased question sets, and stops the gateway. The main stack (engines
and SurrealDB) must be running.

    python3 -I scripts/kb_chunk_size_experiment.py [--sizes 600,1200,2400] [--out benchmarks/kb_chunk_size.json]
"""
import argparse, glob, json, os, random, re, signal, string, subprocess, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8095
BASE = f"http://127.0.0.1:{PORT}"
D = os.path.join(ROOT, "run/chunk-size")


def call(method, path, body=None, raw=None, ctype=None, timeout=600):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": ctype or "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t = r.read().decode()
            return r.status, json.loads(t) if t.strip().startswith(("{", "[")) else t
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        return e.code, json.loads(t) if t.strip().startswith(("{", "[")) else t


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%/-]", " ", s.lower())).strip()


def multipart(files):
    b = "----nafs" + "".join(random.choices(string.ascii_letters, k=16))
    out = b""
    for name, content in files:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n"
    return out + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def config(size):
    s = open(os.path.join(ROOT, "config/nafs-aql.toml")).read()

    def sub(key, val, quote=True):
        nonlocal s
        new = f'"{val}"' if quote else str(val)
        s, n = re.subn(rf'(?m)^({key}\s*=\s*)("[^"]*"|\d+)', lambda m: m.group(1) + new, s, count=1)
        assert n == 1, key
    sub("bind", f"127.0.0.1:{PORT}")
    for k, v in [("calibration_file", "gate_calibration.json"), ("params_file", "gate_params.json"),
                 ("rates_file", "verifier_rates.json"), ("sqlite_path", "nafs.sqlite"), ("trace_path", "trace.jsonl")]:
        sub(k, "run/chunk-size/" + v)
    sub("database", "kb_chunk_size")
    sub("files_dir", "run/chunk-size/files")
    sub("chunk_chars", size, quote=False)
    sub("chunk_overlap", size // 8, quote=False)
    s = re.sub(r"(?m)^\[audit\]\nenabled = true", "[audit]\nenabled = false", s)
    path = os.path.join(D, "nafs-aql.toml")
    open(path, "w").write(s)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="600,1200,2400")
    ap.add_argument("--budget", type=int, default=4800)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/kb_chunk_size.json"))
    a = ap.parse_args()
    os.makedirs(os.path.join(D, "files"), exist_ok=True)
    subprocess.run(["cp", os.path.join(ROOT, "data/runtime/verifier_rates.json"), D])
    docs = sorted(glob.glob(os.path.join(ROOT, "data/runtime/paper-corpus/*.tex")) + glob.glob(os.path.join(ROOT, "data/runtime/paper-corpus/*.md")))
    text = norm(" ".join(open(f, errors="ignore").read() for f in docs))
    qsets = {}
    for name in ("kb_paper_questions_50.json", "kb_paper_questions_para.json"):
        qs = json.load(open(os.path.join(ROOT, "scripts", name)))
        qsets[name.replace("kb_paper_questions_", "").replace(".json", "")] = [
            (q, ph) for q, ph in qs if any(norm(p) in text for p in ph)]
    print({k: len(v) for k, v in qsets.items()}, "questions whose answer is in these documents")
    results = {}
    for size in [int(x) for x in a.sizes.split(",")]:
        cfg = config(size)
        proc = subprocess.Popen([os.path.join(ROOT, "rust/target/release/nafs-gateway"), "--config", cfg],
                                cwd=ROOT, stdout=open(os.path.join(D, "gateway.log"), "w"), stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                try:
                    if call("GET", "/healthz")[0] == 200:
                        break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                time.sleep(1)
            for k in call("GET", "/v1/kb")[1].get("bases", []):
                call("DELETE", f"/v1/kb/{k['id']}")
            s, r = call("POST", "/v1/kb", {"name": f"chunks {size}", "description": "experiment"})
            kb = r["id"]
            body, ctype = multipart([(os.path.basename(f), open(f, "rb").read()) for f in docs])
            call("POST", f"/v1/kb/{kb}/documents", raw=body, ctype=ctype)
            t = time.time()
            while time.time() - t < 1800:
                ds = call("GET", f"/v1/kb/{kb}/documents")[1].get("documents", [])
                if ds and all(d["status"] in ("ready", "failed") and d["n_embedded"] >= d["n_chunks"] for d in ds):
                    break
                time.sleep(5)
            n_chunks = sum(d["n_chunks"] for d in ds)
            slots = max(1, a.budget // size)
            res = {"passages": n_chunks, "slots": slots, "sets": {}}
            for name, qs in qsets.items():
                ranks = []
                for q, ph in qs:
                    hits = call("POST", f"/v1/kb/{kb}/search", {"query": q, "k": 20, "mode": "hybrid"})[1].get("results", [])
                    ranks.append(next((i for i, h in enumerate(hits, 1) if any(norm(p) in norm(h["text"]) for p in ph)), None))
                n = len(ranks)
                res["sets"][name] = {
                    "n": n, "R@1": sum(1 for x in ranks if x == 1) / n, f"R@{slots} (the same {a.budget} characters)": sum(1 for x in ranks if x and x <= slots) / n,
                    "R@4": sum(1 for x in ranks if x and x <= 4) / n, "MRR": round(sum(1 / x for x in ranks if x) / n, 3)}
            results[size] = res
            print(size, json.dumps(res))
        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
    json.dump({"budget_chars": a.budget, "results": results}, open(a.out, "w"), indent=1)
    print("\nchunk size  passages  slots  " + "  ".join(f"{n}: R@1  R@budget  MRR" for n in qsets))
    for size, r in results.items():
        cells = []
        for n in qsets:
            st = r["sets"][n]
            key = "R@%d (the same %d characters)" % (r["slots"], a.budget)
            cells.append(f"{st['R@1']:.2f}  {st[key]:.2f}  {st['MRR']:.3f}")
        print(f"{size:10}  {r['passages']:8}  {r['slots']:5}  " + "   ".join(cells))


if __name__ == "__main__":
    sys.exit(main())
