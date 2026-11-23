#!/usr/bin/env python3
"""Phase 38 reasoning-cache benchmark (benchmarks/arc_phase38).

    python3 scripts/arc_bench.py build      System 2 solves each seed (force_mode slow, cache off), then the
                                            builder turns its answer into an entry (POST /v1/arc/build, wait)
    python3 scripts/arc_bench.py test       every seed and variant through POST /v1/arc/try (cache alone, floor
                                            0.80 so the threshold can be swept afterwards), scored by the oracle
    python3 scripts/arc_bench.py latency    a few hits answered three ways, interleaved: cache (entry trusted),
                                            the usual path with the cache off, System 2 forced
    python3 scripts/arc_bench.py compile    compile the cache to one module and run it on every question
    python3 scripts/arc_bench.py report     the tables (sweep of the meaning-tier floor included)

Results go to benchmarks/arc_phase38/results.json; each step resumes from it. Requests are ephemeral (nothing is
stored in the conversation history or the memory). Gateway at $NAFS_URL (default http://127.0.0.1:8080).
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

URL = os.environ.get("NAFS_URL", "http://127.0.0.1:8080")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "benchmarks/arc_phase38/items.json")
OUT = os.path.join(ROOT, "benchmarks/arc_phase38/results.json")


def call(path, body=None, method=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"}, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"builds": {}, "tests": [], "latency": [], "compile": {}}


def save(res):
    json.dump(res, open(OUT, "w"), indent=1)


def oracle(item, v):
    return eval(item["oracle"], {"v": v})


def right(got, want):
    try:
        g = float(str(got).replace(",", ""))
    except ValueError:
        try:
            from fractions import Fraction
            g = float(Fraction(str(got)))
        except Exception:  # noqa: BLE001
            return False
    return abs(g - float(want)) <= max(1e-6, 1e-4 * abs(float(want)))


def load_avg():
    return float(subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.split()[1])


def chat(q, **kw):
    body = {"messages": [{"role": "user", "content": q}], "ephemeral": True, "retrieve": False, "domain_hint": "math"}
    body.update(kw)
    t = time.time()
    r = call("/v1/chat", body)
    return r, time.time() - t


def step_build(res, items):
    for it in items:
        if it["id"] in res["builds"] and res["builds"][it["id"]].get("entry"):
            continue
        want = oracle(it, it["seed"]["v"])
        la = load_avg()
        r, wall = chat(it["seed"]["q"], force_mode="slow", arc=False, verification={"reference_answer": str(want)})
        rec = {"source_wall_s": round(wall, 1), "source_assurance": r["assurance"], "source_tokens": r["usage"]["completion_tokens"],
               "source_reasoning_tokens": r["usage"]["reasoning_tokens"], "load_before": la}
        print(f"[{it['id']}] System 2: {wall:.0f} s, {r['assurance']}", flush=True)
        t = time.time()
        b = call("/v1/arc/build", {"question": it["seed"]["q"], "answer": r["message"]["content"], "reasoning": r.get("reasoning") or "",
                                   "expected": str(want), "domain": "math", "wait": True})
        rep = b.get("report") or {}
        rec.update({"build_wall_s": round(time.time() - t, 1), "ok": rep.get("ok"), "entry": rep.get("entry"), "rounds": rep.get("rounds"),
                    "builder_tokens": rep.get("builder_tokens"), "reasons": rep.get("reasons"), "load_after": load_avg()})
        e = b.get("entry") or {}
        rec["functions"] = len(e.get("functions") or [])
        rec["reused"] = sum(1 for v in (e.get("links") or {}).values() if not v.startswith(f"{e.get('id')}:"))
        rec["template"] = e.get("template")
        print(f"[{it['id']}] build: ok={rec['ok']} rounds={rec['rounds']} {rec['build_wall_s']} s {rec['reasons']}", flush=True)
        res["builds"][it["id"]] = rec
        save(res)


def step_test(res, items):
    done = {(t["item"], t["q"]) for t in res["tests"]}
    for it in items:
        qs = [dict(kind="seed", **it["seed"])] + it["variants"]
        for v in qs:
            if (it["id"], v["q"]) in done:
                continue
            want = v["answer"] if "answer" in v else oracle(it, v["v"])
            t = time.time()
            r = call("/v1/arc/try", {"question": v["q"], "floor": 0.80})
            rec = {"item": it["id"], "kind": v["kind"], "q": v["q"], "want": want, "hit": r.get("hit"), "tier": r.get("tier"),
                   "similarity": r.get("similarity"), "entry": r.get("entry"), "answer": r.get("answer"), "why": r.get("why"),
                   "ms": r.get("ms"), "run_ms": r.get("run_ms"), "wall_ms": round((time.time() - t) * 1000)}
            rec["right"] = bool(r.get("hit")) and right(r.get("answer"), want)
            own = res["builds"].get(it["id"], {}).get("entry")
            rec["own_entry"] = own
            res["tests"].append(rec)
            save(res)
            print(f"[{it['id']}:{v['kind']}] hit={rec['hit']} tier={rec['tier']} sim={rec['similarity']} right={rec['right']} {rec['why'] or ''}", flush=True)


def step_latency(res, items, n=6):
    # Hits of one kind (ARC_LAT_KIND, default phrasal), one per item.
    kind = os.environ.get("ARC_LAT_KIND", "phrasal")
    pick, seen = [], set()
    for t in res["tests"]:
        if t["hit"] and t["right"] and t["kind"] == kind and t["item"] not in seen:
            pick.append(t)
            seen.add(t["item"])
    pick = pick[:n]
    entries = {t["entry"] for t in pick}
    before = {e["id"]: e["status"] for e in call("/v1/arc")["entries"]}
    for e in entries:
        call(f"/v1/arc/{e}/status", {"status": "trusted"})
    done = {(x["q"], x["arm"]) for x in res["latency"]}
    try:
        for i, t in enumerate(pick):
            arms = ["cache", "usual", "system2"]
            arms = arms[i % 3:] + arms[: i % 3]  # interleave the order
            for arm in arms:
                if (t["q"], arm) in done:
                    continue
                kw = {"cache": {}, "usual": {"arc": False}, "system2": {"arc": False, "force_mode": "slow"}}[arm]
                la = load_avg()
                r, wall = chat(t["q"], **kw)
                from_cache = r["decision"]["rule"] == "reasoning_cache"
                got = r["message"]["content"]
                rec = {"q": t["q"], "item": t["item"], "tier": t["tier"], "kind": t["kind"], "arm": arm, "wall_s": round(wall, 2), "assurance": r["assurance"],
                       "mode": r["reasoning_mode"], "from_cache": from_cache, "load": la, "want": t["want"],
                       "right": right(_final(got), t["want"]), "tokens": r["usage"]["completion_tokens"]}
                res["latency"].append(rec)
                save(res)
                print(f"[{t['item']}:{arm}] {wall:.1f} s cache={from_cache} right={rec['right']} load={la}", flush=True)
    finally:
        for e in entries:
            call(f"/v1/arc/{e}/status", {"status": before.get(e, "candidate")})


def _final(text):
    import re
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def step_compile(res, items):
    r = call("/v1/arc/compile", {"all": True})
    path = os.path.join(ROOT, r["path"]) if not os.path.isabs(r["path"]) else r["path"]
    out = {"entries": r["entries"], "bytes": r["bytes"], "rows": []}
    for t in res["tests"]:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", path, t["q"]], capture_output=True, text=True, timeout=30)
        ms = round((time.time() - t0) * 1000)
        try:
            v = json.loads(p.stdout)
        except ValueError:
            v = {"hit": False, "error": p.stderr[-300:]}
        out["rows"].append({"q": t["q"], "kind": t["kind"], "hit": v.get("hit"), "right": bool(v.get("hit")) and right(v.get("answer"), t["want"]),
                            "same_as_runtime": (not v.get("hit")) or (t["hit"] and right(v.get("answer"), t["answer"] if t["answer"] is not None else float("nan"))),
                            "ms": ms})
    res["compile"] = out
    save(res)
    rows = out["rows"]
    print(f"compiled {out['entries']} entries, {out['bytes']} bytes; hits {sum(1 for x in rows if x['hit'])}/{len(rows)}, "
          f"right {sum(1 for x in rows if x['right'])}, wrong {sum(1 for x in rows if x['hit'] and not x['right'])}")


def step_report(res, items):
    b = res["builds"]
    ok = [x for x in b.values() if x.get("ok")]
    print(f"Builds: {len(ok)}/{len(b)} ok; rounds {[x.get('rounds') for x in b.values()]}")
    if ok:
        sw = sum(x["source_wall_s"] for x in ok) / len(ok)
        bw = sum(x["build_wall_s"] for x in ok) / len(ok)
        print(f"  System 2 solve {sw:.0f} s mean; build {bw:.0f} s mean ({bw / sw:.1f}x the solve); functions {sum(x['functions'] for x in ok)}; reused {sum(x['reused'] for x in ok)}")
    tests = [t for t in res["tests"] if b.get(t["item"], {}).get("entry")]
    kinds = ["seed", "phrasal", "parametric", "combined", "near_miss"]
    for floor in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.01):
        line = []
        for k in kinds:
            ts = [t for t in tests if t["kind"] == k]
            hit = [t for t in ts if t["hit"] and (t["tier"] != "meaning" or (t["similarity"] or 0) >= floor)]
            rt = [t for t in hit if t["right"]]
            line.append(f"{k} {len(hit)}/{len(ts)} hit, {len(rt)} right")
        print(f"floor {floor:.2f}: " + "; ".join(line))
    for tier in ("template", "shape", "meaning"):
        ts = [t for t in tests if t["hit"] and t["tier"] == tier]
        if ts:
            import statistics
            print(f"  tier {tier}: {len(ts)} hits, {sum(t['right'] for t in ts)} right, median {statistics.median(t['ms'] for t in ts):.0f} ms")
    if res["latency"]:
        for tier in ("template", "meaning"):
            for arm in ("cache", "usual", "system2"):
                xs = [x for x in res["latency"] if x["arm"] == arm and x["tier"] == tier]
                if xs:
                    w = sorted(x["wall_s"] for x in xs)
                    import statistics
                    print(f"  {tier} questions, {arm}: n={len(xs)} median {statistics.median(w):.2f} s (min {w[0]:.2f}, max {w[-1]:.2f}), right {sum(x['right'] for x in xs)}/{len(xs)}, from cache {sum(x['from_cache'] for x in xs)}")


def main():
    items = json.load(open(ITEMS))["items"]
    only = os.environ.get("ARC_ITEMS")
    if only:
        items = [i for i in items if i["id"] in only.split(",")]
    res = load()
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"build": step_build, "test": step_test, "latency": step_latency, "compile": step_compile, "report": step_report}[step](res, items)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    main()
