#!/usr/bin/env python3
"""Cheap-sample A/B on factual prompts (P7-T02, NFR-031): what do the k = 2 agreement samples cost on this CPU?

The gate's fixed rule draws k cheap System 1 samples next to the draft on factual prompts. They share the four
cores with the draft, so they lengthen the draft and the time to first token. This script sends each prompt in
alternating order with the samples (the normal path) and without them (`force_mode: fast`, which skips the
samples and the gate), through the running gateway, and reports wall time, time to first token and decode speed
from the draft's stage timings, plus how often the gate escalated with the samples present.

Usage: python3 scripts/samples_ab.py [--rounds 2]   -> benchmarks/samples_ab_factual.json
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:8080"
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 2
PROMPTS = [
    "What is the capital of Japan, and what is it famous for? Two sentences.",
    "Who painted the Mona Lisa, and where is it displayed today?",
    "What is the largest planet in the solar system, and roughly how many Earths would fit inside it?",
    "In what year did the Apollo 11 mission land on the Moon, and who were the two astronauts who walked on it?",
    "What is the chemical symbol for gold, and what is its atomic number?",
    "Which river is the longest in Africa, and through how many countries does it flow?",
]


def ask(prompt, with_samples):
    body = {"messages": [{"role": "user", "content": prompt}], "ephemeral": True, "domain_hint": "factual"}
    if not with_samples:
        body["force_mode"] = "fast"
    req = urllib.request.Request(BASE + "/v1/chat", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        v = json.load(r)
    u = v["usage"]
    draft = next((s for s in u.get("stages", []) if s["stage"] == "draft"), {})
    return {"wall_ms": u["wall_ms"], "ttft_ms": u.get("ttft_ms"), "escalated": v["decision"]["escalate"], "mode": v["reasoning_mode"],
            "draft_ms": draft.get("ms"), "draft_tokens": draft.get("tokens"), "draft_decode_ms": draft.get("decode_ms"), "draft_prefill_ms": draft.get("prefill_ms"),
            "samples": sum(1 for s in u.get("stages", []) if s["stage"] in ("sample", "sample_after")), "elapsed_s": round(time.time() - t, 2)}


def main():
    rows = {"with_samples": [], "without_samples": []}
    for rnd in range(ROUNDS):
        for i, p in enumerate(PROMPTS):
            order = [True, False] if (rnd + i) % 2 == 0 else [False, True]
            for w in order:
                r = ask(p, w)
                r.update({"prompt": p, "round": rnd})
                rows["with_samples" if w else "without_samples"].append(r)
                print(f"r{rnd} {'k=2 ' if w else 'k=0 '} wall {r['wall_ms']:6d} ms  ttft {r['ttft_ms']}  draft {r['draft_tokens']} tok  esc {r['escalated']}  {p[:40]}", flush=True)
    summ = {}
    for k, rs in rows.items():
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        dec = [r["draft_tokens"] / r["draft_decode_ms"] * 1000 for r in rs if r.get("draft_decode_ms") and r.get("draft_tokens")]
        summ[k] = {"n": len(rs), "wall_p50_ms": med([r["wall_ms"] for r in rs]), "ttft_p50_ms": med([r["ttft_ms"] for r in rs if r["ttft_ms"] is not None]),
                   "draft_decode_tok_s": (sum(dec) / len(dec)) if dec else None, "escalated": sum(r["escalated"] for r in rs)}
    out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_average": os.getloadavg(), "rounds": ROUNDS, "summary": summ, "rows": rows}
    path = os.path.join(ROOT, "benchmarks", "samples_ab_factual.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
