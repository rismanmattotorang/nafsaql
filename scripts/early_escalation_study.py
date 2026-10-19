#!/usr/bin/env python3
"""Early-escalation study (P7-T03, FR-080): would stopping an unconfident draft early change the gate's decision?

For each open-ended prompt, System 1 drafts the full answer with the production prompt and sampling settings and
returns per-token log-probs. The fixed rule escalates when the full draft's mean log-prob is below the threshold
(at the default lambda). Early escalation fires at the first token n >= N whose running mean is below the
threshold minus a margin. A disagreement is an early escalation of a draft the full rule would have kept.
The script reports, per (N, margin): agreement, how many escalations were caught early, and draft tokens saved.

Usage: python3 scripts/early_escalation_study.py [--limit 40] [--parallel 2] [--rescore]
--rescore recomputes the grid from the log-probs saved in the previous output, without new drafts.
Writes benchmarks/early_escalation_study.json. Calls llama-server (System 1) directly on 127.0.0.1:8001.
"""
import concurrent.futures as cf, json, math, os, sys, time, urllib.request
try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = tomllib.load(open(os.path.join(ROOT, "config", "nafs-aql.toml"), "rb"))
S1 = "http://127.0.0.1:8001/v1/chat/completions"
OPEN_INSTRUCTION = "Answer helpfully and concisely."  # orchestrator::domain_instruction(Open)

# Open-ended prompts (written for this study; Apache-2.0 with the repository).
PROMPTS = [
    "Explain why the sky looks blue during the day and red at sunset.",
    "Give me three tips for writing a clear email to a busy manager.",
    "What are the trade-offs between renting and buying a home?",
    "Describe how a refrigerator keeps food cold.",
    "Suggest a simple weekly exercise plan for a beginner.",
    "Why do we dream? Summarise the main theories.",
    "How should I prepare for a job interview at a software company?",
    "Explain the difference between weather and climate.",
    "What makes a good teacher?",
    "Write a short paragraph encouraging someone who failed an exam.",
    "How does compound interest work, in plain words?",
    "What are the pros and cons of working from home?",
    "Describe the water cycle for a ten-year-old.",
    "How can a small team improve its meetings?",
    "Explain what a vaccine does in the body.",
    "Give advice for learning a new language as an adult.",
    "Why is sleep important for memory?",
    "How do bees make honey?",
    "What should someone consider before adopting a dog?",
    "Explain the idea of supply and demand with an everyday example.",
    "Write a two-sentence story about a lighthouse keeper.",
    "How can I reduce food waste at home?",
    "What is the difference between a virus and a bacterium?",
    "Describe a good way to give constructive feedback.",
    "Why do leaves fall from trees in autumn?",
    "Explain how a bicycle stays upright when moving.",
    "What are some ways to stay focused while studying?",
    "Describe the benefits of reading fiction.",
    "How do noise-cancelling headphones work?",
    "Explain what inflation is and why it matters.",
    "Tell me about the history of the printing press in a few sentences.",
    "How do I start a vegetable garden on a balcony?",
    "What are the signs of burnout and how can one recover?",
    "Explain the greenhouse effect simply.",
    "Give a friendly reminder message for a team deadline.",
    "Why do cats purr?",
    "Describe how to make a good cup of tea.",
    "What qualities make a city pleasant to live in?",
    "How do airplanes stay in the air?",
    "Explain why exercise improves mood.",
]


def draft(prompt):
    sysmsg = f"{CFG['assistant']['system_prompt']}\n\n{OPEN_INSTRUCTION}"
    d = CFG["deliberation"]
    body = {"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}],
            "max_tokens": d["draft_max_tokens"], "temperature": d["draft_temperature"], "top_p": 0.8,
            "logprobs": True, "top_logprobs": 1, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(S1, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        v = json.load(r)
    lps = [x["logprob"] for x in (v["choices"][0].get("logprobs") or {}).get("content", []) if math.isfinite(x["logprob"])]
    return {"prompt": prompt, "logprobs": lps, "wall_s": round(time.time() - t, 2)}


def evaluate(rows, thr, n_min, margin):
    agree = fired = full_esc = caught = saved = kept_wrongly = 0
    for r in rows:
        lp = r["logprobs"]
        if not lp:
            continue
        full = sum(lp) / len(lp) < thr
        full_esc += full
        fire_at, s = None, 0.0
        for i, x in enumerate(lp, 1):
            s += x
            if i >= n_min and s / i < thr - margin:
                fire_at = i
                break
        early = fire_at is not None
        decision = True if early else full
        agree += decision == full
        if early:
            fired += 1
            saved += len(lp) - fire_at
            caught += full
            kept_wrongly += not full
    n = sum(1 for r in rows if r["logprobs"])
    return {"n_min": n_min, "margin": margin, "n": n, "agreement": agree / n if n else None,
            "early_fired": fired, "full_escalations": full_esc, "caught_early": caught,
            "false_early": kept_wrongly, "tokens_saved": saved,
            "tokens_saved_per_escalation": saved / fired if fired else 0.0}


def main():
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else len(PROMPTS)
    par = int(args[args.index("--parallel") + 1]) if "--parallel" in args else 2
    g = CFG["gate"]
    care = max(0.0, 1.0 - min(1.0, g["lambda_default"] / (2 * g["lambda_default"])))
    thr = g["fixed"]["mean_logprob_threshold"] + 0.25 * care
    rows = []
    out_path = os.path.join(ROOT, "benchmarks", "early_escalation_study.json")
    if "--rescore" in args:
        rows = [{"prompt": d["prompt"], "logprobs": d["logprobs"], "wall_s": d["wall_s"]} for d in json.load(open(out_path))["drafts"]]
    else:
      with cf.ThreadPoolExecutor(par) as ex:
        for i, r in enumerate(ex.map(draft, PROMPTS[:limit]), 1):
            rows.append(r)
            m = sum(r["logprobs"]) / max(1, len(r["logprobs"]))
            print(f"{i:3d} {len(r['logprobs']):4d} tok  mean {m:+.3f}  {r['wall_s']:6.1f}s  {r['prompt'][:50]}", flush=True)
    grid = [evaluate(rows, thr, n, mg) for n in (32, 48, 64, 96) for mg in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.25)]
    chosen = next((x for x in grid if x["n_min"] == g.get("early_escalation_tokens", 48) and abs(x["margin"] - g.get("early_escalation_margin", 0.15)) < 1e-9), grid[0])
    # The best setting that meets FR-080: agreement >= 95%, then the most tokens saved.
    ok = [x for x in grid if (x["agreement"] or 0) >= 0.95]
    best = max(ok, key=lambda x: (x["tokens_saved"], -x["false_early"], x["margin"])) if ok else None
    out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threshold": thr,
           "draft_max_tokens": CFG["deliberation"]["draft_max_tokens"], "host": "shared machine (other workloads running)",
           "configured": chosen, "best_meeting_fr080": best, "grid": grid,
           "rule": "FR-080: early escalation may be on by default only if agreement with full-draft decisions is >= 95%.",
           "on_by_default_allowed": (chosen["agreement"] or 0) >= 0.95,
           "drafts": [{"prompt": r["prompt"], "tokens": len(r["logprobs"]), "mean_logprob": sum(r["logprobs"]) / max(1, len(r["logprobs"])), "wall_s": r["wall_s"], "logprobs": [round(x, 4) for x in r["logprobs"]]} for r in rows]}
    json.dump(out, open(out_path, "w"))
    print("configured:", json.dumps(chosen))
    print("best meeting FR-080:", json.dumps(best))
    for x in grid:
        print(f"N={x['n_min']:3d} margin={x['margin']:.2f}  agreement {x['agreement']:.3f}  fired {x['early_fired']:2d}  caught {x['caught_early']}/{x['full_escalations']}  false {x['false_early']}  saved {x['tokens_saved']}")


if __name__ == "__main__":
    main()
