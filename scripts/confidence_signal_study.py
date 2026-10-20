#!/usr/bin/env python3
"""Confidence-signal study (P8-T01): which token-level confidence signal of a System 1 draft best predicts
that the draft is wrong? (Metacognitive sensitivity: Fleming & Lau 2014; "lowest group confidence": DeepConf 2025.)

Drafts GSM8K items with System 1 under the production prompt and sampling settings (temperature 0.2, top_p 0.8,
no thinking), with per-token log-probs and the top-5 alternatives, scores each draft with the gateway's exact
math verifier, and saves everything to benchmarks/confidence_signal_study.json. Analysis (--analyse) computes,
for each candidate signal, the AUROC for predicting a wrong draft, with a bootstrap 95% CI, and the escalation
rate / wrong-draft recall at matched escalation rates. Nothing in the gateway is changed by this script.

Usage: python3 scripts/confidence_signal_study.py [--limit 150] [--parallel 2] [--offset 0]
       python3 scripts/confidence_signal_study.py --samples [--limit 60]   # k = 2 cheap samples per saved draft (agreement signal)
       python3 scripts/confidence_signal_study.py --analyse
Calls llama-server (System 1) directly on 127.0.0.1:8001 and the gateway on 127.0.0.1:8080 for verification.
"""
import concurrent.futures as cf, json, math, os, random, sys, time, urllib.request
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = tomllib.load(open(os.path.join(ROOT, "config", "nafs-aql.toml"), "rb"))
S1 = "http://127.0.0.1:8001/v1/chat/completions"
GW = "http://127.0.0.1:8080"
OUT = os.path.join(ROOT, "benchmarks", "confidence_signal_study.json")
MATH_INSTRUCTION = "Solve step by step, briefly. End with a final line of the form 'Answer: <number>'."  # orchestrator::domain_instruction(Math)
WINDOW = 16

args = sys.argv[1:]
def arg(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def draft(item):
    t0 = time.time()
    r = post(S1, {
        "model": "system1",
        "messages": [{"role": "system", "content": f"{CFG['assistant']['system_prompt']}\n\n{MATH_INSTRUCTION}"},
                     {"role": "user", "content": item["prompt"]}],
        "max_tokens": CFG["deliberation"]["draft_max_tokens"], "temperature": CFG["deliberation"]["draft_temperature"],
        "top_p": 0.8, "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]
    toks = c["logprobs"]["content"]
    lps = [t["logprob"] for t in toks if isinstance(t.get("logprob"), (int, float)) and math.isfinite(t["logprob"])]
    # Truncated entropy over the top-5 alternatives (renormalised) and the top-1/top-2 margin, per token.
    ents, margins = [], []
    for t in toks:
        alts = [a["logprob"] for a in t.get("top_logprobs", []) if math.isfinite(a.get("logprob", float("nan")))]
        if len(alts) >= 2:
            m = max(alts)
            z = sum(math.exp(a - m) for a in alts)
            ps = [math.exp(a - m) / z for a in alts]
            ents.append(-sum(p * math.log(p) for p in ps if p > 0))
            s = sorted(alts, reverse=True)
            margins.append(s[0] - s[1])
    v = post(f"{GW}/v1/verify", {"answer": c["message"]["content"], "verification": item["verification"]})
    return {"id": item["id"], "reference": item["verification"]["reference_answer"], "tokens": len(lps),
            "finish_reason": c.get("finish_reason"), "passed": bool(v.get("passed")), "verifier": v.get("verifier"),
            "wall_s": round(time.time() - t0, 2), "timings": r.get("timings"), "logprobs": [round(x, 4) for x in lps],
            "entropy5": [round(x, 4) for x in ents], "margin": [round(x, 4) for x in margins],
            "answer_tail": c["message"]["content"][-120:]}


ANSWER_RE = __import__("re").compile(r"answer\s*[:=]?\s*\$?\s*(-?[\d,]*\.?\d+)", __import__("re").I)


def final_number(text):
    m = ANSWER_RE.findall(text or "")
    if m:
        return m[-1].replace(",", "")
    nums = __import__("re").findall(r"-?\d[\d,]*\.?\d*", text or "")
    return nums[-1].replace(",", "") if nums else None


def same_number(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except ValueError:
        return False


def samples_for(item, d):
    """The production agreement feature: k = 2 cheap System 1 samples (sample_temperature, sample_max_tokens_hard,
    seeds 1000 + i), compared with the draft's final number (exact for numeric answers, as answer_similarity does)."""
    t0 = time.time()
    outs = []
    for i in range(CFG["gate"]["k_samples"]):
        r = post(S1, {
            "model": "system1", "seed": 1000 + i,
            "messages": [{"role": "system", "content": f"{CFG['assistant']['system_prompt']}\n\n{MATH_INSTRUCTION}"},
                         {"role": "user", "content": item["prompt"]}],
            "max_tokens": CFG["gate"]["sample_max_tokens_hard"], "temperature": CFG["gate"]["sample_temperature"],
            "top_p": 0.8, "chat_template_kwargs": {"enable_thinking": False}})
        c = r["choices"][0]
        outs.append({"answer": final_number(c["message"]["content"]), "finish_reason": c.get("finish_reason"),
                     "tokens": r.get("usage", {}).get("completion_tokens")})
    draft_ans = final_number(d.get("answer_tail"))
    valid = [o for o in outs if o["finish_reason"] != "length" and o["answer"] is not None]
    agreement = (sum(1 for o in valid if same_number(o["answer"], draft_ans)) / len(valid)) if valid else None
    return {"draft_answer": draft_ans, "samples": outs, "agreement": agreement, "samples_wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- signals
def running_min_window(xs, w):
    if not xs:
        return None
    if len(xs) <= w:
        return sum(xs) / len(xs)
    s = sum(xs[:w]); best = s
    for i in range(w, len(xs)):
        s += xs[i] - xs[i - w]
        best = min(best, s)
    return best / w


def signals(d):
    lp = d["logprobs"]; n = len(lp)
    if n == 0:
        return None
    mean = sum(lp) / n
    tail = lp[-32:]
    bottom = sorted(lp)[: max(1, n // 10)]
    e = d.get("entropy5") or []
    mg = d.get("margin") or []
    out = {
        "mean_logprob": mean,                                   # the fixed rule's signal
        "min_logprob": min(lp),
        "low_window_logprob": running_min_window(lp, WINDOW),   # DeepConf lowest group confidence
        "tail_logprob": sum(tail) / len(tail),                  # DeepConf tail confidence
        "bottom10_logprob": sum(bottom) / len(bottom),          # DeepConf bottom-10% group confidence
        "neg_mean_entropy5": -(sum(e) / len(e)) if e else mean, # higher = more confident
        "neg_max_window_entropy5": -(-running_min_window([-x for x in e], WINDOW)) if e else mean,
        "mean_margin": (sum(mg) / len(mg)) if mg else -mean,
        "low_window_margin": running_min_window(mg, WINDOW) if mg else -mean,
    }
    if d.get("agreement") is not None:
        out["agreement"] = d["agreement"]                      # Theorem 6: self-consistency with the draft
        out["agreement_then_min"] = d["agreement"] + 0.01 * max(-20.0, min(lp))  # ties broken by the min log-prob
    return out


def auroc(scores, labels):
    # P(score_wrong < score_right): a lower confidence should flag a wrong draft.
    pos = [s for s, y in zip(scores, labels) if y]      # wrong drafts
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p < q else (0.5 if p == q else 0.0)
    return tot / (len(pos) * len(neg))


def analyse(drafts):
    drafts = [d for d in drafts if d.get("tokens", 0) > 0]
    labels = [not d["passed"] for d in drafts]
    n_wrong = sum(labels)
    sig = [signals(d) for d in drafts]
    names = [n for n in sig[0].keys() if n not in ("agreement", "agreement_then_min")]
    rng = random.Random(7)
    out = {"n": len(drafts), "wrong": n_wrong, "accuracy": 1 - n_wrong / max(1, len(drafts)), "window": WINDOW, "signals": {}}
    tau = CFG["gate"]["fixed"]["mean_logprob_threshold"] + 0.25 * 0.5  # fixed rule at the default lambda (care = 0.5)
    base_rate = sum(1 for s in sig if s["mean_logprob"] < tau) / len(sig)
    for name in names:
        sc = [s[name] for s in sig]
        a = auroc(sc, labels)
        boots = []
        for _ in range(400):
            idx = [rng.randrange(len(sc)) for _ in sc]
            boots.append(auroc([sc[i] for i in idx], [labels[i] for i in idx]))
        boots = sorted(b for b in boots if b == b)
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        # Recall of wrong drafts when escalating the least confident `rate` fraction (matched to the fixed rule's rate).
        rows = {}
        for rate in sorted({round(base_rate, 3), 0.1, 0.2, 0.3}):
            k = max(1, int(round(rate * len(sc))))
            order = sorted(range(len(sc)), key=lambda i: sc[i])[:k]
            caught = sum(1 for i in order if labels[i])
            rows[str(rate)] = {"threshold": sc[order[-1]], "escalated": k, "wrong_caught": caught,
                               "recall": caught / max(1, n_wrong), "precision": caught / k}
        out["signals"][name] = {"auroc": a, "ci95": [lo, hi], "at_escalation_rate": rows}
    out["fixed_rule"] = {"threshold": tau, "escalation_rate": base_rate,
                         "wrong_caught": sum(1 for s, y in zip(sig, labels) if y and s["mean_logprob"] < tau)}
    # Expected cascade accuracy on unverified mathematics (report, Prop. "cascade difference"): kept drafts keep
    # their own correctness, escalated items get System 2's accuracy p2 (11/11 in run e7607cbb; 0.9 as a
    # conservative bound), at the operating escalation rate of each signal.
    cascade = {}
    for name in names:
        sc = [s[name] for s in sig]
        for rate in (0.1, 0.2):
            k = max(1, int(round(rate * len(sc))))
            order = sorted(range(len(sc)), key=lambda i: sc[i])
            esc = set(order[:k])
            kept_right = sum(1 for i in range(len(sc)) if i not in esc and not labels[i])
            cascade[f"{name}@{rate}"] = {"threshold": sc[order[k - 1]], "escalated": k,
                                         "accuracy_p2_1.0": (kept_right + k) / len(sc),
                                         "accuracy_p2_0.9": (kept_right + 0.9 * k) / len(sc)}
    out["cascade_estimate"] = {"fast_only_accuracy": out["accuracy"], "by_signal_and_rate": cascade}
    # Agreement arm: only the drafts that have samples.
    withs = [(s, y, d) for s, y, d in zip(sig, labels, drafts) if "agreement" in s]
    if withs:
        ys = [y for _, y, _ in withs]
        agr = {"n": len(withs), "wrong": sum(ys), "accuracy": 1 - sum(ys) / len(withs), "signals": {}}
        for name in ("agreement", "agreement_then_min", "min_logprob", "low_window_logprob", "mean_logprob"):
            sc = [s[name] for s, _, _ in withs]
            a = auroc(sc, ys)
            boots = []
            for _ in range(400):
                idx = [rng.randrange(len(sc)) for _ in sc]
                boots.append(auroc([sc[i] for i in idx], [ys[i] for i in idx]))
            boots = sorted(b for b in boots if b == b)
            agr["signals"][name] = {"auroc": a, "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}
        # The production rule on agreement: escalate when fewer than agreement_threshold of the samples agree.
        thr = CFG["gate"]["fixed"]["agreement_threshold"]
        rule = {}
        for cut, label in ((thr, "below_threshold"), (1.0, "any_disagreement")):
            esc = [i for i, (s, _, _) in enumerate(withs) if s["agreement"] < cut]
            caught = sum(1 for i in esc if ys[i])
            kept_right = sum(1 for i in range(len(withs)) if i not in set(esc) and not ys[i])
            rule[label] = {"cut": cut, "escalated": len(esc), "rate": len(esc) / len(withs), "wrong_caught": caught,
                           "recall": caught / max(1, sum(ys)), "precision": caught / max(1, len(esc)),
                           "cascade_accuracy_p2_1.0": (kept_right + len(esc)) / len(withs),
                           "cascade_accuracy_p2_0.9": (kept_right + 0.9 * len(esc)) / len(withs)}
        agr["rules"] = rule
        agr["mean_samples_wall_s"] = sum(d["samples_wall_s"] for _, _, d in withs) / len(withs)
        out["agreement_arm"] = agr
    return out


if __name__ == "__main__":
    if "--analyse" in args:
        data = json.load(open(OUT))
        data["analysis"] = analyse(data["drafts"])
        json.dump(data, open(OUT, "w"), indent=1)
        a = data["analysis"]
        print(f"n={a['n']} wrong={a['wrong']} accuracy={a['accuracy']:.3f}; fixed rule escalates {a['fixed_rule']['escalation_rate']:.2f}, catches {a['fixed_rule']['wrong_caught']}")
        for k, v in sorted(a["signals"].items(), key=lambda kv: -kv[1]["auroc"]):
            r = v["at_escalation_rate"]
            print(f"{k:26s} AUROC {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  " +
                  "  ".join(f"@{rate}: recall {x['recall']:.2f} prec {x['precision']:.2f}" for rate, x in r.items()))
        if "agreement_arm" in a:
            g = a["agreement_arm"]
            print(f"agreement arm: n={g['n']} wrong={g['wrong']} accuracy={g['accuracy']:.3f} samples wall {g['mean_samples_wall_s']:.1f}s")
            for k, v in g["signals"].items():
                print(f"  {k:22s} AUROC {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
            for k, v in g["rules"].items():
                print(f"  rule {k:18s} rate {v['rate']:.2f} recall {v['recall']:.2f} precision {v['precision']:.2f} cascade acc {v['cascade_accuracy_p2_1.0']:.3f} / {v['cascade_accuracy_p2_0.9']:.3f}")
        print("cascade estimate (fast_only accuracy %.3f):" % a["cascade_estimate"]["fast_only_accuracy"])
        for k, v in a["cascade_estimate"]["by_signal_and_rate"].items():
            print(f"  {k:32s} thr {v['threshold']:.3f} esc {v['escalated']:3d}  acc(p2=1) {v['accuracy_p2_1.0']:.3f}  acc(p2=.9) {v['accuracy_p2_0.9']:.3f}")
        sys.exit(0)
    if "--samples" in args:
        data = json.load(open(OUT))
        limit, parallel = arg("--limit", 60), arg("--parallel", 2)
        by_id = {d["id"]: d for d in data["drafts"]}
        items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "bench", "gsm8k.jsonl"))}
        todo = [d for d in data["drafts"][:limit] if "agreement" not in d]
        print(f"{limit - len(todo)} drafts have samples, {len(todo)} to sample", flush=True)
        with cf.ThreadPoolExecutor(parallel) as ex:
            for d, r in zip(todo, ex.map(lambda d: samples_for(items[d["id"]], d), todo)):
                by_id[d["id"]].update(r)
                print(f"{d['id']} {'ok ' if d['passed'] else 'WRONG'} draft {r['draft_answer']} samples {[o['answer'] for o in r['samples']]} agreement {r['agreement']} {r['samples_wall_s']:.1f}s", flush=True)
                json.dump(data, open(OUT, "w"), indent=1)
        print("done", flush=True)
        sys.exit(0)
    limit, parallel, offset = arg("--limit", 150), arg("--parallel", 2), arg("--offset", 0)
    items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "bench", "gsm8k.jsonl"))]
    rng = random.Random(20261020)
    rng.shuffle(items)
    items = items[offset:offset + limit]
    done = {}
    if os.path.exists(OUT):
        for d in json.load(open(OUT)).get("drafts", []):
            done[d["id"]] = d
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} drafts saved, {len(todo)} to draft", flush=True)
    meta = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": "i3-8100 shared machine",
            "model": CFG["backends"]["llama"]["system1"]["hf_id"], "temperature": CFG["deliberation"]["draft_temperature"],
            "max_tokens": CFG["deliberation"]["draft_max_tokens"], "dataset": "gsm8k test (shuffled, seed 20261020)"}
    def save():
        json.dump({**meta, "drafts": list(done.values())}, open(OUT, "w"), indent=1)
    with cf.ThreadPoolExecutor(parallel) as ex:
        for d in ex.map(lambda it: (lambda: draft(it))(), todo):
            done[d["id"]] = d
            print(f"{d['id']} {'ok ' if d['passed'] else 'WRONG'} {d['tokens']:4d} tok {d['wall_s']:6.1f}s mean {sum(d['logprobs'])/max(1,d['tokens']):.3f} low16 {running_min_window(d['logprobs'],16):.3f}", flush=True)
            save()
    print("done", flush=True)
