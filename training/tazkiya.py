#!/usr/bin/env python3
"""Tazkiya: the distillation pipeline of Nafs-Aql (Phase 3; spec Section 6, TR-001..TR-008, FR-033, FR-070, SR-004, OR-005, ER-008).

    build             training set from the ledger: verified System 2 traces (+ approved consensus), redacted,
                      deduplicated, no domain above 30%, 5% abstention examples, 20% replay, train/val split, manifest
    synthesize        System 2 answers to *training-split* prompts (GSM8K train, MBPP train), kept only if an exact
                      verifier passes; labelled synthetic (TR-003); evaluation test splits are never used
    fetch-replay      general instruction data for the 20% replay share (TR-008): first turns of UltraChat 200k
                      (HuggingFaceH4/ultrachat_200k, MIT licence), sampled with a fixed seed and redacted
    regression-suite  the fixed 500-item multi-domain suite for ER-008 (data/bench/regression_500.jsonl)
    promote           TR-007 gate on paired regression runs (lower 95% bound of the gain > 0, point >= 2 points,
                      no domain lower bound below -1 point, calibration and coupling not worse, human sign-off);
                      on success the registry's active System 1 artefact changes (a configuration change, OR-005)
    rollback          restore the previous artefact in one command (OR-005)
    status            print the registry

Standard library only, so it runs on the CPU host. Training itself is training/train_unsloth.py on a GPU host.
"""
import argparse, hashlib, json, os, random, re, sqlite3, sys, time, urllib.parse, urllib.request
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DB = os.path.join(ROOT, "data", "runtime", "nafs.sqlite")
REGISTRY = os.path.join(ROOT, "config", "models.registry.json")
BENCH = os.path.join(ROOT, "data", "bench")
SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------- redaction (SR-004)
_RX = [
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("[URL]", re.compile(r"https?://\S+")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("[SECRET]", re.compile(r"\b(?:sk|pk|ghp|gho|xox[abp]|AKIA)[-_A-Za-z0-9]{12,}\b")),
    ("[SECRET]", re.compile(r"\b[A-Fa-f0-9]{32,}\b")),
    ("[IBAN]", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")),
    ("[PHONE]", re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w.])")),
]
_NAME = re.compile(r"\b(my name is|i am called|call me)\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)", re.I)

def _luhn(digits):
    s, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        s += n; alt = not alt
    return s % 10 == 0

def redact(text):
    """Replace personal data with typed placeholders. Card numbers are checked with Luhn so that
    ordinary long numbers in maths answers survive."""
    if not text:
        return text
    def card(m):
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)
    text = re.sub(r"\b(?:\d[ -]?){13,19}\b", card, text)
    for tag, rx in _RX:
        if tag == "[PHONE]":
            text = rx.sub(lambda m: tag if len(re.sub(r"\D", "", m.group(0))) >= 9 and not re.fullmatch(r"[\d.,]+", m.group(0)) else m.group(0), text)
        else:
            text = rx.sub(tag, text)
    return _NAME.sub(lambda m: f"{m.group(1)} [NAME]", text)

# ---------------------------------------------------------------- helpers
def norm_prompt(p):
    return re.sub(r"\s+", " ", re.sub(r"^\[[^\]]{1,40}\]\s*", "", p.strip().lower()))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def approx_tokens(text):
    return int(len(re.findall(r"\w+|[^\w\s]", text)) * 0.75) + 1

def rationale(reasoning, limit=150):
    """TR-001: a rationale of at most `limit` tokens distilled from the slow trace: the closing
    sentences of the reasoning, where the model states what it concluded and why."""
    if not reasoning:
        return ""
    text = re.sub(r"\[attempt \d+\]\s*", "", reasoning).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    # Budget forcing can cut the trace mid-sentence: drop a trailing fragment.
    if len(sents) > 1 and not re.search(r"[.!?)\]]$", sents[-1].strip()):
        sents = sents[:-1]
    out = []
    for s in reversed(sents):
        if approx_tokens(" ".join([s] + out)) > limit:
            break
        out.insert(0, s)
    if not out and sents:  # one very long sentence: keep its tail
        words = sents[-1].split()
        out = [" ".join(words[-int(limit / 0.75) + 1:])]
    return " ".join(out).strip()

def cap_domains(items, cap, seed):
    """FR-070: no domain may exceed `cap` of the set. Trims the over-represented domains, largest first,
    until every domain is within the cap (possible only with >= ceil(1/cap) domains)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it["domain"]].append(it)
    for v in by.values():
        rng.shuffle(v)
    if len(by) < 2:
        return items, {"feasible": False, "reason": "one domain only; the cap cannot be met"}
    while True:
        total = sum(len(v) for v in by.values())
        over = [d for d, v in by.items() if len(v) > cap * total + 1e-9]
        if not over:
            break
        d = max(over, key=lambda k: len(by[k]))
        others = total - len(by[d])
        keep = int(cap * others / (1 - cap)) if cap < 1 else len(by[d])
        if keep >= len(by[d]) or keep <= 0:
            by[d] = by[d][:max(1, keep)]
            if keep <= 0:
                break
        else:
            by[d] = by[d][:keep]
    out = [x for v in by.values() for x in v]
    total = len(out)
    shares = {d: round(len(v) / total, 3) for d, v in by.items()} if total else {}
    return out, {"feasible": all(s <= cap + 1e-9 for s in shares.values()), "shares": shares}

def bootstrap_ci(diffs, seed=7, n=2000):
    if not diffs:
        return None
    rng = random.Random(seed); k = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return boots[int(0.025 * n)], boots[int(0.975 * n) - 1]

# ---------------------------------------------------------------- build (FR-070, TR-001, TR-002, TR-008, FR-033, SR-004)
def ledger_examples(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in con.execute("PRAGMA table_info(conversations)")}
    if "reviews" not in tables or "training_opt_out" not in cols:
        con.close()
        raise SystemExit("the ledger predates Phase 3 (no reviews table or opt-out column): start the current gateway once to migrate it")
    q = """SELECT r.id, r.domain, r.assurance, r.mode, r.json, m.conversation_id, m.id,
                  (SELECT u.content FROM messages u WHERE u.conversation_id = m.conversation_id AND u.role='user' AND u.id < m.id ORDER BY u.id DESC LIMIT 1),
                  v.decision
           FROM replies r JOIN messages m ON m.reply_id = r.id JOIN conversations c ON c.id = m.conversation_id
           LEFT JOIN reviews v ON v.reply_id = r.id
           WHERE r.ephemeral = 0 AND c.training_opt_out = 0"""
    rows = con.execute(q).fetchall()
    con.close()
    out, skipped = [], Counter()
    for rid, dom, assurance, mode, js, conv, mid, question, review in rows:
        reply = json.loads(js or "{}")
        if not question:
            skipped["no_question"] += 1; continue
        if assurance == "verified" and mode == "slow":
            kind = "verified"
        elif assurance == "consensus" and review == "approved":
            kind = "consensus_reviewed"
        elif assurance == "abstained":
            kind = "abstained"
        else:
            skipped[f"{assurance}_{mode}" + ("_unreviewed" if assurance == "consensus" and not review else "")] += 1
            continue
        out.append({"reply_id": rid, "domain": dom or "other", "kind": kind, "question": question, "answer": reply.get("message", {}).get("content", ""),
                    "reasoning": reply.get("reasoning") or "", "verifier": reply.get("verifier"), "notice": reply.get("uncertainty_notice"),
                    "conversation_id": conv, "adapter_hash": "none"})
    return out, skipped

def to_example(e, synthetic=False):
    q = redact(re.sub(r"^\[[^\]]{1,40}\]\s*", "", e["question"]))
    ans = redact(e["answer"])
    v = e.get("verifier") or {}
    return {"schema_version": SCHEMA_VERSION, "id": hashlib.sha256((e.get("reply_id") or q).encode()).hexdigest()[:16], "domain": e["domain"],
            "prompt": [{"role": "user", "content": q}], "answer": ans, "rationale": redact(rationale(e.get("reasoning", ""))),
            "assurance": e["kind"], "verifier": {"name": v.get("verifier"), "passed": v.get("passed"), "rates": v.get("rates")},
            "source_trace": e.get("reply_id"), "synthetic": synthetic, "adapter_hash": e.get("adapter_hash", "none"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

ABSTAIN_ANSWER = ("I'm not confident I can answer this reliably. {why} If you can share more detail or a way to check the result, "
                  "I'll try again, and you may want to confirm with an authoritative source.")

def build(args):
    seed = args.seed
    rows, skipped = ledger_examples(args.db)
    core = [r for r in rows if r["kind"] in ("verified", "consensus_reviewed")]
    abst = [r for r in rows if r["kind"] == "abstained"]
    synth = []
    if os.path.exists(args.synthetic):
        synth = [json.loads(l) for l in open(args.synthetic) if l.strip()]
    examples = [to_example(r) for r in core]
    # Synthetic examples may make up at most 50% of the verified part (spec Section 6).
    max_syn = len(examples) if examples else len(synth)
    examples += synth[:max_syn]
    # Deduplicate on the normalised prompt, keeping the first (verified before synthetic).
    seen, dedup = set(), []
    for ex in examples:
        k = norm_prompt(ex["prompt"][-1]["content"])
        if k in seen:
            continue
        seen.add(k); dedup.append(ex)
    capped, cap_info = cap_domains(dedup, args.domain_cap, seed)
    n_core = len(capped)
    # 5% abstention examples (TR-008, FR-070): the share of the final set.
    n_abs = round(0.05 * n_core / 0.75) if n_core else 0
    rng = random.Random(seed); rng.shuffle(abst)
    abst_ex = []
    for r in abst[:n_abs]:
        e = dict(r); why = (r.get("notice") or "").split(".")[0]
        e["answer"] = ABSTAIN_ANSWER.format(why=(why + ".") if why else "")
        e["kind"] = "abstention"; e["reasoning"] = ""
        abst_ex.append(to_example(e))
    # 20% replay of general instruction data (TR-008).
    replay = []
    n_rep = round(0.20 * n_core / 0.75) if n_core else 0
    if args.replay and os.path.exists(args.replay):
        pool = [json.loads(l) for l in open(args.replay) if l.strip()]
        rng.shuffle(pool)
        for p in pool[:n_rep]:
            replay.append({"schema_version": SCHEMA_VERSION, "id": "replay-" + hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12], "domain": "replay",
                           "prompt": p.get("prompt") or [{"role": "user", "content": redact(p.get("instruction", ""))}], "answer": redact(p.get("answer") or p.get("output", "")),
                           "rationale": "", "assurance": "replay", "verifier": None, "source_trace": None, "synthetic": False, "adapter_hash": "none",
                           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    full = capped + abst_ex + replay
    rng.shuffle(full)
    os.makedirs(args.out, exist_ok=True)
    val = [x for x in full if int(hashlib.sha256(x["id"].encode()).hexdigest(), 16) % 10 == 0]
    val_ids = {x["id"] for x in val}
    train = [x for x in full if x["id"] not in val_ids]
    for name, xs in (("train", train), ("val", val)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    total = len(full) or 1
    problems = []
    if n_core < args.min_examples:
        problems.append(f"only {n_core} verified examples; Phase 3 entry needs >= {args.min_examples} (roadmap)")
    if not cap_info.get("feasible"):
        problems.append("domain cap of 30% not met (too few domains)")
    if len(abst_ex) < n_abs:
        problems.append(f"abstention examples {len(abst_ex)} < target {n_abs}")
    if len(replay) < n_rep:
        problems.append(f"replay examples {len(replay)} < target {n_rep} (pass --replay FILE with general instruction data, TR-008)")
    manifest = {"schema_version": SCHEMA_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": seed,
                "source_db_sha256": sha256_file(args.db), "counts": {"verified_or_reviewed": n_core, "synthetic": sum(1 for x in capped if x["synthetic"]),
                "abstention": len(abst_ex), "replay": len(replay), "train": len(train), "val": len(val)},
                "shares": {"abstention": round(len(abst_ex) / total, 3), "replay": round(len(replay) / total, 3)},
                "domain_cap": args.domain_cap, "cap": cap_info, "excluded": dict(skipped), "dedup_removed": len(examples) - len(dedup),
                "files": {n: sha256_file(os.path.join(args.out, f"{n}.jsonl")) for n in ("train", "val")},
                "rules": ["TR-002 verified or human-approved consensus only", "SR-004 redacted, opted-out conversations excluded",
                          "FR-070 dedup, domain cap, abstention", "TR-008 replay"], "ready": not problems, "problems": problems}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))
    return 0 if (manifest["ready"] or not args.strict) else 2

# ---------------------------------------------------------------- synthesize (TR-003)
def hf_rows(dataset, config, split, n, offset=0):
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset + len(out), "length": min(100, n - len(out))})
        data = json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60))
        rows = [r["row"] for r in data.get("rows", [])]
        if not rows:
            break
        out += rows
    return out

def synthesize(args):
    tasks = []
    for r in hf_rows("openai/gsm8k", "main", "train", args.n // 2, args.offset):
        tasks.append(("math", r["question"] + "\nEnd with 'Answer: <number>'.", {"reference_answer": r["answer"].split("####")[-1].strip().replace(",", "")}))
    for r in hf_rows("google-research-datasets/mbpp", "full", "train", args.n - len(tasks), args.offset):
        fn = r["test_list"][0].split("assert ")[-1].split("(")[0].strip()
        tasks.append(("code", f"{r['text']}\nWrite a Python function named `{fn}`.", {"tests": r["test_list"]}))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "a") as f:
        for i, (dom, prompt, ver) in enumerate(tasks):
            body = {"messages": [{"role": "user", "content": prompt}], "force_mode": "slow", "verification": ver, "ephemeral": True, "show_reasoning": True, "audit": False}
            try:
                req = urllib.request.Request(args.gateway + "/v1/chat", json.dumps(body).encode(), {"Content-Type": "application/json", **({"Authorization": f"Bearer {args.token}"} if args.token else {})})
                rep = json.load(urllib.request.urlopen(req, timeout=1800))
            except Exception as e:
                print(f"[{i + 1}/{len(tasks)}] error: {e}", file=sys.stderr); continue
            if rep.get("assurance") != "verified":
                print(f"[{i + 1}/{len(tasks)}] {dom}: not verified, dropped"); continue
            e = {"reply_id": rep["reply_id"], "domain": dom, "kind": "verified", "question": prompt, "answer": rep["message"]["content"], "reasoning": rep.get("reasoning") or "",
                 "verifier": rep.get("verifier"), "adapter_hash": "none"}
            f.write(json.dumps(to_example(e, synthetic=True), ensure_ascii=False) + "\n"); f.flush(); kept += 1
            print(f"[{i + 1}/{len(tasks)}] {dom}: verified, kept ({kept})")
    print(f"synthetic examples kept: {kept} -> {args.out}")
    return 0

# ---------------------------------------------------------------- replay data (TR-008)
def fetch_replay(args):
    rng = random.Random(args.seed)
    total = 207865  # rows in the train_sft split
    offsets = sorted(rng.sample(range(0, total - 100, 100), max(1, args.n // 20)))
    out, seen = [], set()
    for off in offsets:
        for r in hf_rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 100, off):
            msgs = r.get("messages") or []
            if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
                continue
            q, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
            if not (10 <= len(q) <= 2000 and 20 <= len(a) <= 4000) or norm_prompt(q) in seen:
                continue
            seen.add(norm_prompt(q))
            out.append({"instruction": redact(q), "output": redact(a), "source": "HuggingFaceH4/ultrachat_200k train_sft", "license": "MIT", "prompt_id": r.get("prompt_id")})
            if len([x for x in out]) >= args.n:
                break
        if len(out) >= args.n:
            break
        print(f"replay: {len(out)}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for x in out[:args.n]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"replay examples: {min(len(out), args.n)} -> {args.out} (MIT, sha256 {sha256_file(args.out)[:16]})")
    return 0

# ---------------------------------------------------------------- regression suite (ER-008)
def regression_suite(args):
    rng = random.Random(2026)
    plan = [("gsm8k", 300), ("mbpp_plus", 100), ("humaneval_plus", 88), ("schema_tasks", 12)]
    out = []
    for name, n in plan:
        rows = [json.loads(l) for l in open(os.path.join(BENCH, f"{name}.jsonl"))]
        rng.shuffle(rows)
        out += rows[:n]
    for t in out:
        t["source"] = f"{t.get('source', '')} [regression_500]"
    path = os.path.join(BENCH, "regression_500.jsonl")
    with open(path, "w") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"regression suite: {len(out)} tasks -> {path} (sha256 {sha256_file(path)[:16]})")
    return 0

# ---------------------------------------------------------------- promotion gate (TR-007) and registry (OR-005)
def load_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            return json.load(f)
    return {"schema_version": SCHEMA_VERSION, "active": {"system1": {"gguf": "models/Qwen3-1.7B-Q4_K_M.gguf", "adapter_hash": "none", "since": None},
            "system2": {"gguf": "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf", "adapter_hash": "none", "since": None}}, "history": []}

def save_registry(reg):
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, REGISTRY)

def get_json(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    return json.load(urllib.request.urlopen(req, timeout=120))

def gate_decision(base_items, cand_items, system, min_gain=0.02, floor=-0.01):
    """Paired comparison of the same system on the same tasks under the baseline and candidate models."""
    b = {i["task_id"]: i for i in base_items if i["system"] == system and i.get("correct") is not None}
    c = {i["task_id"]: i for i in cand_items if i["system"] == system and i.get("correct") is not None}
    keys = sorted(set(b) & set(c))
    diffs = [int(bool(c[k]["correct"])) - int(bool(b[k]["correct"])) for k in keys]
    if not diffs:
        return {"pass": False, "reasons": ["no paired items"], "n": 0}
    lo, hi = bootstrap_ci(diffs)
    gain = sum(diffs) / len(diffs)
    per = {}
    for d in sorted({b[k]["domain"] for k in keys}):
        dd = [int(bool(c[k]["correct"])) - int(bool(b[k]["correct"])) for k in keys if b[k]["domain"] == d]
        per[d] = {"n": len(dd), "gain": sum(dd) / len(dd), "ci95": bootstrap_ci(dd, seed=11)}
    reasons = []
    if not lo > 0:
        reasons.append(f"lower 95% bound of the gain {lo:+.3f} is not above 0")
    if gain < min_gain:
        reasons.append(f"gain {gain:+.3f} < {min_gain:+.2f}")
    for d, v in per.items():
        if v["ci95"] and v["ci95"][0] < floor:
            reasons.append(f"domain {d}: lower bound {v['ci95'][0]:+.3f} < {floor:+.2f}")
    return {"pass": not reasons, "reasons": reasons, "n": len(diffs), "gain": gain, "ci95": [lo, hi], "domains": per}

def promote(args):
    base = get_json(f"{args.gateway}/v1/eval/runs/{args.baseline_run}", args.token)
    cand = get_json(f"{args.gateway}/v1/eval/runs/{args.candidate_run}", args.token)
    for r in (base, cand):
        if not str(r.get("request", {}).get("benchmark", "")).startswith("regression_500") and not args.allow_other_suite:
            print("ER-008: both runs must use the regression_500 suite (or pass --allow-other-suite for a dry rehearsal)"); return 2
    d = gate_decision(base.get("items", []), cand.get("items", []), args.system)
    # NFR-020 and the coupling test must not get worse (TR-007).
    gate = get_json(f"{args.gateway}/v1/gate", args.token)
    if gate.get("promoted") is False and args.require_calibration:
        d["reasons"].append("the gate calibration is not promoted (NFR-020)")
    cp = (cand.get("coupling") or {}).get("draft_blind")
    if cp and cp.get("h5_pass") is False and args.require_coupling:
        d["reasons"].append("coupling test (H5) fails on the candidate run")
    if not args.sign_off:
        d["reasons"].append("human sign-off missing (--sign-off NAME)")
    d["pass"] = not d["reasons"]
    print(json.dumps(d, indent=2))
    if not d["pass"]:
        print("NOT PROMOTED"); return 1
    reg = load_registry()
    prev = reg["active"][args.role]
    new = {"gguf": args.gguf, "adapter_hash": args.adapter_hash or (sha256_file(args.gguf)[:16] if os.path.exists(args.gguf) else "unknown"),
           "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    reg["history"].append({"at": new["since"], "role": args.role, "action": "promote", "from": prev, "to": new, "gguf": args.gguf, "adapter_hash": new["adapter_hash"],
                           "signed_off_by": args.sign_off, "evidence": {"baseline_run": args.baseline_run, "candidate_run": args.candidate_run, "gain": d["gain"], "ci95": d["ci95"], "n": d["n"]}})
    reg["active"][args.role] = new
    save_registry(reg)
    print(f"PROMOTED {args.role} -> {args.gguf}. Restart serving to load it: scripts/serve_llama.sh stop && scripts/serve_llama.sh start")
    return 0

def rollback(args):
    reg = load_registry()
    last = next((h for h in reversed(reg["history"]) if h["role"] == args.role and h["action"] == "promote" and not h.get("rolled_back")), None)
    if not last:
        print(f"nothing to roll back for {args.role}"); return 1
    cur = reg["active"][args.role]
    reg["active"][args.role] = last["from"]
    last["rolled_back"] = True
    reg["history"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "role": args.role, "action": "rollback", "from": cur, "to": last["from"],
                           "gguf": last["from"].get("gguf"), "adapter_hash": last["from"].get("adapter_hash"), "signed_off_by": args.sign_off or "operator"})
    save_registry(reg)
    print(f"ROLLED BACK {args.role} -> {last['from'].get('gguf')}. Restart serving: scripts/serve_llama.sh stop && scripts/serve_llama.sh start")
    return 0

def status(args):
    print(json.dumps(load_registry(), indent=2)); return 0

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--db", default=DB); b.add_argument("--out", default=os.path.join(ROOT, "data", "tazkiya"))
    b.add_argument("--synthetic", default=os.path.join(ROOT, "data", "tazkiya", "synthetic.jsonl")); b.add_argument("--replay", default=os.path.join(ROOT, "data", "tazkiya", "replay.jsonl"))
    b.add_argument("--domain-cap", type=float, default=0.30); b.add_argument("--min-examples", type=int, default=5000); b.add_argument("--seed", type=int, default=17)
    b.add_argument("--strict", action="store_true", help="exit 2 if the set is not ready for training")
    s = sub.add_parser("synthesize"); s.add_argument("--n", type=int, default=50); s.add_argument("--offset", type=int, default=0)
    s.add_argument("--gateway", default="http://127.0.0.1:8080"); s.add_argument("--token", default=os.environ.get("NAFS_TOKEN"))
    s.add_argument("--out", default=os.path.join(ROOT, "data", "tazkiya", "synthetic.jsonl"))
    fr = sub.add_parser("fetch-replay"); fr.add_argument("--n", type=int, default=2000); fr.add_argument("--seed", type=int, default=23)
    fr.add_argument("--out", default=os.path.join(ROOT, "data", "tazkiya", "replay.jsonl"))
    sub.add_parser("regression-suite")
    p = sub.add_parser("promote"); p.add_argument("--baseline-run", required=True); p.add_argument("--candidate-run", required=True)
    p.add_argument("--system", default="fast_only", help="system compared in both runs (fast_only isolates the distilled System 1)")
    p.add_argument("--role", default="system1", choices=["system1", "system2"]); p.add_argument("--gguf", default="models/candidate-system1-q4_k_m.gguf")
    p.add_argument("--adapter-hash"); p.add_argument("--sign-off"); p.add_argument("--gateway", default="http://127.0.0.1:8080"); p.add_argument("--token", default=os.environ.get("NAFS_TOKEN"))
    p.add_argument("--require-calibration", action="store_true"); p.add_argument("--require-coupling", action="store_true", default=True); p.add_argument("--allow-other-suite", action="store_true")
    r = sub.add_parser("rollback"); r.add_argument("--role", default="system1", choices=["system1", "system2"]); r.add_argument("--sign-off")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return {"build": build, "synthesize": synthesize, "fetch-replay": fetch_replay, "regression-suite": regression_suite, "promote": promote, "rollback": rollback, "status": status}[a.cmd](a)

if __name__ == "__main__":
    sys.exit(main())
