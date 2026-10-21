#!/usr/bin/env python3
"""Build the two seeded item sets that every scale run shares (completion plan, section 4).

  calib_set   GSM8K *train* (first 3,200 rows) + MBPP sanitized train and validation: labelled System 1 drafts
              for gate calibration (NFR-020, ECE). Disjoint from every test set, so the test sets stay held out.
  core_set    GSM8K test + HumanEval+ + MBPP+ minus the regression_500 items: the test split for P0-T04,
              H1, ER-002 and ER-004. regression_500 stays a separate fixed suite (ER-008).

Both files are shuffled once with a fixed seed, so a prefix of either (max_items / offset in the eval run)
is a stratified random sample and nightly slices are resumable by offset.
Usage: python3 -I scripts/build_item_set.py [--no-fetch]   (fetching needs the Hugging Face datasets server)
"""
import hashlib, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_datasets as bd  # noqa: E402
BENCH = os.path.join(ROOT, "data", "bench")
N_GSM8K_TRAIN = 3200

def load(name):
    with open(os.path.join(BENCH, f"{name}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]

def fetch():
    items = []
    for i, r in enumerate(bd.rows("openai/gsm8k", "main", "train", N_GSM8K_TRAIN)):
        ref = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k-train-{i:04d}", "domain": "math", "prompt": r["question"],
                      "verification": {"reference_answer": ref}, "source": "openai/gsm8k train", "license": "MIT"})
    bd.write("gsm8k_train", items)
    items = []
    for split, total in (("train", 120), ("validation", 43)):
        for r in bd.rows("google-research-datasets/mbpp", "sanitized", split, total):
            tests = r["test_list"]
            fn = tests[0].split("assert ")[-1].split("(")[0].strip() if tests else "solution"
            prompt = f"{r['prompt']}\nWrite a Python function named `{fn}`. Your code should pass tests like:\n{tests[0] if tests else ''}"
            imports = r.get("test_imports") or []
            items.append({"id": f"mbpp-train-{r['task_id']}", "domain": "code", "prompt": prompt,
                          "verification": {"tests": list(imports) + tests},
                          "source": f"google-research-datasets/mbpp sanitized {split}", "license": "CC-BY-4.0"})
    bd.write("mbpp_train", items)

def build(name, parts, seed, exclude=frozenset()):
    rows = [t for p in parts for t in load(p) if t["id"] not in exclude]
    random.Random(seed).shuffle(rows)
    for t in rows:
        t["source"] = f"{t.get('source', '')} [{name}]"
    path = os.path.join(BENCH, f"{name}.jsonl")
    with open(path, "w") as f:
        for t in rows:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    by = {}
    for t in rows:
        by[t["domain"]] = by.get(t["domain"], 0) + 1
    print(f"{name}: {len(rows)} tasks {by} sha256 {sha(path)}")

if __name__ == "__main__":
    if "--no-fetch" not in sys.argv:
        fetch()
    reg = {t["id"] for t in load("regression_500")}
    build("calib_set", ["gsm8k_train", "mbpp_train"], seed=2026)
    build("core_set", ["gsm8k", "humaneval_plus", "mbpp_plus"], seed=2027, exclude=reg)
