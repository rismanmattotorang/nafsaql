#!/usr/bin/env python3
"""Build the evaluation datasets in data/bench/*.jsonl (one task per line).
Each task: {"id", "domain", "prompt", "verification": {...}, "source", "license"}.
Sources (licences): GSM8K test (MIT), HumanEval (MIT), MBPP sanitized test (CC-BY-4.0), and the
EvalPlus extensions HumanEval+ and MBPP+ (Apache-2.0; many more tests per task) via the Hugging Face
datasets-server API; schema_tasks and open_prompts are written here (Apache-2.0).
Usage: python3 scripts/build_datasets.py            all datasets
       python3 scripts/build_datasets.py --plus     only HumanEval+ and MBPP+ (and their solutions)
"""
import json, os, sys, time, urllib.request, urllib.parse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bench")
os.makedirs(OUT, exist_ok=True)

def rows(dataset, config, split, total):
    out = []
    for off in range(0, total, 100):
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
                    data = json.load(r); break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(2 + attempt * 2)
        batch = [x["row"] for x in data.get("rows", [])]
        out.extend(batch)
        if len(batch) < 100: break
    return out

def write(name, items):
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{name}: {len(items)} tasks")

def gsm8k():
    items = []
    for i, r in enumerate(rows("openai/gsm8k", "main", "test", 1319)):
        ref = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k-{i:04d}", "domain": "math", "prompt": r["question"], "verification": {"reference_answer": ref}, "source": "openai/gsm8k test", "license": "MIT"})
    write("gsm8k", items)

def humaneval():
    items = []
    for r in rows("openai/openai_humaneval", "openai_humaneval", "test", 164):
        prompt = "Complete the following Python function. Return the full function definition.\n\n```python\n" + r["prompt"] + "```"
        items.append({"id": r["task_id"].replace("/", "-").lower(), "domain": "code", "prompt": prompt,
                      "verification": {"test_code": r["test"], "entry_point": r["entry_point"], "code_prefix": r["prompt"]}, "source": "openai/openai_humaneval", "license": "MIT"})
    write("humaneval", items)

def mbpp():
    items = []
    for r in rows("google-research-datasets/mbpp", "sanitized", "test", 257):
        tests = r["test_list"]
        fn = tests[0].split("assert ")[-1].split("(")[0].strip() if tests else "solution"
        prompt = f"{r['prompt']}\nWrite a Python function named `{fn}`. Your code should pass tests like:\n{tests[0] if tests else ''}"
        imports = r.get("test_imports") or []
        items.append({"id": f"mbpp-{r['task_id']}", "domain": "code", "prompt": prompt, "verification": {"tests": list(imports) + tests}, "source": "google-research-datasets/mbpp sanitized test", "license": "CC-BY-4.0"})
    write("mbpp", items)

def schema_tasks():
    T = []
    def add(i, prompt, schema): T.append({"id": f"schema-{i:02d}", "domain": "structured", "prompt": prompt, "verification": {"json_schema": schema}, "source": "nafs-aql", "license": "Apache-2.0"})
    obj = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
    add(1, "Create a JSON object for a library book with fields title (string), author (string), year (integer between 1450 and 2026) and available (boolean).",
        obj({"title": {"type": "string", "minLength": 1}, "author": {"type": "string", "minLength": 1}, "year": {"type": "integer", "minimum": 1450, "maximum": 2026}, "available": {"type": "boolean"}}, ["title", "author", "year", "available"]))
    add(2, "Return a JSON array of exactly three ISO 3166-1 alpha-2 country codes for countries in Southeast Asia.",
        {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string", "pattern": "^[A-Z]{2}$"}})
    add(3, "Produce a JSON object describing a meeting: title, start (ISO 8601 date-time string), duration_minutes (positive integer), attendees (array of at least two email addresses).",
        obj({"title": {"type": "string"}, "start": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}"}, "duration_minutes": {"type": "integer", "minimum": 1},
             "attendees": {"type": "array", "minItems": 2, "items": {"type": "string", "pattern": "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"}}}, ["title", "start", "duration_minutes", "attendees"]))
    add(4, "Give a JSON object with a key 'primes' holding the first eight prime numbers as integers, in increasing order.",
        obj({"primes": {"type": "array", "items": {"type": "integer"}, "const": [2, 3, 5, 7, 11, 13, 17, 19]}}, ["primes"]))
    add(5, "Return a JSON object for an HTTP error with integer field status equal to 404, string field error equal to \"not_found\", and a non-empty string message.",
        obj({"status": {"const": 404}, "error": {"const": "not_found"}, "message": {"type": "string", "minLength": 1}}, ["status", "error", "message"]))
    add(6, "Return a JSON object mapping each of the colours red, green and blue to its hex code string such as \"#FF0000\".",
        obj({"red": {"type": "string", "pattern": "^#(?i:ff0000)$"}, "green": {"type": "string", "pattern": "^#(?i:00ff00|008000)$"}, "blue": {"type": "string", "pattern": "^#(?i:0000ff)$"}}, ["red", "green", "blue"]))
    add(7, "Create a JSON object for a product with sku (string of 3 uppercase letters, a dash and 4 digits, e.g. ABC-1234), price (number greater than 0), currency (one of USD, EUR, IDR) and tags (array of unique strings).",
        obj({"sku": {"type": "string", "pattern": "^[A-Z]{3}-\\d{4}$"}, "price": {"type": "number", "exclusiveMinimum": 0}, "currency": {"enum": ["USD", "EUR", "IDR"]},
             "tags": {"type": "array", "uniqueItems": True, "items": {"type": "string"}}}, ["sku", "price", "currency", "tags"]))
    add(8, "Return a JSON object with keys 'celsius' set to 100 and 'fahrenheit' set to the equivalent temperature as a number.",
        obj({"celsius": {"const": 100}, "fahrenheit": {"type": "number", "minimum": 211.99, "maximum": 212.01}}, ["celsius", "fahrenheit"]))
    add(9, "Describe a user profile as JSON: id (UUID string), name (string), roles (non-empty array whose items are 'admin', 'editor' or 'viewer'), and active (boolean).",
        obj({"id": {"type": "string", "pattern": "^[0-9a-fA-F-]{36}$"}, "name": {"type": "string"}, "roles": {"type": "array", "minItems": 1, "items": {"enum": ["admin", "editor", "viewer"]}}, "active": {"type": "boolean"}}, ["id", "name", "roles", "active"]))
    add(10, "Return a JSON object for a 2x2 identity matrix under the key 'matrix' as an array of arrays of integers.",
        obj({"matrix": {"const": [[1, 0], [0, 1]]}}, ["matrix"]))
    add(11, "Return a JSON array of the days of the week, starting with Monday, as capitalised English names.",
        {"const": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})
    add(12, "Return a JSON object with a key 'answer' whose value is the integer result of 17 multiplied by 23.",
        obj({"answer": {"const": 391}}, ["answer"]))
    write("schema_tasks", T)

def open_prompts():
    P = [("factual", "What is the capital of Indonesia, and on which island is it located?"),
         ("factual", "Who wrote 'The Incoherence of the Philosophers'?"),
         ("factual", "In what year did the first crewed Moon landing take place?"),
         ("open", "Give three practical tips for writing clear technical documentation."),
         ("open", "Explain the difference between a process and a thread in two short paragraphs."),
         ("open", "Summarise the idea of test-time compute in language models for a non-expert."),
         ("open", "Suggest a name and a one-sentence description for a note-taking app for researchers."),
         ("open", "What are the trade-offs of running a language model on a CPU instead of a GPU?"),
         ("factual", "What is the boiling point of water at sea level in Fahrenheit?"),
         ("open", "Write a short, polite email declining a meeting invitation."),
         ("open", "Explain what a hash map is and when you would use one."),
         ("factual", "Which planet in our solar system has the most moons as of 2024?")]
    write("open_prompts", [{"id": f"open-{i:02d}", "domain": d, "prompt": p, "verification": {}, "source": "nafs-aql", "license": "Apache-2.0"} for i, (d, p) in enumerate(P)])

def solutions():
    """Reference solutions for verifier mutation testing (ER-003). Kept apart from the tasks; never sent to a model."""
    os.makedirs(os.path.join(OUT, "_solutions"), exist_ok=True)
    he = rows("openai/openai_humaneval", "openai_humaneval", "test", 164)
    with open(os.path.join(OUT, "_solutions", "humaneval.jsonl"), "w") as f:
        for r in he: f.write(json.dumps({"id": r["task_id"].replace("/", "-").lower(), "solution": r["prompt"] + r["canonical_solution"]}) + "\n")
    mb = rows("google-research-datasets/mbpp", "sanitized", "test", 257)
    with open(os.path.join(OUT, "_solutions", "mbpp.jsonl"), "w") as f:
        for r in mb: f.write(json.dumps({"id": f"mbpp-{r['task_id']}", "solution": r["code"]}) + "\n")
    print(f"solutions: humaneval {len(he)}, mbpp {len(mb)}")

SCHEMA_VALID = {  # one valid instance per schema task, for verifier mutation testing (never sent to a model)
    "schema-01": {"title": "Kitab al-Shifa", "author": "Ibn Sina", "year": 1500, "available": True},
    "schema-02": ["ID", "MY", "TH"],
    "schema-03": {"title": "Sync", "start": "2026-10-18T09:00:00Z", "duration_minutes": 30, "attendees": ["a@x.org", "b@y.org"]},
    "schema-04": {"primes": [2, 3, 5, 7, 11, 13, 17, 19]},
    "schema-05": {"status": 404, "error": "not_found", "message": "No such page"},
    "schema-06": {"red": "#FF0000", "green": "#00FF00", "blue": "#0000FF"},
    "schema-07": {"sku": "ABC-1234", "price": 9.5, "currency": "IDR", "tags": ["new", "sale"]},
    "schema-08": {"celsius": 100, "fahrenheit": 212},
    "schema-09": {"id": "123e4567-e89b-12d3-a456-426614174000", "name": "Sari", "roles": ["editor"], "active": True},
    "schema-10": {"matrix": [[1, 0], [0, 1]]},
    "schema-11": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "schema-12": {"answer": 391},
}

def plus():
    """EvalPlus: same problems, far stronger tests (HumanEval+ ~80x, MBPP+ ~35x the original tests)."""
    he, he_sol = [], []
    for r in rows("evalplus/humanevalplus", "default", "test", 164):
        tid = r["task_id"].replace("/", "-").lower().replace("humaneval", "humaneval_plus")
        prompt = "Complete the following Python function. Return the full function definition.\n\n```python\n" + r["prompt"] + "```"
        he.append({"id": tid, "domain": "code", "prompt": prompt, "verification": {"test_code": r["test"], "entry_point": r["entry_point"], "code_prefix": r["prompt"]},
                   "source": "evalplus/humanevalplus", "license": "Apache-2.0"})
        he_sol.append({"id": tid, "solution": r["prompt"] + r["canonical_solution"]})
    write("humaneval_plus", he)
    mb, mb_sol = [], []
    for r in rows("evalplus/mbppplus", "default", "test", 378):
        tests = r["test_list"]
        fn = tests[0].split("assert ")[-1].split("(")[0].strip().replace("set", "").strip("(") if tests else "solution"
        if not fn.isidentifier():
            import re; m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\(", tests[0].split("assert ")[-1].replace("set(", "", 1)); fn = m.group(1) if m else "solution"
        prompt = f"{r['prompt']}\nWrite a Python function named `{fn}`. Your code should pass tests like:\n{tests[0] if tests else ''}"
        tid = f"mbpp_plus-{r['task_id']}"
        mb.append({"id": tid, "domain": "code", "prompt": prompt, "verification": {"tests": list(r.get("test_imports") or []) + [r["test"]]},
                   "source": "evalplus/mbppplus", "license": "Apache-2.0"})
        mb_sol.append({"id": tid, "solution": r["code"]})
    write("mbpp_plus", mb)
    os.makedirs(os.path.join(OUT, "_solutions"), exist_ok=True)
    for name, sol in (("humaneval_plus", he_sol), ("mbpp_plus", mb_sol)):
        with open(os.path.join(OUT, "_solutions", f"{name}.jsonl"), "w") as f:
            for x in sol: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{name} solutions: {len(sol)}")

def schema_solutions():
    os.makedirs(os.path.join(OUT, "_solutions"), exist_ok=True)
    with open(os.path.join(OUT, "_solutions", "schema_tasks.jsonl"), "w") as f:
        for k, v in SCHEMA_VALID.items(): f.write(json.dumps({"id": k, "solution": v}) + "\n")
    print(f"schema solutions: {len(SCHEMA_VALID)}")

def smoke():
    """A tiny mixed set for quick end-to-end checks on CPU."""
    pick = []
    for name, n in [("gsm8k", 4), ("mbpp", 3), ("schema_tasks", 3), ("open_prompts", 2)]:
        with open(os.path.join(OUT, f"{name}.jsonl")) as f: pick += [json.loads(l) for l in f][:n]
    write("smoke_mixed", pick)

if __name__ == "__main__":
    if "--solutions" in sys.argv: schema_solutions(); solutions(); sys.exit(0)
    if "--plus" in sys.argv: plus(); sys.exit(0)
    schema_tasks(); open_prompts()
    for fn in (gsm8k, humaneval, mbpp, plus):
        try: fn()
        except Exception as e: print(f"{fn.__name__}: FAILED {e}", file=sys.stderr)
    smoke()
    schema_solutions()
    try: solutions()
    except Exception as e: print(f"solutions: FAILED {e}", file=sys.stderr)
