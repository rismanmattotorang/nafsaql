"""Live check of the math and code harness against a running gateway (FR-090..FR-093).

Runs, sequentially:
  1. the code agent (POST /v1/harness/code) on three small tasks (fix, fix, write from nothing);
  2. System 2 repairs with patch repair on (POST /v1/chat, force_mode slow, patch_repair true) on tasks whose
     tests pin a detail the question leaves open, so the first attempt tends to fail and a repair round runs.
It records what happened (status, actions, patches applied or failed, model calls, wall time) to
benchmarks/harness_live_check.json. On a shared host the times are not speed claims; label them.

    python3 -I scripts/harness_live_check.py [--base http://127.0.0.1:8080] [--label "SHARED MACHINE"]
"""
import argparse, json, os, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AGENT_TASKS = [
    {
        "name": "fix_palindrome_case",
        "task": "is_palindrome(s) must ignore case and every character that is not a letter or digit. Fix solution.py.",
        "files": {"solution.py": "def is_palindrome(s):\n    return s == s[::-1]\n"},
        "tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')",
                  "assert is_palindrome('')", "assert is_palindrome('No lemon, no melon')"],
    },
    {
        "name": "fix_second_largest",
        "task": "second_largest(xs) returns the second largest distinct value of the list, or None when there is none. Fix solution.py.",
        "files": {"solution.py": "def second_largest(xs):\n    xs = sorted(xs)\n    return xs[-2]\n"},
        "tests": ["assert second_largest([3, 1, 2]) == 2", "assert second_largest([5, 5, 4]) == 4",
                  "assert second_largest([7]) is None", "assert second_largest([2, 2]) is None"],
    },
    {
        "name": "write_rle",
        "task": "Write rle(s) in solution.py: run-length encode a string as count followed by character, e.g. 'aaab' -> '3a1b'.",
        "files": {},
        "tests": ["assert rle('aaab') == '3a1b'", "assert rle('') == ''", "assert rle('abc') == '1a1b1c'",
                  "assert rle('zzzzzzzzzzzz') == '12z'"],
    },
]

REPAIR_TASKS = [
    {
        "name": "format_price",
        "question": "Write a Python function format_price(x) that formats a number as a price.",
        "tests": ["assert format_price(3.5) == '$3.50'", "assert format_price(1234) == '$1,234.00'",
                  "assert format_price(-2) == '-$2.00'", "assert format_price(0.004) == '$0.00'"],
    },
    {
        "name": "normalize_name",
        "question": "Write a Python function normalize_name(s) that normalizes a person's name.",
        "tests": ["assert normalize_name('  ada   LOVELACE ') == 'Ada Lovelace'", "assert normalize_name('o\\'neil') == \"O'Neil\"",
                  "assert normalize_name('jean-luc picard') == 'Jean-Luc Picard'"],
    },
    {
        "name": "chunk_list",
        "question": "Write a Python function chunks(xs, n) that splits a list into chunks.",
        "tests": ["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunks([], 3) == []",
                  "assert chunks([1, 2], 5) == [[1, 2]]"],
    },
]


def post(base, path, body, timeout=2400):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--label", default="SHARED MACHINE")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/harness_live_check.json"))
    a = ap.parse_args()
    out = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": [], "repair": []}

    def save():
        json.dump(out, open(a.out, "w"), indent=2)

    for t in AGENT_TASKS:
        t0 = time.time()
        try:
            r = post(a.base, "/v1/harness/code", {"task": t["task"], "files": t["files"], "verification": {"tests": t["tests"]}})
            rec = {"name": t["name"], "status": r["status"], "passed": r["passed"], "score": r["score"],
                   "model_calls": r["model_calls"], "wall_s": round(r["wall_ms"] / 1000, 1),
                   "actions": [s["action"] for s in r["steps"]],
                   "patch_failed": sum(1 for s in r["steps"] if s["action"] == "patch" and "not applied" in s["observation"]),
                   "files": r["files"], "steps": r["steps"]}
        except Exception as e:  # noqa: BLE001 - the record says what failed
            rec = {"name": t["name"], "error": str(e), "wall_s": round(time.time() - t0, 1)}
        out["agent"].append(rec)
        print(json.dumps({k: rec.get(k) for k in ("name", "status", "passed", "model_calls", "wall_s", "actions", "error")}), flush=True)
        save()

    for t in REPAIR_TASKS:
        t0 = time.time()
        try:
            r = post(a.base, "/v1/chat", {"messages": [{"role": "user", "content": t["question"]}], "force_mode": "slow",
                                           "patch_repair": True, "ephemeral": True, "verification": {"tests": t["tests"]}})
            notes = [s.get("note") for s in r["usage"]["stages"] if (s.get("note") or "").startswith("candidate")]
            rec = {"name": t["name"], "assurance": r["assurance"], "model_calls": r["usage"]["model_calls"],
                   "wall_s": round(r["usage"]["wall_ms"] / 1000, 1), "rounds": notes,
                   "completion_tokens": r["usage"]["completion_tokens"], "final": r["message"]["content"]}
        except Exception as e:  # noqa: BLE001
            rec = {"name": t["name"], "error": str(e), "wall_s": round(time.time() - t0, 1)}
        out["repair"].append(rec)
        print(json.dumps({k: rec.get(k) for k in ("name", "assurance", "model_calls", "wall_s", "rounds", "error")}), flush=True)
        save()
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()


if __name__ == "__main__":
    main()
