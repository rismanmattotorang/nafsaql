"""Interleaved live A/B of the code agent: the Phase 10 loop against the Codex-style Phase 11 harness.

Arms (same task, same model, same budget, order rotated per task):
  phase10      context "reset", no grammar, no stream cut, patch always offered (the loop of fba3901)
  phase11      context "append" (append-only transcript, compaction), action grammar, stream cut,
               edit_format "auto" (whole file while solution.py is short)
  phase11_any  phase11 with a patch offered at every step (isolates the edit-format policy)
  phase11_u    phase11 with `unstick` (after a repeat: compact and sample more diversely); round 2
  phase11_nou  phase11 without `unstick`; round 2 (the build with value-carrying evidence)
  phase11_b512 / phase11_b256  phase11 with 512 / 256 System 2 reasoning tokens per step; round 2
  phase11_r3 / phase11_adapt   round 3 (grammar without patches inside code blocks): 512 reasoning tokens on every
                               step, or 256 on the first step and 512 after a failure

Per run it records status, verified or not, model calls, wall time, and the server's token accounting summed
over the run (prompt tokens prefilled, prompt tokens served from the cache, tokens decoded), plus every step.
Results are appended to the output file after each run, and runs already in it are skipped, so the script can
be stopped and resumed.

    python3 -I scripts/harness_ab.py --model system1 --arms phase10,phase11,phase11_any [--tasks all]
    python3 -I scripts/harness_ab.py --model system2 --tasks fix_palindrome_case,fix_second_largest,write_rle

On a shared host label the run (--label); wall times then are not speed claims, token counts still are.
"""
import argparse, json, os, time, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ARMS = {
    "phase10": {"context": "reset", "constrain": False, "stream_cut": False, "edit_format": "any"},
    "phase11": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto"},
    "phase11_any": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "any"},
    "phase11_u": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "unstick": True},
    "phase11_nou": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "unstick": False},
    "phase11_b512": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "thinking_budget": 512},
    "phase11_b256": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "thinking_budget": 256},
    "phase11_r3": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "thinking_budget": 512,
                   "first_step_thinking_budget": 0},
    "phase11_adapt": {"context": "append", "constrain": True, "stream_cut": True, "edit_format": "auto", "thinking_budget": 512,
                      "first_step_thinking_budget": 256},
}

TASKS = [
    {"name": "fix_palindrome_case",
     "task": "is_palindrome(s) must ignore case and every character that is not a letter or digit. Fix solution.py.",
     "files": {"solution.py": "def is_palindrome(s):\n    return s == s[::-1]\n"},
     "tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')",
               "assert is_palindrome('')", "assert is_palindrome('No lemon, no melon')"]},
    {"name": "fix_second_largest",
     "task": "second_largest(xs) returns the second largest distinct value of the list, or None when there is none. Fix solution.py.",
     "files": {"solution.py": "def second_largest(xs):\n    xs = sorted(xs)\n    return xs[-2]\n"},
     "tests": ["assert second_largest([3, 1, 2]) == 2", "assert second_largest([5, 5, 4]) == 4",
               "assert second_largest([7]) is None", "assert second_largest([2, 2]) is None"]},
    {"name": "write_rle",
     "task": "Write rle(s) in solution.py: run-length encode a string as count followed by character, e.g. 'aaab' -> '3a1b'.",
     "files": {},
     "tests": ["assert rle('aaab') == '3a1b'", "assert rle('') == ''", "assert rle('abc') == '1a1b1c'",
               "assert rle('zzzzzzzzzzzz') == '12z'"]},
    {"name": "fix_fizzbuzz",
     "task": "fizzbuzz(n) returns the list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number. Fix solution.py.",
     "files": {"solution.py": "def fizzbuzz(n):\n    out = []\n    for i in range(n):\n        if i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n"},
     "tests": ["assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']", "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
               "assert len(fizzbuzz(0)) == 0", "assert fizzbuzz(3) == ['1', '2', 'Fizz']"]},
    {"name": "fix_word_count",
     "task": "word_count(text) returns a dict from each lower-cased word to how often it occurs; words are separated by whitespace and lose leading and trailing punctuation (.,!?). Fix solution.py.",
     "files": {"solution.py": "def word_count(text):\n    counts = {}\n    for w in text.split(' '):\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n"},
     "tests": ["assert word_count('a b a') == {'a': 2, 'b': 1}", "assert word_count('Hi, hi!') == {'hi': 2}",
               "assert word_count('') == {}", "assert word_count('one  two\\nthree') == {'one': 1, 'two': 1, 'three': 1}"]},
    {"name": "fix_binary_search",
     "task": "index_of(xs, x) returns the index of x in the sorted list xs using binary search, or -1 when x is absent. Fix solution.py.",
     "files": {"solution.py": "def index_of(xs, x):\n    lo, hi = 0, len(xs)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] < x:\n            lo = mid\n        else:\n            hi = mid\n    return lo\n"},
     "tests": ["assert index_of([1, 3, 5, 7], 5) == 2", "assert index_of([1, 3, 5, 7], 4) == -1",
               "assert index_of([], 1) == -1", "assert index_of([2], 2) == 0", "assert index_of([1, 3, 5, 7], 9) == -1"]},
    {"name": "write_flatten",
     "task": "Write flatten(xs) in solution.py: flatten arbitrarily nested lists into one list, keeping order, e.g. [1, [2, [3]], 4] -> [1, 2, 3, 4].",
     "files": {},
     "tests": ["assert flatten([1, [2, [3]], 4]) == [1, 2, 3, 4]", "assert flatten([]) == []",
               "assert flatten([[[]]]) == []", "assert flatten([[1, 2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]"]},
    {"name": "fix_roman",
     "task": "to_roman(n) converts 1 <= n <= 3999 to a Roman numeral using subtractive notation (4 = IV, 9 = IX, 40 = XL, 90 = XC, 400 = CD, 900 = CM). Fix solution.py.",
     "files": {"solution.py": "def to_roman(n):\n    vals = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'), (10, 'X'), (5, 'V'), (1, 'I')]\n    out = ''\n    for v, s in vals:\n        while n >= v:\n            out += s\n            n -= v\n    return out\n"},
     "tests": ["assert to_roman(3) == 'III'", "assert to_roman(4) == 'IV'", "assert to_roman(9) == 'IX'",
               "assert to_roman(1994) == 'MCMXCIV'", "assert to_roman(3999) == 'MMMCMXCIX'"]},
    {"name": "fix_merge_intervals",
     "task": "merge(intervals) merges overlapping [start, end] intervals (touching ones too) and returns them sorted by start. Fix solution.py.",
     "files": {"solution.py": "def merge(intervals):\n    out = []\n    for s, e in intervals:\n        if out and s < out[-1][1]:\n            out[-1][1] = e\n        else:\n            out.append([s, e])\n    return out\n"},
     "tests": ["assert merge([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]", "assert merge([[1, 4], [4, 5]]) == [[1, 5]]",
               "assert merge([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]", "assert merge([[1, 10], [2, 3]]) == [[1, 10]]",
               "assert merge([]) == []"]},
    {"name": "write_caesar",
     "task": "Write caesar(s, k) in solution.py: shift each ASCII letter k places through the alphabet (wrapping, keeping case; k may be negative or larger than 26); other characters stay.",
     "files": {},
     "tests": ["assert caesar('abc', 1) == 'bcd'", "assert caesar('xyz', 3) == 'abc'", "assert caesar('Hello, World!', 13) == 'Uryyb, Jbeyq!'",
               "assert caesar('a', -1) == 'z'", "assert caesar('A', 52) == 'A'"]},
]


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def summary(r):
    steps = r.get("steps", [])
    return {
        "status": r.get("status"), "passed": r.get("passed"), "score": r.get("score"),
        "model_calls": r.get("model_calls"), "verifier_calls": r.get("verifier_calls"),
        "compactions": r.get("compactions"), "wall_s": round(r.get("wall_ms", 0) / 1000, 1),
        "prefill_tokens": r.get("prefill_tokens"), "cached_tokens": r.get("cached_tokens"),
        "decoded_tokens": r.get("decoded_tokens"),
        "actions": [s["action"] for s in steps],
        "malformed": sum(1 for s in steps if s["action"] == "none" or s["observation"].startswith("The patch was not applied")),
        "steps": [{k: s.get(k) for k in ("step", "action", "passed", "score", "ms", "usage", "compacted")} | {"observation": s["observation"][:300]} for s in steps],
        "files": r.get("files"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="system1", choices=["system1", "system2"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks/harness_ab.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--arms", default="phase10,phase11")
    a = ap.parse_args()
    tasks = TASKS if a.tasks == "all" else [t for t in TASKS if t["name"] in a.tasks.split(",")]
    doc = json.load(open(a.out)) if os.path.exists(a.out) else {"runs": []}
    doc.setdefault("arms", ARMS)
    done = {(x["model"], x["task"], x["arm"]) for x in doc["runs"]}
    for i, t in enumerate(tasks):
        # Rotate which arm goes first, so a drift in machine load does not favour one arm.
        arms = a.arms.split(",")
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        for arm in order:
            if (a.model, t["name"], arm) in done:
                continue
            body = {"task": t["task"], "files": t["files"], "verification": {"tests": t["tests"]},
                    "model": a.model, "max_steps": a.max_steps, **ARMS[arm]}
            t0 = time.time()
            try:
                r = summary(post(a.base, "/v1/harness/code", body, 1800))
            except (urllib.error.URLError, TimeoutError) as e:
                r = {"error": str(e), "wall_s": round(time.time() - t0, 1)}
            load = os.getloadavg()[0]
            rec = {"model": a.model, "task": t["name"], "arm": arm, "order": order.index(arm),
                   "label": a.label, "load_1m_after": round(load, 1),
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **r}
            doc["runs"].append(rec)
            json.dump(doc, open(a.out, "w"), indent=1)
            print(f"{a.model} {t['name']:22s} {arm}: {r.get('status', r.get('error'))} passed={r.get('passed')} "
                  f"calls={r.get('model_calls')} wall={r.get('wall_s')}s prefill={r.get('prefill_tokens')} "
                  f"cached={r.get('cached_tokens')} decoded={r.get('decoded_tokens')} load={load:.1f}", flush=True)


if __name__ == "__main__":
    main()
