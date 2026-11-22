#!/usr/bin/env python3
"""Phase 35: what the code agent's first step does on a task that starts with no file.

Each task is run with one step only (the quick model's step, where the run starts), and the kind of action the reply
was read as is recorded: `write` (a program), `run` (it tried to run a file), `none` (no usable action), and whether
the program passed. Run once against the build before a change and once after, with the same tasks and seeds.

Usage: python3 -I scripts/first_step_probe.py --label before [--out benchmarks/harness_phase35/first_step_before.json]
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
arg = lambda k, d: args[args.index(k) + 1] if k in args else d
GW, LABEL, OUT = arg("--gw", "http://127.0.0.1:8080"), arg("--label", "run"), arg("--out", "")
TASKS = [
    ("median", "Write a function median(xs) that returns the median of a non-empty list of numbers.",
     ["assert median([1, 3, 2]) == 2", "assert median([1, 2, 3, 4]) == 2.5", "assert median([5]) == 5"]),
    ("rle", "Write rle(s) in solution.py: run-length encode a string as count followed by character, e.g. 'aaab' -> '3a1b'.",
     ["assert rle('aaab') == '3a1b'", "assert rle('') == ''", "assert rle('abc') == '1a1b1c'"]),
    ("flatten", "Write flatten(xs) in solution.py: flatten arbitrarily nested lists into one list, keeping order, e.g. [1, [2, [3]], 4] -> [1, 2, 3, 4].",
     ["assert flatten([1, [2, [3]], 4]) == [1, 2, 3, 4]", "assert flatten([]) == []", "assert flatten([[[]]]) == []"]),
    ("gcd", "Write a function gcd(a, b) that returns the greatest common divisor of two non-negative integers.",
     ["assert gcd(12, 18) == 6", "assert gcd(7, 0) == 7", "assert gcd(17, 5) == 1"]),
    ("is_prime", "Write a function is_prime(n) that returns True when n is a prime number and False otherwise.",
     ["assert is_prime(2)", "assert not is_prime(1)", "assert is_prime(97)", "assert not is_prime(91)"]),
    ("reverse_words", "Write a function reverse_words(s) that returns the words of s in reverse order, separated by single spaces.",
     ["assert reverse_words('a b  c') == 'c b a'", "assert reverse_words('') == ''", "assert reverse_words('one') == 'one'"]),
    ("count_vowels", "Write a function count_vowels(s) that returns how many of the letters a, e, i, o, u (in either case) the string holds.",
     ["assert count_vowels('Education') == 5", "assert count_vowels('xyz') == 0", "assert count_vowels('') == 0"]),
    ("fib", "Write a function fib(n) that returns the n-th Fibonacci number, with fib(0) == 0 and fib(1) == 1.",
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55", "assert fib(20) == 6765"]),
    ("clamp", "Write a function clamp(x, lo, hi) that returns x limited to the range from lo to hi.",
     ["assert clamp(5, 0, 3) == 3", "assert clamp(-1, 0, 3) == 0", "assert clamp(2, 0, 3) == 2"]),
    ("dedupe", "Write a function dedupe(xs) that returns the items of the list xs without repeats, keeping the first of each in order.",
     ["assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]", "assert dedupe([]) == []", "assert dedupe(['a', 'a']) == ['a']"]),
    ("is_anagram", "Write a function is_anagram(a, b) that returns True when the two strings use the same letters the same number of times, ignoring case and spaces.",
     ["assert is_anagram('Listen', 'Silent')", "assert not is_anagram('abc', 'abd')", "assert is_anagram('a b', 'ba')"]),
    ("chunks", "Write a function chunks(xs, n) that splits the list xs into consecutive lists of length n; the last may be shorter.",
     ["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunks([], 3) == []", "assert chunks([1], 5) == [[1]]"]),
]

rows = []
for name, task, tests in TASKS:
    body = {"task": task, "files": {}, "verification": {"tests": tests}, "max_steps": 1}
    req = urllib.request.Request(GW + "/v1/harness/code", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    step = (out.get("steps") or [{}])[0]
    rows.append({"task": name, "action": step.get("action"), "passed": bool(out.get("passed")), "model": step.get("model"),
                 "s": round(time.time() - t, 1), "observation": (step.get("observation") or "")[:160]})
    print(f"{LABEL:7} {name:14} {step.get('action'):6} {'pass' if out.get('passed') else 'fail'}  {rows[-1]['s']:5.1f}s  {rows[-1]['observation'][:70]!r}", flush=True)

kinds = {}
for r in rows:
    kinds[r["action"]] = kinds.get(r["action"], 0) + 1
summary = {"tasks": len(rows), "actions": kinds, "passed_on_the_first_step": sum(r["passed"] for r in rows),
           "median_s": sorted(r["s"] for r in rows)[len(rows) // 2]}
print(json.dumps(summary))
if OUT:
    os.makedirs(os.path.dirname(os.path.join(ROOT, OUT)), exist_ok=True)
    json.dump({"label": LABEL, "note": "one step per task (the quick model's), empty workspace", "load": os.getloadavg(),
               "summary": summary, "rows": rows}, open(os.path.join(ROOT, OUT), "w"), indent=1)
