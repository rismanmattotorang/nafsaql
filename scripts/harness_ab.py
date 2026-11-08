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
    # Phase 14: the cascade (first step on System 1 with 4 candidates in one request, System 2 only if none passes)
    # against the Phase 11 default (System 2 from the first step); everything else at the production defaults.
    "s2": {"cascade": False},
    "cascade": {"cascade": True, "cascade_samples": 4},
    # Phase 16: the guards the research found in modern harnesses, all off (the Phase 15 loop) against all on (the
    # production defaults): the compile check before the verifier, rollback to the best version, the stall guard.
    "p15": {"lint": False, "rollback": False, "stuck_after": 0},
    "p16": {"lint": True, "rollback": True, "stuck_after": 4},
    # Phase 22 (FR-H2): the same run, compaction reached early (1,400 estimated tokens) so it matters on a short task;
    # older long observations cut first (the default) against compaction at once (the Phase 16 behaviour).
    "p22_prune": {"prune": True, "compact_at_tokens": 1400},
    "p22_noprune": {"prune": False, "compact_at_tokens": 1400},
    # The first pair ran 32 times with no pruned step: the failures the agent is shown stay under the 1,200-byte
    # default. The same pair again with the threshold at 600 bytes, where they are cut.
    "p22_prune600": {"prune": True, "prune_over": 600, "compact_at_tokens": 1400},
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


def _words(n):
    base = ("listen silent enlist google gogole cat act tac dog god odg evil vile live veil rat tar art star tars "
            "rats arts stare tears aster inlets tinsel elints stone notes onset tones steno").split()
    return (base * 3)[:n]


def _anagram_groups(words):
    by = {}
    for w in set(words):
        by.setdefault("".join(sorted(w)), []).append(w)
    return sorted(sorted(g) for g in by.values())


# Phase 22: tasks whose failures print a lot. Each test names what it got and what it expected, so a failed run's
# observation is one to two kilobytes, long enough for the prune to have something to cut.
LONG_TASKS = [
    {"name": "long_group_anagrams",
     "task": "group_anagrams(words) returns a list of groups; each group is the sorted list of the distinct words that are anagrams of each other, and the groups are sorted by their first word. Fix solution.py.",
     "files": {"solution.py": "def group_anagrams(words):\n    groups = {}\n    for w in words:\n        groups.setdefault(w, []).append(w)\n    return list(groups.values())\n"},
     "tests": ["words = " + repr(_words(36)) + "\nexp = " + repr(_anagram_groups(_words(36))) +
               "\ngot = group_anagrams(words)\nassert got == exp, f'got {got!r} expected {exp!r}'"]},
    {"name": "long_csv_totals",
     "task": "totals(text) reads CSV text whose first line is the header 'category,amount'; it returns a dict from each category to the sum of its amounts as an int, with the keys in sorted order. Fix solution.py.",
     "files": {"solution.py": "def totals(text):\n    out = {}\n    for line in text.split('\\n'):\n        c, a = line.split(',')\n        out[c] = a\n    return out\n"},
     "tests": ["rows = [('fruit', 3), ('veg', 4), ('fruit', 5), ('dairy', 7), ('veg', 1), ('grain', 9), ('dairy', 2), ('fruit', 6)] * 4\n"
               "text = 'category,amount\\n' + '\\n'.join(f'{c},{a}' for c, a in rows)\n"
               "exp = {'dairy': 36, 'fruit': 56, 'grain': 36, 'veg': 20}\n"
               "got = totals(text)\nassert list(got.items()) == list(exp.items()), f'text {text!r} got {got!r} expected {exp!r}'"]},
    {"name": "long_wrap",
     "task": "wrap(text, width) splits text on whitespace and packs the words greedily into lines of at most width characters, joined by single spaces; it returns the list of lines (a word longer than width gets its own line). Fix solution.py.",
     "files": {"solution.py": "def wrap(text, width):\n    lines, cur = [], ''\n    for w in text.split():\n        if len(cur) + len(w) > width:\n            lines.append(cur)\n            cur = ''\n        cur += w\n    lines.append(cur)\n    return lines\n"},
     "tests": ["text = 'the quick brown fox jumps over the lazy dog while seventeen unusually patient owls watched quietly from an enormous oak tree beside the river'\n"
               "exp = ['the quick brown fox', 'jumps over the lazy', 'dog while seventeen', 'unusually patient owls', 'watched quietly from', 'an enormous oak tree', 'beside the river']\n"
               "got = wrap(text, 22)\nassert got == exp, f'got {got!r} expected {exp!r}'"]},
    {"name": "long_pascal",
     "task": "Write pascal(n) in solution.py: the first n rows of Pascal's triangle as a list of lists of ints (pascal(0) is []).",
     "files": {},
     "tests": ["exp = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1], [1, 6, 15, 20, 15, 6, 1], [1, 7, 21, 35, 35, 21, 7, 1], [1, 8, 28, 56, 70, 56, 28, 8, 1], [1, 9, 36, 84, 126, 126, 84, 36, 9, 1], [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1], [1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1]]\n"
               "got = pascal(12)\nassert got == exp, f'got {got!r} expected {exp!r}'", "assert pascal(0) == []"]},
    {"name": "long_spiral",
     "task": "spiral(m) returns the elements of the rectangular matrix m (a list of equal-length lists) in clockwise spiral order starting at the top left. Fix solution.py.",
     "files": {"solution.py": "def spiral(m):\n    out = []\n    for row in m:\n        out.extend(row)\n    return out\n"},
     "tests": ["m = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]\n"
               "exp = [1, 2, 3, 4, 5, 6, 12, 18, 24, 30, 29, 28, 27, 26, 25, 19, 13, 7, 8, 9, 10, 11, 17, 23, 22, 21, 20, 14, 15, 16]\n"
               "got = spiral(m)\nassert got == exp, f'matrix {m!r} got {got!r} expected {exp!r}'", "assert spiral([]) == []"]},
    {"name": "long_dedupe_runs",
     "task": "compress(xs) collapses every run of equal consecutive items in the list xs into one (xs itself is not changed); compress([1,1,2,2,2,1]) is [1,2,1]. Fix solution.py.",
     "files": {"solution.py": "def compress(xs):\n    return list(set(xs))\n"},
     "tests": ["xs = [3, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 5, 5, 5, 5, 5, 2, 2, 7, 7, 7, 1, 1, 1, 1, 3, 3, 2, 2, 2, 9, 9, 4, 4, 4, 4, 4, 4, 1]\n"
               "exp = [3, 1, 2, 3, 1, 5, 2, 7, 1, 3, 2, 9, 4, 1]\n"
               "got = compress(xs)\nassert got == exp, f'input {xs!r} got {got!r} expected {exp!r}'", "assert compress([]) == []"]},
    {"name": "long_histogram",
     "task": "histogram(xs) returns a string with one line per distinct value of xs in ascending order, each 'value: ' followed by one '#' per occurrence; lines are joined with newlines and there is no trailing newline. Fix solution.py.",
     "files": {"solution.py": "def histogram(xs):\n    out = ''\n    for x in xs:\n        out += str(x) + ': #\\n'\n    return out\n"},
     "tests": ["xs = [4, 2, 4, 1, 3, 4, 2, 5, 4, 1, 3, 3, 5, 4, 2, 2, 1, 4, 5, 5]\n"
               "exp = '1: ###\\n2: ####\\n3: ###\\n4: ######\\n5: ####'\n"
               "got = histogram(xs)\nassert got == exp, f'input {xs!r} got {got!r} expected {exp!r}'"]},
    {"name": "long_matrix_rotate",
     "task": "rotate(m) returns a new square matrix: m rotated 90 degrees clockwise (m itself is not changed). Fix solution.py.",
     "files": {"solution.py": "def rotate(m):\n    return [list(r) for r in zip(*m)]\n"},
     "tests": ["m = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]\n"
               "exp = [[21, 16, 11, 6, 1], [22, 17, 12, 7, 2], [23, 18, 13, 8, 3], [24, 19, 14, 9, 4], [25, 20, 15, 10, 5]]\n"
               "got = rotate(m)\nassert got == exp, f'matrix {m!r} got {got!r} expected {exp!r}'"]},
]


_SHOW = (
    "def show(got, exp, inp):\n"
    "    g = list(got.items()) if isinstance(got, dict) else (got.split('\\n') if isinstance(got, str) else list(got))\n"
    "    e = list(exp.items()) if isinstance(exp, dict) else (exp.split('\\n') if isinstance(exp, str) else list(exp))\n"
    "    rows = [f'{i}: {a!r} vs {b!r}' for i, (a, b) in enumerate(zip(g, e))]\n"
    "    return f'input: {inp!r}\\ngot: {got!r}\\nexpected: {exp!r}\\nelement by element:\\n' + '\\n'.join(rows)\n"
)
_INPUT = {"long_group_anagrams": "words", "long_csv_totals": "text", "long_wrap": "text", "long_pascal": "12",
          "long_spiral": "m", "long_dedupe_runs": "xs", "long_histogram": "xs", "long_matrix_rotate": "m"}
for _t in LONG_TASKS:
    _head, _, _last = _t["tests"][0].rpartition("\n")
    _t["tests"][0] = _SHOW + _head + "\n" + _last.split(", f'")[0] + f", show(got, exp, {_INPUT[_t['name']]})"


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def summary(r):
    steps = r.get("steps", [])
    return {
        "status": r.get("status"), "passed": r.get("passed"), "score": r.get("score"),
        "model_calls": r.get("model_calls"), "verifier_calls": r.get("verifier_calls"),
        "compactions": r.get("compactions"), "pruned_steps": sum(1 for s in steps if s.get("pruned")), "wall_s": round(r.get("wall_ms", 0) / 1000, 1),
        "prefill_tokens": r.get("prefill_tokens"), "cached_tokens": r.get("cached_tokens"),
        "decoded_tokens": r.get("decoded_tokens"),
        "cascade": r.get("cascade"), "cascade_samples": r.get("cascade_samples"),
        "lint_rejections": r.get("lint_rejections"), "rollbacks": r.get("rollbacks"),
        "s1_steps": sum(1 for s in steps if s.get("model") == "system1"), "s2_steps": sum(1 for s in steps if s.get("model") == "system2"),
        "actions": [s["action"] for s in steps],
        "malformed": sum(1 for s in steps if s["action"] in ("none", "rejected") or s["observation"].startswith("The patch was not applied")),
        "steps": [{**{k: s.get(k) for k in ("step", "action", "passed", "score", "ms", "usage", "compacted", "pruned", "model", "samples", "retries", "rolled_back")}, "observation": s["observation"][:300]} for s in steps],
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
    ap.add_argument("--suite", default="base", choices=["base", "long"], help="long: the Phase 22 long-evidence tasks")
    ap.add_argument("--reps", type=int, default=1, help="repetitions of every task and arm (the model samples)")
    a = ap.parse_args()
    pool = LONG_TASKS if a.suite == "long" else TASKS
    tasks = pool if a.tasks == "all" else [t for t in pool if t["name"] in a.tasks.split(",")]
    tasks = [t for _ in range(a.reps) for t in tasks]
    doc = json.load(open(a.out)) if os.path.exists(a.out) else {"runs": []}
    doc.setdefault("arms", ARMS)
    done = {(x["model"], x["task"], x["arm"], x.get("rep", 0)) for x in doc["runs"]}
    for i, t in enumerate(tasks):
        rep = i // (len(tasks) // a.reps)
        # Rotate which arm goes first, so a drift in machine load does not favour one arm.
        arms = a.arms.split(",")
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        for arm in order:
            if (a.model, t["name"], arm, rep) in done:
                continue
            body = {"task": t["task"], "files": t["files"], "verification": {"tests": t["tests"]},
                    "model": a.model, "max_steps": a.max_steps, **ARMS[arm]}
            t0 = time.time()
            try:
                r = summary(post(a.base, "/v1/harness/code", body, 1800))
            except (urllib.error.URLError, TimeoutError) as e:
                r = {"error": str(e), "wall_s": round(time.time() - t0, 1)}
            load = os.getloadavg()[0]
            rec = {"model": a.model, "task": t["name"], "arm": arm, "rep": rep, "order": order.index(arm),
                   "label": a.label, "load_1m_after": round(load, 1),
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **r}
            doc["runs"].append(rec)
            json.dump(doc, open(a.out, "w"), indent=1)
            print(f"{a.model} {t['name']:22s} {arm}: {r.get('status', r.get('error'))} passed={r.get('passed')} "
                  f"calls={r.get('model_calls')} wall={r.get('wall_s')}s prefill={r.get('prefill_tokens')} "
                  f"cached={r.get('cached_tokens')} decoded={r.get('decoded_tokens')} load={load:.1f}", flush=True)


if __name__ == "__main__":
    main()
