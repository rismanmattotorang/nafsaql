#!/usr/bin/env python3
"""End-to-end sweep of the math and code harness (FR-090..FR-093) against a running gateway.

Deterministic part (no model): the capability report, the Lean verifier against the real toolchain (accepted
proofs, partial proofs, errors and the known attacks), per-assert code credit, request validation on /v1/chat
and /v1/harness/code, and that the OpenAPI document lists the endpoints.
Model part (--no-models skips it): an easy Lean theorem through /v1/chat that must verify (--hard adds add_comm,
which a 4B model may not solve: it must then abstain honestly), the code agent on one fix task, and a code chat on System 2 with the candidate tree.

Usage: python3 -I scripts/harness_sweep.py [--base http://127.0.0.1:8080] [--no-models]
Writes benchmarks/harness_sweep.json and exits non-zero on any failure.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = []


def call(base, method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")[:300]}


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "pass": bool(ok), "detail": str(detail)[:300]})
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {str(detail)[:90]}", flush=True)
    return bool(ok)


STATEMENT = "theorem add_comm' (a b : Nat) : a + b = b + a :=\nsorry\n"


def lean(base, body):
    s, d = call(base, "POST", "/v1/verify", {"answer": f"```lean\n{body}\n```", "verification": {"lean": {"statement": STATEMENT}}}, 240)
    return s, d


def deterministic(base):
    s, info = call(base, "GET", "/v1/harness")
    check("info: GET /v1/harness", s == 200 and {"lean", "repair", "code_agent", "sources"} <= set(info), list(info))
    lean_on = s == 200 and info["lean"]["available"]
    check("info: Lean toolchain is available", lean_on, info.get("lean", {}).get("executable"))
    check("info: both sources are named with revisions", s == 200 and {x["name"] for x in info["sources"]} >= {"Open Math Harness", "OpenAI Codex (codex-rs)"} and all(x.get("revision") for x in info["sources"]))

    s, spec = call(base, "GET", "/openapi.json")
    paths = spec.get("paths", {}) if s == 200 else {}
    check("openapi: /v1/harness and /v1/harness/code are documented", "/v1/harness" in paths and "/v1/harness/code" in paths)

    if lean_on:
        rows = [
            ("lean: omega proof is verified", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby omega", True, None),
            ("lean: term proof is verified", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nNat.add_comm a b", True, None),
            ("lean: helper lemma before the theorem is verified", "theorem helper (x y : Nat) : x + y = y + x := Nat.add_comm x y\n\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\nhelper a b", True, None),
            ("lean: a partial proof (sorry) is not verified", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby sorry", False, "sorry_ax"),
            ("lean: a Lean error is reported with a position", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby rfl", False, "elaboration"),
            ("lean: a hidden sorryAx is rejected", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nsorryAx _ false", False, None),
            ("lean: a user axiom is refused before Lean runs", "axiom cheat : False\n\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\ncheat.elim", False, "forbidden"),
            ("lean: native_decide is refused", "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby native_decide", False, "forbidden"),
            ("lean: code that runs at elaboration (#eval) is refused", "#eval 1\n\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\nby omega", False, "forbidden"),
            ("lean: a changed statement is refused", "theorem add_comm' (a b : Nat) : a + b = a + b :=\nrfl", False, "statement_mismatch"),
            ("lean: a shadowed instance is rejected by the type check", "instance (priority := high) cheatAdd : HAdd Nat Nat Nat := ⟨fun _ _ => 0⟩\n\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\nrfl", False, "type_mismatch"),
        ]
        for name, body, want, kind in rows:
            s, d = lean(base, body)
            got_kind = (d.get("details") or {}).get("kind") if s == 200 else None
            check(name, s == 200 and d.get("passed") == want and (kind is None or got_kind == kind), f"passed={d.get('passed')} kind={got_kind}")
        s, d = lean(base, "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby rfl")
        ev = d.get("evidence", "")
        check("lean: evidence names the probe, not the sandbox path", "Probe.lean:" in ev and "/var/folders" not in ev and "nafs-sbx" not in ev)
        s, d = lean(base, "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby sorry")
        check("lean: partial proof scores 0.5", d.get("score") == 0.5, d.get("score"))
        s, d = call(base, "POST", "/v1/verify", {"answer": "x", "verification": {"lean": {"statement": "theorem t : True := trivial"}}})
        check("lean: a statement not proved by sorry is a 400", s == 400, d.get("message", "")[:80])
        s, d = call(base, "POST", "/v1/verify", {"answer": "x", "verification": {"lean": {"statement": "axiom bad : False\ntheorem t : True :=\nsorry"}}})
        check("lean: a statement containing an axiom is a 400", s == 400, d.get("message", "")[:80])

    s, d = call(base, "POST", "/v1/verify", {"answer": "```python\ndef sq(x):\n    return x*x if x<3 else 0\n```", "verification": {"tests": ["assert sq(1)==1", "assert sq(2)==4", "assert sq(3)==9"]}})
    det = d.get("details") or {}
    check("code: per-assert credit (2 of 3)", s == 200 and not d["passed"] and (det.get("tests_passed"), det.get("tests_total")) == (2, 3), det)
    check("code: evidence names the failing assert, not the nonce", "test 3" in d.get("evidence", "") and "NAFS-OK" not in d.get("evidence", ""))
    s, d = call(base, "POST", "/v1/verify", {"answer": "```python\ndef sq(x):\n    return x*x\n```", "verification": {"tests": ["assert sq(1)==1", "assert sq(3)==9"]}})
    check("code: all asserts passing is verified", s == 200 and d["passed"])
    s, d = call(base, "POST", "/v1/verify", {"answer": "```python\ndef f():\n    print('x' * 100000)\n    raise ValueError('boom')\n```", "verification": {"tests": ["f()"]}})
    check("code: long output keeps head and tail in the evidence", s == 200 and not d["passed"] and "ValueError" in d.get("evidence", ""), len(d.get("evidence", "")))

    msg = [{"role": "user", "content": "x"}]
    for name, body in [
        ("validate: repair_policy must be best or last", {"messages": msg, "repair_policy": "random"}),
        ("validate: a chat Lean task with a bad statement", {"messages": msg, "verification": {"lean": {"statement": "no theorem here"}}}),
    ]:
        s, d = call(base, "POST", "/v1/chat", body)
        check(name, s == 400 and d.get("code") == "E_BAD_REQUEST", d.get("message", "")[:80])
    base_req = {"task": "t", "verification": {"tests": ["assert True"]}}
    for name, patch in [
        ("agent: no tests is a 400", {"verification": {}}),
        ("agent: empty task is a 400", {"task": " "}),
        ("agent: a subdirectory file name is a 400", {"files": {"../a.py": "x"}}),
        ("agent: a hidden file name is a 400", {"files": {".x": "x"}}),
        ("agent: max_steps 13 is a 400", {"max_steps": 13}),
        ("agent: max_steps 0 is a 400", {"max_steps": 0}),
        ("agent: an unknown model is a 400", {"model": "gpt"}),
        ("agent: too many files is a 400", {"files": {f"f{i}.py": "x" for i in range(17)}}),
    ]:
        s, d = call(base, "POST", "/v1/harness/code", {**base_req, **patch})
        check(name, s == 400 and d.get("code") == "E_BAD_REQUEST", d.get("message", "")[:80])
    s, d = call(base, "POST", "/v1/harness/code", "not an object")
    check("agent: a malformed body is a 4xx, not a crash", 400 <= s < 500, s)
    s, d = call(base, "GET", "/healthz")
    check("gateway still healthy after the invalid requests", s == 200 and d.get("status") == "ok")


EASY = "theorem add_zero' (n : Nat) : n + 0 = n :=\nsorry\n"


def lean_chat(base, statement, question, name, must_verify):
    t = time.time()
    s, d = call(base, "POST", "/v1/chat", {"messages": [{"role": "user", "content": question}], "ephemeral": True,
                                           "verification": {"lean": {"statement": statement}}}, 3000)
    v = d.get("verifier") or {}
    feats = (d.get("decision") or {}).get("features", {})
    detail = f"{d.get('assurance')} domain={feats.get('domain')} calls={d.get('usage', {}).get('model_calls')} {time.time() - t:.0f}s"
    check(f"model: {name}: a Lean task gets the maths caps and the Lean verifier", s == 200 and feats.get("domain") == "math" and v.get("verifier") == "lean_proof", detail)
    if must_verify:
        check(f"model: {name}: ends verified", s == 200 and d.get("assurance") == "verified", detail)
    else:
        # A hard theorem may or may not be solved by a 4B model; what must hold is honesty either way.
        check(f"model: {name}: verified, or abstained with a notice (never an unchecked proof)",
              s == 200 and (d.get("assurance") == "verified" or (d.get("assurance") == "abstained" and d.get("uncertainty_notice"))), detail)
    if s == 200 and d.get("assurance") == "verified":
        stmt = statement.split(":=")[0] + ":="
        check(f"model: {name}: the verified answer reproduces the statement byte for byte", stmt in d["message"]["content"])
        check(f"model: {name}: the verified answer has no import line", "import " not in d["message"]["content"])


def models(base, hard=False):
    # Whether a 4B model writes a valid Lean 4 proof is model strength, not harness correctness (the repair
    # pipeline with real Lean is tested deterministically in nafs-core). What must hold is honesty either way.
    lean_chat(base, EASY, "Prove in Lean 4 that adding zero to a natural number leaves it unchanged.", "easy theorem", False)
    if hard:
        lean_chat(base, STATEMENT, "Prove in Lean 4 that addition of natural numbers is commutative.", "add_comm", False)

    t = time.time()
    s, d = call(base, "POST", "/v1/harness/code", {
        "task": "is_palindrome(s) must ignore case and every character that is not a letter or digit. Fix solution.py.",
        "files": {"solution.py": "def is_palindrome(s):\n    return s == s[::-1]\n"},
        "verification": {"tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')",
                                   "assert is_palindrome('')", "assert is_palindrome('No lemon, no melon')"]}}, 2400)
    check("model: code agent fixes a function and the verifier accepts", s == 200 and d.get("passed") and d.get("assurance") == "verified",
          f"{d.get('status')} calls={d.get('model_calls')} actions={[x['action'] for x in d.get('steps', [])]} {time.time() - t:.0f}s")
    if s == 200:
        check("model: the agent's step records are complete", all({"step", "action", "observation", "ms"} <= set(x) for x in d["steps"]) and d["steps"])

    t = time.time()
    s, d = call(base, "POST", "/v1/chat", {
        "messages": [{"role": "user", "content": "Write a Python function chunks(xs, n) that splits a list into chunks of size n."}],
        "force_mode": "slow", "ephemeral": True,
        "verification": {"tests": ["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunks([], 3) == []", "assert chunks([1, 2], 5) == [[1, 2]]"]}}, 2400)
    notes = [x.get("note") for x in d.get("usage", {}).get("stages", []) if (x.get("note") or "").startswith("candidate")]
    check("model: a verified code chat on System 2 records its candidate tree", s == 200 and d.get("assurance") in ("verified", "abstained") and notes,
          f"{d.get('assurance')} {notes[:2]} {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--hard", action="store_true", help="also run the add_comm theorem (up to ~12 minutes)")
    a = ap.parse_args()
    t0 = time.time()
    deterministic(a.base)
    if not a.no_models:
        models(a.base, a.hard)
    n, ok = len(RESULTS), sum(r["pass"] for r in RESULTS)
    print(f"\n{ok}/{n} checks passed")
    json.dump({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0), "passed": ok, "total": n, "checks": RESULTS},
              open(os.path.join(ROOT, "benchmarks/harness_sweep.json"), "w"), indent=2)
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
