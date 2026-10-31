"""Generate the Phase 11 results tables and macros of the report from the code-agent A/B records.

Reads benchmarks/harness_ab.json (round 1: Phase 10 loop vs Phase 11) and, when present,
benchmarks/harness_ab_round2.json (round 2: unstick on/off, System 2 reasoning budget) and
benchmarks/harness_ab_round3.json (round 3: a smaller reasoning budget on the first step only), and writes
docs/report/sections/harness_ab_macros.tex (\\HAB... / \\HABR... macros, loaded in the preamble so the abstract can
use them) and docs/report/sections/harness_ab_tables.tex (one table per round).
Every number in the report's Phase 11 results comes from here.

    python3 -I scripts/gen_harness_ab.py
"""
import json, os, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_MACROS = os.path.join(ROOT, "docs/report/sections/harness_ab_macros.tex")
OUT_TABLES = os.path.join(ROOT, "docs/report/sections/harness_ab_tables.tex")
MODEL_LABEL = {"system1": "System~1 (Qwen3-1.7B)", "system2": "System~2 (Qwen3-4B-Thinking)"}
MODEL_KEY = {"system1": "One", "system2": "Two"}
ROUNDS = [
    {
        "src": "benchmarks/harness_ab.json", "prefix": "HAB", "label": "tab:harnessab",
        "arms": {"phase10": ("Phase 10 loop", "Old"), "phase11": ("Phase 11 (auto edits)", "New"),
                 "phase11_any": ("Phase 11, patch always", "Any")},
        "caption": "Code agent, interleaved live A/B on the reference machine (\\texttt{benchmarks/harness\\_ab.json}). "
                   "Same tasks, model and budget (6 steps) per arm; arm order rotated per task. Malformed: steps whose reply "
                   "held no action or a patch that did not apply. Prefill: prompt tokens the server computed; cache: share of "
                   "prompt tokens served from its cache. Wall times are from a shared host and are not speed claims.",
    },
    {
        "src": "benchmarks/harness_ab_round2.json", "prefix": "HABR", "label": "tab:harnessab2",
        "arms": {"phase11_nou": ("Phase 11 without unstick", "NoU"), "phase11_u": ("Phase 11 with unstick", "U"),
                 "phase11_b512": ("Phase 11, 512 reasoning tokens", "Big"),
                 "phase11_b256": ("Phase 11, 256 reasoning tokens", "Small")},
        "caption": "Round 2, on the build with value-carrying evidence (\\texttt{benchmarks/harness\\_ab\\_round2.json}): "
                   "\\texttt{unstick} on System~1 and the reasoning budget per step on System~2. Columns as in "
                   "Table~\\ref{tab:harnessab}.",
    },
    {
        "src": "benchmarks/harness_ab_round3.json", "prefix": "HABRR", "label": "tab:harnessab3",
        "arms": {"phase11_r3": ("512 reasoning tokens every step", "Fixed"),
                 "phase11_adapt": ("256 on the first step, then 512", "Adapt")},
        "caption": "Round 3, on the build whose grammar keeps patches out of code blocks "
                   "(\\texttt{benchmarks/harness\\_ab\\_round3.json}): the System~2 reasoning budget fixed or raised "
                   "after a failure. Columns as in Table~\\ref{tab:harnessab}.",
    },
]


def arm_stats(runs):
    ok = [r for r in runs if "error" not in r]
    steps = [s for r in ok for s in r["steps"]]
    usage = [s["usage"] for s in steps if s.get("usage")]
    prompt = sum(u.get("prompt_tokens") or 0 for u in usage)
    cached = sum(u.get("cached_tokens") or 0 for u in usage)
    verified = sum(1 for r in ok if r["passed"])
    wall = sum(r["wall_s"] for r in ok)
    return {
        "n": len(ok), "errors": len(runs) - len(ok), "verified": verified,
        "calls": sum(r["model_calls"] for r in ok), "wall": wall,
        "wall_median": statistics.median(r["wall_s"] for r in ok) if ok else 0,
        "prefill": prompt, "cached": cached, "hit": cached / (prompt + cached) if prompt + cached else 0,
        "decoded": sum(r["decoded_tokens"] or 0 for r in ok),
        "malformed": sum(r["malformed"] for r in ok), "actions": len(steps),
        "prefill_ms": sum(u.get("prefill_ms") or 0 for u in usage),
        "decode_ms": sum(u.get("decode_ms") or 0 for u in usage),
        "per_verified": wall / verified if verified else None,
    }


def tex_num(x, d=0):
    return f"{x:,.{d}f}".replace(",", "{,}")


def round_tex(rd):
    path = os.path.join(ROOT, rd["src"])
    if not os.path.exists(path):
        return [], []
    runs = json.load(open(path))["runs"]
    macros, rows = [], []
    for model in ("system1", "system2"):
        mr = [r for r in runs if r["model"] == model]
        arms = [a for a in rd["arms"] if any(r["arm"] == a for r in mr)]
        if not arms:
            continue
        tasks = sorted({r["task"] for r in mr})
        # Only tasks every arm finished, so the arms are compared on the same items.
        paired = [t for t in tasks if all(any(r["task"] == t and r["arm"] == a and "error" not in r for r in mr) for a in arms)]
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{MODEL_LABEL[model]}, {len(paired)} tasks}}}} \\\\")
        for a in arms:
            st = arm_stats([r for r in mr if r["arm"] == a and r["task"] in paired])
            key = rd["prefix"] + MODEL_KEY[model] + rd["arms"][a][1]
            for k, v in (("Verified", st["verified"]), ("N", st["n"]), ("Calls", st["calls"]),
                         ("Wall", tex_num(st["wall"])), ("Prefill", tex_num(st["prefill"])),
                         ("Cached", tex_num(st["cached"])), ("Hit", tex_num(100 * st["hit"])),
                         ("Decoded", tex_num(st["decoded"])), ("Malformed", st["malformed"]),
                         ("Actions", st["actions"]), ("WallMedian", tex_num(st["wall_median"], 1)),
                         ("WallPerVerified", tex_num(st["per_verified"]) if st["per_verified"] else "--"),
                         ("DecodeS", tex_num(st["decode_ms"] / 1000)), ("PrefillS", tex_num(st["prefill_ms"] / 1000))):
                macros.append(f"\\newcommand{{\\{key}{k}}}{{{v}}}")
            rows.append(
                f"{rd['arms'][a][0]} & {st['verified']}/{st['n']} & {st['calls']} & {st['malformed']}/{st['actions']} & "
                f"{tex_num(st['prefill'])} & {tex_num(100 * st['hit'])}\\,\\% & {tex_num(st['decoded'])} & {tex_num(st['wall'])} \\\\")
            print(f"{rd['prefix']} {model} {a:13s} verified {st['verified']}/{st['n']} calls {st['calls']} "
                  f"malformed {st['malformed']}/{st['actions']} prefill {st['prefill']} cached {st['cached']} hit {st['hit']:.0%} "
                  f"decoded {st['decoded']} wall {st['wall']:.0f}s (prefill {st['prefill_ms']/1000:.0f}s "
                  f"decode {st['decode_ms']/1000:.0f}s) errors {st['errors']}")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    table = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{rd['caption']}}}", f"\\label{{{rd['label']}}}", "\\small",
        "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
        "Arm & Verified & Calls & Malformed & Prefill tok. & Cache & Decoded tok. & Wall (s) \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table*}",
    ]
    return macros, table


def main():
    head = ["% Generated by scripts/gen_harness_ab.py from benchmarks/harness_ab*.json; do not edit by hand."]
    macros, tables = list(head), list(head)
    for rd in ROUNDS:
        m, t = round_tex(rd)
        macros += m
        tables += t
    open(OUT_MACROS, "w").write("\n".join(macros) + "\n")
    open(OUT_TABLES, "w").write("\n".join(tables) + "\n")
    print("wrote", os.path.relpath(OUT_MACROS, ROOT), "and", os.path.relpath(OUT_TABLES, ROOT))


if __name__ == "__main__":
    main()
