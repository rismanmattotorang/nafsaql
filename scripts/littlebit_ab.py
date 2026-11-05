#!/usr/bin/env python3
"""LittleBit against dense on the same engine (Phase 19). Interleaved, greedy, exact-answer prompts.

    scripts/littlebit_ab.py --dense http://127.0.0.1:8201 --lb http://127.0.0.1:8211 --out benchmarks/littlebit_ab.json

Each prompt has one exact answer (a number or a word) checked by string match, so the pass rate needs no judge. Decode speed is
the server's own `predicted_per_second`. The two servers run on the same four cores: run it on a quiet host and read the speed
as indicative only (benchmarks are interleaved so drift hits both alike).
"""
import argparse, json, re, time, urllib.request

ITEMS = [
    ("What is 17 * 23? Answer with the number only.", "391"),
    ("What is 144 / 12? Answer with the number only.", "12"),
    ("What is 1000 - 387? Answer with the number only.", "613"),
    ("A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. What time does it arrive? Answer like 4:05 pm.", "6:15"),
    ("What is the capital of Indonesia? One word.", "jakarta"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("What is 9 squared? Answer with the number only.", "81"),
    ("How many days are in a week? Answer with the number only.", "7"),
    ("What is the chemical symbol for water? Answer with the formula only.", "h2o"),
    ("What is 15% of 200? Answer with the number only.", "30"),
    ("How many minutes are in 3 hours? Answer with the number only.", "180"),
    ("What is the largest planet in the solar system? One word.", "jupiter"),
]


def ask(url, prompt, max_tokens):
    body = json.dumps({
        "model": "m", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    wall = time.time() - t
    text = r["choices"][0]["message"].get("content") or ""
    tm = r.get("timings", {})
    return text, tm.get("predicted_per_second", 0.0), tm.get("predicted_n", 0), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--lb", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default="benchmarks/littlebit_ab.json")
    a = ap.parse_args()
    rows = []
    for prompt, want in ITEMS:
        row = {"prompt": prompt, "want": want}
        for name, url in (("dense", a.dense), ("littlebit", a.lb)):
            text, tps, n, wall = ask(url, prompt, a.max_tokens)
            ok = want in re.sub(r"[\s,]", "", text.lower())
            row[name] = {"text": text[:200], "pass": ok, "tok_s": round(tps, 2), "tokens": n, "wall_s": round(wall, 1)}
        rows.append(row)
        print(f"{prompt[:48]:50s} dense={'ok' if row['dense']['pass'] else '--'} lb={'ok' if row['littlebit']['pass'] else '--'}", flush=True)
    summary = {}
    for name in ("dense", "littlebit"):
        speeds = [r[name]["tok_s"] for r in rows if r[name]["tok_s"] > 0]
        summary[name] = {"passed": sum(r[name]["pass"] for r in rows), "of": len(rows),
                         "median_tok_s": sorted(speeds)[len(speeds) // 2] if speeds else 0.0}
    out = {"max_tokens": a.max_tokens, "summary": summary, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
