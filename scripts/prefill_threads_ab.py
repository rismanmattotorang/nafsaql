#!/usr/bin/env python3
"""Reading a prompt with every core, writing tokens with one fewer: is it faster end to end? (Phase 27.)

Starts two copies of the engine server on spare ports with the same model: one with `-t T` (prompts and tokens on
the same T threads, as before) and one with `-t T --prefill-threads P`. Sends both the same prompts, alternating
which goes first, each prompt with a different beginning so nothing is reused, and reports the prompt speed, the
decode speed and the time to the last token of each. Stops both servers at the end.

Usage: python3 scripts/prefill_threads_ab.py [--model models/Qwen3-1.7B-Q4_K_M.gguf] [--threads 3] [--prefill 4]
                                             [--prompts 8] [--prompt-chars 4800] [--out benchmarks/engine_phase27/prefill_threads.json]
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXT = ("The gate escalates only when the error it expects to remove is worth the waiting time. "
        "Passages are 1,200 characters long and overlap by 150. Search runs in memory beside the store. "
        "A verified answer is one an exact checker accepted; agreement between samples is a hint, never proof. ")


def call(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait(port):
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise SystemExit(f"server on :{port} did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B-Q4_K_M.gguf")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-chars", type=int, default=4800)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="benchmarks/engine_phase27/prefill_threads.json")
    a = ap.parse_args()
    binary = os.path.join(ROOT, "rust/target/release/nafs-smiley-server")
    arms = {"same": (8211, []), "wide": (8212, ["--prefill-threads", str(a.prefill)])}
    procs = []
    try:
        for name, (port, extra) in arms.items():
            procs.append(subprocess.Popen([binary, "-m", os.path.join(ROOT, a.model), "--alias", name, "-t", str(a.threads),
                                           "--port", str(port), "--slots", "1", "--prefix-cache-mb", "0", *extra],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for port, _ in arms.values():
            wait(port)
        rows = []
        for i in range(a.prompts):
            # A different beginning each time: nothing of an earlier prompt can be reused.
            body_text = f"Document {i * 7919 + 13}. " + (TEXT * (a.prompt_chars // len(TEXT) + 1))[: a.prompt_chars]
            body = {"model": "x", "messages": [{"role": "user", "content": body_text + "\nIn one sentence, what decides that an answer is verified?"}],
                    "max_tokens": a.max_tokens, "temperature": 0}
            row = {"prompt": i}
            for name in (("same", "wide") if i % 2 == 0 else ("wide", "same")):
                t = call(arms[name][0], body)["timings"]
                row[name] = {"prompt_n": t["prompt_n"], "prompt_tok_s": round(t["prompt_n"] / (t["prompt_ms"] / 1000), 1),
                             "decode_tok_s": round(t["predicted_per_second"], 2), "predicted_n": t["predicted_n"],
                             "total_s": round((t["prompt_ms"] + t["predicted_ms"]) / 1000, 2)}
            rows.append(row)
            print(f"{i:2}  same: {row['same']['prompt_tok_s']:6.1f} tok/s prompt, {row['same']['decode_tok_s']:5.2f} decode, {row['same']['total_s']:6.2f} s"
                  f"   wide: {row['wide']['prompt_tok_s']:6.1f} tok/s prompt, {row['wide']['decode_tok_s']:5.2f} decode, {row['wide']['total_s']:6.2f} s", flush=True)
    finally:
        for p in procs:
            p.terminate()
    med = lambda arm, k: statistics.median(r[arm][k] for r in rows)  # noqa: E731
    summary = {k: {"same": med("same", k), "wide": med("wide", k), "ratio": round(med("wide", k) / med("same", k), 3)}
               for k in ("prompt_tok_s", "decode_tok_s", "total_s")}
    summary["wide_faster_prompt_in"] = f"{sum(1 for r in rows if r['wide']['prompt_tok_s'] > r['same']['prompt_tok_s'])} of {len(rows)}"
    summary["wide_faster_total_in"] = f"{sum(1 for r in rows if r['wide']['total_s'] < r['same']['total_s'])} of {len(rows)}"
    load = os.getloadavg()
    out = {"what": "Prompts read by P threads and tokens written by T, against both on T (Phase 27)", "date": time.strftime("%Y-%m-%d %H:%M"),
           "model": a.model, "threads": a.threads, "prefill_threads": a.prefill, "prompt_tokens": rows[0]["same"]["prompt_n"],
           "host_load_average": [round(x, 1) for x in load], "note": "SHARED MACHINE" if load[0] > 4 else "quiet host",
           "summary": summary, "rows": rows}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
