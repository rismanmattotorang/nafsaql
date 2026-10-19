#!/usr/bin/env bash
# Performance tuning experiments for Profile A on the reference CPU (writes benchmarks/reference-machine/perf_tuning_*.{txt,json}).
#   scripts/perf_tuning.sh bench     llama-bench: flash attention and KV-cache types, prefill and decode at depth 0 and 1024
#   scripts/perf_tuning.sh spec      llama-server: System 2 with and without a Qwen3-0.6B draft model (speculative decoding)
# Run on an idle machine; other heavy processes make the numbers noisy (the report records the load average).
set -euo pipefail
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build/bin; OUT=benchmarks/reference-machine; mkdir -p "$OUT"
S2=models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf; S1=models/Qwen3-1.7B-Q4_K_M.gguf; DRAFT=models/Qwen3-0.6B-Q8_0.gguf
case "${1:-bench}" in
bench)
  f="$OUT/perf_tuning_bench.txt"; { echo "# llama-bench $(date -u +%FT%TZ) load: $(sysctl -n vm.loadavg)"; } > "$f"
  for m in "$S1" "$S2"; do
    for cfg in "-fa off" "-fa on" "-fa on -ctk q8_0 -ctv q8_0"; do
      echo "## $(basename $m) $cfg" | tee -a "$f"
      # shellcheck disable=SC2086
      $B/llama-bench -m "$m" -t 4 -r 3 -p 512 -n 128 $cfg -o md 2>/dev/null | grep -E "^\|" | tee -a "$f"
      # shellcheck disable=SC2086
      $B/llama-bench -m "$m" -t 4 -r 3 -p 0 -n 128 -d 1024 $cfg -o md 2>/dev/null | grep -E "^\| .*tg" | tee -a "$f"
    done
  done ;;
spec)
  f="$OUT/perf_tuning_spec.json"; PORT=8012
  python3 - "$f" "$PORT" "$S2" "$DRAFT" "$B" <<'PY'
import json, subprocess, sys, time, urllib.request, os
out, port, s2, draft, b = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
prompts = ["A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip? End with 'Answer: <number>'.",
           "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? End with 'Answer: <number>'.",
           "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters."]
def call(p, prompt, temp):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 320, "temperature": temp, "seed": 1, "thinking_budget_tokens": 256}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{p}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    tm = r.get("timings", {}); return {"wall_s": round(time.time() - t, 2), "predicted_n": tm.get("predicted_n"), "tok_s": tm.get("predicted_per_second"), "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted")}
def serve(extra):
    args = [f"{b}/llama-server", "-m", s2, "-t", "4", "-c", "8192", "-np", "1", "--host", "127.0.0.1", "--port", str(port), "--jinja", "--reasoning-format", "deepseek"] + extra
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open("logs/perf_spec_server.log", "w"))
    for _ in range(180):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p
        except Exception: time.sleep(1)
    p.kill(); raise SystemExit("server did not start")
configs = {"baseline": [],
           "spec_draft_0.6b": ["-md", draft, "--spec-draft-n-max", "6", "--spec-draft-n-min", "1", "--spec-draft-p-min", "0.6", "-td", "4"],
           "spec_ngram": ["--spec-type", "ngram-mod"]}
res = {"load_before": os.getloadavg(), "runs": {k: [] for k in configs}, "protocol": "2 alternating rounds x 3 prompts, temperature 0.6, max 320 tokens, thinking budget 256"}
for rnd in range(2):
    for name, extra in configs.items():
        p = serve(extra)
        try:
            call(port, "Say OK.", 0.0)  # warm-up
            for pr in prompts:
                r = call(port, pr, 0.6); r["round"] = rnd; res["runs"][name].append(r)
                print(rnd, name, r, round(os.getloadavg()[0], 1), flush=True)
        finally:
            p.terminate(); p.wait()
res["summary"] = {k: {"median_tok_s": sorted(x["tok_s"] for x in v)[len(v) // 2], "mean_tok_s": round(sum(x["tok_s"] for x in v) / len(v), 2)} for k, v in res["runs"].items()}
print(json.dumps(res["summary"]), flush=True)
res["load_after"] = os.getloadavg()
json.dump(res, open(out, "w"), indent=1)
PY
  ;;
*) echo "usage: $0 bench|spec"; exit 2 ;;
esac
