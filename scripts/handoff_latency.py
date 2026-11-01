"""How fast the request moves from System 1 to System 2 (the model switch inside a request), measured on the live gateway.

A request whose draft fails its exact check (a wrong `reference_answer` does that on purpose) goes: System 1 draft ->
check -> [System 1 rescue samples, n in one request] -> decision -> System 2. The SSE stream carries the stage events;
this script records, per request:

  draft_end -> decision      the gate's own time (checks, rescue samples when on, arithmetic)
  decision  -> s2_first      the switch: queueing for a System 2 slot, prefill of the System 2 prompt, first token
  s1_busy_after_decision     how long System 1 still reports busy slots after the decision (its streams are dropped;
                             llama-server frees a slot when it next tries to write to the closed connection)

Both llama-servers' /metrics are polled every 100 ms for `llamacpp:requests_processing`. Interleaved arms: rescue on
(the production default, 4 samples) and rescue off. Writes benchmarks/handoff_latency.json.

    python3 -I scripts/handoff_latency.py [--base http://127.0.0.1:8080] [--reps 4]
"""
import argparse, json, os, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS = [("What is 17 + 26? Answer with the number.", "99"), ("How many minutes are there in 3 hours? Answer with the number.", "7"),
           ("What is 12 times 12? Answer with the number.", "5"), ("A dozen eggs plus half a dozen is how many eggs? Answer with the number.", "2")]


def busy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1) as r:
            for l in r.read().decode().splitlines():
                if l.startswith("llamacpp:requests_processing"):
                    return float(l.split()[-1])
    except Exception:
        return None
    return 0.0


def poller(stop, samples, t0):
    while not stop.is_set():
        samples.append((time.time() - t0, busy(8001), busy(8002)))
        stop.wait(0.1)


def stream(base, body, events):
    req = urllib.request.Request(base + "/v1/chat/stream", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        ev = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev:
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                events.append((time.time(), ev, d.get("data", d)))
                if ev == "done":
                    return


def one(base, prompt, ref, rescue):
    body = {"messages": [{"role": "user", "content": prompt}], "verification": {"reference_answer": ref}, "ephemeral": True, "audit": False,
            "retrieve": False, "max_tokens": 160, "thinking_budget": 128, "show_reasoning": True}  # reasoning streamed: its first token marks System 2's start
    if rescue is not None:
        body["rescue"] = rescue
    events, samples, stop = [], [], threading.Event()
    t0 = time.time()
    threading.Thread(target=poller, args=(stop, samples, t0), daemon=True).start()
    try:
        stream(base, body, events)
    finally:
        stop.set()
    rel = [(t - t0, e, d) for t, e, d in events]
    fast_tokens = [t for t, e, d in rel if e == "token" and d.get("mode") == "fast"]
    decision = next((t for t, e, d in rel if e == "decision"), None)
    escalated = next((d.get("escalate") for t, e, d in rel if e == "decision"), None)
    s2_first = next((t for t, e, d in rel if e == "token" and d.get("mode") != "fast" and t > (decision or 0)), None)
    done = next((d for t, e, d in rel if e == "done"), {})
    phases = [(t, d.get("state"), d.get("label")) for t, e, d in rel if e == "phase"]
    s1_after = [t for t, s1, s2 in samples if decision is not None and t > decision and (s1 or 0) > 0]
    return {"prompt": prompt[:40], "rescue": rescue, "escalated": escalated, "draft_end_s": max(fast_tokens) if fast_tokens else None,
            "decision_s": decision, "s2_first_s": s2_first, "draft_to_decision_s": (decision - max(fast_tokens)) if decision and fast_tokens else None,
            "decision_to_s2_first_s": (s2_first - decision) if s2_first and decision else None,
            "s1_busy_after_decision_s": (max(s1_after) - decision) if s1_after else 0.0,
            "s1_busy_max_after_decision": max((s1 for t, s1, s2 in samples if decision and t > decision and s1 is not None), default=0),
            "total_s": rel[-1][0] if rel else None, "mode": done.get("reasoning_mode"), "assurance": done.get("assurance"),
            "stages": [{k: s.get(k) for k in ("stage", "ms", "queue_ms", "prefill_ms", "prompt_n", "cache_n")} for s in done.get("usage", {}).get("stages", [])],
            "phases": phases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_at_start": round(os.getloadavg()[0], 2), "runs": []}
    arms = [4, 0]
    for i in range(a.reps):
        prompt, ref = PROMPTS[i % len(PROMPTS)]
        for arm in (arms if i % 2 == 0 else arms[::-1]):
            r = one(a.base, prompt, ref, arm)
            out["runs"].append(r)
            print(f"rescue={arm} esc={r['escalated']} draft->decision {r['draft_to_decision_s'] and round(r['draft_to_decision_s'], 2)} s  "
                  f"decision->S2 first {r['decision_to_s2_first_s'] and round(r['decision_to_s2_first_s'], 2)} s  S1 busy after {round(r['s1_busy_after_decision_s'], 2)} s  total {round(r['total_s'] or 0, 1)} s", flush=True)
            json.dump(out, open(os.path.join(ROOT, "benchmarks/handoff_latency.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
