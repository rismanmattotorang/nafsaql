#!/usr/bin/env python3
"""Sweep: can the models be prompted into a drawable chart (```chart JSON for Nivo) or Mermaid diagram?

Each prompt is asked of each model under each hint variant (A none, B full, C compact, D compact with kinds, E the gateway's own); the answer is scored with the real
renderer (scripts/visual_score.mjs): the block exists, it draws, it is the right kind, and the data is the data asked for.
Runs are interleaved by prompt so a busy machine hits every arm alike. Results: benchmarks/visual_sweep.json.

  python3 scripts/visual_sweep.py [--model fast|slow] [--variants A,B,C] [--only chart|diagram] [--ids a,b] [--limit N]
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
opt = lambda k, d: args[args.index(k) + 1] if k in args else d
BASE = opt("--base", "http://127.0.0.1:8080"); MODEL = opt("--model", "fast"); VARIANTS = opt("--variants", "A,B,C").split(",")
ONLY = opt("--only", ""); LIMIT = int(opt("--limit", "0")); OUT = opt("--out", os.path.join(ROOT, "benchmarks", f"visual_sweep_{MODEL}.json"))

# The hints are the ones the gateway adds (rust/crates/nafs-core/src/orchestrator.rs `visual_instruction`).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from visual_hints import HINTS

CHARTS = [
 ("bar", "Make a bar chart of monthly sales: Jan 120, Feb 150, Mar 90, Apr 200.", dict(type="bar", numbers=[120, 150, 90, 200], labels=["jan", "feb", "mar", "apr"])),
 ("line", "Plot a line chart of temperature by hour: 6am 12, 9am 16, 12pm 21, 3pm 24, 6pm 19.", dict(type="line", numbers=[12, 16, 21, 24, 19], labels=["6am", "9am", "12pm", "3pm", "6pm"])),
 ("pie", "Show a pie chart of browser share: Chrome 65, Safari 20, Firefox 8, Edge 7.", dict(type="pie", numbers=[65, 20, 8, 7], labels=["chrome", "safari", "firefox", "edge"])),
 ("donut", "Draw a donut chart of a budget: Rent 40, Food 25, Travel 15, Savings 20.", dict(type="pie", numbers=[40, 25, 15, 20], labels=["rent", "food", "travel", "savings"])),
 ("grouped", "Compare revenue and cost by quarter in a bar chart: Q1 revenue 100 cost 70, Q2 revenue 140 cost 90, Q3 revenue 160 cost 110.", dict(type="bar", numbers=[100, 70, 140, 90, 160, 110], labels=["q1", "q2", "q3", "revenue", "cost"])),
 ("radar", "Make a radar chart for two phones on battery, camera, speed and price: Phone A 8, 6, 9, 5; Phone B 6, 9, 7, 8.", dict(type="radar", numbers=[8, 6, 9, 5, 6, 9, 7, 8], labels=["battery", "camera", "speed", "price", "phone a", "phone b"])),
 ("scatter", "Make a scatter plot of height against weight: (150, 50), (160, 58), (170, 66), (180, 77), (190, 88).", dict(type="scatter", numbers=[150, 50, 160, 58, 170, 66, 180, 77, 190, 88], labels=[])),
 ("treemap", "Draw a treemap of storage use: Photos 40 GB, Videos 90 GB, Apps 30 GB, Docs 10 GB.", dict(type="treemap", numbers=[40, 90, 30, 10], labels=["photos", "videos", "apps", "docs"])),
 ("funnel", "Make a funnel chart for signups: Visits 1000, Signups 300, Trials 120, Paid 40.", dict(type="funnel", numbers=[1000, 300, 120, 40], labels=["visits", "signups", "trials", "paid"])),
 ("sankey", "Draw a sankey chart of energy: Coal 30 to Electricity, Gas 40 to Electricity, Electricity 50 to Homes, Electricity 20 to Industry.", dict(type="sankey", numbers=[30, 40, 50, 20], labels=["coal", "gas", "electricity", "homes", "industry"])),
 ("heatmap", "Make a heatmap of sales by day and store: Mon store A 5, store B 8; Tue store A 7, store B 3; Wed store A 9, store B 6.", dict(type="heatmap", numbers=[5, 8, 7, 3, 9, 6], labels=["mon", "tue", "wed"])),
 ("stacked", "Make a horizontal stacked bar chart of votes by city: Paris yes 60 no 40, Rome yes 55 no 45.", dict(type="bar", numbers=[60, 40, 55, 45], labels=["paris", "rome", "yes", "no"])),
]
DIAGRAMS = [
 ("tea", "Draw a flowchart of how to make tea: boil water, add a tea bag, steep for 3 minutes, check if it is strong enough; if not steep more, if yes serve.", dict(kind="flowchart|graph", min=5, words=["boil", "tea bag", "steep", "serve"])),
 ("login", "Make a flowchart for a login process with a decision for a correct password and a locked account after 3 failures.", dict(kind="flowchart|graph", min=5, words=["password", "lock", "login"])),
 ("web", "Draw a sequence diagram of a browser requesting a web page from a server, which queries a database.", dict(kind="sequenceDiagram", min=3, words=["browser", "server", "database"])),
 ("traffic", "Draw a state diagram of a traffic light with green, yellow and red.", dict(kind="stateDiagram", min=3, words=["green", "yellow", "red"])),
 ("mpie", "Draw a Mermaid pie chart of how a day is spent: Work 8, Sleep 8, Leisure 4, Commute 2.", dict(kind="pie", min=4, words=["work", "sleep", "leisure", "commute"])),
 ("ci", "Draw a diagram of a CI pipeline: commit, build, test, deploy to staging, approval, deploy to production.", dict(kind="flowchart|graph", min=6, words=["commit", "build", "test", "staging", "production"])),
 ("pipeline", "Draw a left to right flowchart of a data pipeline: ingest, clean, train, evaluate, publish.", dict(kind="flowchart|graph", min=5, words=["ingest", "clean", "train", "evaluate", "publish"])),
 ("atm", "Show with a diagram how an ATM withdrawal works between the customer, the ATM and the bank.", dict(kind="sequenceDiagram", min=3, words=["customer", "atm", "bank"])),
 ("sub", "Make a flowchart with two groups: a frontend group with web and mobile, and a backend group with api and database.", dict(kind="flowchart|graph", min=4, words=["web", "mobile", "api", "database"])),
 ("water", "Make a diagram of the water cycle.", dict(kind="flowchart|graph", min=4, words=["evaporat", "cloud", "rain"])),
]
prompts = [("chart",) + p for p in CHARTS] + [("diagram",) + p for p in DIAGRAMS]
if ONLY: prompts = [p for p in prompts if p[0] == ONLY]
if opt("--ids", ""): prompts = [p for p in prompts if p[1] in opt("--ids", "").split(",")]
if LIMIT: prompts = prompts[:LIMIT]


def ask(kind, text, variant):
    hint = HINTS[kind].get(variant, "")
    content = f"{hint}\n\nQuestion: {text}" if hint else text
    # E = the gateway's own instruction (the question is sent as it is). Every other variant puts its text in the question and
    # pins the domain, so the gateway adds nothing of its own.
    body = {"messages": [{"role": "user", "content": content}], "ephemeral": True, "retrieve": False, "force_mode": MODEL, "max_tokens": 900 if MODEL == "fast" else 1400}
    if variant != "E": body["domain_hint"] = "open"
    req = urllib.request.Request(BASE + "/v1/chat", data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.load(r)
        return j["message"]["content"], j["usage"]["completion_tokens"], j["usage"]["prompt_tokens"], time.time() - t
    except Exception as e:
        return f"[error {e}]", 0, 0, time.time() - t


rows = []
if os.path.exists(OUT) and "--resume" in args:
    rows = json.load(open(OUT))["rows"]
done = {(r["id"], r["variant"]) for r in rows}
for kind, pid, text, expect in prompts:
    for v in VARIANTS:
        if (pid, v) in done: continue
        ans, ctok, ptok, secs = ask(kind, text, v)
        rows.append(dict(id=pid, kind=kind, variant=v, expect=expect, answer=ans, completion_tokens=ctok, prompt_tokens=ptok, secs=round(secs, 1)))
        print(f"{pid:9s} {v} {ctok:4d} tok {secs:5.0f}s", flush=True)
        json.dump(dict(model=MODEL, load=os.getloadavg(), rows=rows), open(OUT, "w"))
scores = json.loads(subprocess.run(["node", os.path.join(ROOT, "scripts", "visual_score.mjs")], input=json.dumps([dict(id=r["id"] + r["variant"], kind=r["kind"], expect=r["expect"], answer=r["answer"]) for r in rows]), capture_output=True, text=True, check=True).stdout)
for r, s in zip(rows, scores): r["score"] = s
json.dump(dict(model=MODEL, load_at_end=os.getloadavg(), rows=rows), open(OUT, "w"), indent=1)
print(f"\nmodel={MODEL}  load={os.getloadavg()[0]:.1f}  (a busy shared machine slows time, not accuracy)")
for kind in ("chart", "diagram"):
    for v in VARIANTS:
        rs = [r for r in rows if r["kind"] == kind and r["variant"] == v]
        if not rs: continue
        ok = sum(r["score"]["ok"] for r in rs); fence = sum(r["score"]["canonical"] for r in rs); valid = sum(r["score"]["drawn"] for r in rs)
        print(f"{kind:8s} {v}: right {ok}/{len(rs)}  drawn {valid}/{len(rs)}  exact fence {fence}/{len(rs)}  mean {sum(r['secs'] for r in rs)/len(rs):.0f}s  {sum(r['completion_tokens'] for r in rs)/len(rs):.0f} tok")
for r in rows:
    if not r["score"]["ok"]: print(f"  miss {r['kind']:7s} {r['id']:9s} {r['variant']}: {r['score']['note']}")
