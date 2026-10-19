#!/usr/bin/env python3
"""Generate LaTeX tables and backlog exports from data/*.csv (single source of truth).
Run from anywhere: python3 scripts/gen_tables.py"""
import csv, os, collections
ROOT=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(ROOT); DATA=os.path.join(REPO,"data"); DOCS=os.path.join(REPO,"docs")
def esc(s): return (s.replace("\\","\\textbackslash{}").replace("&","\\&").replace("%","\\%").replace("_","\\_").replace("#","\\#").replace("<=","$\\le$").replace(">=","$\\ge$").replace("->","$\\to$"))
reqs=list(csv.DictReader(open(os.path.join(DATA,"requirements.csv"))))
tasks=list(csv.DictReader(open(os.path.join(DATA,"tasks.csv"))))
cats=collections.OrderedDict([("FR","Functional requirements"),("NFR","Non-functional requirements"),("IF","Interface requirements"),("MR","Model requirements"),("TR","Training requirements"),("SR","Security, safety and ethics requirements"),("OR","Operations requirements"),("ER","Evaluation and test requirements")])
os.makedirs(os.path.join(DOCS,"spec","gen"),exist_ok=True); os.makedirs(os.path.join(DOCS,"roadmap","gen"),exist_ok=True)
# --- spec: requirement tables per category (longtable) ---
with open(os.path.join(DOCS,"spec","gen","requirements.tex"),"w") as f:
    for c,title in cats.items():
        rs=[r for r in reqs if r["id"].split("-")[0]==c]
        f.write("\\subsection{%s}\n"%title)
        f.write("\\begin{longtable}{@{}p{1.45cm}p{2.5cm}p{8.1cm}p{1.6cm}p{1.1cm}@{}}\n")
        f.write("\\caption{%s}\\\\\n\\toprule ID & Title & Normative statement & Priority & Phase\\\\\\midrule\\endfirsthead\n"%title)
        f.write("\\toprule ID & Title & Normative statement & Priority & Phase\\\\\\midrule\\endhead\n\\bottomrule\\endfoot\n")
        for r in rs:
            f.write("\\textbf{%s} & %s & %s & %s & P%s\\\\\n"%(r["id"],esc(r["title"]),esc(r["statement"]),r["priority"],r["phase"]))
        f.write("\\end{longtable}\n\n")
# --- spec: traceability matrix ---
with open(os.path.join(DOCS,"spec","gen","traceability.tex"),"w") as f:
    f.write("\\begin{longtable}{@{}p{1.45cm}p{4.4cm}p{4.2cm}p{3.1cm}p{1.1cm}@{}}\n")
    f.write("\\caption{Traceability matrix: requirement, motivating result or decision, implementing component, verifying test, phase}\\\\\n")
    f.write("\\toprule ID & Motivation & Component & Test & Phase\\\\\\midrule\\endfirsthead\n\\toprule ID & Motivation & Component & Test & Phase\\\\\\midrule\\endhead\n\\bottomrule\\endfoot\n")
    for r in reqs:
        f.write("%s & %s & %s & %s & P%s\\\\\n"%(r["id"],esc(r["source"]),esc(r["component"]),esc(r["test"]),r["phase"]))
    f.write("\\end{longtable}\n")
# --- roadmap: phase -> requirements, tasks tables ---
byphase=collections.defaultdict(list)
for r in reqs: byphase[r["phase"]].append(r["id"])
with open(os.path.join(DOCS,"roadmap","gen","phase_reqs.tex"),"w") as f:
    for p in sorted(byphase): f.write("\\newcommand{\\ReqsPhase%s}{%s}\n"%("ABCDEFGH"[int(p)],", ".join(byphase[p])))
with open(os.path.join(DOCS,"roadmap","gen","tasks.tex"),"w") as f:
    for p in sorted(set(t["phase"] for t in tasks)):
        f.write("\\newcommand{\\TasksPhase%s}{\n"%"ABCDEFGH"[int(p)])
        f.write("\\begin{tabular}{@{}p{1.1cm}p{2.3cm}p{6.0cm}p{0.9cm}p{1.9cm}p{1.7cm}@{}}\n\\toprule ID & Epic & Task & PW & Depends & MoSCoW\\\\\\midrule\n")
        for t in tasks:
            if t["phase"]==p: f.write("%s & %s & %s & %s & %s & %s\\\\\n"%(t["id"],esc(t["epic"]),esc(t["title"]),t["est_pw"],esc(t["depends"]) or "--",t["moscow"]))
        f.write("\\bottomrule\\end{tabular}}\n")
    tot=collections.defaultdict(float)
    for t in tasks: tot[t["phase"]]+=float(t["est_pw"])
    f.write("\\newcommand{\\PWTotals}{%s}\n"%"; ".join("P%s: %.1f"%(p,v) for p,v in sorted(tot.items())))
    f.write("\\newcommand{\\PWGrand}{%.1f}\n"%sum(tot.values()))
    f.write("\\newcommand{\\NTasks}{%d}\n\\newcommand{\\NReqs}{%d}\n"%(len(tasks),len(reqs)))
# --- backlog export (CSV + MD) ---
with open(os.path.join(DOCS,"roadmap","backlog.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["task_id","phase","epic","title","estimate_person_weeks","depends_on","requirement_ids","moscow"])
    for t in tasks: w.writerow([t["id"],"P"+t["phase"],t["epic"],t["title"],t["est_pw"],t["depends"],t["reqs"],t["moscow"]])
with open(os.path.join(DOCS,"roadmap","roadmap.md"),"w") as f:
    f.write("# Nafs-Aql backlog (generated from data/tasks.csv and data/requirements.csv)\n\nEstimates are person-weeks (PW) and are estimates, not commitments.\n\n")
    for p in sorted(set(t["phase"] for t in tasks)):
        f.write("## Phase %s\n\n| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |\n|---|---|---|---|---|---|---|\n"%p)
        for t in tasks:
            if t["phase"]==p: f.write("| %s | %s | %s | %s | %s | %s | %s |\n"%(t["id"],t["epic"],t["title"],t["est_pw"],t["depends"],t["reqs"],t["moscow"]))
        f.write("\n")
    f.write("## Requirements by phase\n\n")
    for p in sorted(byphase): f.write("- **Phase %s**: %s\n"%(p,", ".join(byphase[p])))
# consistency checks
req_ids={r["id"] for r in reqs}; covered=set()
for t in tasks:
    for x in [s.strip() for s in t["reqs"].split(",") if s.strip()]:
        if x not in req_ids: print("WARN task",t["id"],"references unknown req",x)
        covered.add(x)
missing=sorted(req_ids-covered); print("requirements not scheduled in any task:",missing)
for t in tasks:
    for d in [s.strip() for s in t["depends"].split(",") if s.strip()]:
        if d not in {x["id"] for x in tasks}: print("WARN unknown dependency",d)
print("generated", len(reqs),"reqs", len(tasks),"tasks")
# --- research/RTM.md (requirements traceability table) ---
with open(os.path.join(DOCS,"research","RTM.md"),"w") as f:
    f.write("# Requirements traceability table (generated from data/requirements.csv)\n\n| Req ID | Statement (short) | Motivating result / decision | Component | Verifying test | Phase |\n|---|---|---|---|---|---|\n")
    for r in reqs: f.write("| %s | %s | %s | %s | %s | P%s |\n"%(r["id"],r["title"],r["source"],r["component"],r["test"],r["phase"]))
