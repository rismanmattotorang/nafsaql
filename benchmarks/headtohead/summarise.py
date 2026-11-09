#!/usr/bin/env python3
"""Median over the rounds of each Go benchmark and each Rust counterpart, side by side."""
import glob, json, os, re, statistics as st
here = os.path.dirname(os.path.abspath(__file__))
go, rs = {}, {}
for f in sorted(glob.glob(os.path.join(here, "round_*_go.txt"))):
    for line in open(f):
        m = re.match(r"(Benchmark\S+?)-\d+\s+\d+\s+([\d.]+) ns/op", line)
        if m:
            go.setdefault(m.group(1), []).append(float(m.group(2)))
for f in sorted(glob.glob(os.path.join(here, "round_*_rust.json"))):
    for k, v in json.load(open(f)).items():
        rs.setdefault(k, []).append(v["ns_per_op"])
med = lambda d, k: st.median(d[k]) if k in d else None
pairs = [
    ("routing signals, 2 KB prompt (their 30-rule BM25 / ours: domain classifier)", "BenchmarkBM25ThirtyRulesLongEnglish", "domain classifier, 2 KB prompt"),
    ("   (their n-gram variant)", "BenchmarkNgramThirtyRulesLongEnglish", "domain classifier, 2 KB prompt"),
    ("extractive compression, 500 tokens", "BenchmarkCompress500Tokens", "sentence packer, 500 tokens to 256"),
    ("extractive compression, 2K tokens", "BenchmarkCompress2KTokens", "sentence packer, 2K tokens to 512"),
    ("extractive compression, 16K tokens", "BenchmarkCompress16KTokens", "sentence packer, 16K tokens to 512"),
    ("dot product n=384, SIMD", "BenchmarkDot/n=384", "dot, n=384 (AVX2 + FMA)"),
    ("dot product n=768, SIMD", "BenchmarkDot/n=768", "dot, n=768 (AVX2 + FMA)"),
    ("dot product n=1024, SIMD", "BenchmarkDot/n=1024", "dot, n=1024 (AVX2 + FMA)"),
    ("dot product n=384, portable", "BenchmarkDot/portable/n=384", "dot, n=384 (portable)"),
    ("dot product n=768, portable", "BenchmarkDot/portable/n=768", "dot, n=768 (portable)"),
    ("dot product n=1024, portable", "BenchmarkDot/portable/n=1024", "dot, n=1024 (portable)"),
]
fmt = lambda ns: "-" if ns is None else (f"{ns/1e6:.2f} ms" if ns >= 1e6 else f"{ns/1e3:.1f} us" if ns >= 1e3 else f"{ns:.0f} ns")
print(f"{'workload':74} {'Semantic Router (Go)':>22} {'Nafs-Aql (Rust)':>16} {'ratio':>8}")
rows = []
for label, g, r in pairs:
    a, b = med(go, g), med(rs, r)
    ratio = (a / b) if a and b else None
    rows.append({"workload": label, "go_ns": a, "rust_ns": b, "go_over_rust": ratio})
    print(f"{label:74} {fmt(a):>22} {fmt(b):>16} {('%.1fx' % ratio) if ratio else '-':>8}")
print()
for k in ("gate decision (fixed rule)", "gate decision (calibrated + certified bins)", "harness host overhead, 100 turns / 800 reads"):
    print(f"{k:74} {'(none upstream)':>22} {fmt(med(rs, k)):>16}")
json.dump({"rounds": len(glob.glob(os.path.join(here, 'round_*_go.txt'))), "pairs": rows,
           "rust_only": {k: med(rs, k) for k in ("gate decision (fixed rule)", "gate decision (calibrated + certified bins)", "harness host overhead, 100 turns / 800 reads")}},
          open(os.path.join(here, "summary.json"), "w"), indent=1)
