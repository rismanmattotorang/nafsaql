# Phase 28: document reading and search (LiteParse and semtools ideas)

Reference machine (Intel i3-8100, 4 cores), shared with other workloads: load 30 to 300 while the bases were embedded,
3 to 9 during the timing runs. Prompt: `docs/prompts/PARSE_SEARCH_PROMPT.md`. Gap analysis:
`docs/research/liteparse_semtools_gaps.md`. Decision 021.

## Files

| File | What | Tool |
|---|---|---|
| `parse_converter.json`, `parse_spans.json` | the PDF reading of 5 project documents scored against their TeX source (order of five-word runs, runs not in the source, section titles marked as headings) | `examples/parse_score` |
| `papers_{converter,spans}_{1,2}.json` | parse and chunk of the 25 PDFs in `papers/`, two interleaved rounds per reader: time, headings, passages, passages without a heading path, table rows, titles | `examples/ingest_bench` |
| `readers.json` | where the answering passage ranks, old reader (base `ylgyxxqms221gnh8ixit`, 847 passages) against new (base `pir9ftsm6eja267j4fg0`, 723 passages), same 9 files, questions answerable in both | `examples/piece_eval` |
| `pieces.json` | the new base with sentence windows of about 240 characters (1,402 windows), the meaning score mixed at m = 0 to 1, fused at weights 0.1 to 1.0 | `examples/piece_eval --pieces 240` |

Both bases were built with `examples/kb_ingest` (the upload path: parse, chunk, store, embed), the old one with
`NAFS_KB_PDF_LAYOUT=converter`.

## Results

**Reading (5 documents with TeX sources).** Section titles marked as headings 11 of 173 → 166 of 173 (mean per
document 0.21 → 0.95); five-word runs of the source found in order 0.692 → 0.735; runs of the reading not in the source
0.507 → 0.470. The two-column paper: order 0.781 → 0.810, headings 2 of 71 → 70 of 71.

**25 papers (421 pages).** 24.0 → 15.5 ms a page; headings 195 → 833; passages 3,873 → 3,367; passages with no heading
path 32 → 19; titles found 21 → 25 of 25; no failures.

**Retrieval, the service's hybrid search (meaning weighted 0.1).**

| Set | Old reader R@1 / MRR | New reader R@1 / MRR | New reader + windows (m = 0.5) R@1 / MRR |
|---|---|---|---|
| Quoted (56) | 0.768 / 0.860 | 0.786 / 0.870 | 0.875 / 0.922 |
| Reworded (142) | 0.099 / 0.152 | 0.148 / 0.194 | 0.162 / 0.201 |
| Two-passage (44) | 0.284 / 0.474 | 0.375 / 0.570 | 0.386 / 0.571 |

The meaning arm alone, MRR: quoted 0.383 → 0.465 → 0.611; reworded 0.089 → 0.107 → 0.120; two-passage 0.154 → 0.281
→ 0.309. The term-graph walk added nothing on either base (within 0.001 MRR), as in Phase 25.

**Not adopted.** A heavier meaning weight (0.3, 0.5): better on reworded questions (MRR up to 0.228 at 0.5 with
windows) and worse on quoted ones (0.857 at 0.5); windows counted fully (m = 1): reworded MRR 0.182, below m = 0.5.
