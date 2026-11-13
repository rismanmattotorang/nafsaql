# NAFS-AQL: DOCUMENT PARSING AND SEARCH PROMPT (v1)

Revised on 2026-11-12 from the owner's request so that it matches Nafs-Aql as it is built today (release 0.1.0, commit
`8e4ef7c`). The project-level name of this work is **Phase 28**.

The owner's request, as written:

> Assess the document parser, semantic search and parsing methods, techniques, and algorithms inside liteparse and
> semtools projects in third_party folder. Do research to identify gaps between Nafs-Aql to both projects. The goal is
> to make better implementations in Nafs-Aql and make it a new state of the art. [...] reuse (or cherrypick) some
> components or rewrite some methods [...] to improve performance and accuracy of document parser, semantic search and
> parsing in Nafs-Aql knowledge management module - both for vector and graph stores. [...] sweep test again, update
> technical paper of Nafs-Aql, revise the README and push and commit to main repo at Gitlab.

What changed from the original request, and why:

| Original request | Revised | Reason |
|---|---|---|
| Assess liteparse and semtools | **LiteParse** (`third_party/liteparse`, run-llama, commit `0548949`) for PDF layout analysis: glyph-to-line projection, reading order, headings, lists, tables, running headers, maths; **semtools** (`third_party/semtools`, run-llama, commit `fe7f6d2`) for search: static embeddings, line windows, distance cut-off, workspace re-indexing, the agent loop | These are the two places the code holds the ideas; their READMEs and evaluation folders hold the claims |
| Reuse or cherry-pick components | **Port algorithms, not dependencies.** LiteParse sits on PDFium (a C++ binary download) and semtools on model2vec weights and LanceDB; Nafs-Aql's knowledge module is pure Rust on four Intel cores. Code is rewritten against what `pdf_oxide` exposes (positioned spans with font size and weight); no new native library | Release 0.1.0 builds from `cargo build` alone, and downloads of binaries or model weights need the owner's approval |
| Improve parsing accuracy and speed | Measured with a **parsing score that needs no download**: the same papers read from PDF and from their TeX source (`examples/parse_agreement`), extended with headings found, table rows kept, running lines left and reading order (word-pair order agreement); plus parse time per page | No public parsing benchmark is on disk; our papers have both readings |
| Improve semantic search (vector and graph) | Measured with the retrieval sets that already exist (`scripts/kb_p25_{quoted,para,multihop,offtopic}.json`, the typo and follow-up sets) through `examples/kb_eval`, for **both** the vector arm and the graph arm, before and after, interleaved | Phases 23 to 27 showed that most borrowed mechanisms do not transfer; a change is adopted only when it wins on our data |
| New state of the art | **No claim without a number.** Each property is claimed only where measured on the reference machine; what was tried and lost is recorded with its number | The project's standing rule |
| Sweep test again | Rust tests, `cargo fmt --check`, `cargo clippy -D warnings` (CI runs clippy 1.99; local is older), then the live sweep (`scripts/live_sweep.sh`) on a fresh release build; timings labelled SHARED MACHINE | As for every phase |
| Update the paper and README | The knowledge section of `docs/report/` and its numbers; README feature list and numbers in step with `docs/roadmap/STATUS.md`. The paper stays topic-organised: no phase log, no requirement ids | The paper is the final-draft paper since Phase 26 |
| Push to the main repo at GitLab | The project's remote is **GitHub** (`github.com/rismanmattotorang/nafsaql`). Branch `phase28-parse-search`, pull request, CI green, rebase-merge into `main`. Never force-push | There is no GitLab remote; `main` is protected by CI by convention |

---

## 0. How you work

Same rules as `docs/prompts/COMPLETION_PROMPT.md` section 0: verify before you claim; label shared-machine timings; never
kill the owner's processes; ask before downloading model weights or binaries; plain language in the interface; Rust at
run time; one change at a time, measured against what it replaces.

## 1. Assess

1. Read LiteParse for: how glyphs become words, lines and blocks; column detection and reading order; heading
   detection; list and inline-style handling; running header and footer removal; table detection from geometry;
   figure and maths handling; its structured (JSON) output; its evaluation metrics.
2. Read semtools for: its embedding model and how text is cut for it; distance cut-off; context windows around a hit;
   workspace store and incremental re-indexing; the parse cache; the agent's search loop and tools; its benchmark.
3. Write `docs/research/liteparse_semtools_gaps.md`: what each has that Nafs-Aql lacks, what Nafs-Aql already has
   that they lack, and for each gap whether it runs in pure Rust on four cores with no further model or binary.

## 2. Build and measure

For each gap that passes the test in 1.3, in this order:

- **PDF layout**: a span-level layout pass over `pdf_oxide`'s positioned text: lines from glyph baselines, columns and
  reading order, headings from font-size ranks and numbering, running lines by position and repetition, tables kept
  as Markdown rows. Score with the extended parse agreement and the retrieval sets (better passages are the point).
- **Chunks for search**: structure-aware passages (a table or list is not cut in the middle; a heading is never the
  last line of a passage), a context window around a hit when the prompt is built, and a document-level unit for
  the graph.
- **Vector arm**: what semtools does cheaply that our 0.6B embedder does not (distance cut-off, small-window vectors,
  merging adjacent hits), measured on the paraphrase set, where we are weakest.
- **Graph arm**: better terms and links from the improved structure (headings, table headers, defined terms),
  measured on the multi-hop set.
- **Re-indexing**: a parse cache keyed by content hash and parser version, so a re-read of an unchanged file does not
  parse or embed again.

Record every mechanism that was measured and not adopted, with its number.

## 3. Validate

Rust tests, `cargo fmt --check`, `cargo clippy --all-targets -D warnings`; the live sweep on a fresh release build;
update `docs/roadmap/STATUS.md`, a decision record, `ASSUMPTIONS.md` and the user guide where behaviour changed.

## 4. The paper and the README

Update the knowledge section, the evaluation tables and the related-work paragraph of `docs/report/`; rebuild the PDF
with no undefined references. Revise the README's knowledge features and numbers.

## 5. Publish

Commit, push the branch, open the pull request, wait for CI, rebase-merge into `main`.
