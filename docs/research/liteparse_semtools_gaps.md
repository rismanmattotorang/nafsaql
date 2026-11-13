# LiteParse and semtools against Nafs-Aql's knowledge module (Phase 28)

Read on 2026-11-12. LiteParse: `third_party/liteparse` (run-llama, commit `0548949`), Rust over PDFium. semtools:
`third_party/semtools` (run-llama, v3.0.0, commit `fe7f6d2`), Rust CLI over model2vec and Qdrant Edge. Paths below are
relative to each repository. The prompt is `docs/prompts/PARSE_SEARCH_PROMPT.md`; results are in `benchmarks/kb_phase28/`.

## 1. LiteParse: how it reads a PDF

LiteParse's quality comes from geometry rules with hand-tuned constants over positioned glyphs. PDFium is only the
glyph source, so the rules port to any reader that gives boxes, font sizes and font names. `pdf_oxide` 0.3.78, which
Nafs-Aql already uses, exposes `extract_spans` and `extract_chars` (box, font name, size, weight, italic, monospace),
`extract_lines`/`extract_rects`/`extract_paths`, the outline and the structure tree.

| Mechanism | Where | What it does |
|---|---|---|
| Glyph segments | `extract.rs:1593-2090` | one pass in content order; a new item on a line change (2 pt), a gap of 15 pt or more, a dot leader, a sub/superscript step; missing-space recovery between two alphanumerics from a per-font median of real space gaps (0.35 em fallback) |
| Reading order: recursive XY-cut | `projection.rs:3280-4910` | 2 pt density histogram weighted by character count; a valley is <= 0.10 x median density, >= 8 pt wide for a column cut, >= 1.4 line heights for a band cut; full-width banners cut first; a vertical cut must beat a horizontal one by 1.1x; fallback column-start histogram with fill and balance gates that also marks tabular regions; stop at depth 12 or 4 items |
| Lines in a region | `projection.rs:4909-5050` | an item joins a line when the y difference is below half the *smaller* height (heights capped at 24 pt) |
| Body size and headings | `markdown_layout/headings.rs:578-735, 365-575` | body size = the largest size holding >= 50 % of the top character weight; larger sizes (> body + 0.5 pt) with >= 10 characters, 8-200 per line and >= 50 % letters map to H1..H6; then a bold body-size fallback (<= 80 chars, no final full stop, not a caption, all spans bold); numbering prefixes (`1.5.2`, `A.2`, `§4`) |
| Paragraph joins | `markdown_layout/paragraphs.rs:66-300` | same size, same boldness, gap <= 1.5 line heights; across regions only when the first line ends without punctuation and the next starts in lower case; end-of-line hyphen before a lower-case letter dropped |
| Running headers/footers | `markdown_layout/repetition.rs:35-330` | lines in the top or bottom 12 % with digits replaced by `#`, on >= 50 % and >= 2 pages; a single-page pattern list (URL, DOI, "Page N of M", a lone number, ©) |
| Tables | `markdown_layout/tables.rs` (8k lines) | borderless: cells split at gaps > 1 em, >= 3 columns, tracks matched within 6 pt, row spacing CV <= 0.5, 70 % of the text kept; ruled: union-find over stroke segments; tables are obstacles for XY-cut |
| Lists, code, maths, figures | `lists.rs`, `classify.rs:1239`, `tex_math/`, `figure_cluster.rs` | confirmed ordered runs; all-monospace lines as code; LaTeX rebuilt from TeX math fonts; figure clusters from vector paths |
| Garbled-layer detectors | `ocr_merge.rs:1855-2300` | vowel ratio < 20 %, symbol and control-character intrusions, one misread glyph in many words |
| Output | `output/json.rs`, `layout.rs:60-120` | per page: text items with boxes, page label; blocks (heading, paragraph, list item, table, code, figure) with level and box |
| Evaluation | `dataset_eval_utils/`, `run_benchmarks.sh` | LLM question answering over page images; external NID (normalised edit distance), TEDS (tables), MHS (heading hierarchy); README claims 2-5 ms/page and 0.886 on opendataloader-bench without OCR |

## 2. semtools: how it searches

| Mechanism | Where | What it does |
|---|---|---|
| Embeddings | `src/search/mod.rs:16,55-69` | static model2vec `potion-multilingual-128M` (256 dimensions, no transformer pass), **one vector per line**, loaded on every call |
| Distance and window | `src/search/mod.rs:84-119` | brute-force cosine (simsimd); `--n-lines` context around the hit; `--max-distance` returns every hit under the cut-off; adjacent hits are not merged |
| Workspace | `src/workspace/store.rs` | Qdrant Edge (HNSW), ids from FNV hashes of path and line; a file is changed when size, mtime or the embedding version differs; old lines of a shrunken file are not deleted (a bug) |
| Parse | `src/parse/*` | LlamaParse cloud API, ten jobs at once, retries with back-off; cache keyed by file *basename* (a collision bug) |
| Ask | `src/ask/*` | a ReAct loop (20 turns by default) over `search`, `grep` and `read` tools with line-cited chunks |
| Benchmark | `benchmarks/arxiv` | 15 questions over 1,000 arXiv texts with and without the search tool, transcripts only, no score |

The project calls its search "fuzzy grep"; the tools ask the agent for keywords. It is not built for reworded questions,
and its own transcripts show the short-line bias of line vectors (a near-zero distance to a short line).

## 3. Gaps

What Nafs-Aql lacked, measured on its own documents before any change (`examples/parse_score`, the PDF reading against
the TeX source of the same document, 5 documents):

| Gap | Before | Source of the idea | Pure Rust, 4 cores, no new model? |
|---|---|---|---|
| Two-column pages read across the columns | five-word runs of the source found in the reading: 0.78 on the two-column report | LiteParse XY-cut | yes |
| Section titles not marked as headings (so passages carry no heading path) | 2 of 71 report titles; 11 of 173 over all documents | LiteParse body size + heading map + numbering | yes |
| Running headers and page numbers inside sentences | part of the 0.51 share of reading runs not in the source | LiteParse repetition by position | yes |
| Hyphens left at column line ends ("assump-tions") | visible on every page of the report | LiteParse paragraph joins | yes |
| Tables read as prose | not scored | LiteParse borderless tables | yes (subset) |
| Passage vectors dilute a one-sentence fact (reworded questions R@1 0.12) | Phase 25 | semtools' small unit + context window | yes, more embedding time |
| Re-read cost on an unchanged file | parse and chunk again, vectors reused by text | semtools' change detection | already: the upload is keyed by SHA-256 and vectors by text |

What Nafs-Aql has that neither has: words and meaning fused (semtools has no word index), a model-free term graph
with Personalized PageRank, an LLM-built entity graph with communities, typo correction, follow-up rewriting,
sentence-level support checks of the answer, hostile-file limits on every reader, an embedder identity guard. semtools'
stale-row and cache-collision bugs do not exist here (a re-read replaces a document's passages; the store keys files by
content hash).

Not ported, and why: PDFium (a native binary download), OCR, LlamaParse (a paid cloud API), model2vec (a download;
semtools itself calls it fuzzy word search, which our BM25F and typo correction already cover), Qdrant (exact scan is
0.2 ms at our scale), the ReAct loop as a default (seconds to tens of seconds per turn on four cores), `tex_math`.

## 4. What was built and what it measured

Details in `benchmarks/kb_phase28/README.md` and decision 021.

| Built | From | Result | Adopted |
|---|---|---|---|
| `nafs_kb::layout`: XY-cut reading order, table regions, running lines, paragraph and hyphen joins, headings, lists, table rows | LiteParse | titles found 11 → 166 of 173; 15.5 vs 24.0 ms/page; hybrid R@1 reworded 0.099 → 0.148, two-passage 0.284 → 0.375, quoted 0.768 → 0.786 | yes |
| `examples/parse_score`: the PDF reading scored against its TeX source (order, junk, headings) | LiteParse's NID/MHS metrics, simplified | the reader's own regression measure | tool |
| Sentence-window vectors scoring their passage (`chunk::pieces`, `Base::similarities_mixed`) | semtools' line vectors and context window | quoted MRR 0.870 → 0.922, reworded 0.194 → 0.201, two-passage 0.570 → 0.571 | yes (m = 0.5) |
| Title dropped from heading paths | prompt-header cost (Phase 27) | shorter vector texts and headers | yes |
| Meaning weight 0.3 / 0.5 | semtools uses meaning alone | reworded up, quoted down | no |
| Term-graph walk on the better-structured base | — | within 0.001 MRR | stays off |
