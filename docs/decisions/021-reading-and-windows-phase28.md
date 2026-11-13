# Decision 021: Read PDF pages by their layout; give passages sentence-window vectors (Phase 28)

Date 2026-11-13. Status: **adopted.** Research [`liteparse_semtools_gaps.md`](../research/liteparse_semtools_gaps.md).
Prompt [`PARSE_SEARCH_PROMPT.md`](../prompts/PARSE_SEARCH_PROMPT.md). Measurements: `benchmarks/kb_phase28/` (SHARED
MACHINE; loads stated there).

## Decisions

1. **PDF pages are read by a layout pass over positioned text** (`nafs_kb::layout`), a Rust rewrite of LiteParse's
   rules over `pdf_oxide` spans: recursive XY-cut with column gutters from the character density of narrow lines and
   full-width lines cut off as bands; tables kept out of the column cut; lines by half the smaller height; running
   headers and page numbers by position and repetition; paragraphs rejoined across hyphens, columns and footnotes;
   headings from the body size, larger sizes, numbering ("I.", "A.", "2.5") and bold; list items; table-of-contents
   entries; table rows as Markdown rows. A failure of the pass falls back to `pdf_oxide`'s converter
   (`NAFS_KB_PDF_LAYOUT=converter` selects it for comparisons). Section titles found 11 → 166 of 173; 15.5 against
   24.0 ms a page; the answering passage first for 0.148 against 0.099 of reworded questions and 0.375 against 0.284 of
   two-passage questions.
2. **Passages get vectors for their sentence windows** (semtools' small search unit; `[knowledge] piece_chars = 240`,
   `piece_mix = 0.5`): a passage's meaning score is the mean of its own similarity and the larger of that and its best
   window's. Quoted MRR 0.870 → 0.922, reworded 0.194 → 0.201, two-passage 0.570 → 0.571; no set worse. Cost: about
   one more embedding pass per document (1,402 windows for 723 passages); a base indexed before gets its windows on
   "Rebuild meaning index".
3. **A document's own title is no longer repeated at the head of every heading path**: it is already the passage's
   title, and repeating it cost tokens in every vector's text and every prompt header.
4. **Fusion stays reciprocal rank with the meaning arm at 0.1.** At 0.3 or 0.5 reworded questions gain and quoted
   questions lose, as in Phase 25.
5. **Not ported:** PDFium (a native binary), OCR, LlamaParse (a cloud API), model2vec static embeddings (a download;
   semtools calls its own search fuzzy word search, which BM25F and typo correction already cover), Qdrant (the exact
   scan is 0.2 ms here), the ReAct search loop (seconds to tens of seconds a turn on four cores), LaTeX maths
   reconstruction.
6. **No "state of the art" claim.** No public parsing or retrieval benchmark was run (downloads need the owner's
   approval). What is claimed is each measured change against the setting it replaced.

## Validation

513 Rust tests pass; `cargo fmt --check` and `cargo clippy -D warnings` (all targets) are clean. Live sweep on the
final build with sentence windows on, 2026-11-13 06:44 to 07:11 (load 4 to 17): contract passed, security 26/26,
Rooms 10/10, end to end 69/69, knowledge 55 passed (1 skipped), knowledge security 33/33, harness 53/53, acceptance
9/9 (first token after 248 ms; a verified reasoning answer in 155 s against a 240 s target).
