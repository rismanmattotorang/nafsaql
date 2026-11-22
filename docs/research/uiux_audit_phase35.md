# UI/UX audit and scorecard (Phase 35)

Audit of the app as it was at commit `4524c8f` (2026-11-22), by reading the code; `A/` is `rust/crates/nafs-web/assets/`.
Each area ends with its checks. A check is **pass** only when a test or a screenshot shows it; the "before" column is
the audit, the "after" column is filled when the work is done (section 8).

## 1. What the app is made of

- `A/app.js` 2,351 lines (vanilla ES module, hash router `route()` :495), `A/app.css` 839, `A/render.js` 735 (pure
  string-to-string: own Markdown, own TeX-to-MathML, own Mermaid subset), `A/nivo.js` (Nivo 0.99 + React, 218 KB
  gzip, loaded on first chart), `A/index.html` 51.
- Pages: Chat, Rooms, Knowledge, Files; with Advanced on: Quality, Learning, Experiments, Speed, Status. Settings,
  Memory, shortcuts, Why?, Compare are modals. Command palette (Ctrl/Cmd+K) navigates only.
- Checks that exist: `node scripts/render_test.mjs` (197), copy lint (`cargo test -p nafs-web`), `e2e_sweep.py
  --ui-only` (plain language, no JS errors, 375 px; not Knowledge or Files), `capture_screenshots.py`.

## 2. Forms

Found:
1. **Knowledge base rename is broken by markup**: `renameBase` (app.js:1910) injects `<form id="kbRen">` into
   `#modalContent`, which is already inside `<form method="dialog">` (index.html:47); the parser drops the nested form.
2. **Enter in a modal input closes the dialog** and loses the draft (Files `editor()` `#fsName`, Settings `#stTok`,
   copy-link input): implicit submission of the dialog form.
3. Segmented controls without state: `modeSeg`, `rmode`, `vTabs`, `fsHow`, `fsLv`, `stTheme`, `vpTabs` have no
   `aria-pressed`; only `kbSearchMode`, `gSeg` and the graph legend do.
4. Verify panel fields (`#vRef`, `#vTests`, `#vSchema`) have placeholders, no labels; bad JSON is reported only at
   send, as a toast (app.js:1092), and `updateChip` swallows it (:1020).
5. Three ways to show an error (inline `.rerr`, toast, page notice); creation is inline for bases and rooms, a modal
   for files; naming rules differ.
6. Chat has no length counter (Rooms has); Experiments' `#eSweepB`/`#eLamB` have a title only.
7. Files `editor()` has no unsaved-changes guard.
8. Undo exists for chat delete and memory changes only; base, document, room and file deletes confirm; a single
   memory note is forgotten with neither.
9. Backend switch and the escalation budget apply for everyone with no confirmation (app.js:2323, `#gBudget`).

## 3. Answers in chat

Supported: headings, lists, task lists, callouts, bold/italic/strike/mark, links, citations `[n]` with a popover;
tables (alignment, numeric right-align, sticky header, Copy as TSV on hover); maths inline and display with Copy
LaTeX; code with label, copy, collapse above 28 lines, 14 highlighted languages; diagrams: flowchart, sequence, pie,
state (own layout) with Source, Copy source, Save SVG, Save PNG, Expand; charts: 11 kinds with Source, Save CSV, SVG,
PNG, Expand, "Show the data".

Missing:
1. Tables: no sort, no CSV download, copy button invisible to the keyboard (opacity 0 until hover, app.css:728), no
   expand for wide tables.
2. Code: no wrap toggle, no line numbers, no download; collapse threshold disagrees with the docs (28 against 24).
3. Diagrams: class, ER, gantt, mindmap, timeline, journey stay code; Expand is not a dialog (no backdrop, focus trap,
   zoom).
4. Answer: no edit and resend; Regenerate appends a duplicate turn; Copy copies raw Markdown only; no export of an
   answer or a conversation other than JSON.
5. Markdown: no footnotes, no images, no definition lists.
6. Streaming: `#msgs role=log` plus replacing `.answer` every 150 ms gives a screen reader repeated or no speech; no
   "answer ready" announcement.
7. Docs disagree with the code (user guide lists three diagram kinds and no charts).

## 4. Documents in and out

| Format | Read into a base | Written | Note |
|---|---|---|---|
| PDF | yes (pdf_oxide + own layout pass, pages, headings, table rows) | no | scanned PDFs refused (no OCR) |
| Word .docx | yes (own reader: headings, tables as `a \| b` rows) | no | lists, bold, footnotes lost |
| PowerPoint .pptx | yes (slide numbers, notes) | no | tables flattened |
| Excel .xlsx/.xls, .csv, .tsv | **no** | chart data as CSV only | spec had spreadsheets out of scope |
| LaTeX .tex | yes (own stripper) | no | macro names leak inside maths |
| Markdown, HTML, text | yes | Markdown from compile, rooms, eval | chat downloads as JSON only |
| RTF, ODT, EPUB, JSON, ipynb, source code, images | no | no | |

- No file can be attached to one question; documents enter only through Knowledge.
- No server-side document writer. `rust/Cargo.lock` already holds `office_oxide 0.1.16` (through `pdf_oxide`): it
  reads DOCX/XLSX/PPTX and the old binary formats to Markdown and writes DOCX/XLSX/PPTX from Markdown; `pdf_oxide`
  writes PDF from Markdown. Their output quality is untested here.
- Smiley (the owner's desktop app): committed code reads text-like files and images and exports a chat as Markdown. A
  full reader/writer (`smiley/src-tauri/src/docs.rs`: DOCX, PPTX, XLSX, PDF, ODF, RTF in; DOCX, PPTX, XLSX, LaTeX,
  Markdown out) is untracked work in progress that does not build yet. Nafs-Aql is ahead on PDF layout, search and
  graph; Smiley's draft is ahead on spreadsheets and on writing files.
- Holes in tests: no live upload of DOCX, PPTX or TeX; `nafs-gateway/src/fs.rs` has no tests; nothing exercises
  `/v1/fs/*` or `/v1/compile`.

## 5. Knowledge base, graph, retrieval

- Upload: only the drop card accepts files; the page filter omits `.xhtml`/`.text` the server accepts; no folder
  upload; no bulk actions.
- Search: k fixed at 8; no document filter; no "more".
- Graph (`drawGraph` app.js:1745): own force layout on the main thread; at most 48 nodes; no save as image; no filter
  by kind; no edge labels; Fit resets the view instead of fitting; no keyboard pan or zoom.
- The graph build's counters (ok, skipped, merged, dropped quotes) are traced, never stored or shown; a passage whose
  reply fails the schema is skipped silently.
- `chunking` status is declared and never set.
- Rooms cannot use a knowledge base.

## 6. Settings and configuration

- Settings live in the browser only; `lambda` is always sent and overrides the server's default; no reset, no value
  shown beside the slider, three options without a hint, no search, no deep link.
- **All three shipped config files fail the generated schema**: `[context]`, `[shadow]`, `knowledge.piece_chars` and
  `piece_mix` are missing from it (last generated 2026-11-11), and CI only checks that the schema is valid JSON.
- Unknown or misspelt config keys are ignored at load; several enumerated options are free strings (`gate.mode`,
  `on_unknown`, `repair_policy`, `judge.backend`, `knowledge.fusion`, `harness.model`).
- Gate mode and backend switch are lost on restart; the dialog does not say so.
- `harness.constrain` and the four-candidate cascade are silently inactive on the default backend (no grammar, one
  sample per request).

## 7. Harness as the person sees it

- The code agent (`POST /v1/harness/code`) has **no page**: one blocking request of up to 900 s, no progress, no
  cancel, runs not stored, every failure returned as 400.
- In chat a multi-round answer shows a journey bar and verification cards only in Advanced; a rejected candidate's
  text is cleared.
- Parsing without a grammar: a fence closes at the first ``` anywhere (agent.rs:443); the last python block wins
  (:362); `python3` is not recognised as a language tag (:422); a file over 6,000 bytes is shown cut in the middle and
  a whole-file reply written from that view loses the middle.
- The verifier sees only `solution.py`.

## 8. Scorecard

Filled in at the end of Phase 35 (`before` from this audit, `after` from the tests and screenshots named).

| Area | Check | Before | After | Evidence |
|---|---|---|---|---|
| Forms | every field has a visible or programmatic label | fail | pass | the UI sweep reads every control of every page for a name (`scripts/e2e_sweep.py`) |
| Forms | no dialog loses a draft on Enter; rename works | fail | pass | dialog is a plain container; rename tried in the browser |
| Forms | segmented controls expose their state | fail | pass | `segSync` sets `aria-pressed`; checked in the UI sweep |
| Forms | invalid input is reported beside the field before sending | fail | pass on the Code form and the check panel; other forms report on sending | UI sweep, Code form check |
| Forms | destructive actions confirm or undo, everywhere | fail | pass | every delete call in `app.js` confirms or offers Undo (the single memory note gained Undo in the review round) |
| Answers | tables: sort, copy, CSV, keyboard-reachable | fail | pass | `scripts/render_test.mjs`, browser |
| Answers | code: copy, wrap, line numbers, download | fail | pass | `scripts/render_test.mjs`, browser |
| Answers | every diagram kind the prompts name renders; expand is a dialog | fail | pass for the four kinds the prompts name (flowchart, sequence, state, pie); other kinds show their source | render tests; decision 027 "Not done" |
| Answers | charts: save image, data as table and CSV | pass | pass | unchanged |
| Answers | maths: inline, display, copy, fallback | pass | pass | unchanged |
| Answers | edit and resend; regenerate replaces | fail | pass | store test `a_turn_replaces_the_tail_in_one_step…`; browser; a failed request loses nothing |
| Answers | a screen reader hears the answer once | fail | pass by inspection of the live region; not tried with a screen reader | `chatSay` |
| Documents | Word, PowerPoint, PDF, LaTeX, Markdown, HTML read | pass | pass | Word keeps lists and tables now |
| Documents | Excel and CSV read | fail | pass | `nafs-kb` parse tests; knowledge sweep |
| Documents | a document can be attached to one question | fail | pass | `/v1/parse`; browser (a CSV, the right number in the answer) |
| Documents | answers export to Markdown, HTML, Word, PowerPoint, Excel, LaTeX, PDF | fail | pass, nine formats with round-trip tests | `nafs-kb` export tests; knowledge sweep |
| Knowledge | drop anywhere; per-document progress and retry | partly | partly: drop anywhere done; progress is per document, retry is "read again" on the row | browser |
| Knowledge | graph: save image, filter, fit, keyboard | fail | pass | browser |
| Knowledge | every knowledge route in the live sweep | fail | pass for the routes the page uses, documents in and out included | `scripts/kb_sweep.py` |
| Settings | explanation, default and reset for every setting | fail | pass | UI sweep (Settings rows, presets, find, put back) |
| Settings | config files validate against the schema in CI | fail | pass | `.github/workflows/ci.yml` runs `scripts/gen_config_schema.py` |
| Settings | unknown config keys are reported | fail | pass | config tests; `/v1/system` `config_warnings` |
| Harness | an agent run can be watched, stopped and inspected in the app | fail | pass | Code page; `/v1/harness/runs`; browser run |
