# NAFS-AQL: PRODUCT QUALITY PROMPT (Phases 34 to 37)

The owner's request of 2026-11-22, revised into a form that can be executed and checked. The original asked for:
finishing the previous session, a production-quality UI/UX pass in five areas, harness research and improvement, new
screenshots, a world-class README, sweeps, review, refactoring, a version bump, a project website better than
vllm-sr.ai with business use cases, a push to GitHub, and a clean-up of `third_party`. What the revision adds: an order
of work, what "done" means for each part, what must be measured, and what must not be claimed.

## 0. How you work

- **Checkpoint.** Keep `docs/status/PROGRESS.md` current: what is done (with evidence and commit hashes), what is
  next. After a context reset, read it first.
- **Verify before you claim.** A feature is done when a test or a screenshot of the running app shows it. A speed-up is
  real when an interleaved comparison stored under `benchmarks/` shows it. A negative result is a result: record it.
- **The machine is shared** (four cores, other sessions and a browser). Run timing comparisons interleaved, in short
  paired rounds, and report the load. Never stop another person's process.
- **Subagents** are for read-only research and independent review. Never for benchmarks: they would compete for the
  same cores.
- **No new heavy dependencies.** The app builds with `cargo build` alone and serves its UI as static files with no
  bundler step at run time. A library is vendored only when writing the feature by hand would be worse, and its licence
  is recorded in `NOTICE`.
- **Plain language in the UI and the docs.** No jargon a first-time user would not know; no claim without a number
  behind it; no number without its source.
- **Privacy is a product feature.** Nothing leaves the machine: no CDN, no remote font, no telemetry.

## 1. Mission

Make Nafs-Aql feel like a finished product to someone who opens it for the first time, without removing what an
expert needs, and make the project legible to someone who lands on its repository or website.

**Definition of done**
1. Phase 34 is closed: tested, measured, reviewed, documented, pushed.
2. Each of the five UI/UX areas in section 3 meets its acceptance list, shown by an automated check or a screenshot.
3. The harness changes that research suggested are each measured against the harness without them.
4. All Rust tests, `cargo fmt --check`, `cargo clippy -D warnings`, the UI checks and every live sweep pass on the
   final build.
5. The paper, README, user guide, features list and status page describe the product as it is.
6. A static website is in the repository and builds on GitHub Pages.
7. The version is raised, the work is on `origin/main`, and `third_party` holds only what the project runs.

## 2. Phase 34 close-out (engine)

Run the six new engine tests and the workspace tests. Measure what is still unmeasured: claimed blocks on the 4B
model, read-ahead on a cold model file (`scripts/cold_start.py`), and the answering model's between-turns reading
(`scripts/turn_reuse.py`, full stack). Have the engine changes reviewed by a reader who has not seen the reasoning.
Fix what the review finds. Write the results into the research note, decision 026, the assumptions, the paper and the
README. A feature that does not pay is turned off by default and reported as such.

## 3. Phase 35: UI/UX to production quality

Start from an audit of the app as it is (pages, forms, rendering, knowledge, settings, accessibility) and a scorecard:
for each area, a list of checks that a production app passes, each marked pass or fail before and after. The
scorecard is the measure of "maximum score"; it is stored in `docs/research/` and its checks are automated where they
can be.

### 3.1 Forms
Every form in the app (composer, new knowledge base, add documents, rooms, memory, settings, lab):
- one visible label per field, a hint where the field is not obvious, the constraint stated before the error;
- validation as the person types or on leaving the field, the message beside the field, in words that say how to
  fix it; the submit control says what it will do and is disabled only with a stated reason;
- sensible defaults, so the shortest path is one field and one button; everything else behind "More options";
- keyboard: Enter submits, Escape cancels, Tab order follows the screen; focus returns to where it was;
- loading, empty, error and success states for every action; no silent failure; destructive actions confirm or undo;
- one implementation of field, validation and dialog used everywhere.

### 3.2 Answers in chat
Tables, maths, code, diagrams and charts are already rendered. Close what is missing against current practice:
- **Tables:** sort by a column, copy as Markdown, download as CSV, a sticky header and horizontal scroll for wide
  tables, numbers right-aligned.
- **Code:** language label, copy, download with the right extension, wrap toggle, line numbers for long blocks.
- **Maths:** inline and display, copy LaTeX, a readable fallback when a formula does not parse.
- **Diagrams:** every diagram type the prompts ask the models for renders; save as SVG and PNG; expand; the source
  can be shown and copied; a broken diagram shows its source and the reason, never a blank.
- **Charts:** the chart types the models are asked for; save as SVG and PNG; the data behind a chart can be shown as
  a table and downloaded as CSV.
- **The answer as a whole:** copy as Markdown, regenerate, edit and resend, stop; sources shown with the answer and
  openable; streaming never shows broken markup; an answer can be exported (section 3.3).
- A rendering test set exercises every element, valid and malformed, in light and dark, and runs without a browser
  where possible.

### 3.3 Documents in and out
- **In.** The knowledge base reads PDF, Markdown, HTML and text today. Add Word (`.docx`), PowerPoint (`.pptx`),
  Excel (`.xlsx`, and `.csv`) and LaTeX (`.tex`) with their structure: headings, lists, tables, slide numbers and
  notes, sheet names, formulas as text. Each format gets a fixture, a parser test and a place in the live sweep.
  Compare with what the owner's Smiley app reads and writes, and say where each is ahead.
- **A document can also be attached to one question** without adding it to a base: it is parsed the same way and put
  before the question, with its size shown and cut to what the context can hold.
- **Out.** An answer or a whole conversation can be saved as Markdown, HTML, Word, PowerPoint, Excel (its tables) and
  LaTeX. The model writes Markdown; a deterministic converter in the gateway makes the file. No model is asked to
  write a file format. Each export has a round-trip test (the exported file parses back to the same text).

### 3.4 Knowledge base, graph and retrieval
- Building a base from nothing takes one name and one drop of files; progress is shown per document and per stage; a
  document that fails says why and can be retried or removed; nothing is half-indexed without saying so.
- Search shows why a passage was found (words, meaning, graph) and opens the passage in its document.
- The graph view reads at a glance: a legend, search, filter by kind, focus on a node and its neighbours, the passages
  behind a node or an edge, fit to screen, save as an image; it stays usable at the size of the largest base in the
  repository's tests.
- Answers from documents show their sources, and say so when the documents do not support the answer.
- Every knowledge route and option is exercised by the knowledge sweep; any feature that the sweep shows broken is
  fixed, not hidden.

### 3.5 Settings
- Grouped by what a person wants to do, not by subsystem; each setting has a one-line explanation, its default, and
  whether it takes effect at once or after a restart; search across settings; reset one setting or all; presets for
  the common choices (faster, more careful, lowest memory).
- A setting that can break the app is validated before it is saved, and the error says what is allowed.
- Every setting in the UI is covered by a test that changes it and reads it back; config keys, the sample file and the
  schema agree, checked in CI.

## 4. Phase 36: the harness

Research first (`docs/research/`): what papers and open-source harnesses of 2024 to 2026 found for small local models
(tool-call formats, edit formats, context under a small window, planning and reflection, cascades inside an agent run)
and for showing agent work to a person (steps, stop, steer, approve, inspect). Then implement the candidates that fit
a 1.7B and a 4B model without grammar-constrained decoding, each behind a switch and each compared with the harness
without it on the existing task sets. The UI shows a run as a list of steps a person can read, stop and inspect.

## 5. Phase 37: release

- Sweep everything on the final build; review the whole change set; refactor what the review marks as duplicated or
  unclear; raise the version; write the release notes.
- New screenshots of every area that changed, light and dark, taken from the running app by the capture script.
- The paper gains the Phase 34 to 36 sections and figures; the README is rewritten as a landing page: what it is in
  one sentence, a picture, proof numbers each linked to its stored run, how it works, what it does not do, quick
  start, documentation.
- **Website.** Assess vllm-sr.ai as the benchmark (structure, documentation, clarity, evidence), then build a static
  site for Nafs-Aql that is better on each point where it can be shown to be: a clear statement of what the product is,
  proof with sources, how it works, use cases by sector with the constraint that makes a local assistant the right
  tool, an honest list of limits, documentation entry points, quick start. Plain HTML and CSS, no tracker, no remote
  asset, works without JavaScript, passes an accessibility and link check, published by GitHub Pages from the
  repository. Use cases must not promise what a small model cannot do.
- Push to `origin/main`. In `third_party`, remove the reference checkouts the project does not build or run, after
  checking that each has an upstream to clone again from and no local changes; keep what scripts and docs depend on,
  and update the docs that point at what was removed.

## 6. What not to do

- Do not claim a score, a speed-up or a quality gain without the stored run.
- Do not turn on by default anything that lost its comparison.
- Do not add a build step, a CDN or a tracker.
- Do not delete the owner's data, conversations or models.
- Do not rewrite what works for the sake of style.
