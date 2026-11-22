# Nafs-Aql 0.2.0

Released 2026-11-22 from `main`. A research preview, as 0.1.0 was. This release makes the engine faster at the points
where a conversation spends its time, takes in and gives out documents in more formats, and goes through every form,
answer and setting of the app for clarity. It also adds a website.

## What is new

**Engine (Smiley Core), after the reasoning of FreeToken (arXiv 2608.16157).** FreeToken serves large
Mixture-of-Experts models on a personal machine with a GPU; none of its mechanisms applies to a dense model on four
CPU cores as written, and its own code has one context anchor, off by default. Its reasoning does apply:

| Change | Measured on the reference machine (shared; interleaved arms) |
|---|---|
| A continued conversation takes its kept state over, with no copy | 240 ms to 0 (quick model) and 457 ms to 0 (reasoning model) at 2,048 tokens |
| The first token comes with the last piece of the prompt | 329 to 229 ms and 1,075 to 568 ms |
| Idle threads take work nobody has started | writing 19 % faster (12 of 12 rounds) and reading 14 % faster on four threads, quick model; 33 % and 10 % on the reasoning model; inside noise on three threads, quick model |
| The model file is read ahead at start | first request of a fresh server 14.6 to 9.2 s |
| The answering model reads the conversation between turns | first token of a later turn 4.25 to 0.71 s (quick, median of 9) and 18.6 to 2.4 s (reasoning, median of 6) |

The cache budget can be read and changed while serving (`/v1/cache`). Kernels are unchanged (95 ns a row of 4-bit
weights in both builds). What was not built, and why, is in `docs/research/freetoken_assessment.md`.

**Documents in.** Excel and CSV join PDF, Word, PowerPoint, LaTeX, HTML, Markdown and text. A sheet is read as tables
that repeat their header every 40 rows; Word keeps its lists and tables. A file can be attached to one question
(`POST /v1/parse`): it is read on the machine and its text goes with the question.

**Documents out.** An answer or a whole chat is saved as Markdown, plain text, a web page, Word, PowerPoint, Excel,
CSV, LaTeX or PDF (`POST /v1/export`), written on the machine by a fixed converter. The model writes Markdown and
never a file format.

**Answers.** Tables sort and export; code has line numbers, wrapping and a save button; footnotes; a chart, diagram
or formula opened larger is a dialog. Regenerate replaces the answer in place and an earlier question can be edited and
sent again. The stored conversation changes only when the new answer exists.

**Forms and settings.** Settings are written as questions a person has, with three presets, a search box and a way to
put everything back. Every control has a name, a hint and, where a value can be wrong, an error beside it. A
configuration file with a misspelt key or an impossible value is reported when the service starts, and the three
shipped files are checked against the generated schema in CI.

**A Code page and a better first step.** Describe a small Python function and its tests; it is written and run in a
sandbox, step by step, and can be stopped (`/v1/harness/runs`). On twelve functions written from nothing the first
step passed its tests 12 times in 12 (1 of 12 before a change to the agent's instructions; median 7.9 s).

**Website.** `site/` is a static site published by GitHub Pages: product, how it works, sixteen researched use cases
with the evidence for each need and its limit, the evidence behind every number, and a documentation map.

## Review

An independent review of the two development phases found nine things to correct (a rewind that could lose a turn,
rows left behind by a rewind, a spreadsheet that could ask for unbounded memory, a run that stayed "going" after a
panic, configuration checks that refused files that used to work, formulas passed unfiltered into LaTeX, maths
delimiters rewritten inside code, a misjudged stored position, smaller things). All were corrected with tests; see
`docs/decisions/027-product-quality-phase35.md`.

## Validation of this release

| Check | Result |
|---|---|
| Rust tests (`cargo test --workspace --release`) | 580 pass, none fail; `cargo fmt --check` and `cargo clippy --all-targets -D warnings` clean |
| Renderer checks (`node scripts/render_test.mjs`) | 222 pass |
| Configuration against the generated schema | the three shipped files pass |
| Website (`python3 scripts/site_check.py`) | 6 pages, 0 problems |
| Live: API contract | passed |
| Live: security probes | 26 of 26, no open high-severity finding |
| Live: rooms | 10 of 10 |
| Live: end to end (every endpoint and page, with the new form, Settings, Code and renderer checks) | 102 of 102 (97 of 102 on the first run: three real faults fixed and two checks corrected) |
| Live: knowledge (documents in and out included) | 81 passed, 1 skipped |
| Live: knowledge security | 33 of 33 |
| Live: code-agent harness | 53 of 53 |
| Live: acceptance | 8 of 9 (see below) |

## Acceptance MVP-A1 did not pass

MVP-A1 asks for a quick reply to "Say hello in one short sentence." with the first token inside 2 s. On this build the
gate sent it to the reasoning model (rule `uncertified_bin`, estimated error 0.109) and it took 67 s and 147 s, under
a load average of 40 to 47 from another project's builds. The previous run (2026-11-14, rule `fixed`) passed. Since
then the calibrated gate has been fitted (2026-11-15) and an open-domain bin has no certificate, so it escalates. The
test was not run between those dates, so this was not caught earlier, and it is not a result of the 0.2.0 changes as
far as can be told: the change came with the calibration, but no build before this one was run against the new
state. It is the gate behaving as configured (`on_unknown = escalate`); whether to treat "hello" as a quick question is
a decision for the next release. The other eight acceptance tests pass.

## Known limits

Those of 0.1.0 stand (see `RELEASE-0.1.0.md`), and:

- Every engine number is from one shared four-core machine with 6 to 12 paired rounds or 6 to 9 turns; they show
  direction, not release-grade statistics. The between-turns gain needs the pause between turns to outlast the reading.
- The code agent's first-step result is one model, twelve small tasks, one run per arm.
- Diagrams other than flowcharts, sequences, state diagrams and pies are shown as their source.
- Scanned PDFs and pictures are not read (they need a recognition model).
- A regenerated answer replaces the old one; both are not kept.
- Settings stay in the browser, not per person on the server.
- With the default engine three start-up notes are reported on the Status page (the options for llama-server's
  grammar, raw continuation and four first-step candidates do nothing with Smiley Core); they apply if you switch
  engines.

## Run it

```bash
git clone https://github.com/rismanmattotorang/nafsaql.git && cd nafsaql
make models
make up
```

Open http://127.0.0.1:8080. `make down` stops everything.
