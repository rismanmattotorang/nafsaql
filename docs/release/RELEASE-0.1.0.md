# Nafs-Aql 0.1.0

Released 2026-11-11 from `main`. A research preview: everything a four-core CPU can settle is in it, and what needs a
GPU or a much larger evaluation is listed at the end.

Nafs-Aql is an assistant that runs on four CPU cores with no GPU. A 1.7B model answers at once; a 4B reasoning model
steps in when a gate, derived from expected loss, judges the wait worth it; where an answer can be checked (maths,
code against tests, data against a schema, Lean proofs) a checker decides, and every answer says how sure it is. It
answers from your own documents with numbered sources, says which sentences of an answer are not in those sources,
and remembers earlier conversations in notes you can read and delete.

## What is in it

- **One Rust gateway** with a web interface, a chat API described in OpenAPI, chat rooms, an evaluation lab and a quality
  page; 13 crates, 507 tests.
- **Its own inference engine** (Smiley Core): several users in one pass over the weights, a store of earlier
  prompts' beginnings, prompts read with every core and tokens written with one fewer.
- **A gate and verifiers**: expected-loss routing, certified per-bin risk, exact verifiers whose verdict is final,
  a code agent that ends only on a verifier's pass or a spent budget.
- **Knowledge**: readers for PDF, Word, PowerPoint, LaTeX, HTML, Markdown and text; an in-process index (words and
  meaning, fused by rank); a graph of every base with no model call; typing slips and follow-up questions handled by
  rule; a per-sentence check of every answer against its passages.
- **Memory**: verified answers; notes of earlier conversations, read only when a question asks for them; memory
  tokens (Memorilla) with a guard against a module that ignores the question.
- **The paper**: `docs/report/nafs_aql_report.pdf`, 23 pages, with the formal model, the architecture and every
  measurement, including what did not work.

## Measured on the reference machine (Intel i3-8100, 4 cores, shared)

| | |
|---|---|
| Hard questions at the accuracy of always reasoning | 36 % of the cost |
| Reading a prompt | 61 tokens/s on the quick model, 23 on the reasoning model (+23 % from reading with all four cores) |
| Search over 847 passages, words and meaning | 0.15 s, of which the search itself is 0.2 ms |
| Held-out questions answered from passages | 42 of 50, against 0 of 50 without |
| A follow-up question finds its passage first | 10 times as often when read with the turn before it |
| Answers made up without reading, flagged by the source check | 38 of 44; it flags 1 of 42 right answers |
| Memory of earlier conversations, live check | 14 of 14 |

## Validation of this release

| Check | Result |
|---|---|
| Rust tests (`cargo test --workspace --release`) | 507 pass; `cargo fmt --check` and `cargo clippy -D warnings` (all targets) clean |
| CI (Ubuntu 24.04 and macOS 15: format, lints, tests; dependency audit; API and schema validation) | green on the release commit |
| API contract | passed |
| Security probes | 26 of 26, no open high-severity finding |
| Rooms | 10 of 10 |
| End to end (every endpoint and page) | 69 of 69 |
| Knowledge | 55 passed, 1 skipped |
| Knowledge security | 33 of 33 |
| Memory of earlier conversations | 14 of 14 |
| Code-agent harness | 53 of 53 |
| Acceptance | 9 of 9 (first token after 152 ms; a verified reasoning answer in 198 s against a 240 s target) |

Live suites ran on the release build on 2026-11-11 between 21:41 and 22:16, on a machine shared with other work.

## Known limits

- Questions worded differently from the document find their passage first 12 % of the time. The remedies need
  models or benchmarks that have not been downloaded.
- The source check compares words and numbers: it does not catch a wrong number taken from the right passage.
- The learned gate is not in force: its ledger holds 1,294 of the 2,900 checked answers its own rule asks for.
- No memory-token module carries knowledge; pre-training is a GPU job.
- Outside chat rooms there are no accounts: one installation, one person's memory.
- Every number is from one shared four-core machine and tens to hundreds of items; no public benchmark was run.

## Roadmap gates at release

G1 (acceptance and latency), G4 (evaluation suite and load tests) and G5 (security and deployment) pass; G1
depends on the machine's load at the time of the run (it failed its two latency targets under load 30 to 40 earlier
the same day). G0, G2, G3 and G6 are pending: they need a full-scale baseline, 2,900
labelled drafts, a GPU-trained adapter and a 700-pair non-inferiority test. The gate table as the service reports it
is in [`RELEASE-CANDIDATE.md`](RELEASE-CANDIDATE.md).

## Run it

```bash
git clone https://github.com/rismanmattotorang/nafsaql.git && cd nafsaql
make models
make up
```

Open http://127.0.0.1:8080. `make down` stops everything.
