# NAFS-AQL: ONE CONTEXT FILESYSTEM PROMPT (v1)

Revised on 2026-11-13 from the owner's request so that it matches Nafs-Aql after Phase 28 (layout reader, sentence
windows). The project-level name of this work is **Phase 29**.

The owner's request, in short: study OpenViking's context database for agents (`third_party/OpenViking` and
docs.openviking.ai, "context layers"), find Nafs-Aql's gaps, write specifications and a phased plan, and build better
versions of four capabilities: one filesystem for knowledge, memory and skills with a URI per item; search scoped to a
directory, with `find` (direct) and `search` (planned from the session); summaries before sources (L0 abstracts, L1
overviews, L2 content); sessions committed as readable files with memories extracted as Markdown, and `compile` into a
wiki, graph or report. Start with Phases 0 and 1, then continue; screenshots, paper, README, push to main on GitHub.

What changed from the original request, and why:

| Original request | Revised | Reason |
|---|---|---|
| Adopt, cherry-pick or port OpenViking's C++, Rust and TypeScript code | **Re-implement in Rust; copy no code or prompt text from the Python core.** The Python core and its prompt templates are AGPL-3.0; Nafs-Aql is Apache-2.0. Only `crates/` (ragfs, ov_cli) is Apache-2.0, and its file-store and CLI ideas are what we need, not its code | Licence compatibility |
| A `viking://` URI per item | A **`nafs://` URI scheme** over the store Nafs-Aql already has (SurrealDB + in-process index): `nafs://kb/{base}/{doc}#p{ord}` for documents and passages, `nafs://memory/{kind}/{key}` for memories, `nafs://skills/{name}`, `nafs://sessions/{id}`; `ls`, `tree`, `read`, `glob` and `grep` over it, in the API and the agent harness | The store stays the source of truth; the tree is a view, not a second store |
| Search a directory, not the whole index | A **scope prefix** on every search (`nafs://kb/{base}/{doc}`, `nafs://memory/preferences`, ...) applied as a bitset inside the index scan (OpenViking's C++ index does the same with a directory trie). `find` runs the query; `search` plans 0 to 2 scoped queries from the session, **only when the follow-up rule fires or the first retrieval is weak**: a planner call per turn costs seconds on four cores | OpenViking removed its recursive directory walk (commit 69e71e4ff) after shipping it with propagation off; a scoped flat search is what survives |
| Generated L0 abstracts and L1 overviews | **Model-free first**: L1 = heading tree + first paragraph + table captions + top terms (from the Phase 28 layout reader); L0 = title + first sentence + top terms. **Model-written** L0/L1 only on request or for documents above a size, in the background, with OpenViking's freshness rule (regenerate a parent only when a child's L0 changed and 10 % of a wide directory changed) | One summary call per document is 20 to 60 s on the quick model; 800 passages would be hours |
| Read the summary before the source | A **tiered context assembler**: every hit enters the prompt at its cheapest tier (L0 or its best sentence), leftover budget upgrades the best hits to L1 then L2, with category quotas (passages, memories, skills, ledger answers). Measured against the current four whole passages on the existing question sets and the 50-question answer set | Prefill is 80 % of an answer's time (Phase 27); the change must not cost right answers (Phase 27 lost 3 of 50 to excerpts) |
| Sessions become files | **Session commit**: the conversation archived as Markdown/JSONL under `nafs://sessions/{id}`; memories extracted into **typed Markdown files with identity in the path** (`preferences/{topic}`, `entities/{name}`, `events/{date}-{slug}`) and a merge rule per field; every write recorded in a **memory diff** the person can read and undo in Settings → Memory. Extraction is one grammar-constrained call to the quick model per commit, never per turn; explicit "remember ..." statements are kept by rule with no model | The Phase 26 recall store holds episodes, not facts; a diff makes writes visible and reversible |
| Skills define how to perform tasks | `SKILL.md` packages (front matter: name, description, tools, tags) under `nafs://skills`, indexed by their description, loaded progressively by the harness (abstract first, body on demand) | The harness has tools but no skill store |
| VikingBot and `ov compile` into wiki, graph or report | `compile` with **model-free targets first**: a wiki (one page per document and per heading section with L1, links by shared terms), a knowledge graph (the existing model-built and term graphs exported as `entities/*.md` + `relations.jsonl` with evidence URIs), a report (Markdown from L1s). A model-written wiki is optional and runs on the reasoning model in the background | Agent-run compilation needs a large hosted model |
| Make Nafs-Aql far superior | **No claim without a number.** Each property is claimed only where measured against the setting it replaces; OpenViking's LoCoMo figures use a lenient judge and leave out a category, so they are not a target | The project's standing rule |
| Push to main | Branch per phase group, pull request, CI green, rebase-merge into `main` | As before |

## 0. Foundation (Phase 29.0)

Research note `docs/research/openviking_gaps.md`, specification `docs/spec/context_fs_spec.md` (URIs, operations,
layers, scopes, memory schemas, skills, session commit, compile), decision record, this prompt.

## 1. One filesystem and scoped search (Phase 29.1)

`nafs_core::uri` (parse, normalise, scope prefixes); a `ContextFs` view in `nafs-kb` over bases, documents, passages
and memories (`ls`, `tree`, `read`, `stat`, `glob`, `grep`); `/v1/fs/*` routes; index scope bitsets; `find` with a
scope; harness tools. Measure: scoped vs unscoped retrieval on questions about one document; latency.

## 2. Context layers (Phase 29.2)

Model-free L0/L1 per document and per base, stored and refreshed with the freshness rule; L0/L1 in `ls`/`tree`; L1
as a search field and a vector; optional model-written layers in the background. Measure: document-level questions.

## 3. Tiered assembly and `search` (Phase 29.3)

The tiered budget assembler with quotas; the planner gated by the follow-up rule and retrieval weakness. Measure:
answer accuracy and prompt size against four whole passages, interleaved.

## 4. Sessions, memories and skills (Phase 29.4)

Session commit and archive; typed Markdown memories with merge rules and the diff; event decay; skills store and
progressive loading. Measure: recall checks, memory duplicate rate.

## 5. Compile and the interface (Phase 29.5)

`compile` targets; a Files page in the app (tree with L0 lines, read view, memory editor, diff); screenshots.

## 6. Validate and publish

Rust tests, fmt, clippy, live sweep, paper and README, pull request, CI, merge.
