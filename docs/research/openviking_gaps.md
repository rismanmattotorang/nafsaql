# OpenViking against Nafs-Aql: one context filesystem (Phase 29)

Read on 2026-11-13: `third_party/OpenViking` at `13738af5e` (Volcengine), including its own copy of the concept pages
(`docs/en/concepts/03-context-layers.md` and neighbours). Prompt: `docs/prompts/CONTEXT_FS_PROMPT.md`. Specification:
`docs/spec/context_fs_spec.md`.

**Licence.** The Python core and its prompt templates are AGPL-3.0 (`LICENSE`, `pyproject.toml:19`); `crates/` (ragfs,
ov_cli) is Apache-2.0. Nafs-Aql is Apache-2.0, so the design is re-implemented in Rust and no code or prompt text is
copied.

## 1. What OpenViking is

| Part | What it does | Where |
|---|---|---|
| `viking://` tree | scopes `resources`, `user/{id}` (memories, resources, skills, sessions), `agent` (skills); `viking://~` is the caller's root; a URI maps to a store path by prefix rewrite; `ls`, `tree`, `glob`, `read`, `write`, `rm`, `mv`, `grep` | `docs/en/concepts/04-viking-uri.md`, `openviking/storage/viking_fs/_ops.py`, `_access.py:903-918` |
| Layers | L0 `.abstract.md` (~256 chars), L1 `.overview.md` (~4,000 chars), L2 the content; sidecars **per directory**; a 50-150 word LLM summary per non-code file, a tree-sitter skeleton for code; one LLM call per directory for L1, L0 cut from L1; bottom-up | `openviking/storage/queuefs/semantic_processor.py:1339-1804` |
| Freshness | md5 reuse; a parent is refreshed only when a child's L0 changed; a directory of more than 32 entries waits for 10 % of them; evenly spaced sampling of 32 entries | `semantic_ops/freshness_policy.py:28-61`, `abstract_overview.py:500-510` |
| `find` | one vector search filtered by directory (a Roaring bitmap per path prefix in the C++ index), context type and layer; optional single rerank on the abstracts; L0, L1 and L2 compete in one pool | `openviking/retrieve/hierarchical_retriever.py:102-299`, `src/index/detail/scalar/bitmap_holder/dir_index.h` |
| Recursive retrieval | a beam over directories with parent-score propagation, **removed** in commit `69e71e4ff` (2026-09-29); propagation had been off by default | `git show 69e71e4ff^:openviking/retrieve/hierarchical_retriever.py` |
| `search` | an LLM planner over the session summary and last five messages emits 0..n typed queries (memory, resource, skill); fine-tuned small planner models are offered | `openviking/retrieve/intent_analyzer.py:38-137` |
| Context assembler | category quotas, a 1,600-token budget filled breadth first (every hit at its cheapest tier) then deepened, fallback to a lower tier rather than truncation, no score threshold (scores cluster at 0.38-0.50) | `openviking/retrieve/context_assembler/{params,budget,tiers}.py` |
| Sessions | commit archives `history/archive_NNN/messages.jsonl`, then asynchronously summarises, extracts memories and writes `memory_diff.json` (adds, updates with before/after, deletes, skips with reasons) | `docs/en/concepts/08-session.md`, `openviking/session/session.py` |
| Memories | typed Markdown files whose path is their identity (`preferences/{user}/{topic}.md`, `entities/{category}/{name}.md`, dated events); merge per field (immutable, replace, patch, sum); events append-only with a 7-day half-life after a protection window | `system_assets/memory/memory_templates/builtin/*.yaml`, `viking_vector_index_backend.py:2126-2166` |
| Skills | `SKILL.md` with front matter (name, description, allowed tools, tags) and files; L0 from the metadata; vector retrieval; progressive loading | `docs/en/api/04-skills.md`, `retrieve/skill_package_retriever.py` |
| Compile | an agent run (VikingBot) with a skill that writes a wiki, a knowledge graph (`entities/*.md` + `relations.jsonl` with evidence) or a report; built for a large hosted model | `openviking/system_assets/compile/example-skills/` |
| Evaluation | LoCoMo with a lenient LLM judge and the adversarial category left out; tau2-bench; no ablation of L0/L1 against flat retrieval | `benchmark/locomo/vikingbot/judge.py:42-54`, `run_eval.py:542-544` |

## 2. Gaps and the answer to each

| Capability | OpenViking | Nafs-Aql before | Phase 29 answer | Model calls |
|---|---|---|---|---|
| One namespace | `viking://` over a file store | record ids in two stores | `nafs://` URIs over the stores that exist (knowledge, memory, skills, sessions); a view, not a copy | none |
| Browsing | `ls`, `tree` with L0 inline | the Knowledge page lists documents | `/v1/fs/{ls,tree,read,stat,glob,grep}` and a Files page | none |
| Scoped search | directory bitmap in the vector index | whole base or one base | a scope prefix (a base, a document, a memory kind) masks the in-process index; words and meaning both | none |
| L0/L1 | one LLM call per file and per directory | none | **model-free** L0/L1 from the layout reader's structure (headings, first paragraph, captions, defining sentences, distinctive terms); per document and per base with the freshness rule; optional model-written L0 in the background | none by default |
| Summary before source | tier per hit, budget, quotas | four whole passages | the same breadth-then-depth assembler, measured against four whole passages | none |
| Planned search | an LLM planner every turn | follow-up rule, typo correction | rule planner (scope from the question's names, follow-up rule, memory words); an LLM planner only when retrieval is weak | at most one, rarely |
| Memories | typed files with identity in the path, LLM extraction | recall episodes (turns, windows, notes) | typed Markdown memory files with keys, merge rules, a diff log the person can undo; extraction by rule first ("I prefer", "remember that", "my X is"), a grammar-constrained quick-model pass on commit | one per commit, optional |
| Sessions as files | archive + memory diff | conversations in SQLite | commit writes a Markdown transcript under `nafs://sessions/{id}` and the extracted memories with their diff | as above |
| Skills | `SKILL.md` store | harness tools only | `SKILL.md` packages under `nafs://skills`, found by `find`, loaded abstract-first | none |
| Compile | agent run on a large model | none | wiki, graph (the existing graphs with evidence URIs) and report rendered from layers, no model | none |

What Nafs-Aql already has that OpenViking lacks: BM25F fused with vectors in one ranking (OpenViking runs keywords as a
separate mode), typo and follow-up rules, sentence-window vectors, a layout reader, per-sentence support checks of an
answer, a gate and verifiers, hostile-file limits, and everything on four CPU cores with no hosted model.

Not ported: the recursive directory beam (removed upstream), an LLM summary per file at ingest by default, agent-run
compilation, VikingBot's channels, multi-tenant accounts (Nafs-Aql has one owner), LoCoMo's lenient judge as a target.
