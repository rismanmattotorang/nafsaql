# NAFS-AQL: ROUTING, KNOWLEDGE AND MEMORY CLOSE-OUT PROMPT (v1)

Revised on 2026-11-11 from the owner's request so that it matches Nafs-Aql as it is built today (after Phase 25, commit
`518a374`). The project-level name of this work is **Phase 26**. It is the last development phase before the release.

What changed from the original request, and why:

| Original request | Revised | Reason |
|---|---|---|
| Assess "semantic-router" and "memorilia" | **vLLM Semantic Router** (`third_party/semantic-router`, commit `029bf54ed`) and **Memorilla** (`third_party/memorilla`, commit `d8f1889`), read for what the two earlier reads did not cover | Phases 20 and 22 already read the decision engine, calibrated cascade, prototype banks, reask signal, cache polarity guard, prompt compression and the memory module's forward pass |
| "Semantic duo routing" | The **gate between the quick model and the reasoning model**: the quick model drafts, exact verifiers and a calibrated confidence decide whether the reasoning model answers | Nafs-Aql does not choose among many models; it decides when one small model is enough. A selector over many models has nothing to select here |
| Identify gaps in vectorization, graph, RAG and routing | Gaps are **measured before they are closed**: each candidate mechanism runs against the setting it would replace, on our own data, and is adopted only when it wins | Phases 23 to 25 showed that most borrowed mechanisms do not transfer (convex fusion, feedback terms, field weights, prototype banks) |
| Make Nafs-Aql the superior solution | **No claim without a number.** "Superior" is claimed per property that was measured on the reference machine; the rest is listed as open | No public benchmark has been downloaded, and the owner's approval is needed for downloads |
| Fix any gaps and implement better solutions | Only mechanisms that run with **no extra model** on four cores (rules, word statistics, the 0.6B embedder already running) | Semantic Router's reranker, hallucination detector and query rewriter each need a further 150M to 307M encoder or a chat call per request |
| Train a better Memorilla | **Fix the trainer and guard activation**; a full retrain is hours to days of this CPU and is the owner's call | The one trained module carries no information about the question; the cause is the training recipe, not the port |
| Sweep test again and finalise the release | Rust tests, `fmt`, `clippy`, then the live sweep (`scripts/live_sweep.sh`) on a fresh build; record what was and was not re-run | Timings are labelled SHARED MACHINE |
| Revise the report to be final; tone down religious aspects; remove plans and specs | One **topic-organised paper**: the inspiration is one paragraph in the introduction; components carry plain names (gate, audit ledger, distillation); no phase log, requirement ids, roadmap or evaluation plan | The owner's instruction |
| Push and commit to main | Branch `phase26-router-knowledge-memory`, pull request, CI green, rebase-merge into `main` | `main` is never force-pushed; CI is the release gate |

---

## 0. How you work

Same rules as `docs/prompts/COMPLETION_PROMPT.md` section 0 and `MEMORILLA_KB_PROMPT.md` section 0. In particular: verify
before you claim; label shared-machine timings; never kill the owner's processes; ask before downloading model weights;
plain language in the interface; Rust at run time.

## 1. Assess

1. Read Semantic Router for: vector stores and hybrid search, chunking, embeddings and their caches, semantic cache,
   conversation memory (extraction, retrieval gate, reflection, consolidation), model selection, hallucination
   detection, topic continuity, tool retrieval, and any committed measurements.
2. Read Memorilla against `nafs-memorilla` for: training recipe (stages, losses, schedule, dropout, data), inference
   (document selection, caching, prompt, decoding), memory management, evaluation and baselines.
3. Write the gap list in `docs/research/router_memory_gaps.md`: what each has that Nafs-Aql lacks, what Nafs-Aql has
   that they lack, and for each gap whether it can run on four cores with no further model.

## 2. Close the gaps

For each gap that passes the four-core test, build it, test it, and measure it against what it replaces:

- **Knowledge search**: a mistyped question word; a follow-up question that names nothing on its own; how many
  passages a prompt carries.
- **Answering from documents**: whether each sentence of the answer is in the passages it was given, with no model.
- **Routing**: whether the question alone predicts that the quick model will be wrong (the deferred difficulty prior),
  on the labelled drafts the ledger now holds; whether an unsupported quick answer should go to the reasoning model.
- **Memory**: a memory of earlier conversations (store, gate, search, decay, duplicates, forgetting) that a person can
  read and delete; the Memorilla trainer brought to the reference recipe, and a guard that refuses a module whose
  memory does not depend on the question.

Record every mechanism that was measured and not adopted, with its number.

## 3. Validate and release

Rust tests, `cargo fmt --check`, `cargo clippy -D warnings`; live sweep on a fresh release build; update
`docs/roadmap/STATUS.md`, the decision record, the user guide and the README; set the release version.

## 4. The paper

Rewrite `docs/report/` as the final-draft paper for publication: design, architecture, features, capabilities and
measured results. One paragraph of inspiration; no theology; no development plan, specification, requirement ids or
phase history. Build the PDF with no undefined references.

## 5. Publish

Commit, push the branch, open the pull request, wait for CI, rebase-merge into `main`.
