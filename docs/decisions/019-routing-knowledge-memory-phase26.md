# Decision 019: Close the routing, knowledge and memory gaps with no further model; the paper becomes final (Phase 26)

Date 2026-11-11. Status: **adopted.** Research [`router_memory_gaps.md`](../research/router_memory_gaps.md), the prompt
[`ROUTING_KNOWLEDGE_MEMORY_PROMPT.md`](../prompts/ROUTING_KNOWLEDGE_MEMORY_PROMPT.md). Measurements:
`benchmarks/kb_phase26/`, `benchmarks/router_phase26/`. All timings SHARED MACHINE (another build used the same four
cores throughout; load 40 to 150).

## Decisions

1. **Only mechanisms that need no further model are taken** from vLLM Semantic Router and Memorilla. Its reranker,
   hallucination detector, fact-check sentinel and query rewriter each add an encoder of 150M to 307M parameters or a
   chat call per request; on four cores that is the request's whole budget.
2. **A mistyped question word is corrected against the base's own words** (`fuzzy_terms`, on). Answer within 8 for
   0.967 → 1.000 of 60 mistyped questions; no word of the same questions as written is touched.
3. **A follow-up question is searched with the previous turn** (`followup_context`, on), by a rule with no model.
   Answer first for 0.025 → 0.250 of 40 follow-ups; the rule fires on 1 of 256 self-contained questions.
4. **Every answer from documents says which of its sentences are in its passages** (`support_check`, on): content
   words and numbers, per sentence, no model. On 50 questions,
   41 of 42 right answers from passages have every sentence supported, and 38 of 44 answers made up without passages
   are flagged (AUC 0.90); it does not catch a wrong number taken from the right passage (3 of 5 wrong answers pass).
   Words the answer repeats from the question count as found (5 false flags → 1).
5. **A memory of earlier conversations** (`[recall]`, on): a note per finished turn, read only when the question is
   in the first person or points back, searched by words and by meaning, with the polarity guard, an elbow cut,
   decay, number-aware duplicates and a size limit; never written or read by a private chat, a room or the Lab; every
   note readable and deletable (`/v1/memory`, Settings → Memory). Live check 14 of
   14 (`recall_check.json`); the memory step costs 0.13 to 0.58 s when the gate opens and nothing when it does not.
   Recall quality over a long history is not measured.
6. **The Memorilla trainer takes the reference's from-scratch recipe**, measures whether a module's memory depends on
   the question, and the gateway refuses a module where it does not. The Phase 20 module is such a module (one vector
   for every slot and question). No retrain was run: it is hours of this CPU per small base with an uncertain result
   without pre-training, and it is the owner's call.
7. **The first embedding model to write a vector names a base**; a base named by another model is searched by words
   until its meaning index is rebuilt.
8. **The gate keeps deciding from the draft, and the fixed rule stays in force.** The question alone says little that
   the draft does not (see the table). The label stream has raised the ledger from 203 to 1,294 labelled drafts (280
   wrong); a refit on them has held-out ECE 0.023 and Brier 0.115, with one of five bins certified under the 0.15 bound
   (0.106) and two just above it (0.157, 0.161). The system did not promote it: its rule asks for 2,900 held-out drafts
   (`benchmarks/router_phase26/gate_refit.json`). At 90 labels an hour that is about 18 more idle hours.
9. **The technical report is rewritten as the final paper**: organised by topic (model, architecture, gate, harness,
   inference, knowledge, memory, evaluation, comparison, limitations), 34 section files reduced to 13, one paragraph
   on the origin of the name, no theology, no phase history, requirement ids, roadmap or evaluation plan. Its
   bibliography holds the 86 cited works; all 217 verified entries stay in `docs/research/refs_all.bib`.

## Not adopted, with the number

| Idea | Measurement | Why |
|---|---|---|
| A difficulty signal from the question alone (nearest labelled questions, prototype banks) | AUC for a wrong draft over 810 labelled drafts: 0.615 and 0.573, against 0.682 for the draft's features and 0.690 for both | adds 0.008; skipping the draft would touch 12 of 810 requests |
| Elbow cut of the passages in a prompt | 3.90 → 3.77 passages (quoted), 3.33 → 2.99 (reworded); one answering passage lost | a tenth of a passage for an answer |
| Reading every question with the previous turn | follow-ups 0.275 first against 0.250 by the rule | the rule keeps 255 of 256 self-contained questions as typed |
| Character 3-gram arm (Semantic Router's answer to typing slips) | not built | the edit-distance correction recovers every lost passage of the test set and adds no index |
| Model selection algorithms (Elo, KNN, Thompson, session-aware switching) | not built | two models; the gate's waiting-time term and the engine's prefix store already carry what stay bias and cache affinity describe |
| Escalating an unsupported quick answer to the reasoning model (`support_escalate_below`) | at 0.5 it would escalate 2 of 50 quick answers, both wrong | built and off: two answers do not show that the reasoning model then answers them right |

## Validation

504 Rust tests pass; `cargo fmt --check` and `cargo clippy -D warnings` (all targets) are clean. Live sweep on the
final build, 2026-11-11 16:23 to 17:19 (load 30 to 370): contract passed, security 26/26, Rooms 10/10, end to end
69/69, knowledge 55 passed (1 skipped), knowledge security 33/33, harness 55/55, acceptance 7/9. The two acceptance
failures are latency targets under this load (first token of a quick reply 2.6 s against 2.0 s; a verified reasoning
answer in 420 s against 240 s); both answers were right, and neither request reads the new memory (they are private
requests). The memory's own live check passes 14/14, and its dialog was exercised in a browser.

## What stays open

- A reworded question still finds its passage first 12 % of the time. The remedies need a reranker or a stronger
  embedder (downloads the owner has not approved) or hours of the quick model per base.
- No module trained with the corrected Memorilla recipe exists; pre-training is a GPU job.
- The support check cannot see a wrong sentence built from the passages' own words.
- The memory of earlier conversations is one person's: outside rooms there are no accounts.
- No public benchmark has been run; every comparison is against our own earlier settings.
