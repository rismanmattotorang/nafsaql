# Routing, knowledge and memory: what vLLM Semantic Router and Memorilla have that Nafs-Aql lacked (Phase 26)

Date 2026-11-11. Third read of vLLM Semantic Router (`third_party/semantic-router`, commit `029bf54ed`) and second read
of Memorilla (`third_party/memorilla`, commit `d8f1889`), limited to what the earlier reads
([`router_harness_comparison.md`](router_harness_comparison.md), [`semantic_router_deepseek_h2h.md`](semantic_router_deepseek_h2h.md),
[`memorilla_knowledge.md`](memorilla_knowledge.md)) did not cover. Decision record:
[`019-routing-knowledge-memory-phase26.md`](../decisions/019-routing-knowledge-memory-phase26.md). Measurements:
`benchmarks/kb_phase26/`, `benchmarks/router_phase26/`. Timings are SHARED MACHINE (load 60 to 100 during this phase:
another build ran on the same four cores).

The rule for this phase: a mechanism is taken only if it runs on four cores with **no further model** (rules, word
statistics, or the 0.6B embedder that is already running), and it is switched on only if it **wins a measurement**
against the setting it replaces.

## 1. Semantic Router: what it does that was not read before

| Area | What Semantic Router does | Needs | In Nafs-Aql before this phase |
|---|---|---|---|
| Chunking | Paragraph and heading split, then fixed windows of 800 characters with 400 overlap | nothing | Heading-aware passages of 1,200 with 150 overlap, measured against 600 and 2,400 |
| Formats | txt, md, json, csv, html | nothing | PDF, Word, PowerPoint, LaTeX, HTML, Markdown, text |
| Hybrid search | Vector top 4k, then BM25 and character 3-gram Jaccard computed over those candidates only; weighted sum 0.7/0.2/0.1 with per-query min-max; RRF optional | embedder | BM25F as an independent arm over the whole base, exact vector scan, RRF at 0.1 (confirmed over 256 questions) |
| Typing slips | Character 3-gram arm | nothing | **nothing** |
| Reranker | Cross-encoder, 307M | a further model | measured earlier: nothing to gain on our sets |
| Embedding identity | Refuses to search a store built by another model, layer or dimension | nothing | the model name was stored with the base and **not checked**; every write renamed the base |
| Embedding cache | sha256 key, 32 shards, in-flight sharing | nothing | question-vector cache (256), prefix-row reuse inside the engine |
| Semantic cache | HNSW over question embeddings, threshold 0.8 | embedder | memory of **verified** answers only, with the polarity guard |
| Conversation memory | A note per turn and a window every third turn; a retrieval gate (first-person word, no retrieved context, no greeting); elbow threshold on scores; reflection gate (recency decay, duplicate removal that respects numbers, token budget); consolidation by word overlap; per-user isolation | embedder; an optional chat call to rewrite the query | **nothing**: a fact told in one chat could not be found in the next |
| Hallucination | Token classifier (149M to 307M) over context, question and answer; a fixed-string explanation | a further model | **nothing**: an answer from documents was always "not checked" |
| Topic continuity | Containment of the live turn's terms in the previous 8 turns with decay 0.85; reference and change-marker lexicons; three-way verdict | nothing | reask signal only; **a follow-up question was searched alone** |
| Model selection | Elo, RouterDC, KNN/KMeans/SVM/MLP over embedding plus category, Thompson sampling, session-aware switching with stay bias and switch gates, cache affinity | embedder; offline labels where trained | one decision (quick or reasoning) by expected loss, calibrated error and certified bins |
| Graphs | none (TextRank over sentences in prompt compression only) | | rule-built graph with Personalized PageRank; model-built entity graph with verified quotes |
| Committed quality numbers | none for retrieval, memory, cache precision, selection or hallucination; speed records only | | every setting has a stored run |

Not taken, and why: the reranker, hallucination detector, fact-check sentinel and query rewriter each add an encoder
or a chat call per request on the same four cores; RouterDC's feedback never reaches its selection; GMTRouter, the POMDP
solver and the hybrid selector's weights are hand-set with no evidence in the repository; 2D Matryoshka needs their
trained encoder; their own record (`bench/grounded_fusion/FINDINGS.md`) shows hard NLI filtering making fused answers
worse. A selector among many models has nothing to select in a two-model system: the session-aware terms (stay bias,
switch margin, cache affinity) describe what the gate's waiting-time term and the engine's prefix store already do.

## 2. Memorilla: why our trained module answered nothing

The module trained in Phase 20 (one epoch, 234 examples) was run through a NumPy forward that matches the PyTorch
fixture to 9e-7. **All 16 memory tokens are one vector** (cosine 0.9999999 between slots, 0.999998 between
questions), with norm 63 against a mean token-embedding norm of 1.54. The residual branches drifted to large constants
and the projection is close to rank one. The module had learnt a question-independent prompt for the answer format,
which is why validation loss fell from 8.77 to 1.46 while claim accuracy stayed at 0.55 with and without it, and why
`best/`, chosen by validation loss, was the collapsed checkpoint.

| Item | Reference | Ours before | Now |
|---|---|---|---|
| Learning rate, examples per step, epochs | 1e-4, 16, 5 (`configs/factkg.yaml`) | 3e-4, 4, 3 | 1e-4, 16, 5 |
| Warm-up | 10 % of steps, from 1e-8 of peak | 10 % of steps, starting at peak/10 steps | 10 % of steps, linear from zero, cosine to 10 % |
| Dropout | 0.1 at four sites | none | 0.1 at the same four sites, training only (inference and parity unchanged) |
| Loss | token-level cross-entropy over answer tokens | mean per example (a 2-token "True" weighed as a 75-token restatement) | token-level over an accumulation group |
| Checkpoint choice | generation metrics on validation | validation loss | a checkpoint whose memory depends on the question first, then validation loss |
| Decoding | greedy | draft temperature, top-p 0.8 | greedy for answers read from memory tokens |
| Pre-training (Stages 1 and 2, 3.3M rows) | yes | no | no: a GPU job |

Our addition, absent from the reference: **collapse statistics** (mean cosine between the slots of one question, mean
cosine between the memories of different questions, mean norm) computed at every validation and saved beside the
weights, and an **activation guard**: a module whose memories for different questions have cosine 0.98 or more is not
used. `auto` then answers from passages, and an explicit `memory` request falls back to passages with a sentence that
says why. A module saved before this phase is probed once with three fixed questions.

Not done: a retrain with the new recipe. At 137 to 272 s an example it is hours for one small base and the outcome
without pre-training is uncertain; it is the owner's call. The guard makes the present state safe: the collapsed
module is not used.

## 3. What was built, and what each measurement said

All retrieval numbers are on the 847-passage base made from this project's report (`benchmarks/kb_phase26/retrieval.json`).

| Mechanism | Against | Result | Decision |
|---|---|---|---|
| **A mistyped question word is replaced by the nearest word the base holds** (same first letter, one edit, two for a word of eight letters or more; identifiers and numbers are taken as typed) | no correction; 60 quoted questions with one slip each | words only: answer within 8 for 0.967 → 1.000, first 0.867 → 0.883; hybrid: within 4 for 0.950 → 0.983, within 8 for 0.967 → 1.000. 40 words corrected; **0 words changed** in the 60 questions as written | on (`fuzzy_terms`) |
| **A follow-up question is searched together with the previous turn** when it names nothing on its own (fewer than three content words, or a reference word or opener with at most six) | the follow-up alone; 40 pairs | answer first 0.025 → 0.250, within 8 for 0.225 → 0.375, MRR 0.073 → 0.285. Joining every question gives 0.275 / 0.400 / 0.310. The rule fired on 37 of 40 follow-ups and on 1 of 256 self-contained questions (4 of 20 off-topic ones) | on (`followup_context`) |
| Elbow cut of the passages put in a prompt (Semantic Router's memory threshold) | the first four that pass the relevance filter | passages per prompt 3.90 → 3.77 (quoted), 3.33 → 2.99 (reworded) at a gap of 0.03; the answering passage left one reworded prompt | **off** (`rag_elbow_gap = 0`): it saves a tenth of a passage and loses an answer |
| **Sentence-level support check** of an answer from documents: each sentence's content words must be in one passage and each number in the passages or the question | none before | see section 4 | on, reported with the answer (`support_check`) |
| Escalating an unsupported quick answer to the reasoning model | keeping it | see section 4 | built, **off** (`support_escalate_below = 0`) |
| Question-only difficulty signal: nearest labelled questions (k = 15), and Semantic Router's prototype banks | the gate's draft features; 810 labelled drafts, 238 wrong, five folds | AUC for "the draft is wrong": domain 0.566, banks 0.573, nearest questions 0.615, draft features 0.682, draft features plus nearest questions 0.690. Skipping the draft at a score of 0.6 would skip 12 questions, 9 of them wrong | **not built**: the question adds 0.008 to what the draft already says, and a skipped draft saves under 2 % of requests |
| **A base records the embedding model that wrote its first vector**; a base made by another model is searched by words until its meaning index is rebuilt | comparing vectors of two models | unit test; the two bases in use carry the running model's name | on |
| Refit of the gate's calibration on the ledger (1,294 labelled drafts, was 203) | the fixed rule | held-out ECE 0.023, Brier 0.115; one of five bins certified under 0.15 (0.106), two at 0.157 and 0.161 | **not promoted** by the system's own rule (2,900 held-out drafts) |
| **Memory of earlier conversations** | none before | see section 5 | on (`[recall]`) |
| Memorilla trainer recipe, collapse statistics, activation guard | the Phase 20 trainer | unit and parity tests pass; no retrain | on |

## 4. The support check

A reply from documents now carries which of its sentences are in the passages it was given. The check splits the
answer into sentences, drops citation marks, list numbers, questions and sentences that decline ("the passages do not
say"), stems the content words, and asks two things of each remaining sentence: does one passage hold at least 60 % of
its content words, and is every number it states in the passages or in the question ("1,200" equals "1200", "four"
equals "4"). A sentence that fails is listed with the reason. It costs microseconds and no model. It says "this
sentence is not in your sources"; it does not say the sentence is false, and a sentence that rewords a passage in
other words can fail it.

Measured by `scripts/support_study.py` (results below, `benchmarks/kb_phase26/support.json`): the quick model answered
each question twice, from the passages and with none, and the made-up answers were checked against the same passages.

| Group (50 questions over the 847-passage base, quick model) | Answers with a claim | Every sentence supported | Mean share of supported sentences |
|---|---|---|---|
| Answered from passages, right (42) | 42 | **41** | 0.99 |
| Answered from passages, wrong (8) | 5 (3 said the base does not hold the answer) | 3 | 0.60 |
| Made up: answered with no passages, checked against the same passages (50, all wrong) | 44 (6 declined) | 6 | 0.21 |

- **A made-up answer is flagged 38 times in 44** and a right answer from passages once in 42. The area under the curve
  for "made up" against "from passages" is 0.90.
- **It does not catch a wrong reading of the right passage.** Three of the five wrong answers with a claim pass: each
  quotes a number that is in a passage, and it is the wrong number. Two are flagged.
- **A word the answer repeats from the question counts as found.** Without that rule 5 of 42 right answers had a
  sentence flagged ("SurrealDB listens on port 8300" against a passage that writes the port as `:8300`); with it, 1.
  Requiring every non-question word to be in a passage was worse (16 of 42 flagged). Sentences that say the base does
  not hold the answer ("does not explicitly mention") are set aside; the first run counted them as unsupported claims.
- **Escalation stays off.** With `support_escalate_below = 0.5`, 2 of 50 quick answers would go to the reasoning
  model, both of them wrong. Two answers are not a measurement of whether the reasoning model then gets them right, so
  the setting exists and its default is 0.

The answers are those of one live run (rag 42 of 50, closed-book 0 of 50; a rag answer took 54 s at the median under
load 14 to 60); the support rows were recomputed from the stored answers and passages with the final rule.

## 5. The memory of earlier conversations

Built in `nafs-memory::recall`, with no model call.

- **Writing.** Each finished turn of a stored conversation becomes a note `Q: … A: …` (the answer clipped to 1,200
  characters, reasoning removed). Every third turn of a conversation also stores a window of its last five. A greeting,
  a turn under 30 characters and an answer that declined are not stored. A message that opens with "Remember that …"
  is kept as a note that the size limit never removes.
- **When it is read.** Only when the question speaks in the first person or points back ("my", "we", "last time",
  "you said", "remember"), and never when passages from a knowledge base are already in the prompt. An impersonal
  question does not pay for a memory search or for the prompt tokens of notes it does not need.
- **How.** BM25 over the notes, and cosine over the notes' vectors when the embedding server runs, fused by reciprocal
  rank; a note that answers the opposite question is dropped (the polarity guard); the tail below the largest drop in
  relevance is cut (the elbow, which was *not* adopted for passages but suits a short list of notes); what remains is
  ordered by relevance, kind and age (half-life 30 days), cleared of duplicates that have the same words **and the same
  numbers** ("budget 10K" and "budget 12K" stay apart), and cut to three notes and 1,500 characters.
- **Whose memory.** A private (ephemeral) request, the Lab and a room neither write nor read it: what is said in a room
  is shared by its members. Deleting a conversation deletes its notes. `GET/POST/DELETE /v1/memory`,
  `DELETE /v1/memory/{id}` and `POST /v1/memory/tidy` list, add, forget and merge; Settings has a Memory section that
  shows every note with a Forget button.
- **Limits.** At most 5,000 notes; the never-recalled oldest go first.

Beyond Semantic Router's memory: the polarity guard, windows that are not returned beside a turn they contain, a size
limit that uses the access counts (theirs tracks the counts and never prunes), room and private-chat exclusion, and a
person can read and delete every note. Behind it: no per-user isolation (Nafs-Aql is single-user outside rooms) and no
model-written query rewrite.

**Live check** (`scripts/recall_check.py`, `benchmarks/kb_phase26/recall_check.json`, load 20 to 54): 14 of 14.
A fact kept with "Remember that …" and a fact stated inside an ordinary turn were both recalled from other
conversations and used in the answer; a question in other words ("which pet does my sibling own?" for "my sister's
dog is called Biscuit") found the note through the meaning arm; an impersonal question and a private request read
nothing; the conversation that held the fact answered from its own history; a forgotten note and the notes of a
deleted conversation were not recalled; deleting the conversations left the memory as it was before. The memory step
took 0.13 to 0.58 s, nearly all of it the one embedding call; a question that does not open the gate pays nothing.
Twenty-one unit tests cover the store, the gate, the ranking, the duplicates, the budget and the deletions.

What this does not measure: recall quality over a long history. The check is a handful of facts; no set of real
multi-session conversations with known answers exists here, and none was downloaded.

## 6. Where Nafs-Aql stands against the two after this phase

| Property | Semantic Router | Memorilla | Nafs-Aql |
|---|---|---|---|
| Decides when a small model is enough, with a bound on the error it keeps | calibrated cascade, bins | no | expected-loss gate, exact verifiers, certified bins, assurance on every reply |
| Checks an answer from documents against its sources | 149M to 307M classifier | no | by rule, no model, per sentence, numbers included |
| Search over documents | BM25 re-ranks the vector top 4k | top-5 cosine baseline | independent BM25F and exact vector arms, RRF, identifiers kept whole, typing slips corrected, follow-ups read with their turn |
| Graph of a document set | none | none | by rule in under a millisecond; by the model with verified quotes |
| Memory of earlier conversations | yes (per user, embedder, optional rewrite call) | per-collection latent memory | yes (no model; gate, decay, number-aware duplicates, room exclusion, readable and deletable) |
| Latent memory tokens | no | trained on GPUs for 8B decoders | port with parity, trainer on the reference recipe, collapse guard; **no trained module that carries knowledge** |
| Runs on four cores with no GPU | router yes, its encoders are a Python runtime | no | yes, one Rust gateway and its own engine |
| Committed quality measurements | none | paper benchmarks | every adopted and rejected setting |

What is still open: a reworded question finds its passage first 12 % of the time (needs a reranker or a stronger
embedder, both downloads the owner has not approved); no public benchmark has been run; the memory module needs GPU
pre-training; the support check cannot see a wrong answer that uses the passages' own words.
