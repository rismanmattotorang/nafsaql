# Reasoning cache specification (Phase 38)

Status: built (2026-11-23). Code: `rust/crates/nafs-core/src/arc/` (`mod.rs` store, matching, builder; `serve.rs` request
path; `runtime.py` sandbox runtime; `compile.py` describe and compile; `builder_prompt.txt`), `rust/crates/nafs-gateway/src/arc.rs`
(API), the Reasoning cache page in `nafs-web/assets/app.js`. Assessment: `docs/research/reasoning_cache.md`.

## 1. Terms

| Term | Meaning |
|---|---|
| entry | one cached problem: question q(o), template q(.), parameters, code, status, statistics |
| parameter | `name`, `type` (int, float, Fraction, bool, list, str), `description`, `value` in q(o) |
| semantic function | a Python function decorated `@semantic("<description>", root=?)`, reading inputs with `load(ctx, d, T)`, calling others with `do(ctx, d, payloads)`, recording steps with `narrate`, returning a dict with natural-language keys; the root returns `answer` and `summary` |
| tier | how an entry was found: `template`, `shape`, `meaning` |
| status | `candidate` (runs beside the models), `trusted` (answers), `quarantined` (never runs) |

## 2. Request path (orchestrator, before the System 1 draft)

**FR-ARC-01 Eligibility.** The cache is tried when `[arc] enabled`, the request does not set `arc: false`, has no
knowledge base, tests, test code, JSON schema or Lean task, and is not forced slow. The template and shape lookups
(no model) run for every such question, whatever kind it was classified as: a match is the whole of a stored,
self-contained question. The meaning lookup, and admission of the answer to the build queue, also need the domain to
be in `[arc] domains` and the question not to lean on the previous turn (`continuity::leans_on_previous` with history
present). Both were found on the live chat page: a perimeter question classified as "open", and a question whose "its"
the follow-up rule read as pointing back.

**FR-ARC-02 Entry search.** In order, over entries not quarantined:
1. *template*: the template compiled to an anchored pattern (literal words with flexible spaces; numeric placeholders
   capture a number, others capture text) matches the question; values are the captures;
2. *shape*: the question with numbers masked equals the original's, and has as many numbers; numeric parameters take
   the new numbers by their position in the original;
3. *meaning*: cosine of the question's embedding (`Knowledge::embed`) with each entry's vectors; the best entry is a
   hit when the similarity reaches its floor (FR-ARC-07).

**FR-ARC-03 Extraction.** Template and shape give the values. On the meaning tier, when `model_extraction`, System 1
receives the parameter list (name, type, description), the original question with its values as a one-shot example,
and the new question; it replies with JSON `{"same": bool, "<name>": "<value>", ...}` (temperature 0, no thinking).
`same: false` is a miss.

**FR-ARC-03b Meaning-tier checks.** Before extraction on the meaning tier: (1) contrast words, no model: a pair of words that turns one question into another (odd/even, area/perimeter, simple/compound, least/greatest, price/saved, added/removed, total/average and others) where the entry's question has one side only and the new question the other side only is a miss; (2) when `same_check`, System 1 is asked "Does question B ask for exactly the same quantity as question A, computed the same way, only with other wording or other numbers?" (one word, temperature 0); anything but yes is a miss.

**FR-ARC-04 Guards (no model).** A hit is a miss when any value is not written in the question (text, or the same
number), or when the question writes a number that no value carries and the template does not write itself.

**FR-ARC-05 Dispatch.** The entry's code and the code of every entry it links to run in the verifier sandbox
(`nafs_verify::sandbox::run`, no network, `[arc] run_timeout_s`), with stdin
`{"root": "<entry>:<fn>", "params": {description: value}, "links": {...}, "code": [[id, code], ...]}`. The runtime
statically refuses imports outside the standard-library allow list, `open`, `exec`, `eval`, dunder attributes and
`print`. Any error, unconvertible value or missing answer is a miss.

**FR-ARC-06 Outcome.**
- *trusted* and `serve`: the reply is built from the run: content = summary plus `\boxed{answer}` for math; reasoning =
  the narration; decision rule `reasoning_cache`; mode fast; trace states `C` (answered from the cache) then `T`; one `arc` source row (entry, tier, similarity, values,
  steps). Assurance: **verified** when the request carries a reference answer and `verify_math_full` passes; if it
  fails, the entry records a checked disagreement and the request goes on to the models. Otherwise **consensus** when
  the entry has agreed with the models at least once, else unverified. The turn is stored, traced and counted
  (`nafs_requests_total{mode="cache"}`).
- *candidate*: the run's outcome is kept and the request continues on the usual path.
- *miss*: an `arc` stage records why.

**FR-ARC-07 Comparison and trust.** After the usual path answers, a candidate's outcome is compared: against the
reference answer when there is one (checked), else against the final answer when it is verified, consensus or from
System 2 (checked only when verified); unverified quick answers are no evidence. Numbers agree to the precision the
less precise side was written with. Agreement: `agreements += 1`; on the meaning tier `min_agreed = min(similarity)`;
promoted to trusted at `agreements >= trust_after` with no disagreement. Disagreement: `disagreements += 1`; on the
meaning tier `max_disagreed = max(similarity)`; quarantined when checked on the template or shape tier, or when
disagreements reach 2 and exceed agreements. Floor of an entry:
`max(min_similarity, min(serve_similarity, min_agreed), max_disagreed + 0.005)`.

**FR-ARC-08 Admission.** When the usual path answered with System 2 in a stored conversation that is not private, not a room and not Lab traffic (an entry keeps its question and every user of the server can be answered from it), the request was eligible and no entry covered it,
and the assurance is in `[arc] admit`, the question, answer, reasoning, reference and reply id join the build queue
(deduplicated by shape, capped at `queue_cap`, saved in `queue.json`).

## 3. Build path (idle worker)

**FR-ARC-10 Scheduling.** Every `poll_s` seconds, when no request is in flight, one job is taken. A request that
arrives cancels the System 2 call; the job returns to the front of the queue (at most 3 attempts).

**FR-ARC-11 Builder call.** System 2 receives `builder_prompt.txt` (three sections TEMPLATE, PARAMS, CODE, the rules for
semantic functions, one worked example) and: the question, the end of its own reasoning (`build_reasoning_chars`), its
answer, the expected answer, and up to four existing function descriptions closest to the question (for `do()` reuse).
Thinking budget `build_thinking_budget`, streamed.

**FR-ARC-12 Checks, each failure fed back for the next round (`build_rounds`).**
1. The three sections parse; PARAMS has at least one line `name | type | description | value`.
2. Template check: every placeholder declared and every declared one used; every value written in the question; the
   template filled with the values equals the question (spaces and end punctuation aside). On failure, the template
   is rebuilt from the question by replacing each value written exactly once; the rebuilt one must pass.
3. `describe` in the sandbox: no forbidden construct; exactly one root; every function narrates.
4. Links: each `do()` description resolves to a function of this entry (word overlap >= 0.5), else to another entry's
   function (embedder cosine >= `link_similarity`, else word overlap >= 0.6), else the round fails.
5. The root on the original values returns the expected answer (FR-ARC-07 comparison).
6. Robustness: for each numeric parameter, two other values (int: n+1, 2n; float: 1.5x, x+1) must run without error
   (a division by zero or a non-integer for an int input is allowed).
7. Round trip: the original question must hit the new entry with its own values (template tier, else System 1).

**FR-ARC-13 Registration.** The entry gets the question's vector and one vector per function description, status
candidate (trusted when `trust_after = 0`), and joins `cache.json` (written atomically). Past `max_entries` the entry
with the fewest hits that no other entry calls is evicted.

## 4. Compilation

**FR-ARC-20.** `POST /v1/arc/compile {"all": bool}` compiles trusted entries (and candidates with `all`) into one
standard-library module `data/arc/arc_cache.py`: every semantic function becomes a named function whose `load(ctx, d)`
calls are parameters, `do()` calls direct calls with payloads bound to the callee's parameters by description, and
`load(result, d)` a literal key of the callee's returned dict (else a word-overlap pick). The module carries each
entry's template and parameters: `python3 arc_cache.py "<question>"` answers a template match with no model;
`--entry <id> '<json>'` runs one entry on values by description.

## 5. API

| Method and path | Purpose |
|---|---|
| `GET /v1/arc` | settings, counts, entries (summaries), queue |
| `GET /v1/arc/{id}` | one entry with code, narration and its floor |
| `DELETE /v1/arc/{id}` | delete |
| `POST /v1/arc/{id}/status` | `{"status": "trusted" | "candidate" | "quarantined"}` (human review) |
| `POST /v1/arc/build` | `{question, answer, reasoning?, expected?, domain?}` or `{reply_id}`; `wait: true` builds now |
| `POST /v1/arc/try` | `{question, floor?}`: the cache alone, trust ignored, no fallback |
| `POST /v1/arc/compile` | write the compiled module |

`ChatRequest.arc: false` turns the cache off for one request. Metrics: `nafs_arc_total{result}`,
`nafs_arc_shadow_total{agree}`, `nafs_arc_builds_total{ok}`, `nafs_arc_queued_total`.

## 6. Configuration `[arc]`

`enabled`, `serve`, `build`, `dir`, `domains`, `admit`, `trust_after` (2), `min_similarity` (0.80),
`serve_similarity` (0.85), `link_similarity` (0.80), `model_extraction`, `build_rounds` (3),
`build_thinking_budget` (1024), `build_max_tokens` (3072), `build_reasoning_chars` (2500), `poll_s` (20),
`queue_cap` (64), `max_entries` (500), `run_timeout_s` (10). Shipped on, math only.

## 7. Security

Entry code is model-written and runs only in the verifier sandbox (macOS: seatbelt with no network and no file access
outside the run directory; Linux: see the README's sandbox note). Static refusal (FR-ARC-05) is defence in depth, not
the boundary. The compiled module is meant for inspection and edge deployment; it refuses the same constructs at
compile time. Entries never hold personal data beyond the question they were built from; deleting an entry removes it
from `cache.json` at once.

## 8. Tests

Unit (`arc::tests`, no model): template synthesis and ambiguity, template and shape matching, value guards,
answer comparison precision, draft parsing and template check, extraction parsing, floor learning, promotion and
quarantine, eviction keeping callees, sandbox run with semantic calls, forbidden code, compile and run without a model.
Live: `scripts/arc_bench.py` (build, test, latency, compile, report) over `benchmarks/arc_phase38/items.json`.
