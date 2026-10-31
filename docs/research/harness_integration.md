# Harness integration: Open Math Harness and Codex-RS in Nafs-Aql

Date 2026-10-24. Status: **implemented** (crate `nafs-harness`, wiring in `nafs-verify`, `nafs-core`, `nafs-gateway`).
Decision record: [005](../decisions/005-harness-integration.md). Sources assessed:

| Harness | Revision | Licence | Size |
|---|---|---|---|
| Open Math Harness ("Nucleus", Open Math OS runtime) | `687f2a2` (2026-10-23), release 0.41.0 | Apache-2.0 | ~112k lines of Rust, Linux + bubblewrap only |
| OpenAI Codex CLI, `codex-rs` | `970111f6` (2026-10-23) | Apache-2.0 | ~150 crates; `core` alone ~259k lines |

## 1. What each harness is

**Open Math Harness** is a capability microkernel (`kernel/`, in-memory, no I/O) with one confined process per
service, length-prefixed JSON IPC, and bubblewrap confinement. Its mathematics machinery sits in three services:

- **Lean verification** (`program-files/lean`, `interfaces/lean`). The theorem statement is *frozen* on the
  service side. A candidate is refused on text alone if it contains a forbidden token (`sorry`, `axiom`,
  `native_decide`, `#eval`, `macro`, `elab`, `IO`, `unsafe`, ...) or does not contain the statement byte for byte.
  Lean then checks a file built from the *frozen* preamble, a separator, the candidate body, an unguessable marker,
  `#print axioms thm` and `#check @thm` under `pp.all`. Only output after the marker counts; axioms outside
  `propext`, `Classical.choice`, `Quot.sound` reject; the elaborated type must be character-identical to the
  pristine statement's. Whether a proof holds is decided by Lean, never by a model.
- **Proof search** (`program-files/discovery`, `prover`). Candidates form a tree. Each node is scored
  (complete and verified 1.0, complete 0.9, sorries 0.5 + 0.4·closed/(closed+open), errors 0.05, invalid 0). The
  initial policy orders continuable nodes by (score desc, depth asc, id asc), drops invalid ones, reserves one slot
  for the best *repair prospect*, and fills the rest best-first. A worker's prompt shows the statement, the
  current file, the verdict, and its siblings as "already tried, do something different".
- **Ultramath** (`chat-agent/src/swarm`). Eight members at maximum reasoning effort each register a distinct
  direction (a model-based duplicate check refuses repeats), post findings to a shared board, and submit to a
  coordinator who accepts or rejects.

**Codex-RS** is a submission/event-queue agent: a turn samples the model, executes the tool calls of the
stream, records their outputs and samples again until a reply carries no tool call. The reusable parts are
leaves of that design:

- **`apply_patch`**: a small patch language (`*** Begin Patch`, `*** Add/Delete/Update File:`, `@@` context,
  `+`/`-`/space lines) with a lenient parser and `seek_sequence`, a four-pass fuzzy matcher (exact, trailing
  whitespace, surrounding whitespace, Unicode punctuation folded to ASCII). Errors go back to the model as text.
- **Exec**: a one-shot command with a deadline (exit 124 on timeout), the process group killed as a whole, and
  output reported as `Exit code / Wall time / Total output lines / Output`, **middle-truncated** (head and tail
  kept, `…N chars truncated…`).
- **Seatbelt**: `sandbox-exec -p <policy> -DNAME=/path ...`; paths are policy *parameters*, never interpolated
  into the policy text.
- **Context**: tool outputs truncated on entry to history; compaction near 90% of the window.

## 2. Fit with Nafs-Aql

Nafs-Aql already holds the rule both harnesses are built around (exact checkers decide; model text is data),
and it already runs a verifier-gated System 2 loop. What it lacks maps onto the harnesses directly:

| Gap in Nafs-Aql (evidence) | Source in the harnesses | What we take |
|---|---|---|
| No formal-proof verifier; maths is only checkable against a reference answer | OMH Lean service | Lean verifier (`verification.lean`), frozen statement, textual rules, marker-fenced axiom and type checks |
| A retry always repairs the **last** candidate, even when an earlier one was closer | OMH discovery tree + initial policy | Candidate tree with partial credit; the next round repairs the best node and sees its siblings |
| Code verdicts are all-or-nothing, so a policy cannot tell "1 of 8 asserts failed" from "nothing works" | OMH scoring (partial credit for open goals) | Per-assert credit from the sandbox runner, protected by the run's nonce |
| A-067: System 2 code time goes into full-budget retries that regenerate the whole program at ~6 tok/s | Codex `apply_patch` | Patch repair: a retry may answer with a patch to `solution.py`; whole code stays accepted |
| Verifier evidence keeps only the tail; the first failing assertion is often in the head | Codex output truncation | Head + tail truncation with an explicit marker |
| No agent that can run its own code before answering | Codex turn loop + OMH prover-agent stop rule | `POST /v1/harness/code`: a bounded run/patch/submit loop in the sandbox that ends only on the verifier's acceptance |
| Sandbox profile interpolates the scratch path into policy text | Codex seatbelt | Path passed as a `-D` policy parameter |

What we **do not** take, and why:

- The OMH kernel, platform, IPC and launcher: Linux + bubblewrap only, and Nafs-Aql is one process with an OS
  sandbox per untrusted run. The capability model is the right idea at platform scale, not at ours.
- Ultramath's eight-member swarm at maximum effort: on four cores at 6–8 tok/s one member is minutes; eight are
  not a product. We keep its cheapest idea, "siblings already tried", in the repair prompt.
- Codex's Responses-API client, tool router, MCP, approvals and compaction: Nafs-Aql speaks chat completions to
  llama.cpp, the models are 1.7B/4B, and a request is bounded by the wall-clock cap rather than a context window.
  Codex has no text-protocol fallback for models without native tool calls; ours is text-only (fenced blocks), as
  in OMH's fixed prover loop, because small models follow a single fenced block more reliably than a JSON schema.
- `execpolicy` (Starlark): the agent cannot run a shell at all, only Python files of its own workspace inside the
  existing sandbox, so there is nothing for a command policy to decide.
- The Lean REPL `analyze`/`complete` passes: they need leanprover-community/repl and Mathlib; one Lean process
  per check is enough for our request sizes.

## 3. Research anchors

- Best-first search over verified candidates is competitive with heavier tree search when the verifier is exact
  (BFS-Prover, ACL 2025, arXiv 2502.03438); OMH's initial policy is an instance.
- Verifier-guided self-correction with Lean's compiler feedback adds solves at equal sample count
  (Goedel-Prover-V2, arXiv 2508.03613). Nafs-Aql's evidence loop (FR-025) is the same mechanism.
- Edit formats: diffs cost far fewer output tokens than whole files, but weaker models fail diff formats more
  often (aider edit-format notes; Diff-XYZ, arXiv 2510.12487). Hence patch repair **falls back to whole code**,
  uses Codex's fuzzy matcher, and stays **off by default until measured** on the reference machine (rule 3,
  "measure, then keep").
- Self-Debugging (Chen et al., ICLR 2024) and SWE-agent's agent-computer interface (Yang et al., NeurIPS 2024):
  concise, bounded execution feedback is what makes small models repair code; hence Codex's exec format and
  middle truncation in every observation.

## 4. Implementation plan (as built)

1. **`rust/crates/nafs-harness`** (new, pure logic, no I/O except through injected closures):
   - `patch`: Codex `apply_patch` parser, `seek_sequence` and replacement computation, applied to an in-memory
     workspace (`BTreeMap<String, String>`), with path checks (relative, no `..`).
   - `truncate`: Codex middle truncation and the exec output format.
   - `lean`: OMH textual rules (comment/string blanking, forbidden tokens, statement extraction, preamble copy) and
     the verify protocol over an injected Lean runner, plus OMH's error classifier and repair feedback.
   - `search`: OMH candidate tree, scoring and initial policy, adapted to Nafs-Aql verdicts.
   - `agent`: a bounded code agent loop over an injected model and injected tools.
2. **`nafs-verify`**: Lean verifier in the hub (fails closed when Lean is not configured); per-assert credit for
   code tests; head + tail evidence; seatbelt `-D` parameter; a generic sandboxed program runner for Lean.
3. **`nafs-core`**: the System 2 loop keeps a candidate tree and repairs the best node (`deliberation.repair_policy`,
   default `best`, identical to the old behaviour when no partial credit exists); optional patch repair
   (`deliberation.patch_repair`, request `patch_repair`, default off); the code agent runner bound to System 2.
4. **`nafs-gateway`**: `POST /v1/harness/code`, `GET /v1/harness`; OpenAPI and config schema.
5. **Tests**: unit tests per module (ported Codex patch cases, OMH rule and policy cases), hub tests with a fake
   Lean, orchestrator tests on the mock backends, a gateway test, and a live run on the reference machine.
6. **Documents**: this note, decision 005, report section, README, STATUS, NOTICE.

## 5. Phase 11: the Codex agent harness, adapted to small models on a CPU

Date 2026-10-25/26. Status: **implemented** (`nafs-harness`: `grammar.rs`, `context.rs`, `diff.rs`, `agent.rs`;
host wiring in `nafs-core/src/harness.rs` and `backend.rs`). Decision record: [006](../decisions/006-codex-harness.md).
Report: section "Phase 11".

Phase 10 took leaves of Codex. Phase 11 assessed its **agent core** (`codex-rs/core`, revision `970111f6`) for the
mechanisms that make the loop fast and accurate, and ported those that survive the move to chat completions, a
1.7B/4B model and four CPU cores.

### 5.1 What the Codex core does (with sources)

| Mechanism | Where | What it does |
|---|---|---|
| Turn loop | `core/src/session/turn.rs` (`run_turn`, `try_run_sampling_request`) | Samples, records each output item, follows up while the response held tool calls; stops on a response without one |
| Dispatch at item done | `core/src/stream_events_utils.rs` (`handle_output_item_done`), `tools/parallel.rs` | A tool call is spawned the moment its item completes, while the model still streams; results are drained in order (`FuturesOrdered`) |
| Parallel gating | `tools/parallel.rs` | One `RwLock` per turn: read-only tools (`exec_command`, `view_image`, read-only MCP) share it, `apply_patch` and `update_plan` take it exclusively |
| Append-only history | `context_manager/history.rs`, `normalize.rs` | Items are only appended; state changes are new items; synthetic outputs are deterministic "to preserve prompt-cache reuse" |
| Incremental requests | `client.rs` (`get_incremental_items`) | When the new input extends the previous one, only the suffix is sent with `previous_response_id`; any property change forces a full request |
| Truncation on insertion | `history.rs`, `utils/output-truncation` | Tool output capped (models.json: 10,000 tokens; 4 bytes per token) as it enters the history |
| Compaction | `compact.rs`, `openai_models.rs` | At `min(config, 90 % of the window)`: a model-written handoff summary plus user messages up to 20,000 tokens replace the history |
| Freeform `apply_patch` | `core/assets/tools/apply_patch.lark`, `tools/handlers/apply_patch_spec.rs` | The only way to edit; the tool's input is constrained by a Lark grammar |
| Budget reminders | `rollout_budget.rs`, `session/token_budget.rs` | Remaining-budget notes appended to the history at thresholds |
| Turn diff | `turn_diff_tracker.rs` | The turn's net change as a unified diff, without rereading the filesystem |
| Retries | `responses_retry.rs`, `async-utils/src/backoff.rs` | 200 ms initial delay, factor 2, jitter 0.9-1.1; malformed calls go back to the model as failed tool output |

### 5.2 Measurements that shaped the port (reference machine, 2026-10-25)

- **Prefill matters.** System 2 (Qwen3-4B-Thinking-2507 Q4_K_M) prefilled a cold 1,969-token prompt in 64 s
  (31 tokens/s); the same prompt with one exchange appended took 3 s (21 tokens computed, 1,972 from the cache).
  Phase 10 rebuilt the task message (with the files) every step, so after any edit almost the whole prompt was
  recomputed.
- **A plain grammar breaks a reasoning model.** With `grammar` set, llama-server (797de63) constrains from the first
  sampled token, so Qwen3-4B-Thinking emitted the answer in place of its reasoning. A grammar that starts with the
  reasoning span (any text up to `</think>`) works, including with `thinking_budget_tokens` budget forcing.
- **Small models and patches.** Phase 10 (System 2): a unified-diff header, an update of a missing file, an update
  hunk without prefixes. First Phase 11 smoke run (System 1, grammar on, patch offered): well-formed patches that
  appended a second copy of the file. Hence the `auto` edit format.

### 5.3 What was ported, and how

| Codex | Nafs-Aql | Why the change |
|---|---|---|
| Append-only history | `Transcript` (append only); the observation carries the edited files and the evidence | llama-server's slot cache is a token-prefix match, like Codex's incremental requests |
| Model-written compaction | Harness-written: task + current files + last verdict + "already tried" list, same prefix as the first message | A model summary costs minutes at 6-8 tok/s; the facts are already structured |
| Lark grammar for `apply_patch` | GBNF per step: existing files only, no `@@ -n,m`, allowed actions, reasoning span first | llama-server takes GBNF; per-step specialisation removes whole classes of failure |
| Dispatch at item done | `stop_after`: the stream ends when the action is complete (unconstrained requests) | Nothing after the action is decoded; under the grammar the server already stops |
| Truncation on insertion (10k tokens) | 2 kB observations and evidence, 6 kB files | 10k tokens would take minutes to prefill here |
| Budget reminders | "Steps left: n"; last step offers no `run` | Cheap, appended (cache-safe) |
| Turn diff tracker | `diff.rs`: the run's net unified diff in the reply | Reviewable result |
| Turn abort | the run's cap enforced inside the model call (`timeout_at`) | Phase 10 overran a 900 s cap by one 1,041 s call |
| — (Nafs-Aql's own) | verdict memo; `edit_format = "auto"`; relative paths in tracebacks | Accuracy and tokens for small models |

Not ported: parallel tool calls (one action per reply; the verifier runs after every change, so a second action would
mostly duplicate it), websocket reuse and remote/async compaction (server state llama.cpp lacks), MCP and approvals
(the agent has no shell).

### 5.4 Research anchors

- Prefix caching: SGLang's RadixAttention (Zheng et al., NeurIPS 2024) and vLLM's automatic prefix caching rest on the
  same invariant as Codex's incremental requests: never rewrite what came before.
- Constrained decoding is cheap on the server (Outlines, Willard & Louf 2023; XGrammar, Dong et al. 2024) but
  constraining the *reasoning* can cost accuracy (Tam et al., "Let Me Speak Freely?", EMNLP 2024 Industry), hence the
  free reasoning span.
- Edit formats: weaker models fail diff formats more than whole files (Aider's edit-format notes; Diff-XYZ, arXiv
  2510.12487).

Results: section 5.5 below and `benchmarks/harness_ab.json`.

### 5.5 Results (interleaved live A/B, shared host)

| Round | Model, tasks | Arms | Verified | Calls | Malformed steps | Wall |
|---|---|---|---|---|---|---|
| 1 | System 1, 10 | Phase 10 loop / Phase 11 / Phase 11 patch-always | 2 / 4 / 0 | 50 / 41 / 60 | 23 / 1 / 16 | 356 / 504 / 1,468 s |
| 1 | System 2, 3 | Phase 10 loop / Phase 11 | 0 / 3 | 15 / 3 | 13 / 0 | 2,695 / 451 s |
| 2 | System 1, 10 | `unstick` off / on | 5 / 7 | 35 / 27 | 0 / 0 | 523 / 305 s |
| 2 | System 2, 6 | 512 / 256 reasoning tokens per step | 6 / 6 | 7 / 14 | 0 / 5 | 718 / 858 s |
| 3 | System 2, 6 | 512 every step / 256 first step then 512 | 6 / 6 | 6 / 6 | 0 / 0 | 560 / 282 s |

What the data say, and what they do not:

- **Accuracy came from the action language.** Under the Phase 10 loop most steps of both models were unparseable or
  unappliable; under the grammar almost none were. The edit policy matters as much: offering patches at every step
  verified nothing on System 1, because its patches applied and still broke the file.
- **Prefill was not the bottleneck it looked like.** llama-server's `--cache-reuse 256` and host-memory prompt cache
  served 75–85 % of the Phase 10 loop's prompt tokens anyway; the append-only transcript raises that a little
  (and to 93–100 % on System 2 in round 3, with the warm-up, partly because both arms send the same task message), but decoding is where the time goes.
- **Speed came from fewer and shorter calls**: one call per task instead of six, nothing decoded after the action, and
  a cheap first reasoning budget. A verified System 2 fix of a small function now takes 33–59 s on this host.
- Samples are small (10 and 6 tasks, small single-file functions) and the host was shared: verified counts are
  indicative, not significant. Generated report tables: `docs/report/sections/harness_ab_tables.tex`
  (`scripts/gen_harness_ab.py`).
