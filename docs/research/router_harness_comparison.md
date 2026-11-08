# vLLM Semantic Router and DeepSeek Harness read against Nafs-Aql: the gate and the agent loop

2026-11-08. Sources: `third_party/semantic-router` at 029bf54ed (vLLM Semantic Router, Go, Apache-2.0) and
`third_party/deepseek-harness` at fcc0ad4621 (DeepSeek Harness `dsh`, TypeScript, MIT), both read in full by two
read-only deep reads against our `nafs-core` gate and `nafs-harness` crate. Written for Phase 22. Companion
specification: [`router_harness_upgrade_spec.md`](../spec/router_harness_upgrade_spec.md).

Two different comparisons, so two halves. The router half asks *how a request is sent to a model and at what price*;
the harness half asks *how a task is driven to a verified end*. In both, the honest finding is the same shape as the
OntoSmith read: we are ahead on the part that decides correctness, and behind on a handful of mechanisms that cost
nothing to port and that we had simply not built.

## Part A — the agent harness

### A.1 What `dsh` is

A ReAct loop over an **event-sourced session log**. Every model request is derived from the log (`deriveMessages()`
over a "surface" node list), and the log's model-visible surface changes only by `append` or by a
`replace{startSeq,endSeq}` operation, so any prompt the model ever saw can be rebuilt byte for byte by a detached
reader ("model-visible means logged"). Everything else — the model adapter, tools, sandbox, compaction, plan mode,
subagents — is a Cordis plugin on a shared context, patched by layered YAML.

### A.2 Where we are already ahead

The deep read's summary line is the comparison in one sentence: *dsh has no verifiers, no search, no rollback, no
compile gate, no grammar-constrained decoding and no step limit; its benchmarks measure host overhead, not task
success.* Specifically:

| | `dsh` | `nafs-harness` |
|---|---|---|
| When a run ends | The model's own claim (`update_goal complete`); `defaultMaxGoalRounds 256`; **no maximum steps per turn** anywhere in the loop; the Ralph mode's README says its reports "are not independently verified" | Only when an exact verifier accepts (unit tests, Lean, JSON Schema) or the budget is spent; never because the model says it is done |
| Candidates | One trajectory | A best-first tree scored by the verifier, with per-assert credit on code; a cascade of four cheap candidates judged before the expensive model runs (−26 % wall time, Phase 14) |
| Edits | Literal `old_string`/`new_string` that must match exactly once; `apply_patch` exists and is **off** (`applyPatch: false`) | Codex's `apply_patch` with its fuzzy matcher, as a per-step grammar that keeps the reasoning span free; a whole-file format for short files |
| Bad edits | Applied | A compile check before the verifier (37 ms), rollback to the best version, a stall guard that ends the run `stuck` on the fourth identical failure |
| Sandbox | bwrap `--ro-bind / / --unshare-pid` with **no network unsharing**; "network is outside the policy vocabulary"; in-process file tools confined by path checks, not the OS; read-before-edit observation **disabled by default** | Network denied, CPU/memory/wall limits, namespaces plus seccomp on Linux, Seatbelt on macOS; 26 live escape probes |
| Measurement | Benchmarks of Node CPU, heap and I/O with no model (e.g. 100 turns / 800 reads under 1,125 ms); no task-success numbers | Interleaved A/Bs on verified task counts, published with the ones that did not pay |

### A.3 What is genuinely better there

Five mechanisms, each pure algorithm, each cheap on a CPU because none of them spends a model call:

1. **Two-phase compaction.** At `0.8 × window` a **model-free pruner** runs first (tool results over 8,192 chars keep
   4,096 of head and 1,024 of tail), and only if the transcript is still over the threshold is a summary written;
   the cut never separates a tool call from its result (`toolPairingBalancedBefore`), and a summary that is not
   smaller than what it replaces is rejected. Ours compacts in one step into a fresh task message when the
   transcript passes `compact_at_tokens`, so a long test log takes the whole transcript down with it.
2. **An anchored token meter.** The transcript's size is the provider's **reported** `prompt_tokens` of the last call
   with an identical request envelope, plus a signed heuristic delta (chars/4 with block and role overheads) for
   what changed since. Ours is bytes/4 over the whole transcript. On a CPU where prefill is the cost, the
   difference between an estimate and the server's own count is the difference between compacting at the right
   step and the wrong one.
3. **Deterministic repair of a failed step.** A step that throws appends a synthetic `tool/result` for every
   unanswered call with a fixed wording (`TOOL_NOT_STARTED`: "retry it if it is still needed";
   `TOOL_OUTCOME_UNKNOWN`: "retry only if the operation is read-only or idempotent"), partial output is kept marked
   `interrupted`, and **every tool call is dropped from a message that ended on `max-tokens`**. We record a
   `model_error` action; we do not say what the model should conclude from it, and a reply cut at the token
   budget is parsed as if it were whole.
4. **Spill to file with a head-and-tail preview.** A tool result over `maxInlineTokens 12,500` goes whole to a
   0600 file and the model gets a preview and the path. We middle-truncate to `max_observation_bytes` and the rest
   is gone; a test log's middle can hold the one assertion that matters.
5. **A repeat-call chain.** The same tool with the same canonical arguments is counted (thresholds 3, 5, 8), a
   gentle reminder first and a detailed one naming the call and its arguments after, **denied calls counted too**,
   reset on any user message. Our stall guard reacts to the same *failure*; theirs reacts to the same *attempt*
   before it fails, and says which one.

Close behind: prefix stability as a discipline — the runtime-context message is appended **only when it changed**,
tool-catalogue changes become developer messages rather than a rewritten prefix, and the compaction summary is
requested as an extension of the last request so it reuses the cache. We built the first of these in Phase 11; the
other two are worth copying as rules.

### A.4 What not to copy

- **The plugin architecture.** Two of its four postmortems are composition bugs (`export default apply` dropping
  `inject`; a `!!js` expression in a `disabled:` field silently never evaluated, so the filesystem tools were off and
  the snapshot refresh recorded the failure as expected output). In Rust: fixed traits, a fixed pipeline.
- **Model-claimed completion and unbounded loops.** We have verifiers and budgets; self-report is strictly weaker.
- **Model-driven control** — a model-written summary as the main compaction, a model judging tool safety, workflows
  of up to 1,000 subagents. On four cores with a 1.7B/4B model each of those is a whole decode spent on a weak
  judgement.
- **`dsh_session_log`**, on by default, which uploads the session log suffix (up to 8 MiB per request) to the
  provider. Nothing leaves this machine.

## Part B — the router

### B.1 What vLLM Semantic Router is

A decision layer in front of many models: a request yields **signals** (twenty declared types — keyword, embedding,
domain, complexity, jailbreak, PII, fact-check, reask, context length, and typed questions answered by a "Decision
model"), Boolean **rules** over those signals pick a **decision** (a tier, a priority, a set of candidate models, an
algorithm and plugins), and the algorithm picks the model. It ships for Envoy and Kubernetes but has a native HTTP
front and the routing core is pure Go.

The deep read's first finding is that **the paper lags the code in both directions**: the code has three-valued
rule logic, tiers, projections and a calibrated cascade the paper does not mention, and the paper describes fuzzy
evaluation, coverage analysis, logic minimisation and a mean-confidence AND that the code does not do. Its own
`bench/README.md` says `bench/` holds harnesses, not results, and every `results/` directory is git-ignored; the
paper's latency and cache tables have no script or dataset in the repository.

### B.2 Where we are already ahead

| | Semantic Router | Nafs-Aql's gate |
|---|---|---|
| The decision rule | Lexicographic and hand-tuned: tier, then priority, then evidence; inside a decision a chosen algorithm (`multi_factor` min-max utility with 0.25 weights, `latency_aware` without exploration, `automix` `quality − cost/10·tradeoff`). The paper's `argmax Quality − λ·Cost` is "a framing only; no selector implements λ as a price". | One rule derived from expected loss, `L·(ê₁ − ê₂) > λ·Δc`, with λ a price of latency in real units, ê₁ a calibrated estimate promoted only at ECE ≤ 0.05, and a budget mode that solves for λ* from a target escalation rate. |
| What decides correctness | **No exact verifiers in production.** Seven verifier kinds are declared; only AutoMix, faithfulness and peer-consistency exist, and the latter two never approve. `self_verify` asks the same model to rate its own answer 0–1. The reference AutoMix verifier returns `random.uniform(0.5, 0.9)` when `transformers` is missing. | Exact verifiers (numeric and SymPy, unit tests in a sandbox, JSON Schema, Lean) whose verdict is final; agreement is a flag, never proof; measured true- and false-accept rates per verifier. |
| The cascade's stop rule | `confidence` looper: stop at the first response over a threshold; the default `avg_logprob` gate is `clamp((avgLP+3)/3, 0, 1)` against threshold −1.0, "effectively a no-op unless a threshold is set"; **if none pass, the last response is served, not the best.** | System 2 stops when the exact verifier accepts; when nothing passes the best-scoring candidate is returned, and the system can abstain. |
| The gate's cost on a CPU | Measured by them: 79 ms with a 0.3B signal model on 12 vCPU, about 2.8 s with 0.8B on 12 cores, and **9.8 s median per request** for the knowledge recipe on 4 cores. The maintainers switched to the slower model "although the latency condition fails on CPU". | 0.08–0.12 s of gate time per escalated request, from features the draft already produced (log-probabilities, agreement, the verifier's verdict). |
| Semantic cache | Keyed on everything but the last user turn; **no quality gate on writes** — every 2xx is cached. | Retrieval memory stores verified answers only (FR-061). |

### B.3 What is genuinely better there

1. **Calibrated acceptance with certified per-bin risk.** The System One cascade accepts a cheap model's answer
   only if its top probability falls in a score bin whose **one-sided Clopper–Pearson upper bound** on the error
   rate, at `α / #bins` (Bonferroni over every pre-declared bin), is at or below `max_risk`; the calibration artefact
   is pinned by SHA-256 and its policy hash must equal the live policy's, so stale calibration refuses to load; an
   unmatched bin means "not accepted", never an uncalibrated success. Our calibrated mode promotes on ECE ≤ 0.05,
   which bounds the *average* miscalibration; a per-bin upper bound is the stronger, and cheaper, guarantee, and our
   exact verifiers supply the error counts for free.
2. **Three-valued rule logic with a declared unknown policy.** A signal that errors or times out makes its leaf
   *unknown*, AND/OR propagate it, and the decision declares `on_unknown: match | no_match | fail_request`. A
   probability is never ranked against a cosine: selection by confidence happens only when every candidate's score
   is of the same kind. Our gate treats a missing feature as its default silently.
3. **Contrastive prototype banks.** A rule's examples are embedded once, clustered at cosine 0.9 to at most 8
   medoids, and scored `0.75·best + 0.25·mean(top-2)`; **complexity** is the margin between a hard bank and an easy
   bank with hard/medium/easy boundaries; **reask** is cosine ≥ 0.8 to the previous user turn. Cheap priors from an
   embedding server we already run.
4. **Cache safety.** The polarity guard rejects a hit whose token set differs by at most two tokens when a negation
   cue or an antonym pair is on one side only; truncated queries are skipped; personalised, RAG and PII requests
   bypass the cache.
5. **Extractive context compression without a model.** BM25 over line chunks of `max(8, target/4)` tokens with a
   +0.35 boundary bonus, least-relevant blocks first, JSON compressed only in its string leaves, originals kept for
   15 minutes behind a retrieve tool.

One measured number is worth keeping as the comparison target: their cascade (Kai-0.6B → Vega-27B) scored 190 of
231 against 152 alone and 203 for the large model alone, escalating 106 of 231 (41 errors rescued, 3 introduced),
with the gate frozen at top-probability ≥ 0.606 on a separate calibration set; coverage target 70 %, achieved 54 %.

### B.4 What not to copy

- **A ≥ 0.3B model as the per-request gate.** On four cores it costs more than our small model's answer.
- **Model-judged cascades** — self-rated confidence, judge-written fusion, ReMoM, a random-fallback verifier,
  serving the last response rather than the best. No measured gain in the repository.
- **Hand-weighted min-max utilities.** Scores shift with the candidate set and there is no price of latency in real
  units; our explicit expected-loss gate is the rule we want.
