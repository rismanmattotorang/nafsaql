# ShadowPrefill for Nafs-Aql (Phase 30)

Source: Qiu, Mahapatra, Fonseca, Russinovich, *ShadowPrefill: Keeping Your Next Model Warm for Per-Step Model Routing
in Agentic LLM Serving*, arXiv 2611.13176 (Microsoft Azure Research, 9 Nov 2026). Specification:
`docs/spec/shadow_prefill_spec.md`. Measurements: `benchmarks/engine_phase30/`. Decision 023.

## 1. What the paper does

A per-step router sends each call of an agent to one of two models. KV caches belong to one model, so a switch must
prefill the session's context before its first token: the token-weighted prefix hit rate falls from 80-89 % on
same-model calls to 44-65 % after a switch, and a switch call takes 1.5-3.2x as long. ShadowPrefill keeps a *shadow
job* per session on the model the router did not choose and fills it in capacity that normal traffic leaves unused:

| Mechanism | Paper | Section |
|---|---|---|
| Incremental shadow jobs | the shadow appends each call's input, and its output as it is generated, so the other model never re-reads what it holds | 3.1 |
| Sticky placement | each session keeps one instance per model; switch calls go to the instance with the shadow | 3.1 |
| Idle and SLO-slack harvest (C1) | shadow tokens fill idle iterations (capped at beta = 0.5 TPOT) and, at peak, each iteration's slack below every in-flight request's TPOT SLO (Algorithm 1) | 3.2 |
| Routed-call protection (C2) | an iteration that carries a routed call spends only a small credit (delta = 0.025 of decode time) | 3.2 |
| Shadow KV class (C3) | at most 25 % of the KV pool, free or idle-for-5 s blocks only, evicted first; host-memory spill tier | 3.3 |
| Ordering by switch likelihood (C4) | pending shadows run in order of predicted switch probability | 3.1 |
| Safety (C5) | shadow work never blocks; a switch never waits for a shadow; a partial shadow is used as far as it goes | 3.4 |

Results (simulation over 22 pairs, plus a 16-H100 vLLM prototype): switch-call latency -8.5 to -21.4 %, switch-call
TTFT -44 to -47 %, LLM-call latency -5.0 to -9.7 %, token cost -22.5 to -46.1 %, energy per request +0.4 %. The gain
grows with context length (0.7 % below 2K tokens, 20.5 % beyond 64K) and shrinks as load rises unless the host tier
keeps long shadows.

## 2. How it maps to Nafs-Aql

| Paper | Nafs-Aql |
|---|---|
| A per-step router between a small and a large model | The gate between the quick model (S1, Qwen3-1.7B) and the reasoning model (S2, Qwen3-4B-Thinking), per question |
| Instances serving normal traffic, with SLO slack | Two Smiley Core servers on the same four CPU cores; no other traffic |
| Idle capacity | The time a person reads an answer, and the time between conversations |
| Shadow KV class, evict first | The engine's prefix store (768 MB per model, least recently used); a shadow's cache enters it like any finished request's |
| Host tier | Not needed: the prefix store is already in host memory |
| Incremental shadow job | A shadow sends the conversation so far; the server restores the longest stored prefix and reads only the rest |
| Sticky placement | One server per model |
| Switch never waits; partial shadow used | A real request cancels a shadow that shares its beginning, and restores what it read |

The difference that matters is contention. The paper's shadows run on GPUs that are already powered and partly idle;
here a shadow and a draft compete for the same four cores and the same memory bandwidth, and escalation is rare
(about one reply in four in real chats, one in ten over all requests in the ledger).

## 3. What was built

See the specification. Engine: a shadow class (lowest priority, never decoded, taken over by a real request, stopped
when its caller goes, duty-capped). Server: `"shadow"` on a chat completion, open-ended conversations. Gateway:
between-turns shadow, arrival shadow, stable-prefix layout, per-request overrides, Settings → Speed.

## 4. Results

From `benchmarks/engine_phase30/` (decision 023):

| Question | Answer |
|---|---|
| What does a cold switch cost? | 46 s to the reasoning model's first token for a 925-token prompt; 25.8 to 51.9 s after three quick turns (median 36.6 s) |
| Does a shadow during the draft pay? | No: the draft took 75 % longer (19.0 to 33.3 s) to save 46 s on the quarter of questions that escalate |
| Does a shadow between turns pay? | Only with a stable prompt beginning: 2.0 s to the first token (34 tokens read, 493 to 526 restored); with today's layout 28.0 s, since the system prompt changes with every question |
| Does the stable beginning cost answers? | No: 42 of 50 answers from documents right with either layout, 2 discordant pairs each way (p = 1.0) |
| Anything else? | The quick model reuses its own conversation with the stable beginning: a third turn reads about half as many tokens |

The paper's gains are on GPUs with SLO slack under concurrent traffic. On one CPU the slack is the reader's time, and
the precondition the paper takes for granted, that a session's next prompt begins with its last, had to be made true
first.
