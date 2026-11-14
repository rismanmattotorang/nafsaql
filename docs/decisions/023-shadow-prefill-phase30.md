# Decision 023: Keep the other model warm between turns, with a stable prompt beginning (Phase 30)

Date 2026-11-14. Status: **adopted.** Research [`shadow_prefill.md`](../research/shadow_prefill.md). Specification
[`shadow_prefill_spec.md`](../spec/shadow_prefill_spec.md). Measurements: `benchmarks/engine_phase30/` (quiet machine,
load 2 to 7). After Qiu et al., *ShadowPrefill* (arXiv 2611.13176, 2026).

## Decisions

1. **The engine has a shadow class** (Smiley Core): a request that only reads its prompt into the cache, at the lowest
   priority, never decoded, taken over by a real request that shares its beginning, stopped when its caller goes, and
   kept to a duty cap while it is the only work. Server: `"shadow"` on a chat completion; messages may end with the
   assistant's turn. A test against the real model checks that a warm request answers exactly as a cold one.
2. **Between turns, the model that did not answer reads the conversation** (`[shadow] between_turns = true`), once
   nothing has been in flight for 1.5 s; the next request stops it, and the server keeps what it read.
3. **The prompt keeps a stable beginning** (`[shadow] stable_prefix = true`): the system prompt is the assistant's own
   text alone; the domain instruction, earlier answers, passages and notes go before the question. Without it the
   system prompt changes with every question and neither model can reuse a conversation it read.
   - Together (four-turn sessions, three rounds): the switch to the reasoning model took 2.0 s to its first token
     against 36.6 s today (1.5 to 2.3 s in every round; 34 tokens read instead of 493). Either change alone did not
     help (28.0 s and 41.3 s).
   - The stable layout also lets the quick model reuse its own conversation (a third turn read 165 to 182 tokens
     instead of 333 to 394).
   - Answers from documents are unchanged: 42 of 50 right with either layout, 2 discordant pairs each way.
4. **No shadow during the draft** (`at_arrival = false`, built and available): on four shared cores it made every
   draft 75 % slower (19.0 to 33.3 s) to save 46 s on the quarter of questions that escalate.
5. **Not ported:** the paper's SLO-slack controller (there are no concurrent users with token deadlines here), the
   host-memory tier (the prefix store already lives in host memory), switch-probability ordering (escalation is
   predicted no better than chance from the question alone, Phase 26), and the disaggregated variant.
6. Settings → Speed lets a person turn the shadow and the stable beginning off or the arrival shadow on.

## Validation

539 Rust tests pass (including a shadow take-over test against the real model); `cargo fmt --check` and `cargo clippy
-D warnings` (all targets, local and the newer nightly clippy) are clean. Live sweep on the final build with the new
defaults, 2026-11-14 (load 4 to 8): contract passed, security 26/26, Rooms 10/10, end to end 74/74, knowledge 55
passed (1 skipped), knowledge security 33/33, harness 53/53, acceptance 9/9 (first token after 136 ms; a verified
reasoning answer in 53 s against a 240 s target).
