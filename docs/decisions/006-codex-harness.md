# Decision 006: Codex-style agent harness for the code agent (Phase 11)

Date 2026-10-26. Status: **adopted** as the default of `POST /v1/harness/code`; the Phase 10 loop stays selectable
(`harness.context = "reset"`). Research note: [`docs/research/harness_integration.md`](../research/harness_integration.md)
section 5. Report: section "Phase 11". Evidence: `benchmarks/harness_ab.json` (and `_round2.json`), generated
table `docs/report/sections/harness_ab_gen.tex`.

## Context

The owner asked for a Codex-like harness with inference speed and accuracy as the goals. Phase 10 had taken leaves of
Codex (patch format, truncation, sandbox parameters) but kept a simple loop whose live run lost most of its time and
steps to re-prefilled prompts, malformed actions and a model call that overran the run's cap.

## Decision

| Setting | Default | Mechanism (Codex source) |
|---|---|---|
| `harness.context` | `"append"` | append-only transcript with harness-written compaction (`context_manager`, `compact.rs`) |
| `harness.compact_at_tokens` | 2600 | compaction threshold, leaves room for a reply in a 4,096-token System 1 slot |
| `harness.constrain` | `true` | per-step GBNF action grammar with a free reasoning span (`apply_patch.lark`, freeform tool) |
| `harness.stream_cut` | `true` | end the stream when the action is complete; only for unconstrained requests (dispatch at item done) |
| `harness.edit_format` | `"auto"` | whole file while `solution.py` is missing or at most `whole_max_lines` (80) lines; patch or whole file above |
| `harness.unstick` | `true` | after a step that reproduced judged files: compact and sample more diversely (+0.3 per repeat) |
| `harness.first_step_thinking_budget` | 256 | System 2 reasoning tokens on the first step; `thinking_budget` (512) after a failure |
| warm-up | on | the agent's system prompt is primed in the server cache at startup |
| always | | verdict memo, steps-left reminders, no `run` on the last step, no `SUBMIT`, run cap inside the model call, relative paths and `got <value>` in evidence, the run's net diff |

## Evidence (interleaved live A/B, shared host; wall times are not speed claims)

System 1 (Qwen3-1.7B), 10 tasks, 6 steps, arm order rotated:

| Arm | Verified | Calls | Malformed steps | Cache share of prompt tokens | Decoded tokens | Wall |
|---|---|---|---|---|---|---|
| Phase 10 loop | 2/10 | 50 | 23/50 | 75 % | 2,793 | 356 s |
| Phase 11, `edit_format = "auto"` | 4/10 | 41 | 1/41 | 78 % | 4,146 | 504 s |
| Phase 11, patch at every step | 0/10 | 60 | 16/60 | 76 % | 10,706 | 1,468 s |

- The grammar removes unparseable replies (23 of 50 Phase 10 steps were prose or `SUBMIT`).
- Offering patches at every step is the worst arm: patches applied and still broke the file (doubled `+` prefixes, a
  mangled `End Patch` line inside the file, a file replaced by `python check.py`). Hence `"auto"`.
- Verified 4 vs 2 rests on two discordant tasks (exact McNemar p = 0.5): indicative, not significant.
- Prefill was not System 1's bottleneck: llama-server's `--cache-reuse 256` and host prompt cache already served 75 %
  of the Phase 10 loop's prompt tokens.

System 2 (Qwen3-4B-Thinking-2507), 3 tasks: Phase 11 verified 3/3 in 3 calls (451 s); the Phase 10 loop 0/3 in 15
calls (2,695 s; two runs ended at the 900 s cap, now enforced inside the call), 13/15 steps malformed.

Round 2 (`benchmarks/harness_ab_round2.json`, build with `got <value>` evidence):

| Comparison | Arm A | Arm B |
|---|---|---|
| System 1, 10 tasks, `unstick` | on: 7/10 verified, 27 calls, 305 s | off: 5/10, 35 calls, 523 s |
| System 2, 6 tasks, reasoning tokens per step | 512: 6/6, 7 calls, 718 s | 256: 6/6, 14 calls, 858 s, 5/14 malformed |

- `unstick` stays on (both discordant tasks favour it). `thinking_budget` stays 512: 256 halves one-call tasks but
  costs more calls elsewhere and more time overall.
- The 256 arm's malformed steps were Codex patches written *inside* the ```python block (allowed by the grammar then);
  the grammar now forbids code lines that start with `*`, so `"auto"` is enforced by the grammar.

Round 3 (`benchmarks/harness_ab_round3.json`, grammar without patches in code blocks, agent-prompt warm-up), System 2,
6 tasks: 512 reasoning tokens every step 6/6 in 6 calls, 560 s; 256 on the first step then 512: 6/6 in 6 calls, 282 s.
Hence `first_step_thinking_budget = 256`. The escalation to 512 was not exercised (every task passed on the first
step). The warm-up made the first call of the round find the agent prompt cached (207 tokens, against 5 in round 1).
The escalation was exercised in the end-to-end check of the final build (System 2, `fix_merge_intervals` with
defaults): the 256-token first step failed with `got [[5, 6]]` in the evidence, the 512-token second step verified.

## Re-open / flip conditions

- `edit_format = "any"`: only with a model that applies patches reliably (an A/B where the patch arm verifies at least
  as many tasks).
- `context = "reset"`: if an A/B on a quiet host shows the append transcript losing verified tasks.
- `first_step_thinking_budget = 0`: if harder tasks show the cheap first step costing more calls than it saves.
- The defaults are indicative at n = 10 (System 1) and n = 6 (System 2); a 30-task interleaved run on a quiet host is
  the confirmation step.
