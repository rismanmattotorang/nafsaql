# Decision 010: close the code agent's gaps against modern harnesses (Phase 16)

Date 2026-11-02. Status: **adopted.** Research note: [`docs/research/harness_gaps.md`](../research/harness_gaps.md). Report: section "Phase 16".

## Context

The owner asked for research on modern harnesses (papers and open implementations) and a comparison with Nafs-Aql's code agent. The
research (SWE-agent, Agentless, observation masking, Anthropic's context engineering, Gemini CLI, Claude Code, DeepSeek-V3.2, SWE-Search,
OpenHands critics, loop guards) found four gaps that matter on four CPU cores, and several that do not.

## Decision

| What | Default |
|---|---|
| Compile check before the verifier (`harness.lint`): an edit whose Python file does not compile is refused, the files stay, the model sees the error and the lines around it | on |
| Rollback to the best version (`harness.rollback`): a change that scores below the best so far is dropped and the model is told | on |
| Stall guard (`harness.stuck_after = 4`): the second identical failure carries a nudge, the fourth ends the run as `stuck`; `run` steps count, the feedback is hashed without its step counter | on |
| Caller guidelines (`guidance`, 2,000 characters), the role of AGENTS.md / CLAUDE.md / GEMINI.md | per request |
| The "no such file" message lists the files that exist | always |

Each has a request field, a configuration key, a reply field and a metric.

## Evidence

Interleaved on the ten-task set, guards off against on (research note, section 6): identical verified counts (10/10 on System 2; 4/10
and 5/10 on System 1), identical model calls (decoded tokens equal except for one sampled task), wall times within the shared host's noise; the compile check costs 37 ms
per edit. The guards changed no outcome on this set because neither model produced a file that does not compile; they are justified by the
research and by the failure seen in Phase 15 (a 900 s cap spent on repeated `IndentationError`s), and pinned by five unit tests and an
integration test through the real sandbox. The A/B itself exposed a stall-guard defect (six identical `run` requests for a missing
file), fixed before round 2.

## Not adopted

- **Self-generated reproduction tests (Agentless):** they work against an unpatched bug that fails them; a new function has no such
  oracle, and a test written by a 1.7B model would raise the label without raising the trust.
- **Reasoning persistence across steps (DeepSeek-V3.2):** the Qwen3-Thinking template drops earlier reasoning unless tool responses are
  sent in its form; keeping a 500-token span costs about 25 s of prefill per step at 20 tokens/s. Named as the next experiment.
- **Observation masking, sub-agents, plan mode, hooks:** no gap at this scale (compaction already replaces observations with a
  one-line outcome; the runs are at most 12 steps; the cascade is the sub-agent idea; the sandbox is a hook that cannot be skipped).

## Re-open conditions

- A task set where the models produce files that do not compile, or regress: re-run the A/B to measure the guards' effect.
- A host that keeps reasoning cheaply (a faster prefill): try reasoning persistence.
