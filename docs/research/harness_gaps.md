# Modern agent harnesses and the gaps in Nafs-Aql's (Phase 16)

Date 2026-11-02. The question: what do the current harnesses and papers do that Nafs-Aql's code agent does not, which of those
differences matter on four CPU cores with 1.7B and 4B models, and which can be closed and measured here. Decision record:
[010](../decisions/010-harness-gaps.md). Report: section "Phase 16".

## 1. What was read, and how far to trust it

Primary sources were fetched and read: the SWE-agent paper (ablation table), the Agentless paper (selection ablation), the
JetBrains observation-masking paper, and Anthropic's context-engineering article. The rest comes from search results (vendor
documentation pages, blog posts, crate documentation) and is marked as such; a number from a vendor's own announcement is a
vendor number. ChatGPT's and Gemini's *models* are not inspectable; what is comparable is the open harness around them
(Codex, Gemini CLI, Claude Code) and the published reports (DeepSeek-V3.2). Where a claim is only from a secondary source, the
table says so.

## 2. What modern harnesses do

| Technique | Who | What it is | Evidence found |
|---|---|---|---|
| **Edit-time guardrail** | SWE-agent (ACI) | after each edit the file is checked (flake8 with a fixed set of syntax, indentation and undefined-name codes); a failing edit is **reverted** and the agent sees the error, the proposed result and the original | primary: SWE-bench Lite, GPT-4 Turbo: 18.0 % with the check, 15.0 % without, 10.3 % with no edit command |
| **Context management** | Anthropic (compaction, tool-result clearing, notes, sub-agents), Claude Code, JetBrains | compact at the limit; drop old tool results ("the safest, lightest-touch compaction"); write notes outside the window; isolate exploration in sub-agents that return 1-2 k tokens | primary (article); JetBrains: masking old observations (window of 10 turns) **halves cost** with a solve rate equal to LLM summarization across five model configurations; vendor: context editing -84 % tokens on a 100-turn task |
| **Verifier-based selection** | OpenHands critic, SWE-Gym, Agentless | sample several candidates and pick by a verifier, a critic model, or generated tests | primary (Agentless): majority voting 25.67 %, + regression tests 27.00 %, + reproduction tests 32.00 % on SWE-bench Lite; search results: OpenHands critic Best@8 73.8 % against 57.9 % random on the mixed-outcome subset |
| **Search over trajectories** | SWE-Search (MCTS) | backtrack to a better node instead of continuing from a worse one | search result: +23 % relative over the same agents without search |
| **Checkpoint and restore** | Gemini CLI | a snapshot before every file-modifying action, restored on demand (`/restore`) | documentation (search result) |
| **Persistent project instructions** | Codex (AGENTS.md), Claude Code (CLAUDE.md), Gemini CLI (GEMINI.md, hierarchical) | the conventions of a project travel with every task | documentation (search results) |
| **Stall detection** | loop-guard libraries and agent runtimes | key on tool name **and** arguments **and** result; repetition warns, repetition with stagnant results stops | engineering sources only (crate documentation); no study found |
| **Reasoning persistence** | DeepSeek-V3.2 | reasoning is kept across tool calls until a new user message arrives; the tool-call history survives when reasoning is dropped | primary (report), as quoted by search results |
| **Sub-agents, plan mode, hooks** | Claude Code, Gemini CLI | isolated contexts, read-only planning, deterministic pre-tool hooks | documentation (search results) |
| **Constrained decoding** | ToolDec, NVIDIA's Bash work | a grammar removes malformed calls; small models gain most (62.5 -> 75.2 % over 13 models), with a reported validity-versus-correctness trade-off below 3B | search results; a "constraint tax" paper warns about sub-3B models |
| **Edit formats** | Aider, Diff-XYZ | the format the model must write matters more than its size: search/replace against udiff against whole file; small models stay limited whatever the format | search results |

## 3. What Nafs-Aql had

The code agent (Phases 10, 11, 14) already had: an append-only transcript that keeps the server's prompt cache valid; compaction
written by the harness; an action grammar (Codex's patch grammar) for hosts that can constrain sampling; a whole-file edit policy for
short files; Codex's patch parser; an **exact** verifier (hidden tests in a sandbox) after every change, which is a stronger
selection signal than any critic model or generated test; the cascade (four quick candidates judged by the verifier, the thorough
model only when none passes); retries with Codex's backoff; a repeat guard that notices a workspace the verifier already judged;
budget reminders; a wall-clock cap that holds inside a model call.

## 4. The gaps

| Gap | Nafs-Aql before | Decision |
|---|---|---|
| Edit-time guardrail | a file that does not compile went to the verifier like any other: one sandbox run and one whole step per syntax error, and the same `IndentationError` came back step after step in the Smiley Core runs | **built** (section 5.1) |
| Checkpoint / backtrack | the next step built on the files of the last, even when they scored lower than an earlier version | **built** (5.2) |
| Stall detection | only an identical *workspace* was noticed; the same *failure* from different code, or a reply with no action, repeated until the step budget or the 900 s cap (one agent run spent 900 s that way) | **built** (5.3) |
| Project instructions | the task text only | **built** (5.4), small |
| Verifier-based selection | **ahead**: an exact verifier, with candidates judged in the cascade | none needed |
| Context management | compaction rewrites the transcript into the task plus a one-line outcome per attempt, which is stronger than masking; the runs are at most 12 steps, inside the 10-turn window where masking changes nothing | none: no gap at this scale |
| Self-generated tests (Agentless's reproduction tests) | a task without hidden tests is not accepted by the code agent | **not built**: reproduction tests work against an *unpatched* bug that fails them; for a new function there is no such oracle, and a test written by a 1.7B model would raise the *label* ("verified") without raising the trust |
| Reasoning persistence (DeepSeek-V3.2) | each step's reasoning is dropped: the harness sends the reply's final text only, and the observation as an ordinary user message, which the Qwen3-Thinking template (read from the model file) treats as a new query, so it would drop earlier reasoning anyway (it keeps an assistant turn's reasoning only after the last user query) | **not adopted now**: V3.2 keeps it across the tool calls of one turn. Doing the same here means sending observations as `<tool_response>` messages and the reasoning back, and a kept 500-token span costs about 25 s of prefill per step at the 20 tokens/s measured; named as the next experiment |
| Sub-agents, plan mode, hooks | the cascade is the sub-agent idea at this scale (isolated candidates, a condensed result); the sandbox is a hook that cannot be skipped | none: tasks are one function in a flat workspace |
| Critic model | none | not built: the exact verifier is the critic; a trained critic is for tasks without one |

## 5. What was built

All four are in `nafs-harness` (the loop, tested with a scripted host) and `nafs-core` (the host); each is a request field and a
configuration key, on by default, and reported in the reply.

1. **Compile check before the verifier** (`harness.lint`, `AgentEnv::lint`). After a patch or a whole-file write, every edited
   `.py` file is compiled in the sandbox (`compile()`, never executed). On failure the edit is **not applied**, the files stay as
   they were, and the model is shown the error and the five lines around it (`IndentationError: unexpected indent (line 3)` with the
   offending line marked), as SWE-agent does. No verifier run is spent, and a cascade candidate that does not compile is not judged. The
   step is recorded as action `rejected`; `lint_rejections` counts them. A sandbox failure passes the edit on to the verifier: it is
   not the model's fault.
2. **Rollback to the best version** (`harness.rollback`). When a change is judged and scores below the best version so far, the
   files are restored to that version, the model is told what happened ("scored 0.05, below the best version so far (0.50): restored;
   do not repeat this change"), and the failed attempt stays in the transcript. `rollbacks` counts them; the step carries
   `rolled_back`.
3. **Stall guard** (`harness.stuck_after`, default 4). A step's signature is the hidden tests' evidence when the files were judged
   (whatever edit produced them), or the action and its first line of feedback when they were not. The second consecutive identical
   failure adds "the same failure as the previous step: try a different approach" to the observation; the `stuck_after`-th ends the
   run with status `stuck` and the best files, instead of spending the budget. A `run` step does not count and does not reset the
   count.
4. **Caller guidelines** (`guidance`, 2,000 characters). Shown with the task, before the request for an action, and again after a
   compaction: the role AGENTS.md, CLAUDE.md and GEMINI.md play.

Tests: five unit tests in `nafs-harness` (refusal and its cost, rollback, stall nudge and stop, guidelines) and an integration test in
`nafs-core` through the real sandbox (`code_agent_refuses_an_edit_that_does_not_compile`).

## 6. Measurements

All runs on the shared host (the owner's other workloads at load 5-15), Smiley Core, the same ten tasks (`scripts/harness_ab.py`: eight repairs
and writes of one function with hidden tests), arms interleaved with the order alternated per task: **p15** = the three new guards off
(`lint`, `rollback`, `stuck_after = 0`), **p16** = the production defaults.

| Set | Arm | Verified | Model calls | Wall time, sum | Decoded tokens | Guards fired |
|---|---|---|---|---|---|---|
| System 2, 6 steps (`harness_ab_p16.json`) | p15 | 10/10 | 17 | 747 s | 4,520 | - |
| | p16 | 10/10 | 17 | 772 s | 4,520 | 0 |
| System 1, 8 steps, round 1 (`harness_ab_p16_s1.json`) | p15 | 4/10 | 58 | 307 s | 2,418 | - |
| | p16 | 4/10 | 58 | 297 s | 2,362 | 0 |
| System 1, 8 steps, round 2, after the stall fix (`harness_ab_p16_s1_r2.json`) | p15 | 5/10 | 47 | 350 s | 3,315 | - |
| | p16 | 5/10 | 47 | 347 s | 3,315 | 1 stall stop (at the last step) |

What this shows, and what it does not:

- **The guards cost nothing.** Same verified counts and model calls, decoded tokens equal except for one System 1 task whose sampling differed
  (2,418 against 2,362 in round 1; equal in the other two sets), wall times within the host's noise (the same code ran in both arms for
  most steps: 747 vs 772 s on System 2 is the shared host, not the guards). The compile check costs
  **37 ms** per edit (35 ms of it is Python's own start-up), against steps of 5 to 230 s.
- **They did not change an outcome here either.** Lint rejected no edit and rollback restored nothing: on these tasks neither model
  produced a file that does not compile, and System 1 either solved a task or never improved on it. The benefit is on the failure modes the
  research names, which this set rarely produces; those are pinned by tests (section 5) and were seen in practice once: a Smiley Core
  agent run that spent its whole 900 s cap on repeated `IndentationError`s (Phase 15), which the unfixed loop could not stop.
- **The A/B found a defect in the stall guard.** Round 1's trace showed System 1 asking to run a file that does not exist six times in a
  row; the guard skipped `run` steps. It now counts them, hashes only the stable lines of the feedback (the step counter had made every
  signature different), and the "no such file" message lists the files that exist. Round 2 ran on the fix.
- **Run-to-run variation is real**: the same ten System 1 tasks verified 4 times in round 1 and 5 in round 2 in both arms (sampling), so a
  difference of one task is noise.

## 7. What this does not claim

- The comparison is with the *harnesses* of Codex, Claude Code, Gemini CLI and the published agents, from their documentation and
  papers; it is not a benchmark against ChatGPT, Gemini, Claude or DeepSeek as models, which would measure the models.
- SWE-agent's numbers are for GPT-4 Turbo on SWE-bench Lite and are quoted for the direction of the effect, not as an expected gain
  here: the tasks here are one function with hidden tests, and the models are 1.7B and 4B.
- Self-generated tests, reasoning persistence and sub-agents are named as gaps and not closed, with the reason in the table.
