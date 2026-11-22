# Harness and agent UX for small local models: what the literature says (Phase 36)

Research of 2026-11-22. "Measured" means numbers in a paper or write-up that was opened; "guidance" means advice
without a controlled measurement. The fit is judged for a 1.7B and a 4B Qwen3 without grammar-constrained decoding, one
sample per request, 8K of context, 20 and 7 tokens a second.

## 1. Getting an action out of a small model

| Finding | Evidence | Source | Here |
|---|---|---|---|
| Let the model reason in prose, then give the action last; do not force the reasoning into a format | Measured: an 8B model on GSM8K 74.7 in free text against 48.9 under a JSON instruction; parse errors did not explain the gap | Tam et al., *Let Me Speak Freely?*, 2024, arXiv 2408.02442 | Already the shape of the agent's reply; keep it, never ask for JSON |
| Code goes in a fenced block, never inside a JSON string | Measured (charts): every model tested wrote worse code in a JSON tool call | Aider, *LLMs are bad at returning code in JSON*, 2024 | Already so |
| Few-shot examples are the largest lever for format below 4B, and the effect depends on the model | Measured: a 1.3B model's parsable output rose from 7 % to 89 % with three examples; another stayed at 0 % | Kavathekar et al., *Small Models, Big Tasks*, 2025, arXiv 2504.19277 | Candidate: one worked example in the constant system prompt |
| Fewer choices per turn | Measured: an 8B Q4 model failed with 46 tools and succeeded with 19 | Paramanayakam et al., *Less is More*, 2024, arXiv 2411.15399 | The agent has three actions; no change |
| One action per turn; an error message that says exactly what to fix; a bounded repair re-ask | Guidance (llama.cpp function-calling notes; Anthropic, *Writing tools for agents*) | | Candidate: the parser's messages; tolerant fences |
| Expect the 1.7B to be wrong about half the time on multi-turn calls | Measured: BFCL v3 56.6 (1.7B) and 65.9 (4B) with thinking | Qwen3 Technical Report, 2025, arXiv 2505.09388 | The harness must absorb malformed replies, not assume them away |

## 2. Edit formats

| Finding | Evidence | Source | Here |
|---|---|---|---|
| Whole-file output is what weak models get right | Measured: small local models produce 97 to 100 % well-formed whole-file edits on Aider's 133 exercises (a 1.5B: 100 % well formed, 31.6 % correct) | Aider edit leaderboard | The default for files up to 80 lines; keep |
| When a diff is needed, search and replace beats unified diff and patch formats for small models | Measured: exact match for a 1.5B model 0.20 with search/replace against 0.00 with unified diff; at 7B, 70 % of unified diffs parsed and 19 % applied | Glukhov et al., *Diff-XYZ*, JetBrains, 2025, arXiv 2510.12487 | **Candidate: a search/replace action for long files in place of the patch format**, which has never been measured here on a long file |
| Apply edits with layered, more permissive matching, and require a unique match | Measured: 9 times more edit errors without flexible patching | Aider, *Unified diffs*; Roo Code docs warn fuzzy matching can pick the wrong place | The patch matcher takes the first match; an ambiguous match must be refused |
| A lint gate inside the edit | Measured: 18.0 % against 15.0 % resolved without it | Yang et al., *SWE-agent*, 2024, arXiv 2405.15793 | Present (decision 010) |

## 3. Context under a small window

| Finding | Evidence | Source | Here |
|---|---|---|---|
| Masking old observations matches summarising them, at about half the cost | Measured: 24.6 % against 24.8 % on SWE-bench Verified with a 32B model | Lindenbauer et al., *The Complexity Trap*, JetBrains, 2025, arXiv 2508.21433 | Prune-then-compact exists and was inert on short tasks (decision 015) |
| Keep the last few observations whole; cap what a tool returns and say how to narrow it | Measured: last five observations 18.0 % against full history 15.0 %; a 100-line viewer beat 30 lines and the full file | SWE-agent | Observations are cut at 2,000 bytes with no way to read the rest |
| Context rot is strongest in small models | Measured: an 8B model's output degraded from about 5,000 words | Hong et al., *Context Rot*, Chroma, 2025 | Treat 8K as a ceiling; four passages stay the default |
| A stable, append-only prompt beginning | Guidance (Manus, 2025) | | Measured here in Phases 17, 30 and 34: it is the largest single saving on a CPU |

## 4. Control flow

| Finding | Evidence | Source | Here |
|---|---|---|---|
| Self-correction pays only with a strong verifier | Measured: a small self-verifier gave +0.3 to +0.4 points; an oracle +8.8 to +11.1 | Zhang et al., *Small Language Models Need Strong Verifiers*, 2024, arXiv 2404.17140 | The design's premise; reflect only on a verifier's signal |
| Retry and escalate match complex agent designs | Measured: retry 92.0 % and escalation 85.0 % at a fraction of the cost of search-based agents | Kapoor et al., *AI Agents That Matter*, 2024, arXiv 2407.01502 | The cascade (decision 008); keep |
| Spend thinking by phase | Measured: maximum reasoning throughout lost to timeouts (53.9 %) against high effort for plan and check only (66.5 %) | LangChain, *Improving deep agents with harness engineering*, 2026 | First-step budget 256, later 512 (measured, decision 006) |
| A check before finishing, a loop detector, time warnings | Measured together, not apart: 52.8 % to 66.5 % | same | Stall guard and repeat notice exist; the verifier is the completion check |
| A plausible plan can mislead the person | Measured: a 248-person study found over-trust in agents with plausible plans | He et al., *Plan-Then-Execute*, CHI 2025, arXiv 2502.01390 | Show actions and verdicts, not a model-written plan |

## 5. Showing agent work to a person

- Say what the system is doing in specific words; keep the model's reasoning behind a disclosure: step-by-step reasoning
  is often rationalisation after the fact (Apple HIG, generative AI; Nielsen Norman Group, *Explainable AI*; Magentic-UI,
  2025, arXiv 2507.22358, where 12 participants disliked verbose reasoning).
- Stop, edit, retry and undo next to the output (Microsoft's *Guidelines for Human-AI Interaction*, CHI 2019, G8 and G9;
  Google PAIR).
- Approve by reversibility; a sandbox removes most prompts (Anthropic reports 84 % fewer in internal use). Here every
  action runs in the sandbox on an in-memory copy of the files, so nothing needs approval; the person approves the
  result (the diff).
- People rarely open citations and over-trust well-cited answers (NN/g): put the source beside the claim and say
  plainly when an answer was not checked. A categorical label backed by an exact verifier is the honest form.
- WCAG 2.2: a status message must be exposed without moving focus (SC 4.1.3): announce "writing" and "ready", never the
  token stream; targets of 24 px (2.5.8); an alternative to dragging (2.5.7); focus not hidden by a sticky composer
  (2.4.11).

## 6. Documents

- Read into structure-preserving Markdown; slides as one unit each with their number and notes (MarkItDown; Anthropic's
  document skills).
- Do not paste a large table into a small context; keep the header and a capped number of rows, and say what was cut
  (SpreadsheetLLM, 2024, arXiv 2407.09025, reaches 25-fold compression by structure).
- Generate through a high-level form, not the file format: an 8B model ran 2.1 % of raw slide-library programs and
  54.4 % with a high-level helper (Ge et al., *AutoPresent*, 2025, arXiv 2501.00912). The model writes Markdown; a
  deterministic converter makes the file.

## 7. Evaluating a harness change

- Report malformed replies apart from wrong ones (Aider's benchmark).
- Pair the comparison by task and repeat it; with 200 tasks and one sample each, only a difference above 13 points is
  detectable (Miller, *Adding Error Bars to Evals*, 2024, arXiv 2411.00640). On four cores only large effects can be
  shown: a change that cannot be shown to help is recorded as "no measured difference", and is kept only if it is a
  guard that costs nothing.
- Pin the quantisation: it alone moved a function-calling score by more than 20 points (*Less is More*).

## 8. What follows for Nafs-Aql

Ranked by expected benefit over risk, given what the audit found (`uiux_audit_phase35.md` §7):

1. **A page for the agent**: the run as a list of steps with what was done and what the verifier said, live; stop;
   the diff at the end; a list of earlier runs. (No measurement needed: the feature did not exist.)
2. **A parser that survives what small models write**: fences that close on a fence line, not at the first three
   backticks anywhere; `python3`/`py3` tags; the first solution block when a later one is only an example; a clear
   message for `pytest` or shell commands. Measured as malformed steps per run, before and after.
3. **Refuse a whole-file reply written from a cut view** (a file over the view limit, or a reply that contains the cut
   marker); refuse an ambiguous patch match.
4. **Search/replace for long files** against the patch format, on tasks with files over 80 lines (a task set that did
   not exist).
5. **One worked example in the system prompt**, measured on the 1.7B model.
6. **Config checked at load**: ranges for the harness keys, named values instead of free strings, and a startup line
   that says which harness options the active engine cannot honour.
