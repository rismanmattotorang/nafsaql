# Decision 028: A reasoning cache, after CPU-ARC (Phase 38)

Date 2026-11-23. Status: **adopted, on by default for maths questions.** Prompt
[`REASONING_CACHE_PROMPT.md`](../prompts/REASONING_CACHE_PROMPT.md). Assessment
[`reasoning_cache.md`](../research/reasoning_cache.md); specification
[`reasoning_cache_spec.md`](../spec/reasoning_cache_spec.md); measurements `benchmarks/arc_phase38/`.

## Context

NVIDIA's CPU-ARC (arXiv 2611.09891) turns a strong model's reasoning trace into parameterised Python *semantic
functions*; a small model reads the new values, the code runs on CPU, and a miss falls through to the strong model. On
this machine an answer from System 2 takes about a minute even for a grade-school problem, so the cost the paper removes
is the largest cost Nafs-Aql has. Nothing we had reused reasoning: the KV prefix store breaks at the first changed
token, and the store of verified answers is a hint the model must still read and answer from.

## Decisions

1. **System 2 is source and builder; System 1 reads values; the verifiers are the judges.** A build is one System 2 call
   (three sections: template, parameters, code), checked in the sandbox and fed back up to three rounds, while no
   request is in flight. Semantic functions make no model call (A-197).
2. **Three ways to find an entry**: the stored template as a pattern and the question's shape with its numbers masked
   (no model), then the meaning vector. A re-numbered question never touches a model.
3. **Guards without a model**: every value must be written in the question and every number in the question used;
   descriptions match exactly before they match approximately; a pair of words that turns one question into another
   (area/perimeter, odd/even) refuses a meaning-tier match.
4. **On the meaning tier, System 1 is asked directly whether it is the same question** before it reads the values. The
   extraction's own "same" field let 6 of 14 near misses through; the direct question, with the contrast words, let none.
5. **Trust is earned.** An entry is a candidate that runs beside the models until it agrees with their answer twice;
   a checked disagreement on the template or shape tier stops it; each entry learns its own meaning-tier floor
   (vCache, Krites).
6. **Compilation** writes one standard-library module whose `do()` calls are direct calls, which checks itself against
   every entry's answer and leaves out (with the reason) an entry it cannot compile faithfully.
7. **Defaults**: `[arc] enabled`, maths only, `trust_after = 2`, `serve_similarity = 0.85`, `min_similarity = 0.80`,
   `same_check = true`.

## Measured (Phase 38 item set, 12 seeds, 64 variants; reference machine)

| | Result |
|---|---|
| Builds | 12 of 12; 7 in one round, 5 in two (each first round refused by a check: an ambiguous description, no root, no narration, a wrong answer); 20 functions |
| Build time | 461 s mean, 7.2x System 2's own solve (64 s mean); CPU-ARC reports 12-84x in dollars |
| Re-numbered questions | 24 of 24 answered, 24 right, by the template, median 95 ms, no model |
| Reworded questions | 13 of 14 answered, 13 right (the miss writes "Five" as a word, refused by design) |
| Reworded and re-numbered | 4 of 12 answered, 4 right; the rest fall below a meaning similarity of 0.80 |
| Near misses (a different question on the same topic) | before the decision-4 checks 6 of 14 answered, all wrong; after, **0 of 14** |
| Answers given, after the checks | 53, all right |
| Meaning tier | median 7.7 s through the try endpoint (three short System 1 calls) |
| Time in chat, other numbers (6 each, interleaved) | cache 0.12 s; usual path 20.8 s; System 2 85.5 s; all right |
| Time in chat, other words (6 each) | cache arm 9.7 s (4 of 6 from the cache, 2 fell through after System 1 said "different"); usual 18.1 s; System 2 81.8 s; all right |
| Break-even | a build (461 s) is repaid after about 6 re-numbered questions that would go to System 2, 23 that the usual path would answer |
| Compiled module | 12 entries, 28 KB; 36 of 36 template questions right with no model, 78 ms a run; same answer as the runtime on 76 of 76 |

The contrast-word list was written while looking at this item set, so its share of the result is optimistic; the
direct System 1 question was measured on it unchanged (21 of 23 judgements right).

## Also fixed

- **PDF reader:** a centred heading no longer runs on into the line under it when LaTeX shrinks the space at a
  column's foot. Found by the test that reads this project's own paper, after its abstract grew by two lines. The four
  other project documents score exactly as before; the paper reads 75 of 78 section titles as headings (order 0.817,
  was 0.810).
- **Sandbox interpreter:** the cache picks a Python of version 3.9 or later instead of the first `python3` on the
  PATH (a conda 3.8 here), which cannot compile the cache.
- **Found on the live chat page, after the benchmark** (which passed a maths hint on every request): the chat page
  sends none, and the question classifier read a perimeter question as "open", so the cache was never consulted; and in
  a continued conversation the follow-up rule read "its perimeter" as pointing back. The template and shape lookups,
  which need no model and match only a whole stored question, now run for every eligible question; the domain rule and
  the follow-up rule apply to the meaning lookup and to admission only. A template that is mostly placeholders is
  refused at build and never matched. A cached answer's trace is `C` (answered from the cache) then `T`; the contract
  suite caught the one-state trace the first build wrote.
- **Privacy:** questions from private requests, rooms and Lab traffic are never built into entries (A-202).
- **Test messages:** three knowledge-base tests cut a failure message at a byte count that could fall inside a
  character, panicking instead of reporting; they now cut by characters.

## Not adopted

Model calls inside semantic functions; generator-model perturbation tests at build time; System 1 drafting for System 2
on a miss (speculative decoding measured no gain here, Phases 4 and 17); merging similar functions across entries;
caching open or code questions by default. Reasons in the assessment, section 6.
