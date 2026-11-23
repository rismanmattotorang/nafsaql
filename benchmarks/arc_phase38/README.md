# Phase 38: reasoning cache benchmark

Driver: `scripts/arc_bench.py` (steps `build`, `test`, `latency`, `compile`, `report`); items: `items.json`; results:
`results.json` (each step resumes from it; it holds every row the tables are made from, including the first test run before the meaning-tier checks, `tests_before_guards`). The step logs stay local. Reference machine: Intel i3-8100,
4 cores, Smiley Core engine, System 1 Qwen3-1.7B Q4_K_M, System 2 Qwen3-4B-Thinking-2507 Q4_K_M, embedder
Qwen3-Embedding-0.6B Q8_0. The machine lost power twice during the build step (19:44 and 20:06); the
build resumed from the stored results.

## Method

- **12 seed problems** (GSM8K/MATH style: change, divisor count, distance, simple interest, perimeter, sum of odd numbers,
  work rate, discount, removed number, combinations, least common multiple, reading time). Each has an independent
  oracle (a Python expression over its values) that gives the right answer to every variant.
- **Build.** System 2 answers the seed with the cache off and the oracle's answer as the reference (so the reply is
  checked), then `POST /v1/arc/build` with that reply, its reasoning and the reference, `wait: true`. Times are wall
  times through the gateway.
- **64 variants**: 15 *phrasal* (other words, same numbers), 23 *parametric* (same words, other numbers), 12
  *combined* (both), 14 *near misses* (same topic and usually the same numbers, a different question that needs other
  reasoning, such as the area instead of the perimeter, or an extra 2-dollar eraser). Two phrasal/combined variants
  write a value as a word ("Five numbers"), which the value guard must refuse.
- **Test.** Every seed and variant through `POST /v1/arc/try` (the cache alone, no fallback, trust ignored) with the
  meaning-tier floor at 0.80, so the floor can be swept afterwards: at floor t a meaning-tier hit counts only when its
  similarity is at least t. A hit is *right* when its answer equals the oracle's (relative 1e-4). A near miss that hits
  with the right answer is counted as right (the question was answerable by the same code); one that hits with a wrong
  answer is a **false hit**.
- **Latency.** For up to six hits on different items: the cache answering (entry made trusted for the run), the usual
  path with the cache off, and System 2 forced, interleaved in rotating order, ephemeral requests.
- **Compile.** `POST /v1/arc/compile {"all": true}`, then the module run as `python3 arc_cache.py "<question>"` on every
  question; its answers are compared with the oracle and with the runtime's.
