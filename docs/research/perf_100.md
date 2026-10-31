# Performance run: 100 prompts through the live service

Date 2026-10-27. Script `scripts/perf_100.py`, raw data `benchmarks/perf_100.json` (every request, its stage
timings, and 10-second machine samples). Build: `main` at 453a104 (UI pass and Phase 11 harness), llama.cpp
797de63, Qwen3-1.7B (quick) and Qwen3-4B-Thinking-2507 (thorough), both Q4_K_M, 4 CPU cores, no GPU.

## Setup

- **100 prompts, all in Auto mode**, shuffled with seed 20261027: 40 maths (30 GSM8K, 10 short word problems),
  20 code (MBPP+ with their tests), 12 structured (exact JSON format), 28 open or factual (12 English, 8
  Indonesian, 8 short chat). 72 carry an exact check.
- **2 simultaneous users** (the recommended limit; both models have 2 System 2 and 4 System 1 slots). Requests
  were ephemeral and skipped the audit stream, so only the prompts loaded the models.
- **Machine:** only the Claude desktop app (about 0.6 of a core) was otherwise busy. Model servers averaged 305%
  CPU of 400% (peak 400%); load average 7.4 (the models' threads outnumber the cores).

## Caveat: the Mac slept during the second half

78 prompts finished in the first 3,682 s. After that the Mac fell asleep in 21 cycles of about 310 s each (the
keep-awake the desktop app holds ended with its turn), 6,383 s in all, so elapsed time is 11,442 s but **active
time is 5,058 s (84 min)**. Per-request model times (`wall_ms`) run on a clock that does not count sleep and agree
between the two halves (quick answers: median 10.6 s before, 9.4 s during), so they are valid; throughput and
client-side times below use active time. `perf_100.py` now holds `caffeinate -i` while it runs and reports active
time itself.

## Results

| | |
|---|---|
| Answered | 98 of 100 (2 hit the gateway's 15-minute request cap: one GSM8K, one code) |
| Active time | 84 min; **1.16 answers/min**; 6.9 tokens written per second across both models |
| Thought longer (thorough path) | 19 of 98 (19%): code 8 of 19, GSM8K 5 of 29, Indonesian 3 of 8, structured 1 of 12, chat and open 2 |
| Checkable answers that passed | **66 of 70 answered (94.3%)**; 66 of 72 counting the two timeouts |
| First words appear after | median **0.91 s**, slowest 5% 2.9 s (quick path: 0.89 s and 1.75 s; two cold starts at 9.6 s) |

| Total answer time | typical (median) | slowest 5% | slowest 1% | average |
|---|---|---|---|---|
| All 98 | 15.9 s | 445 s | 895 s | 86 s |
| Quick (79) | 10.1 s | 31.9 s | 59.4 s | 14.7 s |
| Thorough (19) | 277 s | 896 s | 914 s | 383 s |

| Topic | Answers | Thought longer | Passed their check | Quick median | Thorough median |
|---|---|---|---|---|---|
| Word problems (GSM8K) | 29 | 5 | 29 of 29 | 22.4 s | 278 s |
| Short word problems | 10 | 0 | 10 of 10 | 9.4 s | – |
| Code (MBPP+) | 19 | 8 | 15 of 19 | 13.8 s | 566 s |
| Structured (format) | 12 | 1 | 12 of 12 | 5.1 s | 113 s |
| Factual (English) | 5 | 0 | – | 4.4 s | – |
| Open (English) | 7 | 1 | – | 8.2 s | 164 s |
| Open (Indonesian) | 8 | 3 | – | 13.1 s | 174 s |
| Short chat | 8 | 1 | – | 4.5 s | 162 s |

Quick answers that could be checked: **56 of 56 passed** (a quick answer that fails its check is sent to the
thorough path, so this is by design). Thorough answers that could be checked: 10 of 14 passed; the 4 failures
were all code, each after 3 attempts of about 4–5 minutes.

### Where the time goes (share of model time, 136 model calls, 97 checks)

| Step | Share | Calls | Average per call | Prompt reused |
|---|---|---|---|---|
| Thinking (thorough model) | 77.6% | 27 | 243 s | 90% |
| Quick draft | 17.9% | 98 | 15 s | 65% |
| Review of open answers | 2.2% | 5 | 37 s | – |
| Progress checks while thinking | 1.4% | 20 | 5.8 s | 99% |
| Everything else (checks, extra samples) | 0.9% | 95 | under 1 s | – |

Waiting for a free thorough-model slot added 476 s in total (17 s per thorough call); the quick model never queued.
The exact checks are free: 97 of them took 24 s in all.

## What it says

1. **A few hard prompts cost most of the compute.** 6 of 100 prompts (4 code answers that failed all 3 thorough
   attempts, and the 2 that hit the cap) used about **52% of all model time** and delivered no checked answer. The
   79 quick answers (79% of the prompts) used about 11% of the model time, at a median of 10 s each; the other 15
   thorough answers used about 37%.
2. **Code is the expensive topic:** 42% of code prompts went to the thorough path (against 17% of word problems
   and 8% of structured tasks), at a median of 9.4 minutes each, and 4 of the 8 failed. A cap on thorough attempts
   per code task (or on time per request, below the 15-minute gateway limit) would bound this waste; whether a
   shorter cap loses answers needs a measurement before it becomes a default.
3. **Two users do not add throughput.** Each stream ran at about 4–7 tokens/s while the idle machine measures 22.9
   (quick) and 10.3 (thorough) tokens/s for one stream: two streams share the same 4 cores. This agrees with the
   earlier NFR-007 finding (p95 at 2 users is about 2.3× the single-user value); a run at one user would give
   shorter per-request times at the same total throughput.
4. **Time to first words is good** (median 0.9 s, within the 2 s target at the 95th percentile for the quick path),
   helped by prompt reuse (65% for drafts, 90% when thinking, 99% for progress checks). The two slow starts were the
   first requests after a restart.
5. **Reference targets (this computer):** first words within 2 s for 95%: met on the quick path (1.75 s). A short
   quick answer within 12 s for 95%: not met here (31.9 s), but these prompts are longer than the 150-token reply
   the target was written for (median 84 tokens written, and word problems need steps). A thorough answer within
   240 s for 95%: not met (896 s), driven by code.

## Limits

One run, one machine, 2 users, a fixed shuffled order. Latencies of individual prompts depend on what ran beside
them. The 100 prompts are not a statistical sample of any real workload; the accuracy figures describe this set
only. Re-run with `python3 -I scripts/perf_100.py run` (resumable) and `report`.
