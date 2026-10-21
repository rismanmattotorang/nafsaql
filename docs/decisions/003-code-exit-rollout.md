# Decision 003: when the exact-verifier exit on code and schema tasks becomes the production default

Date 2026-10-21. Status: **implemented and measured at small n; production switch deferred to a stated condition**.

## What is in the code (P7-T11, FR-085, FR-086)

- Full-answer probes on code-test and JSON-Schema tasks, judged by the exact verifier; a passing probe is the final answer. Per request: `adaptive_exit_domains`; Lab systems `slow_exit_all`, `slow_exit_all_decay`.
- Verified retries get `budget * decay^r` with a floor, and no attempt starts that cannot finish before the 900 s cap at the previous attempt's token rate (`retry_budget_decay`, `retry_min_budget`).
- Production configuration today: exit on math and factual only, decay 1.0; the time guard is always on.

## Evidence so far (run `8a0266ee`, MBPP+, paired, SHARED MACHINE with the owner's mistral.rs and Ollama at 170 to 210% CPU)

10 paired items, identical verdicts in all three arms (9 of 10 correct; the wrong item is wrong everywhere). Mean wall 428 s (`slow_only`) against 164 s (exit) and 109 s (exit + decay 0.5); mean reasoning tokens 1351 against 428 and 399; median wall ratio 0.19 and 0.20. Per-item table in `benchmarks/p7_t11_code_exit_study.json` and `docs/status/PROGRESS.md`. **Condition 1 below is met; condition 2 is not yet.**

## The switch

`deliberation.adaptive_exit.domains = ["math", "factual", "code", "structured"]` and `retry_budget_decay = 0.5` in `config/nafs-aql.toml`, followed by a gateway restart with `inflight` 0.

## Condition

Flip it when **both** hold:

1. The paired study has at least 10 items with no accuracy loss on the hidden tests (every exit-arm answer that `slow_only` gets right is also right, allowing at most one discordant pair in the exit arm's disfavour), and the exit arm's median wall ratio is below 0.7.
2. The P0-T04 baseline run `553a4eff` (core set, offset 0) has finished, so its `slow_only` arm keeps one configuration from the first item to the last. The next slice (offset 100) then runs on the new default and is labelled as such.

The owner can flip earlier by editing the two keys; the Lab systems stay for later paired checks. If the study shows a loss, the switch stays off and the Lab systems remain the way to measure it (A-067).
