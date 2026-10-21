# Progress checkpoint

Branch `feat/complete-nafs-aql`, started 2026-10-21 07:20 (+07) from `246c063`. Read this first after any context reset.

## Done

| When | Unit of work | Evidence |
|---|---|---|
| 07:20 | Step 1: live gates fetched and compared with the prompt (identical); gap analysis written | `docs/status/gap-analysis.md`, commit `3d70899` |
| 07:31 | Item sets built: `calib_set` 3,363 (sha256 c1c62380d01b57ee), `core_set` 1,373 (f1dc0bcce70ffe13); inputs committed, derived files regenerated | commit `71a061d` |
| 07:34 | Track A part 1: MVP acceptance **9/9** (A1 TTFT 158 ms, A3 slow path 53 s with 274 of 1,024 reasoning tokens, A9 System 2 restart ok). Load average 2.3 at start, 4.9 at end, no foreign ML workloads (Ollama idle, no mistral.rs): **SHARED MACHINE** by the prompt's rule (Chrome and the Claude app keep load1 above 1.5). `GET /v1/gates`: **G1 pass** | `benchmarks/mvp_acceptance.json` |
| 07:39 | Track A part 2: load protocol x3 (runs `8c160814`, `dddc4c94`, `dc96ce8e`; load1 3.3 to 7.3, no foreign ML workloads, SHARED MACHINE). Single-session p95 1,229 / 1,438 / 1,314 ms; 4-session p95 4,363 / 5,563 / 4,465 ms; **ratios 3.55, 3.87, 3.40**. Quieter host made the single session faster and the ratio worse: the 4-session cost is intrinsic (compute-bound at batch 4 on 4 cores). G4 NFR-007 stays **fail** (median over the 5 most recent eligible runs now 3.40). Track D next | gateway bench runs |
| 07:46 | Track D: slots A/B (`-np 2` vs `-np 4`, interleaved x3): median ratio 2.75 vs 3.16, neither <= 2. Fallback applied: single-session profile and a 2-session limit documented (runbook, decision 002, A-066); threshold change proposed to the owner (HUMAN-NEEDED 4). G4 stays fail | `benchmarks/slots_ab_system1.json`, `docs/decisions/002-nfr007-single-session-profile.md` |
| 07:50 | Track G: security 26/26 (0 open high), deployment 11/0/1, artifacts valid, `cargo audit` clean (267 crates). G5 pass | commit `64faaff` |
| 07:52 | Track H: `docs/release/RELEASE-CANDIDATE.md` written; scholar packet sections 5 to 8 added (methodology, evaluation design, sample outputs with failures, checklist); sign-off left pending | this commit |
| 09:05 | Scaled snapshot of run `553a4eff` at 41/400 results written here and into the release-candidate document; PR description updated | this commit |
| 07:55 | Branch pushed; PR opened: https://github.com/rismanmattotorang/nafsaql/pull/2 (gate table and changelog in the description); CI pending | PR #2 |
| 16:13 | P7-T11 implemented, swept and measured: exact-verifier exit on code/schema with probe reuse, shrinking retries, deadline-aware adaptive path; PR #4. Study: 10/10 identical verdicts, mean wall 428 s -> 164 s (exit) -> 109 s (exit + decay) | `benchmarks/p7_t11_code_exit_study.json`, decision 003 |
| 07:33 | Track E: `test_tazkiya.py` 9 tests OK; `tazkiya.py build` reports 4 verified examples (needs 5,000); `train_unsloth.py --dry-run` valid on the built set; `RUNBOOK.md` commands match the scripts. No registry drill: decision 001 | `docs/decisions/001-registry-drill-and-tr007.md` |

## Running

- Eval run `553a4eff` resumed at 16:14 (+07) after the study (112 kept results; `slow_only` keeps the production configuration).

- (finished) P7-T11 paired study (`mbpp_plus` offset 0, 10 items, `slow_only` vs `slow_exit_all` vs `slow_exit_all_decay`), started 13:27 (+07) on the new gateway binary; log: scratchpad `study_code_exit.log`. Eval run `553a4eff` was auto-resumed by the gateway after the restart (it resumes interrupted eval runs at startup) and then cancelled at 112 stored results so the study runs alone; it is resumed after the study (`POST /v1/eval/runs/553a4eff-620b-4150-86de-c13fe1bc0977/resume`).
- Track C run 1: eval run `553a4eff` on `core_set` (offset 0, 100 items) with `fast_only, slow_only, nafs_blind, nafs_aql`, started 07:46. Feeds H1, ER-002, ER-004 (blind arm), P0-T04. Resume after a gateway restart: `POST /v1/eval/runs/553a4eff-620b-4150-86de-c13fe1bc0977/resume`.

## Per-item times measured in this run (first items of `553a4eff`, quiet shared host, load1 5 to 6)

| System | Wall per item | Note |
|---|---|---|
| `fast_only` | 5 to 15 s | GSM8K |
| `slow_only` | 57 s | adaptive exit on |
| `nafs_blind` | 140 s | escalated a right draft (wasted); draft + 2 samples + System 2 |
| `nafs_aql` | 11 s | verifier accepted the draft |

**Code items are the cost.** `mbpp_plus-620` (a draft System 1 got wrong): `slow_only` exceeded the 900 s request cap (error, no answer); `nafs_blind` escalated and spent 916 s, 2,046 reasoning tokens and 7 model calls (System 2 with N = 1, R = 2 verified retries) and was still wrong. System 2 on code has no adaptive exit (`deliberation.adaptive_exit.domains` is math and factual), so a hard code item costs the whole budget per round. Expected `slow_only` cost per core-set item is therefore about 0.74 x 57 s + 0.26 x 600 s = 200 s, not 57 s.

About 220 s per math item for the four systems together, and up to 45 min for a hard code item: 100 items in about 6 h. Revised estimates (to be refined as n grows): P0-T04 fast 4,071 x 10 s = 11 h, slow 4,573 x 200 s = 254 h (72 h if the code path gets the adaptive exit); H1 100 blind items = 4 h; ER-002 700 pairs x 70 s = 14 h; ER-004 about 950 items x 72 s (fast + slow) = 19 h; ECE 2,900 `nafs_blind` items x 60 to 140 s = 48 to 110 h.

## Scaled G2 / G0 results from run `553a4eff` (SHARED MACHINE, load1 5 to 7; the run continues in the gateway)

Run `553a4eff` (core_set, offset 0), status running, 42/400 results, load1 6.4 at 09:04.

| System | n scored | Accuracy (95% CI) | Escalated | Mean wall | p50 wall |
|---|---|---|---|---|---|
| `fast_only` | 11 | 73% (39 to 94) | 0% | 11 s | 10 s |
| `nafs_aql` | 9 | 100% (66 to 100) | 22% | 35 s | 10 s |
| `nafs_blind` | 10 | 80% (44 to 97) | 30% | 137 s | 22 s |
| `slow_only` | 10 | 100% (69 to 100) | 100% | 104 s | 91 s |

- **H1 routing (blind gate vs oracle):** 78% on n = 9 (needs 100; target >= 70%); confusion {'escalate_needed': 1, 'escalate_wasted': 1, 'keep_correct': 6, 'keep_missed': 1}; escalation 30%.
- **ER-004 coupling (blind):** 0/2 wrong drafts adopted, Clopper-Pearson upper bound 84% (needs 200 trials and <= 5%).
- **ER-002 non-inferiority (nafs_aql vs slow_only):** n = 9 pairs, delta 0.0, CI None, McNemar p 1.0 (needs 700 pairs, lower bound > -3 points).
- **Raw e1_hat calibration on labelled drafts:** n = 19, ECE 0.164, Brier 0.192, draft error rate 26%.

All four sub-gates stay `pending` (n far below their thresholds); the direction matches the earlier scaled runs: the gated system matches deliberate-only accuracy at a third of its mean wall time, the blind gate is above the 0.70 target on 9 items, and no wrong draft has been adopted. These are indicative, not decisions.

### P7-T11 paired study, final (run `8a0266ee`, MBPP+ items, SHARED MACHINE: owner's mistral.rs and Ollama at 170 to 210% CPU, load1 6 to 23)

| Arm | Correct of 10 | Mean wall | Mean reasoning tokens | Mean model calls |
|---|---|---|---|---|
| `slow_only` (production: no exit on code) | 9 | 428 s | 1351 | 1.4 |
| `slow_exit_all` (FR-085) | 9 | 164 s | 428 | 1.2 |
| `slow_exit_all_decay` (FR-085 + FR-086, decay 0.5) | 9 | 109 s | 399 | 1.2 |

Paired against `slow_only`: accuracy delta 0 points in both arms (no discordant pair in 10; the one wrong item, `mbpp_plus-7`, is wrong in every arm); median wall ratio 0.19 (exit) and 0.20 (exit + decay); mean ratio 0.35 and 0.26. The two exit arms differ only on retries, so their gap on single-attempt items is load noise; on the retried item the decay arm took 523 s against 820 s. Exit reasons on the deadline-aware binary (items 7 to 14): `verified_probe` 10, `model` 2, `budget` 2, `deadline` 1. Before the deadline fix, `slow_exit_all` on `mbpp_plus-7` exceeded the 900 s cap (two full probes plus the answer in one attempt); after it, 820 s. Per item (verdict / wall / reasoning tokens / model calls):

| Item | `slow_only` | `slow_exit_all` | `slow_exit_all_decay` |
|---|---|---|---|
| mbpp_plus-2 | ok / 509 s / 1023 / 1 | ok / 309 s / 194 / 1 | ok / 62 s / 214 / 1 |
| mbpp_plus-3 | ok / 412 s / 1023 / 1 | ok / 128 s / 341 / 1 | ok / 83 s / 341 / 1 |
| mbpp_plus-4 | ok / 607 s / 1023 / 1 | ok / 80 s / 219 / 1 | ok / 62 s / 219 / 1 |
| mbpp_plus-6 | ok / 550 s / 2046 / 2 | ok / 56 s / 207 / 1 | ok / 58 s / 207 / 1 |
| mbpp_plus-7 | wrong / 745 s / 2873 / 3 | wrong / 820 s / 2058 / 3 | wrong / 524 s / 1750 / 3 |
| mbpp_plus-8 | ok / 81 s / 403 / 1 | ok / 47 s / 240 / 1 | ok / 47 s / 240 / 1 |
| mbpp_plus-9 | ok / 371 s / 1023 / 1 | ok / 45 s / 228 / 1 | ok / 55 s / 228 / 1 |
| mbpp_plus-11 | ok / 448 s / 2046 / 2 | ok / 60 s / 230 / 1 | ok / 94 s / 230 / 1 |
| mbpp_plus-12 | ok / 213 s / 1023 / 1 | ok / 42 s / 303 / 1 | ok / 62 s / 303 / 1 |
| mbpp_plus-14 | ok / 340 s / 1023 / 1 | ok / 57 s / 259 / 1 | ok / 41 s / 259 / 1 |

Decision 003: condition 1 (>= 10 items, no loss, median ratio < 0.7) is met; condition 2 (baseline run `553a4eff` finished) is not, so the production switch stays off until that run completes. Record: `benchmarks/p7_t11_code_exit_study.json`.

## Next three actions

0. **Flip the production switch (decision 003)** once run `553a4eff` finishes: `deliberation.adaptive_exit.domains = ["math", "factual", "code", "structured"]`, `retry_budget_decay = 0.5`, restart with `inflight` 0. Then re-run `make acceptance` and the e2e sweep. Add `"code"` to `deliberation.adaptive_exit.domains`, restart with `inflight` 0, run `core_set` with `slow_only, slow_noexit` on 40 code items; adopt only if non-inferior on the hidden tests and faster. Evidence for the need: the `mbpp_plus-620` rows above.
1. Track G: security tests, deployment and artifact checks (G5 must stay pass after the changes).
2. Track H: `docs/release/RELEASE-CANDIDATE.md`, scholar packet update; PR.
3. When run `553a4eff` has enough items: per-item times, scaled H1 / ER-002 / ER-004 numbers; then start the `calib_set` `nafs_blind` run: `core_set`, systems `fast_only, slow_only, nafs_blind, nafs_aql`, 100 items (H1 routing needs fast and slow in the same run as the blind gate; ER-004's blind arm needs fast and slow; ER-002 needs nafs_aql and slow). Then `calib_set` with `nafs_blind` for calibration labels.
3. Track G (G5 re-checks) and Track H (release candidate, scholar packet).

## Open questions

- The host is never idle by the prompt's rule (load1 < 1.5 for five minutes): Chrome and the Claude app alone keep load1 at 2 to 4. Results are labelled SHARED MACHINE with the load average; the idle-host command is in `HUMAN-NEEDED.md`.
