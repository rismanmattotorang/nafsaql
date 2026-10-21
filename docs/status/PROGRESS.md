# Progress checkpoint

Branch `feat/complete-nafs-aql`, started 2026-10-21 07:20 (+07) from `246c063`. Read this first after any context reset.

## Done

| When | Unit of work | Evidence |
|---|---|---|
| 07:20 | Step 1: live gates fetched and compared with the prompt (identical); gap analysis written | `docs/status/gap-analysis.md`, commit `3d70899` |
| 07:31 | Item sets built: `calib_set` 3,363 (sha256 c1c62380d01b57ee), `core_set` 1,373 (f1dc0bcce70ffe13); inputs committed, derived files regenerated | commit `71a061d` |
| 07:34 | Track A part 1: MVP acceptance **9/9** (A1 TTFT 158 ms, A3 slow path 53 s with 274 of 1,024 reasoning tokens, A9 System 2 restart ok). Load average 2.3 at start, 4.9 at end, no foreign ML workloads (Ollama idle, no mistral.rs): **SHARED MACHINE** by the prompt's rule (Chrome and the Claude app keep load1 above 1.5). `GET /v1/gates`: **G1 pass** | `benchmarks/mvp_acceptance.json` |
| 07:33 | Track E: `test_tazkiya.py` 9 tests OK; `tazkiya.py build` reports 4 verified examples (needs 5,000); `train_unsloth.py --dry-run` valid on the built set; `RUNBOOK.md` commands match the scripts. No registry drill: decision 001 | `docs/decisions/001-registry-drill-and-tr007.md` |

## Running

- Load protocol x3 (Track A), started 07:35 (chat-short, fast, levels 1/2/4, 8 requests per session). Log: scratchpad `protocol_x3.log`; results in the gateway store (`GET /v1/bench/runs`).

## Next three actions

1. Read the three protocol ratios; recompute G4. If the median of the five most recent eligible runs is still > 2, Track D (slot and thread study) and the roadmap fallback.
2. Start the combined Track C run: `core_set`, systems `fast_only, slow_only, nafs_blind, nafs_aql`, 100 items (H1 routing needs fast and slow in the same run as the blind gate; ER-004's blind arm needs fast and slow; ER-002 needs nafs_aql and slow). Then `calib_set` with `nafs_blind` for calibration labels.
3. Track G (G5 re-checks) and Track H (release candidate, scholar packet).

## Open questions

- The host is never idle by the prompt's rule (load1 < 1.5 for five minutes): Chrome and the Claude app alone keep load1 at 2 to 4. Results are labelled SHARED MACHINE with the load average; the idle-host command is in `HUMAN-NEEDED.md`.
