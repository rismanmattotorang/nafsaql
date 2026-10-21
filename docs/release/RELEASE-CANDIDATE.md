# Release candidate status (no tag)

Written 2026-10-21 during the completion run on branch `feat/complete-nafs-aql`. Gate statuses are copied from `GET /v1/gates`, which computes them from stored runs and records (`rust/crates/nafs-gateway/src/gates.rs`, unchanged). **No release tag exists and none is created:** G0, G2, G3, G4 and G6 are not all `pass`.

Hardware for every number below: Intel Core i3-8100 (4 cores, no SMT, AVX2 only), 32 GB DDR4-2400, macOS 15.4.1, no GPU, llama.cpp `797de63` CPU-only, both models GGUF Q4_K_M (`config/models.manifest.json`), 4 threads. Host label **SHARED MACHINE** throughout: the owner's other workloads were idle during this run (Ollama idle, no mistral.rs), but Chrome and the Claude app kept load1 between 2.3 and 7.3, above the 1.5 that "idle" requires.

## Gate table

| Gate | Before (2026-10-21 07:20) | After | Why it is not `pass` |
|---|---|---|---|
| G0 Phase-0 exit | pending | pending | P0-T04 needs 4,600 fast-only and 4,600 slow-only scored items; stored: see the latest counts in `GET /v1/gates`. Nightly slices on `core_set` (section 4). H7 passes. |
| G1 MVP and latency | **fail** | **pass** | Acceptance 9/9; TTFT 158 ms; slow path 53 s (`benchmarks/mvp_acceptance.json`). |
| G2 gate quality | pending | pending | ECE needs a promoted calibration on 2,900 labelled drafts; H1 needs 100 blind-gate items; ER-002 700 pairs; ER-004 200 wrong-draft trials. Run `553a4eff` (section 4) is producing all four; scaled results in `docs/status/PROGRESS.md`. |
| G3 adapter and rollback | pending | pending | TR-007 needs a GPU-trained adapter (owner deferred). OR-005: an honest drill is blocked by the promotion gate and by TR-007's rule; decision 001. |
| G4 load tests | **fail** | **fail** | NFR-007 ratio 3.4 to 3.9 on a quiet host at the production slots, 2.6 to 3.2 at 2 slots: not reachable on 4 cores. Fallback applied (single-session profile, 2-session limit); threshold change is the owner's decision (decision 002). |
| G5 security and deployment | pass | pass | Re-verified after every change: 26/26 security tests, 0 open high; deployment 11 pass, 0 fail, 1 skipped (Linux host); `cargo audit` clean on 267 crates; artifacts validate. |
| G6 release | pending | pending | ER-002 at 1 point needs 6,200 pairs (not reachable here; the criterion's fallback is to release on the 3-point result with a documented limitation). ER-008 passes (baseline 69.4% on 500 items). SR-005 needs a scholar's sign-off. |

## Headline CPU numbers

| Measure | Before (stored, owner's load 9 to 15) | After (this run, quiet shared host) |
|---|---|---|
| Fast-path TTFT (acceptance A1) | 553 ms | 158 ms |
| Fast path p50 / p95, 1 session, short chat | 1.1 to 2.0 s / 2.3 to 4.7 s | 0.62 to 0.89 s / 1.2 to 1.4 s |
| Fast path p95, 4 sessions | 5.7 to 10.1 s | 4.4 to 5.6 s (ratio 3.4 to 3.9) |
| Slow path, GSM8K item with 1,024-token budget (A3) | 354 s (820 reasoning tokens, before the adaptive exit) | 53 s (274 reasoning tokens) |
| System 1 decode in service (`/v1/perf/stages`, 24 h window) | 2.6 to 4.3 tok/s | 12.1 tok/s |
| System 2 decode in service | about 3 tok/s | 6.9 tok/s |
| Idle llama-bench decode (unchanged reference) | S1 22.9, S2 10.3 tok/s | not re-measured (host never idle) |
| RAM | S1 1.35 GB RSS, S2 3.7 GB RSS | same |

## What changed in this run

- No change to gate logic, thresholds, stored records or fixtures.
- New seeded item sets for every scale run (`scripts/build_item_set.py`): `calib_set` (3,363 items, GSM8K train and MBPP train, calibration only) and `core_set` (1,373 items, test split, disjoint from `regression_500`).
- Evidence added: acceptance 9/9; three quiet-host protocol runs; the slot study (`benchmarks/slots_ab_system1.json`); refreshed security and deployment records.
- Documents: `docs/status/` (gap analysis, progress, human-needed), decisions 001 and 002, A-066, runbook capacity note, the scholar packet's sections 5 to 8.

## Limitations, stated plainly

1. **Concurrency.** Four concurrent sessions cost 3.4 to 3.9x the single-session p95 on four cores; the CPU profile is a one-to-two-user profile. NFR-007 fails and will keep failing on this hardware.
2. **Scale.** Every G2 and G0 criterion is reported at a fraction of its required n. The direction of the evidence (the gated system matches deliberate-only accuracy at a fraction of its cost; no wrong draft adopted) is consistent across runs, but the gates stay `pending` until the nightly slices reach the required counts.
3. **Calibration.** The gate runs on the fixed rule; no calibration is promoted. Calibration samples come only from blind gated runs (`nafs_blind`), not from `fast_only`, and confident code drafts skip the samples, so the calibration set over-represents uncertain drafts (section 4).
4. **Open-ended answers** have no exact verifier; the judge on System 2 accepts 4.5% of wrong answers (upper bound 23%). Such answers are shown as *unverified* or *consensus*, never *verified*.
5. **Training.** No adapter has been trained; the ledger holds 4 verified System 2 traces against about 5,000 needed.
6. **Linux sandbox** does not isolate the filesystem or fork (macOS does); documented in the README.

## 4. Commands that produce the missing evidence

Per-item times measured in this run are in `docs/status/PROGRESS.md` (updated as run `553a4eff` progresses); estimates there replace the stored-run estimates in `docs/status/gap-analysis.md`.

```bash
# P0-T04, ER-002, ER-004, H1: nightly resumable slices of the core set (offset advances by 100)
curl -s -X POST localhost:8080/v1/eval/runs -H 'Content-Type: application/json' \
  -d '{"benchmark":"core_set","systems":["fast_only","slow_only","nafs_blind","nafs_aql"],"max_items":100,"offset":100,"shuffle":false,"seed":2026}'
# ECE: calibration labels (blind gate on the calibration split), then recalibrate from the stored runs
curl -s -X POST localhost:8080/v1/eval/runs -H 'Content-Type: application/json' \
  -d '{"benchmark":"calib_set","systems":["nafs_blind"],"max_items":200,"offset":0,"shuffle":false,"seed":2026}'
curl -s -X POST localhost:8080/v1/gate/recalibrate -H 'Content-Type: application/json' -d '{"run_ids":["<calib run ids>"]}'
# resume after a gateway restart
curl -s -X POST localhost:8080/v1/eval/runs/<run_id>/resume
```

## Sign-off

- Scholar review (SR-005): pending, no reviewer assigned; packet in `docs/release/scholar_review_packet.md`.
- Items for the owner: `docs/status/HUMAN-NEEDED.md`.
