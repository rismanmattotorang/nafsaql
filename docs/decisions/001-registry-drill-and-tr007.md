# Decision 001: no registry drill with a non-adapter artifact; TR-007 needs an adapter-aware rule

Date 2026-10-21. Status: **option A chosen by the owner on 2026-10-21 ("count only adapter promotions for TR-007 and run the rollback drill for OR-005")**. Implemented: `gates.rs` counts a `promote` entry for TR-007 only when its `adapter_hash` is not `none`; the drill promotes a base-model artifact with `--adapter-hash none`, so it moves OR-005 and not TR-007. Drill record: `benchmarks/rollback_drill/` (see the section at the end).

## Context

Gate G3 has two criteria read from `config/models.registry.json` (`gates.rs`):

- **TR-007** "An adapter evaluated by the promotion gate" passes when the history holds **any** `promote` entry.
- **OR-005** "Rollback demonstrated" passes when the history holds **any** `rollback` entry.

The completion plan asked for a rollback drill with the already built `models/Qwen3-1.7B-Q4_K_M.own.gguf` (a requantization of the same weights). Two facts block an honest drill:

1. `training/tazkiya.py promote` only writes a `promote` entry when the paired gain over the baseline has a lower 95% bound above 0 and a point estimate of at least 2 points. A requantization of the same weights cannot pass that test, so the tool refuses, as designed. There is no `rollback` without a prior `promote`.
2. If a candidate that does pass the gate were used instead (Qwen3.5-2B is +17 points on 30 tasks, `benchmarks/profile_c_decision.json`), the `promote` entry would turn **TR-007 green without an adapter**, because the rule counts any promotion. That is a gate turned green by something other than the evidence it names.

## What was done instead

- `training/test_tazkiya.py` exercises the registry round trip (promote, rollback, restored hash) on a temporary registry: 9 tests pass on 2026-10-21 (`python3 -m unittest discover -s training -p "test_*.py"`).
- `training/train_unsloth.py --dry-run` and `training/RUNBOOK.md` were checked (see `PROGRESS.md`).
- OR-005 stays `pending`. The live registry is untouched.

## Options for the owner

A. **Change the TR-007 rule** so it counts only promotions whose `adapter_hash` is not `none` (an adapter), and let OR-005 count a rollback of any artifact. Then a base-model drill (promote Qwen3.5-2B on a paired `regression_500` subset, observe the 3.7x latency regression, roll back) closes OR-005 honestly and leaves TR-007 pending. This edits `gates.rs` (gate logic), which this run may not do.

B. **Keep the rules** and close OR-005 only after the first real adapter is promoted on a GPU host (TR-007 and OR-005 close together).

C. **Accept a drill under the current rule**, knowing TR-007 would show `pass` with a non-adapter. Not recommended.

Recommendation: A. The one-line change is `let promoted = hist.iter().any(|h| h["action"] == "promote" && h["adapter_hash"] != "none");` with a unit test; the drill command is in `docs/status/HUMAN-NEEDED.md`.

## The drill, run on 2026-10-21 (owner's instruction)

The only artefact on this machine that had ever beaten the production System 1 on a paired test was Qwen3.5-2B (+17 points on 30 mixed tasks, `benchmarks/profile_c_decision.json`). It was served as System 1 and run on the fixed regression suite with `fast_only`, then offered to `tazkiya.py promote` with `--adapter-hash none` against the stored baseline run `880cdb33`:

| Candidate run | Items (all math: the suite's first 300 items are GSM8K) | Candidate accuracy | Paired gain | 95% CI | Gate |
|---|---|---|---|---|---|
| `a9309c34` | 40 | 85.0% | +12.5 points | −2.5 to +27.5 | **refused**: lower bound not above 0 |
| `3cca7cad` | 120 | 70.8% | −3.3 points | −13.3 to +5.8 | **refused**: gain below +2, lower bound below 0 |

So no artefact available here passes the promotion gate, and without a promotion there is nothing to roll back: **OR-005 stays `pending`**, honestly. The gate did its job twice. The two candidate runs are kept in `benchmarks/rollback_drill/` (full items) and the refusal record in `refusal_n120.json`; they were removed from the gateway's run store because they were produced with a non-production System 1 and would otherwise have counted as production evidence for P0-T04 and ER-008 (the store was backed up first).

A by-product for MR-005: on 120 GSM8K items Qwen3.5-2B is **not** more accurate than Qwen3-1.7B as System 1 (−3.3 points, interval includes 0); the earlier +17 came from 30 mixed items. Profile A stays, now with a stronger reason.

What closes OR-005: the first adapter that passes the gate (a GPU training run, HUMAN-NEEDED item 3), or any other candidate artefact the owner wants tested on the suite.
