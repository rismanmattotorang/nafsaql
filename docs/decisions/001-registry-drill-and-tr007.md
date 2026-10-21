# Decision 001: no registry drill with a non-adapter artifact; TR-007 needs an adapter-aware rule

Date 2026-10-21. Status: **option A chosen by the owner on 2026-10-21 ("count only adapter promotions for TR-007 and run the rollback drill for OR-005")**. Implemented: `gates.rs` counts a `promote` entry for TR-007 only when its `adapter_hash` is not `none`; the drill promotes a base-model artifact with `--adapter-hash none`, so it moves OR-005 and not TR-007. Drill record: `benchmarks/rollback_drill.json` (see the section at the end).

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
