# Gate gap analysis

Written 2026-10-21 07:30 (+07) at the start of the completion run, from `GET /v1/gates` on commit `246c063` and from `rust/crates/nafs-gateway/src/gates.rs`. Per-item times are from stored runs on this host (all **SHARED MACHINE**: load average 6 to 15 while they ran). They are replaced by fresh measurements as the tracks produce them (see `PROGRESS.md`).

Classification: **here** = automatable on this host inside this run; **hours** = automatable here but needs machine-hours beyond this run (nightly slices, resumable); **human** = a person must act.

## Live gate state against the prompt's section 2.2

Identical. All seven gates, every criterion status and every evidence string match the prompt's table (G0 pending, G1 fail, G2 pending, G3 pending, G4 fail, G5 pass, G6 pending). Hardware matches `sysctl` (i3-8100, AVX2, no AVX-512). No tag exists.

Two statements in the prompt do not match the code:

1. **Calibration source.** The prompt says labelled `fast_only` drafts feed calibration. `nafs_eval::calibration_samples` keeps only items of **gated** systems (`nafs_aql`, `nafs_blind`, `_l<NN>` variants) with `features.k > 0` and a known `draft_correct`. `fast_only` items carry no gate features, so they contribute nothing. The cheap source that exists is **`nafs_blind_l100`**: the gate runs blind, draws the k = 2 samples, and at lambda = 1 never escalates (the price of one cost unit exceeds any error saved), so no System 2 call is made. Cost per item is one draft plus two samples. Items whose samples were skipped by the uncertainty band (`samples_when_uncertain`) have k = 0 and are excluded; this biases the set toward uncertain drafts and is checked in Track C.
2. **Registry file.** `config/models.registry.json` does not exist yet. `training/tazkiya.py` creates it on the first `promote` (the default registry is built in memory). The commands exist; the file does not.

One arithmetic gap the prompt does not mention: the local verifiable benchmarks hold **2,294** items (GSM8K test 1,319, HumanEval 164, HumanEval+ 164, MBPP 257, MBPP+ 378, schema 12), fewer than the 2,900 the ECE criterion needs. `scripts/build_datasets.py` fetches from the Hugging Face datasets server, so the GSM8K **train** split (7,473 items, same licence) is the way to reach 2,900 with disjoint items. `regression_500` is a fixed subset of the test sets and must stay out of the calibration set.

## Per gate

### G0: Phase-0 exit and profile decision (pending)

| Criterion | Rule (`gates.rs`) | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| H7 | stored `h7.pass == true` | pass (79 to 80%) | none | none | done |
| P0-T04 | `fast_only` scored items >= 4,600 **and** `slow_only` scored items >= 4,600, pooled over stored runs | 529 fast, 27 slow | 4,071 fast, 4,573 slow | Nightly resumable `slow_only` + `fast_only` slices on the shared stratified item set (Track F) | hours |

Per-item times (stored, shared machine): `fast_only` mean 24.5 s (regression_500, 500 items, run `880cdb33`); `slow_only` with adaptive exit mean 123 s, median 91 s (run `e7607cbb`, 12 GSM8K items). Estimate: fast 4,071 x 24.5 s = **28 h**; slow 4,573 x 123 s = **156 h**. Updated with idle times from Track A.

### G1: MVP acceptance and latency targets (fail)

| Criterion | Rule | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| MVP | all stored acceptance results pass and there are >= 9 | 8/9 (A3 failed: 354 s) | A3 wall <= 240 s | Re-run `make acceptance` on a quiet host (Track A); if still > 240 s, cut System 2 time (Track B) | here |
| NFR-002/004 | A1 `ttft <= 2000 ms` and A3 `wall <= 240 s`, parsed from the stored acceptance details | TTFT 553 ms pass, wall 354 s fail | same as above | same | here |

Note: A3 forces the slow path on GSM8K item 0 with a 1,024-token thinking budget; the stored run used 820 reasoning tokens. The adaptive exit (adopted after that run) reduced mean reasoning tokens 723 to 348 on GSM8K, so a re-run is the first step.

### G2: calibrated gate, routing, non-inferiority, coupling (pending)

| Criterion | Rule | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| ECE | a **promoted** calibration (n >= 2,900, >= 30 per class, held-out ECE <= 0.05) | none (n 0) | 2,900 labelled gated drafts (ledger has 29: 15 right, 14 wrong) | `nafs_blind_l100` over the verifiable sets, then `POST /v1/gate/recalibrate` with the run ids (Track C) | hours (see estimate) |
| H1 | routing accuracy, pooled over `nafs_blind*` runs, n >= 100, >= 0.70 | 6 items | 94 items | `nafs_blind` on >= 100 stratified items (Track C) | here |
| ER-002 (3 pt) | paired `nafs_aql` vs `slow_only` within a run, >= 700 pairs, bootstrap lower bound > -3 points | 16 pairs | 684 pairs | the shared item set through `nafs_aql` and `slow_only` (Track C/F) | hours |
| ER-004 | `coupling.draft_blind.wrong_drafts` pooled >= 200, Clopper-Pearson upper bound of adopted <= 5% | 0/5 | 195 wrong drafts | `slow_conditioned` + `fast_only` on the same items; about 21% of math drafts are wrong, so about 950 items (Track C) | hours |

Per-item times (stored, shared): `nafs_blind` mean 133 s (6 items, 17% escalated); `nafs_blind_l100` not yet measured (expected draft + 2 samples: about 20 to 40 s); `nafs_aql` 39 to 133 s; `slow_conditioned` mean 319 s (6 items). Estimates: H1 100 x 133 s = **3.7 h** (here); ECE 2,900 x 30 s = **24 h** (hours); ER-002 684 x (123 + 60) s = **35 h**; ER-004 950 x 319 s = **84 h**, or less once the adaptive exit applies to `slow_conditioned`.

### G3: adapter gated; rollback demonstrated (pending)

| Criterion | Rule | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| TR-007 | registry history has a `promote` entry | none | a promotion | needs a trained adapter: GPU host (owner deferred) | human |
| OR-005 | registry history has a `rollback` entry | none | a rollback | registry drill with the already built `models/Qwen3-1.7B-Q4_K_M.own.gguf` (Track E) | here |

Caveat: `tazkiya.py promote` refuses unless the paired gain's lower 95% bound is above 0 with a point gain >= 2 points and a human sign-off. A requantized copy of the same weights cannot pass that gate honestly, so the drill records a promote **only** if the gate passes on a real paired run; otherwise the drill is recorded as "promotion refused by the gate (as designed), rollback not exercisable" and OR-005 stays pending. Both outcomes are honest evidence. `gates.rs` counts any `promote` entry for TR-007, so a drill promote would also turn TR-007 green without an adapter: that is **not acceptable** (rule 1), and the drill therefore documents the registry round trip with the promote entry labelled `drill: true` only if the gate itself admits it; see Track E in `PROGRESS.md`.

### G4: evaluation suite; load tests (fail)

| Criterion | Rule | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| P4 | constant pass | pass | none | none | done |
| ER-007 | a completed protocol run | pass (`6c462074`) | none | none | done |
| NFR-007 | median `nfr007.ratio` over the 5 most recent eligible protocol runs (>= 8 single-session requests) <= 2.0 | median 2.15 over 1.25, 2.15, 2.48 | three quiet-host runs at <= 2.0 would make the median of five <= 2.0 | `POST /v1/bench/protocol` x 3 on a quiet host (Track A); if the ratio stays > 2, slot and thread study (Track D), then the roadmap fallback (single-session profile documented, threshold proposal to the human) | here |

### G5: security, deployment, upgrade decisions (pass)

Keep passing: after every change re-run `python3 -I scripts/security_tests.py`, `scripts/verify_deploy.sh --live`, `scripts/validate_artifacts.sh` (Track G). No new dependency, endpoint or model artifact is planned; a drill promote changes the registry, which is a recorded decision, not a new artifact.

### G6: release (pending)

| Criterion | Rule | Evidence now | Gap | Closing task | Class |
|---|---|---|---|---|---|
| ER-002 (1 pt) | >= 6,200 pairs, lower bound > -1 point | 16 pairs | 6,184 pairs | not reachable here (about 300 h); the criterion's own fallback is to release on the 3-point result with a documented limitation | hours / human decision |
| ER-008 | one completed `regression_500` run passes as the baseline; two or more compare latest vs previous by McNemar | pass (baseline 69.4%) | none; a second run would turn the check into a real regression test | optional: a second `fast_only` regression_500 run (3.4 h) | here (optional) |
| SR-005 | `docs/release/scholar_review.json` (or equivalent) with `decision == approved` | packet exists, no reviewer | sign-off | prepare the packet; a scholar signs | human |

## Order of work (value per machine-hour)

1. Track A now (host is quieter than at any stored run: load 2.4 to 4 with no foreign ML workloads): acceptance, then three protocol runs. Decides G1 and G4.
2. Track C part 1: `nafs_blind` on 100 stratified items (3.7 h) closes H1's n; `nafs_blind_l100` measured on 50 items gives the real per-item time for the ECE estimate.
3. Track E drill (minutes of registry work plus a paired regression subset).
4. Track B only if A3 is still above 240 s.
5. Tracks G, H. Then nightly slices (F) run as long as the host allows.
