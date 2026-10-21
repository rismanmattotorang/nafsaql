# Items that need a human

Each entry: what, why it matters, the exact command or action, the evidence it produces, the criterion it moves.

## 1. Idle-host measurement of the slow path and the load protocol

- **What:** run the MVP acceptance and the load protocol on this machine with no other workloads (close Chrome, Ollama, mistral.rs and other llama-servers; the Claude app also counts).
- **Why:** NFR-004 (slow path <= 240 s) and NFR-007 (p95 at 4 sessions <= 2x) are defined on an idle reference machine. Every stored run so far is SHARED MACHINE.
- **Action:**
  ```bash
  uptime   # load1 must be < 1.5 for five minutes
  make acceptance
  for i in 1 2 3; do curl -s -X POST localhost:8080/v1/bench/protocol -H 'Content-Type: application/json' -d '{"prompt_set":"chat-short","mode":"fast","levels":[1,2,4],"requests_per_session":8}'; sleep 600; done
  curl -s localhost:8080/v1/gates | python3 -m json.tool
  ```
- **Evidence:** `benchmarks/mvp_acceptance.json`, protocol runs in the gateway store.
- **Moves:** G1 (NFR-002/004, MVP), G4 (NFR-007).

## 2. Scholar sign-off (SR-005)

- **What:** a scholar of the tradition reviews `docs/release/scholar_review_packet.md` and records a decision.
- **Why:** G6 cannot pass without it; the team must not sign for the reviewer.
- **Action:** the reviewer fills section 4 of the packet; the operator stores `{"decision": "approved" | "changes_requested", "reviewer": NAME, "date": YYYY-MM-DD, "notes": ...}` at the path `gates.rs` reads (see `docs/release/RELEASE-CANDIDATE.md` for the exact file).
- **Evidence:** the sign-off record.
- **Moves:** G6 (SR-005).

## 3. GPU training run (TR-007)

Still open. Decision 001 (option A, chosen 2026-10-21) means the rollback drill does not touch TR-007: only an adapter promotion closes it.

### 3a. Rollback drill (OR-005): attempted 2026-10-21, promotion refused

Qwen3.5-2B, the only artefact that had ever beaten System 1 here, was offered to the promotion gate on 40 and then 120 regression items; the gate refused both (−3.3 points at n = 120). No promotion, so no rollback: OR-005 waits for the first adapter or any candidate the owner names (decision 001).

- **What:** train the first Tazkiya adapter on a rented GPU and evaluate it with the promotion gate.
- **Why:** G3's TR-007 needs an adapter evaluated by `tazkiya.py promote`; the owner deferred GPU work.
- **Action:** `training/RUNBOOK.md` (data build, `train_unsloth.py`, `tazkiya.py promote --baseline-run <id> --candidate-run <id> --sign-off NAME`). The ledger holds far fewer verified System 2 traces than the roughly 5,000 the runbook asks for, so `tazkiya.py synthesize` must run first (System 2 on GSM8K and MBPP train splits: about 2 min per item on this CPU).
- **Evidence:** registry `promote` entry with the paired gain.
- **Moves:** G3 (TR-007).

## 4. Threshold decision for NFR-007 (gate G4): **decided 2026-10-21** (restated at the 2-session limit, decision 002)

- **What:** decide whether NFR-007 stays a CPU-profile requirement.
- **Why:** on four cores the four-session p95 ratio is 3.4 to 3.9 on a quiet host and no slot count reaches 2x (`docs/decisions/002-nfr007-single-session-profile.md`); G4 will stay `fail` on this machine whatever else is done.
- **Action:** choose one of the options in decision 002 (mark NFR-007 as GPU-profile only, or restate it at the recommended concurrency limit of 2 sessions). Either needs an edit to `data/requirements.csv` and `rust/crates/nafs-gateway/src/gates.rs`, which this run does not make.
- **Evidence:** the stored protocol runs already show the 2-session ratio (1.7 to 2.0x).
- **Moves:** G4 (NFR-007).
