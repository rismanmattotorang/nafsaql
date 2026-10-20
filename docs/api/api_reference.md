# Nafs-Aql API reference (v1.5.1)

Generated from `api/openapi.yaml` by `scripts/gen_api_reference.py`; do not edit by hand.

External API of the Nafs-Aql Rust gateway (crate `nafs-gateway`). Every reply exposes the
reasoning mode that produced it (`fast` or `slow`), the assurance level, and the controller
decision identifier. Streaming uses Server-Sent Events (SSE); chat rooms use WebSocket.
Requirement IDs (IF-001..IF-010, FR-002, FR-006) refer to the Nafs-Aql Technical Specification.
Version 1.1 adds conversation management, the verifier playground, system/backend control, dataset browsing, run cancellation and calibration fitting, as implemented by nafs-gateway 0.1 (Phase 1).
Version 1.2 (Phase 2) adds the rubric judge, the audit stream and self-model (Prop. 5), gate modes (fixed, calibrated, budget), recalibration from the ledger with held-out promotion, measured gate parameters (Beta-fit caps, budget curve), verifier-rate measurement, new evaluation systems (nafs_blind, slow_conditioned, slow_b<N>), request bounds enforced with HTTP 400, and same-origin enforcement (HTTP 403 for foreign Origin or Host).
Version 1.3 (Phases 3-4) adds human feedback, review and opt-out (Tazkiya), Both-trial preferences, the gates dashboard, drift monitoring, the load-test protocol, McNemar/Bonferroni statistics and report export.
Version 1.4 (Phase 5) adds alerts with a drill, lab insights, resumable evaluation runs, retrieval memory (retrieve / retrieved) and the security posture in /v1/system.

Version 1.5 (Phase 7) adds per-stage latency in usage.stages (queue, prefill, decode and KV-cache reuse for model stages) and GET /v1/perf/stages, the latency breakdown by stage.

Authentication: send `Authorization: Bearer <token>` when the gateway runs with `NAFS_AUTH_TOKENS`. Browser requests must be same-origin (cross-origin requests get 403).

## bench

### `GET /v1/bench/prompt-sets`

Built-in prompt sets for load tests

Responses: `200`, `403`.

### `POST /v1/bench/protocol`

Recorded load-test protocol at 1, 2 and 4 sessions with p50/p95/p99 and the NFR-007 verdict (ER-007)

Request body: `prompt_set`, `mode`, `levels`, `requests_per_session`, `max_tokens` (* = required).

Responses: `202`, `400`, `403`.

### `GET /v1/bench/runs`

List performance-test runs

Responses: `200`, `403`.

### `POST /v1/bench/runs`

Start a load test (FR-052)

Request body: `BenchRunRequest` (* = required).

Responses: `202`, `403`.

### `GET /v1/bench/runs/{run_id}`

Fetch a performance-test run with latency quantiles

Parameters: `run_id` (path, required).

Responses: `200`, `404`, `403`.

### `POST /v1/bench/runs/{run_id}/cancel`

Cancel an active run after in-flight items finish

Parameters: `run_id` (path, required).

Responses: `202`, `404`, `403`.

## chat

### `POST /v1/chat`

Complete one chat turn (non-streaming)

Request body: `ChatRequest` (* = required).

Responses: `200`, `400`, `401`, `429`, `503`, `403`.

### `POST /v1/chat/stream`

Complete one chat turn as an SSE stream of tokens and controller events

Request body: `ChatRequest` (* = required).

Responses: `200`, `400`, `401`, `429`, `403`.

### `GET /v1/conversations`

List persisted conversations (most recent first)

Responses: `200`, `403`.

### `DELETE /v1/conversations/{conversation_id}`

Delete a conversation and its messages

Parameters: `conversation_id` (path, required).

Responses: `204`, `404`, `403`.

### `GET /v1/conversations/{conversation_id}`

Fetch a conversation with per-message reasoning mode and assurance

Parameters: `conversation_id` (path, required).

Responses: `200`, `404`, `403`.

### `PATCH /v1/conversations/{conversation_id}`

Rename a conversation and/or set its training opt-out (SR-004)

Parameters: `conversation_id` (path, required).

Request body: `title`, `training_opt_out` (* = required).

Responses: `204`, `404`, `403`.

### `GET /v1/decisions/{decision_id}`

Fetch the controller decision record for a reply

Parameters: `decision_id` (path, required).

Responses: `200`, `404`, `403`.

### `POST /v1/replies/{reply_id}/feedback`

Human rating of a reply (+1 / -1): a soft signal shown in the ledger, never a calibration label

Parameters: `reply_id` (path, required).

Request body: `rating`*, `note` (* = required).

Responses: `204`, `400`, `404`, `403`.

## eval

### `GET /v1/eval/datasets`

Evaluation datasets with task counts, domains and licences

Responses: `200`, `403`.

### `GET /v1/eval/datasets/{name}`

First tasks of a dataset

Parameters: `name` (path, required), `limit` (query).

Responses: `200`, `404`, `403`.

### `GET /v1/eval/runs`

List evaluation runs

Responses: `200`, `403`.

### `POST /v1/eval/runs`

Start an evaluation run (FR-050)

Request body: `EvalRunRequest` (* = required).

Responses: `202`, `403`.

### `GET /v1/eval/runs/{run_id}`

Fetch an evaluation run with metrics and paired statistics

Parameters: `run_id` (path, required).

Responses: `200`, `404`, `403`.

### `POST /v1/eval/runs/{run_id}/calibrate`

Fit the e1_hat calibration from a run (k-fold held-out ECE); promoted only if NFR-020 holds (FR-015)

Parameters: `run_id` (path, required).

Request body: `include_ledger` (* = required).

Responses: `200`, `400`, `404`, `403`.

### `POST /v1/eval/runs/{run_id}/cancel`

Cancel an active run after in-flight items finish

Parameters: `run_id` (path, required).

Responses: `202`, `404`, `403`.

### `GET /v1/eval/runs/{run_id}/export`

Report export: Markdown summary, item CSV or full JSON (P4-T01)

Parameters: `run_id` (path, required), `format` (query).

Responses: `200`, `404`, `403`.

### `POST /v1/eval/runs/{run_id}/resume`

Resume an interrupted or cancelled evaluation run, keeping completed items (runs interrupted by a restart resume automatically)

Parameters: `run_id` (path, required).

Responses: `202`, `400`, `404`, `403`.

## gate

### `GET /v1/gate`

Gate mode, promotion verdict, calibration and parameter provenance

Responses: `200`, `403`.

### `POST /v1/gate`

Set the gate mode (FR-014 budget mode) and escalation budget at runtime

Request body: `mode`, `escalation_budget` (* = required).

Responses: `200`, `400`, `403`.

### `POST /v1/gate/derive`

Derive e2_hat, delta_c, Beta-fit caps (Prop. 3) and b* (Prop. 9) from evaluation runs; apply = true installs them (FR-011, FR-021)

Request body: `run_ids`, `min_n`, `apply` (* = required).

Responses: `200`, `400`, `404`, `403`.

### `DELETE /v1/gate/params`

Discard measured parameters and use the configuration placeholders

Responses: `200`, `403`.

### `GET /v1/gate/params`

Measured parameters and their provenance

Responses: `200`, `403`.

### `POST /v1/gate/recalibrate`

Refit e1_hat from the ledger (verified drafts and exact-labelled audits) plus optional eval runs; promote only if NFR-020 holds (FR-015)

Request body: `run_ids`, `min_n`, `apply` (* = required).

Responses: `200`, `400`, `403`.

## lab

### `GET /v1/lab/insights`

Pooled policy map: best System 1 / System 2 policy per domain and price of latency (Prop. 2), value of deliberation, human preferences

Parameters: `lambda` (query).

Responses: `200`, `403`.

### `GET /v1/preferences`

Value of deliberation per domain from Both trials (share preferring System 2, Clopper-Pearson CI)

Responses: `200`, `403`.

### `POST /v1/preferences`

Record a Both trial: the same prompt answered by System 1 and System 2 and the person's choice

Request body: `prompt`*, `domain`, `fast_reply_id`*, `slow_reply_id`*, `preferred`*, `note` (* = required).

Responses: `201`, `400`, `404`, `403`.

## ledger

### `GET /v1/audits`

Audit stream records and counts (FR-040)

Parameters: `limit` (query).

Responses: `200`, `403`.

### `POST /v1/audits/run-next`

Run the oldest pending audit now (normally done by the idle worker)

Responses: `200`, `503`, `403`.

## phase4

### `GET /v1/drift`

CUSUM drift charts of calibration error and fast-path accuracy per domain with a shuffled negative control (FR-042)

Parameters: `k` (query), `h` (query), `reference` (query).

Responses: `200`, `403`.

### `GET /v1/gates`

Roadmap gates G0-G4: go/no-go per criterion from stored runs, acceptance results, H7, the model registry and protocol runs (FR-053)

Responses: `200`, `403`.

## phase5

### `GET /v1/alerts`

Alert rules over the recent window: fast-path latency, error rate, escalation rate, verifier acceptance, calibration drift (OR-004)

Responses: `200`, `403`.

### `POST /v1/alerts/drill`

Alert drill: every rule reports firing for the given seconds (10..3600, default 120)

Request body: `seconds` (* = required).

Responses: `200`, `403`.

## phase7

### `GET /v1/perf/stages`

Latency breakdown by stage and engine speed (prefill, decode, cache reuse) over recent replies (NFR-030, P7-T01)

Parameters: `hours` (query), `lab` (query).

Responses: `200`, `403`.

## rooms

### `GET /v1/rooms`

List chat rooms, most recently active first

Responses: `200`, `403`.

### `POST /v1/rooms`

Create a chat room

Request body: `name`*, `profile` (* = required).

Responses: `201`, `400`, `409`, `403`.

### `DELETE /v1/rooms/{room_id}`

Delete a room and everything stored from it

Responses: `200`, `404`, `403`.

### `GET /v1/rooms/{room_id}`

One room

Responses: `200`, `404`, `403`.

### `PATCH /v1/rooms/{room_id}`

Rename a room

Request body: `name`* (* = required).

Responses: `200`, `400`, `404`, `409`, `403`.

### `GET /ws/rooms/{room_id}`

WebSocket upgrade for a chat room (frames are `RoomFrame` JSON objects)

Parameters: `room_id` (path, required), `user` (query).

Responses: `101`, `400`, `404`, `403`.

## system

### `GET /healthz`

Liveness and backend readiness

Responses: `200`, `403`.

### `GET /metrics`

Prometheus metrics (NFR-021)

Responses: `200`, `403`.

### `GET /openapi.json`

This document (IF-001)

Responses: `200`, `403`.

### `GET /v1/self-model`

Per-domain capability table with the audit-weighted estimator (FR-041, Prop. 5); unmeasured below 30 audits

Responses: `200`, `403`.

### `GET /v1/system`

Backends (live health), gate, deliberation caps, calibration, host

Responses: `200`, `403`.

### `GET /v1/system/activity`

Most recent replies (mode, assurance, latency) for dashboards

Responses: `200`, `403`.

### `POST /v1/system/backend`

Switch the active backend set at runtime (llama | vllm) Refused with 503 when the target backend is not reachable, unless force is true.

Request body: `name`*, `force` (* = required).

Responses: `200`, `400`, `403`, `503`.

### `POST /v1/verify`

Run the exact verifiers on an answer (verifier playground, FR-030)

Request body: `VerifyRequest` (* = required).

Responses: `200`, `400`, `403`.

## tazkiya

### `GET /v1/tazkiya`

Training-data readiness per domain, opt-outs, feedback, reviews and the model registry (Phase 3)

Responses: `200`, `403`.

### `GET /v1/tazkiya/review`

Consensus answers awaiting human review before they may enter training (FR-033)

Parameters: `limit` (query).

Responses: `200`, `403`.

### `POST /v1/tazkiya/review/{reply_id}`

Approve or reject a consensus answer for the distillation set (FR-033)

Parameters: `reply_id` (path, required).

Request body: `approved`*, `reviewer`, `note` (* = required).

Responses: `204`, `403`.

## verifiers

### `POST /v1/judge`

Score an answer with the rubric judge (FR-034): pass iff h, a, u >= 0.5 and V >= 0.7

Request body: `question`*, `answer`* (* = required).

Responses: `200`, `400`, `503`, `403`.

### `GET /v1/verifiers`

Published (r, phi) per verifier (FR-032)

Responses: `200`, `403`.

### `GET /v1/verifiers/runs`

Verifier measurement runs

Responses: `200`, `403`.

### `POST /v1/verifiers/runs`

Mutation-test the exact verifiers (and optionally the judge) and publish (r, phi) (ER-003)

Request body: `items_per_dataset`, `include_judge`, `concurrency`, `code_datasets`, `judge_only` (* = required).

Responses: `202`, `403`.

### `GET /v1/verifiers/runs/{run_id}`

One measurement run

Parameters: `run_id` (path, required).

Responses: `200`, `404`, `403`.

### `POST /v1/verifiers/runs/{run_id}/cancel`

Cancel a measurement run

Parameters: `run_id` (path, required).

Responses: `202`, `404`, `403`.

## Schemas

- `Error`
- `ModelProfile`: Model profile (MR-001)
- `ReasoningMode`: Path that produced the reply (IF-003)
- `Assurance`: FR-031
- `Domain`
- `ChatRequest`
- `Message`
- `Usage`
- `ChatReply`
- `VerifierResult`
- `DecisionRecord`
- `StreamEvent`: SSE frame payload; the SSE event name equals `event`. Discriminated by `event`.
- `Conversation`
- `Room`
- `RoomFrame`: A WebSocket text frame. `welcome` (to the new member only, always first): `{you, room, users, assistant, limits, cursor, pending}`. `presence`: `{joined, members, users}` with the member in `user`. `user_message`: `{content, id, ts}` with the sender in `user`. `assistant_event`: `{reply_to, event}` where `event` is a stream event (`{event, data}`) for the question with id `reply_to`. `queue`: `{busy, working_for, queued}`. `typing`: the typist in `user`. `room_renamed`: `{name}`. `room_deleted`: the socket closes with code 4004 right after. `heartbeat` every 25 s; `pong` answers a client `ping`. `error`: `{code, message}` to one member (`E_RATE_LIMITED`, `E_QUEUE_FULL`, `E_TOO_LONG`, `E_ROOM_FULL`). `resync`: `{missed}` frames were skipped; reconnect.
- `BenchRunRequest`
- `BenchRun`
- `Quantiles`
- `EvalRunRequest`
- `EvalRun`
- `SelfModel`
- `TraceState`: Automaton states: draft, flagged, deliberate, accepted, abstain
- `ConversationSummary`
- `Dataset`
- `Calibration`
- `VerifyRequest`
- `SystemInfo`: Backends with live health, gate and deliberation configuration, calibration, host facts
- `Rates`: Mutation-measured verifier quality (FR-032)
- `CalibrationResult`
- `GateState`
- `StageTiming`: One stage of a request (NFR-030). Model stages carry the server timings.
