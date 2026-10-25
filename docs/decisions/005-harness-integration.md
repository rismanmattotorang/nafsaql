# Decision 005: Math and code harness from Open Math Harness and Codex

Date 2026-10-24. Status: **adopted**, with patch repair **off** until measured. Research note:
[`docs/research/harness_integration.md`](../research/harness_integration.md). Report: section "Phase 10".

## Context

The owner asked to assess Open Math Harness (revision `687f2a2`) and OpenAI Codex's `codex-rs` (revision
`970111f6`), both Apache-2.0, and integrate what serves Nafs-Aql's mathematics and coding paths. Neither can be
embedded: Open Math Harness's runtime needs Linux and bubblewrap and is a microkernel of confined processes; Codex
speaks only the Responses API, assumes native tool calls and is about 150 crates. Both have leaves that are pure
logic and fit Nafs-Aql's rule that exact checkers decide.

## Decision

| Requirement | What | Default |
|---|---|---|
| FR-090 | Lean 4 as an exact verifier (`verification.lean`), Open Math Harness's frozen-statement protocol, Lean in the OS sandbox; requests refused when no toolchain is configured | available when `verifiers.lean` is set |
| FR-091 | Best-first repair over a candidate tree scored by the verifier (OMH scoring; per-assert credit on code; Lean's graded verdicts) | `repair_policy = "best"` (equals the old loop under all-or-nothing verdicts, report Proposition 18) |
| FR-092 | Patch repair: a code retry may answer with a Codex `apply_patch` to the candidate | `patch_repair = false` |
| FR-093 | Code agent `POST /v1/harness/code`: patch / whole file / run / submit in the sandbox, ends only on the verifier's acceptance, context reset every step | enabled, System 2, 6 steps |
| — | Verifier evidence keeps head and tail (Codex middle truncation); Seatbelt paths passed as `-D` parameters (Codex pattern) | always |

The derived code lives in `rust/crates/nafs-harness` (pure logic, 56 unit tests including the ported upstream
suites), with each file's source and modifications in its header and both notices in the crate's `NOTICE`.

## Not adopted

- Open Math Harness's kernel, platform, IPC and launcher (Linux + bubblewrap; Nafs-Aql is one process).
- Ultramath (eight members at maximum effort; minutes per member on this CPU). Its "already tried" idea is kept in
  repair prompts.
- Codex's Responses-API client, tool router, approvals, MCP and compaction (frontier-model assumptions).
- Codex `execpolicy` (Starlark): the agent cannot run a shell, only Python files of its own workspace in the
  sandbox, so a command policy has nothing to decide.
- The Lean REPL `analyze` / `complete` passes (need leanprover-community/repl and Mathlib).

## Re-open / flip conditions

- **Patch repair on:** an interleaved A/B (`patch_repair` true vs false, same items, alternating) of at least 20
  code items whose first System 2 attempt fails, showing no loss of verified answers and lower mean wall time;
  record the patch failure rate f and check Eq. (patch) of the report.
- **Lean in production:** install a toolchain (`elan`, core Lean needs no Mathlib), set `verifiers.lean`, and run the
  verifier tests against the real binary; Mathlib statements additionally need a Mathlib build and `lean_path`.
