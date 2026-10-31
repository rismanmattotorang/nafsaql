//! nafs-harness: the parts of two agent harnesses that Nafs-Aql integrates, as pure logic.
//!
//! - From **Open Math Harness** (Apache-2.0): Lean's textual rules and strict verify protocol (`lean`), the
//!   discovery scoring protocol (`scoring`) and a best-first candidate tree (`search`).
//! - From **OpenAI Codex** (Apache-2.0, Copyright 2025 OpenAI): the `apply_patch` format and fuzzy matcher
//!   (`patch`), middle truncation and the exec output format (`truncate`), and the shape of the turn loop
//!   (`agent`).
//!
//! Nothing here runs a process, opens a socket or touches the filesystem: Lean, Python and the models are
//! reached through closures and traits the host provides (`nafs-verify` runs both interpreters in its OS
//! sandbox). See `docs/research/harness_integration.md` and this crate's `NOTICE`.
pub mod agent;
pub mod context;
pub mod diff;
pub mod grammar;
pub mod lean;
pub mod patch;
pub mod scoring;
pub mod search;
pub mod truncate;
