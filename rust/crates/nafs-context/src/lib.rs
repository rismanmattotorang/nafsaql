//! nafs-context (Phase 29): one context filesystem for knowledge, memory, skills and sessions.
//!
//! The design follows OpenViking's (`third_party/OpenViking`, Volcengine) context database: one URI namespace, layered
//! summaries before sources, search scoped to a subtree, sessions committed as files with the memories drawn from
//! them, and compilation into a wiki or a graph. OpenViking's core is AGPL-3.0, so nothing is copied; this is a Rust
//! re-implementation that needs no model for any of it. This crate is pure logic (no I/O): the gateway joins it to
//! the stores that hold the items (`nafs-kb` for documents, `nafs-memory` for memory files, skills and sessions).
//! Specification: `docs/spec/context_fs_spec.md`.

pub mod assemble;
pub mod compile;
pub mod memory;
pub mod skill;
pub mod uri;

pub use uri::{Space, Uri, UriError};
