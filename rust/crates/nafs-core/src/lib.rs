//! nafs-core: the request automaton (A -> L -> M -> T/X), the Aql Gate and the backends.
pub mod backend;
pub mod calibration;
pub mod config;
pub mod continuity;
pub mod domain;
pub mod drift;
pub mod gate;
pub mod harness;
pub mod knowledge;
pub mod orchestrator;
pub mod parallel;
pub mod sequential;
pub mod support;

pub use backend::{Backend, BackendHealth, GenOutput, GenRequest};
pub use config::Config;
pub use orchestrator::Orchestrator;
