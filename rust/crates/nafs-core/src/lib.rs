//! nafs-core: the request automaton (A -> L -> M -> T/X), the Aql Gate and the backends.
pub mod backend;
pub mod calibration;
pub mod config;
pub mod domain;
pub mod drift;
pub mod gate;
pub mod harness;
pub mod live_cost;
pub mod orchestrator;
pub mod sequential;

pub use backend::{Backend, BackendHealth, GenOutput, GenRequest};
pub use config::Config;
pub use orchestrator::Orchestrator;
