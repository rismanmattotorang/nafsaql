//! nafs-memorilla: Memorilla for Nafs-Aql (Phase 20, decision 014) on Candle.
//!
//! Phase 2 holds the **reference text encoder**: Qwen3-Embedding in f32 from its safetensors, last-token pooling and
//! L2 normalisation, exactly as Memorilla's `embed_texts` runs it through vLLM. Serving uses Smiley Core's quantised
//! copy (`nafs-smiley-server --embedding`); this one checks it (parity test) and is the encoder the memory-module
//! trainer of Phase 6 reads.
//!
//! Phase 4 adds the **memory module** (`memory`), matching `memorilla/memory.py` (parity fixtures in `tests/`).

pub mod encoder;
pub mod memory;

pub use encoder::{Encoder, END_OF_TEXT};
pub use memory::{MemoryConfig, MemoryModule};
