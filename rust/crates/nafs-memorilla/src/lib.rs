//! nafs-memorilla: Memorilla for Nafs-Aql (Phase 20, decision 014) on Candle.
//!
//! Phase 2 holds the **reference text encoder**: Qwen3-Embedding in f32 from its safetensors, last-token pooling and
//! L2 normalisation, exactly as Memorilla's `embed_texts` runs it through vLLM. Serving uses Smiley Core's quantised
//! copy (`nafs-smiley-server --embedding`); this one checks it (parity test) and is the encoder the memory-module
//! trainer of Phase 6 reads. The memory module itself comes in Phase 4.

pub mod encoder;

pub use encoder::{Encoder, END_OF_TEXT};
