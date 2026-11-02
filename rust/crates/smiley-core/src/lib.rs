//! Smiley Core — a CPU inference engine that reports the bytes it moved.
//!
//! The design and the reasoning behind it are in `smiley/docs/SMILEY-CORE-SPEC.md`.
//! The short version: single-stream decode is memory-bandwidth-bound by an order of
//! magnitude, so the interesting quantity is not tokens per second but bytes moved
//! per token — and no incumbent engine reports it. This one does, which lets a run
//! be checked against its own roofline instead of taken on faith.
//!
//! The crate is a library with no knowledge of Tauri, SQLite or the UI. The host
//! adapter in `src-tauri/src/smileycore.rs` is the only file that knows both sides.

// nafs-aql: the workspace builds with `-D warnings`; these are upstream idioms, kept as they are.
// `chunks_exact_to_as_chunks` exists only in newer clippy (CI's stable is ahead of older local toolchains).
#![allow(
    unknown_lints,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    clippy::chunks_exact_to_as_chunks
)]

pub mod dspark;
pub mod gguf;
pub mod kv;
pub mod model;
pub mod pool;
pub mod prefix;
pub mod quant;
pub mod safetensors;
pub mod session;
#[cfg(target_arch = "x86_64")]
pub mod simd;
pub mod suffix;
pub mod telem;
pub mod template;
pub mod tokenizer;

pub use gguf::{Format, Gguf};
pub use session::{Session, SessionCfg};

/// Whether this engine can run the GGUF at `path`, read from its header alone.
///
/// `Ok(())` means it will load and chat correctly; `Err` names the first thing it
/// cannot do. This is the check the catalogue and the host run *before* offering a
/// pairing, because every way of failing it — another architecture, another
/// tokenizer, another chat template — otherwise surfaces at the first token as an
/// error or, worse, as fluent text from a prompt the model never saw.
///
/// What is implemented, and therefore what this accepts:
/// * the `llama` and `qwen2` architectures (`model::ARCHS`);
/// * a `gpt2` byte-level BPE vocabulary with the `smollm`, `llama-bpe` or `qwen2`
///   pre-tokenizer (`tokenizer::PRE_TYPES`) — each transcribed from llama.cpp and
///   checked against `llama-tokenize`; a near-miss tokenizer produces confident
///   wrong text, so nothing else is approximated;
/// * a ChatML or Llama 3 chat template (`template::TEMPLATES`), detected from the
///   file's own `tokenizer.chat_template`;
/// * rope scaling only as Llama 3's factor table or a linear factor;
/// * tensors in F32, F16, Q5_0, Q8_0, Q4_K or Q6_K (`Gguf::open` refuses others).
pub fn compatibility(path: &std::path::Path) -> Result<(), String> {
    let g = Gguf::open(path)?;
    check_compatible(&g)
}

pub(crate) fn check_compatible(g: &Gguf) -> Result<(), String> {
    let arch = g.str("general.architecture").unwrap_or("unknown");
    if model::Arch::from_name(arch).is_none() {
        return Err(format!(
            "architecture {arch:?} — Smiley Core implements {}",
            model::ARCHS.join(", ")
        ));
    }
    let tok = g.str("tokenizer.ggml.model").unwrap_or("unknown");
    if tok != "gpt2" {
        return Err(format!(
            "tokenizer {tok:?} — Smiley Core reads gpt2 byte-level BPE vocabularies"
        ));
    }
    let pre = g.str("tokenizer.ggml.pre").unwrap_or("smollm");
    if tokenizer::PreType::from_name(pre).is_none() {
        return Err(format!(
            "pre-tokenizer {pre:?} — Smiley Core implements {}",
            tokenizer::PRE_TYPES.join(", ")
        ));
    }
    if let Ok(t) = g.str(&format!("{arch}.rope.scaling.type")) {
        if t != "none" && t != "linear" {
            return Err(format!("rope scaling {t:?} is not implemented"));
        }
    }
    template::Template::detect(g)?;
    Ok(())
}

/// Which kernel path this build will actually take on this machine.
///
/// Reported in telemetry so that a slow run on a machine without AVX2 is never
/// mistaken for a slow engine — and so that a run claiming SIMD throughput can be
/// checked against whether SIMD was in fact available.
pub fn simd_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            if std::arch::is_x86_feature_detected!("fma") {
                return "avx2+fma";
            }
            return "avx2";
        }
        return "scalar";
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return "neon";
        }
        return "scalar";
    }
    #[allow(unreachable_code)]
    "scalar"
}
