//! Acceptance criterion A1: the kernels agree with llama.cpp on real weights.
//!
//! This is the test that matters most in the crate, because the failure it catches
//! is invisible by any other means. A quantised block layout read with the wrong
//! interleaving — high nibbles where low ones belong, a super-block scale applied to
//! the wrong 32 of its 256 weights — still decodes to finite, plausibly-distributed
//! floats. The model still loads. It still generates fluent English. It is simply
//! the wrong model, and nothing short of comparing against a known-good decoder
//! will say so.
//!
//! The golden values in `fixtures/dequant-golden.f32` were produced by llama.cpp's
//! own dequantisation code, not by a second transcription of it:
//!
//! ```text
//! llama-quantize --allow-requantize SmolLM2-360M-Instruct.Q4_K_M.gguf out.gguf F32 4
//! ```
//!
//! and then a 2048-element window of each of five tensors — one per weight format
//! the model uses — was copied out. The windows start at element 4096 rather than 0
//! so that they exercise block-offset arithmetic rather than only the first block.
//!
//! The model file itself is 270 MB and is not in the repository. Point
//! `SMILEY_CORE_MODEL` at it, or leave it at the default path below.

use smiley_core::gguf::{Format, Gguf};
use smiley_core::quant::dequant_row;
use std::path::PathBuf;

fn model_path() -> PathBuf {
    std::env::var_os("SMILEY_CORE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

fn fmt_of(s: &str) -> Format {
    match s {
        "F32" => Format::F32,
        "F16" => Format::F16,
        "Q5_0" => Format::Q5_0,
        "Q8_0" => Format::Q8_0,
        "Q4_K" => Format::Q4_K,
        "Q6_K" => Format::Q6_K,
        other => panic!("fixture names an unknown format {other:?}"),
    }
}

#[test]
fn dequantisation_matches_llama_cpp_on_the_real_model() {
    let path = model_path();
    if !path.exists() {
        // Skipping is a real cost — this is the only test that can catch a wrong
        // block layout — so say so loudly rather than passing quietly.
        eprintln!(
            "\n  SKIPPED: A1 (kernels vs llama.cpp) needs the model at {path:?}.\n  \
             Set SMILEY_CORE_MODEL to run it. No other test covers block layout.\n"
        );
        return;
    }
    let g = Gguf::open(&path).expect("open model");

    let tsv = include_str!("fixtures/dequant-golden.tsv");
    let blob: &[u8] = include_bytes!("fixtures/dequant-golden.f32");

    let mut checked = 0usize;
    let mut formats = Vec::new();
    for line in tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 5, "malformed fixture line: {line}");
        let (name, fmt, start, count, off) = (
            f[0],
            fmt_of(f[1]),
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
            f[4].parse::<usize>().unwrap(),
        );

        let ti = g.tensor(name).expect(name);
        assert_eq!(
            ti.kind,
            fmt,
            "{name} is {} in the model but the fixture says {}; the fixture is stale",
            ti.kind.label(),
            fmt.label()
        );

        // Decode exactly the window the fixture covers. `start` is block-aligned, so
        // the byte offset is exact and no partial block is involved.
        let blk = fmt.block();
        assert_eq!(start % blk, 0);
        assert_eq!(count % blk, 0);
        let bytes = g.bytes(ti);
        let b0 = start / blk * fmt.block_bytes();
        let b1 = (start + count) / blk * fmt.block_bytes();
        let mut got = vec![0.0f32; count];
        dequant_row(fmt, &bytes[b0..b1], &mut got);

        for i in 0..count {
            let want = f32::from_le_bytes(blob[off + i * 4..off + i * 4 + 4].try_into().unwrap());
            // Both sides decode the same stored codes with the same arithmetic, so
            // this is an equality check up to f32 rounding order, not a tolerance
            // on a numerical method. A layout error is off by orders of magnitude,
            // not by an ulp.
            let tol = 1e-6 * want.abs().max(1e-3);
            assert!(
                (got[i] - want).abs() <= tol,
                "{name} [{}] element {}: smiley-core {} vs llama.cpp {}",
                fmt.label(),
                start + i,
                got[i],
                want
            );
        }
        // A window of all zeros would pass the comparison while proving nothing.
        let nonzero = got.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > count / 2,
            "{name}: only {nonzero}/{count} decoded values are non-zero — \
             the window is not exercising the decoder"
        );
        checked += count;
        formats.push(fmt);
    }

    // The fixture must actually cover every format the model contains, or A1 is
    // only evidence about the formats it happens to include.
    let mut present: Vec<Format> = g.tensors.values().map(|t| t.kind).collect();
    present.sort_by_key(|f| f.label());
    present.dedup();
    for f in present {
        assert!(
            formats.contains(&f),
            "model contains {} but the golden fixture does not cover it",
            f.label()
        );
    }
    eprintln!(
        "A1: {checked} weights checked against llama.cpp across {} formats",
        formats.len()
    );
}

#[test]
fn the_model_loads_with_the_shape_the_spec_describes() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model at {path:?}");
        return;
    }
    let g = Gguf::open(&path).expect("open model");

    assert_eq!(g.str("general.architecture").unwrap(), "llama");
    assert_eq!(g.u32("llama.block_count").unwrap(), 32);
    assert_eq!(g.u32("llama.embedding_length").unwrap(), 960);
    assert_eq!(g.u32("llama.attention.head_count").unwrap(), 15);
    assert_eq!(g.u32("llama.attention.head_count_kv").unwrap(), 5);
    assert_eq!(g.u32("llama.feed_forward_length").unwrap(), 2560);
    assert_eq!(g.u32("llama.rope.dimension_count").unwrap(), 64);
    assert_eq!(g.tensors.len(), 290);

    // Tied embeddings: this model has no separate output projection, and the
    // loader must notice by absence rather than assume either way.
    assert!(g.tensors.contains_key("token_embd.weight"));
    assert!(!g.tensors.contains_key("output.weight"));

    // Every tensor's declared bytes must lie inside the map; `Gguf::open` checks
    // this, so reaching here at all is the assertion. Confirm the total is sane.
    let total = g.total_weight_bytes();
    let file = std::fs::metadata(&path).unwrap().len();
    assert!(
        total < file && total > file * 9 / 10,
        "weights {total} vs file {file} — directory and data disagree"
    );
}
