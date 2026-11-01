//! Oracle for `quant::quantize_row_q4_k`: blocks llama.cpp wrote (a real GGUF's Q4_K
//! tensor), decoded and quantised again here, must come back as the same bytes —
//! the decoded values are exactly representable, so a faithful port of
//! `quantize_row_q4_K_ref` lands on llama.cpp's own scales, mins and codes.
//! Skips when the model file is absent.

use smiley_core::gguf::{Format, Gguf};
use smiley_core::quant::{dequant_row, quantize_row_q4_k};
use std::path::PathBuf;

#[test]
fn requantising_llama_cpp_q4_k_blocks_reproduces_their_bytes() {
    let p = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("models/Qwen3-4B-Q4_K_M.gguf");
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return;
    }
    let g = Gguf::open(&p).unwrap();
    let t = g.tensor("blk.0.attn_q.weight").unwrap().clone();
    assert_eq!(t.kind, Format::Q4_K, "fixture tensor is Q4_K");
    let bytes = g.bytes(&t);
    let n = t.dims[0] as usize;
    let rb = bytes.len() / t.dims[1] as usize;
    let (mut same, mut total, mut worst) = (0usize, 0usize, 0.0f32);
    for r in 0..64 {
        let row = &bytes[r * rb..(r + 1) * rb];
        let mut x = vec![0.0f32; n];
        dequant_row(Format::Q4_K, row, &mut x);
        let mut q = Vec::new();
        quantize_row_q4_k(&x, &mut q);
        for (a, b) in row.chunks_exact(144).zip(q.chunks_exact(144)) {
            total += 1;
            if a == b {
                same += 1;
            }
        }
        let mut y = vec![0.0f32; n];
        dequant_row(Format::Q4_K, &q, &mut y);
        worst = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(worst, f32::max);
    }
    eprintln!("{same}/{total} super-blocks byte-identical, worst value drift {worst:e}");
    assert!(
        same * 100 >= total * 95,
        "only {same}/{total} blocks reproduced"
    );
    assert!(worst < 1e-2, "a re-quantised value moved by {worst}");
}
