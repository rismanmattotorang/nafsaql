// nafs-aql: upstream test kept as it is; the workspace lints with -D warnings.
#![allow(clippy::needless_range_loop)]
//! Acceptance criterion A4: the AVX2 kernels agree with the scalar twins.
//!
//! Checked on the *real* weight file, not on synthetic bytes. Random bytes exercise
//! the arithmetic but not the distribution: real quantised weights cluster, real
//! block scales span a narrow range, and a bug in the high-bit expansion of Q5_0 or
//! the 2-bit gather of Q6_K can easily be invisible on uniform noise while being
//! systematic on trained weights.
//!
//! The scalar path is the specification. When these disagree, the SIMD kernel is
//! wrong by definition.

use smiley_core::gguf::{Format, Gguf};
use smiley_core::quant::{dot, dot_scalar, row_bytes, QuantAct};
use std::path::PathBuf;

fn model_path() -> PathBuf {
    std::env::var_os("SMILEY_CORE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

/// Deterministic activations with a realistic spread, including values that push the
/// int8 quantiser to its clamp.
fn activations(n: usize, seed: u64) -> QuantAct {
    let mut s = seed | 1;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0
        })
        .collect();
    let mut a = QuantAct::with_capacity(n);
    a.fill(&x);
    a
}

#[test]
fn simd_and_scalar_agree_on_every_row_of_real_tensors() {
    let path = model_path();
    if !path.exists() {
        eprintln!("\n  SKIPPED: A4 (SIMD ≡ scalar) needs the model at {path:?}\n");
        return;
    }
    if !smiley_core::simd_name().starts_with("avx2") {
        eprintln!("  SKIPPED: no AVX2 on this machine, both paths are the scalar one");
        return;
    }
    let g = Gguf::open(&path).expect("open model");

    // One tensor per quantised format, chosen so all four kernels are exercised.
    let names = [
        "token_embd.weight",      // Q8_0
        "blk.0.attn_q.weight",    // Q5_0
        "blk.0.ffn_down.weight",  // Q6_K
        "blk.12.ffn_down.weight", // Q4_K
    ];
    let mut seen: Vec<Format> = Vec::new();
    let mut rows_checked = 0usize;

    for name in names {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = row_bytes(t.kind, n);
        let bytes = g.bytes(t);
        let a = activations(n, 0xC0FF_EE00 + n as u64);

        // Every row would be slow on a 49k-row embedding table; take a stride that
        // still covers thousands of rows and both ends of the tensor.
        let step = (rows / 512).max(1);
        let kq = matches!(t.kind, Format::Q4_K | Format::Q6_K);
        let mut deq = vec![0.0f32; n];
        for r in (0..rows).step_by(step) {
            let row = &bytes[r * rb..(r + 1) * rb];
            let want = dot_scalar(t.kind, row, &a);
            let got = dot(t.kind, row, &a);
            // Both accumulate the same integer products and differ only in the
            // order of the float accumulation — but a row whose terms nearly
            // cancel can leave a small result with an error that is a few ulps
            // of the *terms*. So each path is held to the exact (f64) value of
            // the decoded dot product within a few ulps of the terms' magnitude,
            // which catches a wrong lane (an error of a few percent of that
            // magnitude) and tolerates summation order.
            smiley_core::quant::dequant_row(t.kind, row, &mut deq);
            let (mut exact, mut mag) = (0.0f64, 0.0f64);
            for i in 0..n {
                let act = if kq {
                    a.qk[i] as f64 * a.dk[i / 256] as f64
                } else {
                    a.q[i] as f64 * a.d[i / 32] as f64
                };
                let term = deq[i] as f64 * act;
                exact += term;
                mag += term.abs();
            }
            let tol = 4e-6 * mag + 1e-7;
            assert!(
                (got as f64 - exact).abs() <= tol && (want as f64 - exact).abs() <= tol,
                "{name} [{}] row {r}: simd {got} vs scalar {want} vs exact {exact} (|terms| {mag})",
                t.kind.label()
            );
            rows_checked += 1;
        }
        if !seen.contains(&t.kind) {
            seen.push(t.kind);
        }
    }
    assert_eq!(
        seen.len(),
        4,
        "not every quantised format was exercised: {seen:?}"
    );
    eprintln!(
        "A4: {rows_checked} rows agree across {} formats",
        seen.len()
    );
}

/// A dot product of all-zero activations must be exactly zero in both paths — the
/// bias corrections have to cancel rather than leave a residue proportional to the
/// block scale.
#[test]
fn both_paths_are_exactly_zero_on_a_zero_activation() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.12.ffn_down.weight",
    ] {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let mut a = QuantAct::with_capacity(n);
        a.fill(&vec![0.0f32; n]);
        let row = &g.bytes(t)[..row_bytes(t.kind, n)];
        assert_eq!(dot_scalar(t.kind, row, &a), 0.0, "{name} scalar");
        assert_eq!(dot(t.kind, row, &a), 0.0, "{name} simd");
    }
}

// ---------------------------------------------------------------------------
// The batched kernels. `dot_batch` decodes a weight row once per tile of four
// activations; it must agree with the per-activation scalar specification for every
// batch width, including the 1/2/3 remainders of the tile loop.
// ---------------------------------------------------------------------------

fn prng_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

/// Raw random bytes hit half-precision Inf/NaN about one block in 128, and a NaN scale
/// makes both sides NaN, so the comparison would pass or fail for no reason that has
/// to do with the kernel. Clear the exponent's top bits: a normal half in [0.125, 0.25).
fn sane(fmt: Format, w: &mut [u8]) {
    let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
    for blk in w.chunks_exact_mut(fmt.block_bytes()) {
        match fmt {
            Format::Q8_0 | Format::Q5_0 => fix(blk, 0),
            Format::Q4_K => {
                fix(blk, 0);
                fix(blk, 2);
            }
            Format::Q6_K => fix(blk, 208),
            _ => {}
        }
    }
}

#[test]
fn batched_kernels_match_the_scalar_specification_at_every_batch_width() {
    use smiley_core::quant::dot_batch;
    for (fmt, n) in [
        (Format::Q8_0, 960),
        (Format::Q5_0, 960),
        (Format::Q4_K, 2560),
        (Format::Q6_K, 2560),
    ] {
        let mut w = prng_bytes(0xBEEF_0000 + n as u64, row_bytes(fmt, n));
        sane(fmt, &mut w);
        // 1..=9 covers the empty remainder, each of 1/2/3, and two full tiles plus one.
        for nb in 1..=9usize {
            let acts: Vec<QuantAct> = (0..nb)
                .map(|i| activations(n, 0x1000 + i as u64 * 7919))
                .collect();
            let mut got = vec![0.0f32; nb];
            dot_batch(fmt, &w, &acts, &mut got);
            for (b, a) in acts.iter().enumerate() {
                let want = dot_scalar(fmt, &w, a);
                // Same integer products; only the float accumulation order and the
                // bias being subtracted at the end rather than per block differ.
                let tol = 2e-4 * want.abs().max(1.0);
                assert!(
                    (got[b] - want).abs() <= tol,
                    "{} nb={nb} act {b}: batched {} vs scalar {want}",
                    fmt.label(),
                    got[b]
                );
            }
        }
    }
}

#[test]
fn batched_kernels_agree_on_rows_of_the_real_model() {
    use smiley_core::quant::dot_batch;
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model at {path:?}");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    let mut checked = 0;
    for name in [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.12.ffn_down.weight",
    ] {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = row_bytes(t.kind, n);
        let bytes = g.bytes(t);
        let acts: Vec<QuantAct> = (0..6).map(|i| activations(n, 0xAB00 + i)).collect();
        let step = (rows / 128).max(1);
        for r in (0..rows).step_by(step) {
            let row = &bytes[r * rb..(r + 1) * rb];
            let mut got = vec![0.0f32; acts.len()];
            dot_batch(t.kind, row, &acts, &mut got);
            for (b, a) in acts.iter().enumerate() {
                let want = dot_scalar(t.kind, row, a);
                // Relative, plus a small absolute floor: a near-cancelling dot product
                // (a result of 0.02 built from partial sums in the tens) carries float
                // rounding from the *order* of accumulation, which is a property of the
                // sum and not of the kernel. A wrong nibble order or scale is off by
                // 0.1 or more, five orders above this floor.
                let tol = 2e-4 * want.abs() + 2e-5;
                assert!(
                    (got[b] - want).abs() <= tol,
                    "{name} row {r} act {b}: {} vs {want}",
                    got[b]
                );
                checked += 1;
            }
        }
    }
    eprintln!("batched ≡ scalar on {checked} (row, activation) pairs of real weights");
}
