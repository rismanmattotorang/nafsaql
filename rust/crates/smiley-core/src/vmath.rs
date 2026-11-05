//! Vector transcendental math: `exp`, softmax and SwiGLU.
//!
//! Before this module every `exp` in the engine was a scalar libm call — one per
//! attention score, one per SwiGLU element, one per vocabulary entry when sampling.
//! At 4 096 positions that is ~34 µs per (head, layer) of softmax alone on this CPU,
//! more than the dot products it normalises, and it grows with context.
//!
//! The vector `exp` is ggml's (`ggml-cpu/vec.h`, `ggml_v_expf`, itself adapted from
//! Arm's optimized-routines): max error 1.45 ulp + 0.5, flushing above 88.38 to
//! infinity and below −103.97 to zero. Porting llama.cpp's own routine rather than
//! writing another polynomial is deliberate — llama.cpp is this engine's correctness
//! oracle, and matching its softmax and SwiGLU arithmetic keeps the two engines'
//! outputs as close as the rest of the pipeline allows. The softmax sum accumulates
//! each eight-wide chunk into an `f64`, as `ggml_vec_soft_max_f32` does.
//!
//! Every function has a scalar path with identical semantics (tails and non-AVX2
//! machines), and the tests pin the vector path against it.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_ok() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// Eight `exp`s at once — ggml's `ggml_v_expf`, transcribed.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp8(x: __m256) -> __m256 {
    let r = _mm256_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8_aa3b)), r); // log2(e)
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
        _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f31_7200)), x), // 0x1.62e4p-1
    );
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(126.0),
        _CMP_GT_OQ,
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3c07_2010)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3d2b_9f17)),
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3e2a_af33)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3eff_fedb)),
            ),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7f_fff6)), b),
    );
    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(192.0),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

/// `x[i] = exp(x[i] − shift)` in place; returns `Σ x[i]` after the update.
///
/// The softmax numerator and normaliser in one pass. The caller divides.
pub fn exp_shift_sum(x: &mut [f32], shift: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked.
            return unsafe { exp_shift_sum_avx2(x, shift) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - shift).exp();
        sum += *v as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_shift_sum_avx2(x: &mut [f32], shift: f32) -> f64 {
    let s = _mm256_set1_ps(shift);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), s));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(v) as f64;
        i += 8;
    }
    while i < x.len() {
        x[i] = (x[i] - shift).exp();
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// `x[i] = exp((x[i] − shift) · scale)` in place; returns the sum. Sampling's
/// temperature form: `scale = 1/T`.
pub fn exp_scaled_sum(x: &mut [f32], shift: f32, scale: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked.
            return unsafe { exp_scaled_sum_avx2(x, shift, scale) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = ((*v - shift) * scale).exp();
        sum += *v as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_scaled_sum_avx2(x: &mut [f32], shift: f32, scale: f32) -> f64 {
    let s = _mm256_set1_ps(shift);
    let k = _mm256_set1_ps(scale);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = exp8(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), s),
            k,
        ));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(v) as f64;
        i += 8;
    }
    while i < x.len() {
        x[i] = ((x[i] - shift) * scale).exp();
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// The largest element (`−∞` for an empty slice).
pub fn max(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() && x.len() >= 16 {
            // SAFETY: features checked.
            return unsafe { max_avx2(x) };
        }
    }
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_avx2(x: &[f32]) -> f32 {
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= x.len() {
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < x.len() {
        r = r.max(x[i]);
        i += 1;
    }
    r
}

/// SwiGLU in place: `gate[i] = silu(gate[i]) · up[i]`, `silu(x) = x / (1 + e^−x)` —
/// ggml's `ggml_vec_swiglu_f32`, so the feed-forward activation matches llama.cpp's.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked; lengths equal.
            return unsafe { swiglu_avx2(gate, up) };
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + (-*g).exp())) * u;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &mut [f32], up: &[f32]) {
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= gate.len() {
        let x = _mm256_loadu_ps(gate.as_ptr().add(i));
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
        let silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(
            gate.as_mut_ptr().add(i),
            _mm256_mul_ps(silu, _mm256_loadu_ps(up.as_ptr().add(i))),
        );
        i += 8;
    }
    while i < gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
        i += 1;
    }
}

/// In-place orthonormal Walsh–Hadamard transform with a fixed sign flip first
/// (`x ← H·D·x / √n`). `n` must be a power of two. Its own inverse is
/// [`hadamard_inv`] (`x ← D·H·x / √n`), because `H/√n` is symmetric orthonormal and
/// `D` is a ±1 diagonal.
///
/// This is the rotation uzu applies before quantising its int8 KV cache
/// (`backends/cpu/kernel/attention/prepare_int8.rs`): it spreads an outlier channel
/// over every channel of the vector, so one per-vector absmax scale no longer has to
/// stretch to cover it, and the other channels keep their resolution. Dot products
/// are unchanged by an orthonormal rotation, so `q·k` can be taken entirely in the
/// rotated space once `q` is rotated the same way.
pub fn hadamard(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    for (i, v) in x.iter_mut().enumerate() {
        if sign_bit(i) {
            *v = -*v;
        }
    }
    fwht(x);
}

/// The inverse of [`hadamard`].
pub fn hadamard_inv(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    fwht(x);
    for (i, v) in x.iter_mut().enumerate() {
        if sign_bit(i) {
            *v = -*v;
        }
    }
}

/// A fixed pseudo-random sign pattern (a 64-bit constant, repeated): any pattern
/// that is not structured along the channel axis does the job, and a fixed one keeps
/// the cache deterministic across runs.
#[inline]
fn sign_bit(i: usize) -> bool {
    const SIGNS: u64 = 0x9E37_79B9_7F4A_7C15;
    (SIGNS >> (i & 63)) & 1 == 1
}

fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    #[test]
    fn vector_exp_is_within_two_ulp_of_libm_over_the_softmax_range() {
        // Softmax inputs are max-shifted, so ≤ 0; the far tail underflows to 0 in both.
        let xs = prng(7, 4099, -90.0, 0.0);
        let mut v = xs.clone();
        let s = exp_shift_sum(&mut v, 0.0);
        let mut want_sum = 0.0f64;
        for (i, (&x, &got)) in xs.iter().zip(&v).enumerate() {
            let want = x.exp();
            want_sum += want as f64;
            let ulp = (want.to_bits() as i64 - got.to_bits() as i64).abs();
            assert!(
                ulp <= 2 || (want - got).abs() < 1e-37,
                "exp({x}) = {got}, libm {want}, {ulp} ulp at {i}"
            );
        }
        assert!((s - want_sum).abs() <= 1e-5 * want_sum.max(1e-30));
    }

    #[test]
    fn exp_handles_overflow_and_underflow_like_ggml() {
        let mut v = vec![100.0f32, -200.0, 0.0, 88.0, -87.0, -104.5, 1.0, -1.0, 50.0];
        exp_shift_sum(&mut v, 0.0);
        assert!(v[0].is_infinite());
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 1.0);
        assert!((v[3] / 88.0f32.exp() - 1.0).abs() < 1e-6);
        assert!((v[6] / 1.0f32.exp() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scaled_exp_is_exp_of_the_scaled_shift() {
        let xs = prng(11, 1003, -30.0, 4.0);
        let m = max(&xs);
        let mut a = xs.clone();
        let k = 1.0f32 / 0.7;
        let sa = exp_scaled_sum(&mut a, m, k);
        // The same rounded argument the kernel forms: `exp` multiplies an argument's
        // relative error by its magnitude, so `/0.7` vs `·(1/0.7)` alone is ~40 ulp.
        let mut b: Vec<f32> = xs.iter().map(|x| (x - m) * k).collect();
        let sb = exp_shift_sum(&mut b, 0.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 2e-6 * y.max(1e-30), "{x} vs {y}");
        }
        assert!((sa - sb).abs() < 1e-4 * sb);
    }

    #[test]
    fn max_matches_the_scalar_fold() {
        for n in [0, 1, 7, 8, 9, 31, 1000] {
            let xs = prng(n as u64 + 3, n, -5.0, 5.0);
            assert_eq!(
                max(&xs),
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                "n = {n}"
            );
        }
    }

    #[test]
    fn swiglu_matches_the_scalar_definition() {
        let g0 = prng(5, 1539, -12.0, 12.0);
        let up = prng(9, 1539, -3.0, 3.0);
        let mut g = g0.clone();
        swiglu(&mut g, &up);
        for i in 0..g0.len() {
            let want = (g0[i] / (1.0 + (-g0[i]).exp())) * up[i];
            assert!(
                (g[i] - want).abs() <= 1e-6 * want.abs().max(1e-3),
                "{i}: {} vs {want}",
                g[i]
            );
        }
    }

    #[test]
    fn hadamard_is_orthonormal_and_inverts_exactly_enough() {
        for n in [64usize, 128, 256] {
            let x = prng(n as u64, n, -4.0, 4.0);
            let y = prng(n as u64 + 1, n, -4.0, 4.0);
            let dot = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(p, q)| (*p as f64) * (*q as f64))
                    .sum::<f64>()
            };
            let (mut hx, mut hy) = (x.clone(), y.clone());
            hadamard(&mut hx);
            hadamard(&mut hy);
            assert!(
                (dot(&hx, &hy) - dot(&x, &y)).abs() < 1e-3,
                "dot products survive the rotation"
            );
            hadamard_inv(&mut hx);
            for (a, b) in hx.iter().zip(&x) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn hadamard_spreads_an_outlier_channel() {
        // One channel 100× the rest — the case per-vector int8 handles worst.
        let mut x = vec![0.01f32; 128];
        x[17] = 1.0;
        let amax_before = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mean_before = x.iter().map(|v| v.abs()).sum::<f32>() / 128.0;
        hadamard(&mut x);
        let amax_after = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mean_after = x.iter().map(|v| v.abs()).sum::<f32>() / 128.0;
        assert!(amax_before / mean_before > 30.0);
        assert!(
            amax_after / mean_after < 2.0,
            "peak-to-mean {} after rotation",
            amax_after / mean_after
        );
    }
}
