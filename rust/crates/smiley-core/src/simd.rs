//! AVX2 fused dequantise-and-dot.
//!
//! These are the kernels that decide whether the engine's roofline claims apply to
//! itself. With the scalar path the measured compute roof on the reference machine
//! is 28.5 GFLOP/s against 30.6 GB/s of bandwidth — a ridge point of 0.93 FLOP/byte,
//! *below* this workload's 6.2 FLOP/byte. A scalar decode step is therefore
//! compute-bound, and every statement it makes about memory bandwidth describes our
//! own arithmetic instead. Reaching the vector units moves the compute roof by more
//! than an order of magnitude and puts the workload back on the bandwidth slope
//! where the model says it belongs.
//!
//! # The instruction that shapes everything
//!
//! `VPMADDUBSW` multiplies **unsigned** bytes by **signed** bytes, summing adjacent
//! pairs into 16-bit lanes. Every quantised format here stores unsigned codes, so
//! the codes go in the unsigned operand and the int8 activation in the signed one,
//! with the format's offset applied afterwards via the bias identity in `quant.rs`.
//! That is why the kernels never reconstruct a signed weight.
//!
//! Saturation is the trap. `VPMADDUBSW` saturates its 16-bit result, and it sums two
//! products, so the largest code times the largest activation must stay under 16384
//! to be safe. Q6_K has the widest codes at 63, and `63 × 127 × 2 = 16002`, which
//! fits — but only because activations are clamped to ±127 rather than ±128 when
//! they are quantised. That clamp in `QuantAct::fill` is load-bearing here.
//!
//! # Correctness
//!
//! Every kernel has a scalar twin in `quant.rs`, and `tests/simd.rs` asserts they
//! agree on the real weight file (acceptance criterion A4). The scalar path is the
//! specification; this file is an optimisation of it and is never the reference.

#![cfg(target_arch = "x86_64")]

use crate::gguf::{f16, Format};
use crate::quant::QuantAct;
use std::arch::x86_64::*;

/// Whether the AVX2 path may run. `is_x86_feature_detected!` caches internally, so
/// this is a load of a relaxed atomic after the first call.
#[inline]
pub fn have_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    hsum_i32_128(_mm_add_epi32(lo, hi))
}

#[inline(always)]
unsafe fn hsum_i32_128(v: __m128i) -> i32 {
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
    _mm_cvtsi128_si32(s)
}

/// `Σ codes·acts` for 32 unsigned codes against 32 signed activations.
#[inline(always)]
unsafe fn dot32_u8_i8(codes: __m256i, acts: __m256i) -> i32 {
    let prod = _mm256_maddubs_epi16(codes, acts); // 16 × i16
    hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)))
}

#[inline(always)]
unsafe fn acts32(a: &QuantAct, block: usize) -> __m256i {
    _mm256_loadu_si256(a.q.as_ptr().add(block * 32) as *const __m256i)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_q8_0(w: &[u8], a: &QuantAct) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(34).enumerate() {
        let wq = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
        let aq = acts32(a, b);
        // Q8_0 codes are signed, so the unsigned operand must be |w| and the sign
        // moved onto the activation. `_mm256_sign_epi8(x, y)` negates x where y is
        // negative and zeroes it where y is zero — exactly the two moves needed.
        let absw = _mm256_sign_epi8(wq, wq);
        let sa = _mm256_sign_epi8(aq, wq);
        let s = dot32_u8_i8(absw, sa);
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * s as f32;
    }
    acc
}

/// Spread the 32 fifth-bits of a Q5_0 block into one `0x10`-or-`0` byte per weight.
///
/// Byte `i` of the result must carry bit `i` of `qh`. `_mm256_shuffle_epi8` works
/// inside 128-bit lanes, so the broadcast `qh` is gathered per lane: bytes 0..8 take
/// `qh` byte 0, 8..16 take byte 1, and the upper lane takes bytes 2 and 3.
#[inline(always)]
unsafe fn q5_high_bits(qh: u32) -> __m256i {
    let spread = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, // upper lane: bytes 3,2
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // lower lane: bytes 1,0
    );
    let bit = _mm256_set_epi8(
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1,
        -128, 64, 32, 16, 8, 4, 2, 1,
    );
    let v = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), spread);
    let masked = _mm256_and_si256(v, bit);
    // cmpeq against the bit pattern gives 0xFF exactly where the bit was set; a
    // non-zero test would be wrong because the masked byte is the bit's own value.
    let set = _mm256_cmpeq_epi8(masked, bit);
    _mm256_and_si256(set, _mm256_set1_epi8(0x10))
}

#[target_feature(enable = "avx2")]
unsafe fn dot_q5_0(w: &[u8], a: &QuantAct) -> f32 {
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(22).enumerate() {
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = _mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i);
        // Weights 0..16 are the low nibbles, 16..32 the high nibbles — the same
        // interleaving the scalar twin spells out.
        let lo = _mm_and_si128(qs, _mm256_castsi256_si128(lownib));
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm256_castsi256_si128(lownib));
        let nib = _mm256_set_m128i(hi, lo);
        let codes = _mm256_or_si256(nib, q5_high_bits(qh));
        let s = dot32_u8_i8(codes, acts32(a, b));
        // Codes are [0,31] with offset 16: one correction for the whole block.
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * (s - 16 * a.sums[b]) as f32;
    }
    acc
}

/// The byte-wise reference `unpack_q4k_scales` is tested against.
#[cfg(test)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// All eight Q4_K scales and mins of a super-block at once: `get_scale_min_k4` for
/// `j = 0..8`, done as four 32-bit word operations (ggml's `kmask` unpacking) instead
/// of eight branchy byte extractions. Bytes 0..8 of the result are the scales, 8..16
/// the mins.
#[inline(always)]
fn unpack_q4k_scales(q: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let w = |i: usize| u32::from_le_bytes([q[i], q[i + 1], q[i + 2], q[i + 3]]);
    let (u0, u1, u2) = (w(0), w(4), w(8));
    let m_hi = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
    let m_lo = u1 & KMASK1;
    let s_hi = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
    let s_lo = u0 & KMASK1;
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&s_lo.to_le_bytes());
    out[4..8].copy_from_slice(&s_hi.to_le_bytes());
    out[8..12].copy_from_slice(&m_lo.to_le_bytes());
    out[12..16].copy_from_slice(&m_hi.to_le_bytes());
    out
}

/// The eight Q4_K scales as broadcast 16-bit vectors — `_mm256_set1_epi16(sc[k])`
/// for `k = 0..8` — from one widening load and one byte shuffle each, instead of a
/// byte extract, a GPR→vector move and a broadcast (two port-5 µops) per scale.
#[inline(always)]
unsafe fn q4k_scale_vecs(sm: &[u8; 16]) -> [__m256i; 8] {
    // Lanes 0..8 hold the scales, 8..16 the mins; duplicate the scale half.
    let s16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(sm.as_ptr() as *const __m128i));
    let sc = _mm256_permute4x64_epi64::<0x44>(s16);
    // The controls are a static table, loaded: built with `_mm256_setr_epi8` from
    // loop-computed bytes they were assembled byte by byte on every super-block
    // (measured: the Q4_K kernel ran 7× slower).
    let ctl = BCAST16.as_ptr() as *const __m256i;
    // nafs-aql: spelled out, not `std::array::from_fn`. Whether LLVM inlines `from_fn` into the kernel depends on
    // how much unrelated code the crate holds; when it does not, every super-block pays a call and an array
    // returned through memory (the Q4_K kernel ran 7x slower once the batching engine was added to the crate).
    let t = |k: usize| _mm256_shuffle_epi8(sc, _mm256_loadu_si256(ctl.add(k)));
    [t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)]
}

/// `pshufb` controls that broadcast 16-bit lane `k` of each 128-bit half to the
/// whole half, `k = 0..8`.
static BCAST16: [[u8; 32]; 8] = {
    let mut t = [[0u8; 32]; 8];
    let mut k = 0;
    while k < 8 {
        let mut i = 0;
        while i < 32 {
            t[k][i] = (2 * k + i % 2) as u8;
            i += 1;
        }
        k += 1;
    }
    t
};

/// Q6_K's controls: index `c·4 + g` broadcasts 16-bit lane `2g` to the low half and
/// lane `2g + 1` to the high half (both halves of the source hold the same 8 lanes).
static Q6K_SPLIT16: [[u8; 32]; 4] = {
    let mut t = [[0u8; 32]; 4];
    let mut g = 0;
    while g < 4 {
        let mut i = 0;
        while i < 32 {
            let lane = if i < 16 { 2 * g } else { 2 * g + 1 };
            t[g][i] = (2 * lane + i % 2) as u8;
            i += 1;
        }
        g += 1;
    }
    t
};

/// The eight Q4_K mins as sixteen 16-bit lanes, each repeated for the two 16-weight
/// runs of its 32-weight sub-block, so one `madd_epi16` against an activation's sixteen
/// run sums yields the min term as eight i32 partial sums with no `hadd`.
/// Built once per super-block and shared by every activation.
#[inline(always)]
unsafe fn q4k_mins16(mins: &[u8]) -> __m256i {
    let m = _mm_loadl_epi64(mins.as_ptr() as *const __m128i);
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m, m))
}

/// One activation's super-block result, entirely in vector form:
/// `Σ w·a = dk·(d·Σ sc·q·a − dmin·Σ m·a)`. The scalar version reduced the min term to
/// a scalar per super-block per activation — a horizontal sum, an int→float convert and
/// a scalar multiply-add, about a dozen instructions against the eight the dot product
/// itself costs — and kept a second accumulator for it. Here both stay as eight lanes
/// and the one horizontal reduction happens once per row.
#[inline(always)]
unsafe fn q4k_fold(
    acc: __m256,
    sumi: __m256i,
    mins16: __m256i,
    bsums: *const i16,
    dv: __m256,
    dminv: __m256,
    dk: f32,
) -> __m256 {
    let minterm = _mm256_madd_epi16(mins16, _mm256_loadu_si256(bsums as *const __m256i));
    let x = _mm256_fnmadd_ps(
        dminv,
        _mm256_cvtepi32_ps(minterm),
        _mm256_mul_ps(dv, _mm256_cvtepi32_ps(sumi)),
    );
    _mm256_fmadd_ps(x, _mm256_set1_ps(dk), acc)
}

/// # Safety
/// Caller must have checked `have_avx2()`. `w` must be exactly `row_bytes(fmt, a.n)`
/// long and `a` must hold `a.n` codes; both hold for every call from `matmul`.
pub unsafe fn dot(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    match fmt {
        Format::Q8_0 => dot_q8_0(w, a),
        Format::Q5_0 => dot_q5_0(w, a),
        // The k-quants run on the per-256 tiles at width one: integer
        // accumulation across the super-block, one float scale per 256.
        Format::Q4_K => {
            let mut o = [0.0f32];
            tile_q4_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        Format::Q6_K => {
            let mut o = [0.0f32];
            tile_q6_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        // Norm vectors, not a matmul path; the scalar twin is already adequate and
        // duplicating it here would be two implementations of a cold path.
        Format::F32 | Format::F16 => crate::quant::dot_scalar(fmt, w, a),
    }
}

// ---------------------------------------------------------------------------
// Batched kernels: one weight row, many activations.
//
// The kernels above are one row against one activation, and they end every 32-weight
// block with a horizontal sum — about eight shuffles and adds to turn a vector of
// partial products into a scalar, per block, per row. Two things follow from that
// shape and both were measured before this file changed:
//
//   * decode spent more instructions reducing than multiplying, and
//   * prefill, which hands `matmul` a batch of up to 64 activations, called the
//     one-activation kernel 64 times per row — so the weight row was *fetched* once
//     (the byte accounting is right) but *decoded* 64 times. A 42-token prompt took
//     1.08 s, about the same per token as a decode step, when reading the weights
//     once was supposed to be what made prefill cheap.
//
// These kernels fix both. The weight block is decoded once into a register, then
// reused against up to `TILE` activations; partial products stay in vector
// accumulators (`cvt` + `fmadd` per block instead of a horizontal sum) and are
// reduced once per row. The per-format bias terms (the offset of Q5_0/Q6_K, the min
// of Q4_K) accumulate separately in scalars and are subtracted at the end, which is
// the same bias identity as `quant.rs`, evaluated in float instead of per block in
// integers — algebraically identical, differing only in rounding.
//
// `dot` above stays: it is the fallback on a CPU with AVX2 but no FMA, and the
// second implementation `tests/simd.rs` checks these against.
// ---------------------------------------------------------------------------

/// Activations processed per pass over a weight row. Four keeps every accumulator
/// and the decoded block in the sixteen vector registers; wider tiles spill.
pub const TILE: usize = 4;

/// Whether the batched kernels can run. They use fused multiply-add and the F16C
/// scale conversion (every AVX2 CPU since Haswell has both).
#[inline]
pub fn have_batched() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

#[inline(always)]
pub(crate) unsafe fn hsum_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
    _mm_cvtss_f32(s)
}

/// `Σ codes·acts − offset·Σacts` as eight exact i32 lane sums, converted to f32.
/// The lane layout matches `QuantAct::lane_sums`.
#[inline(always)]
unsafe fn madd_offset_ps<const SHIFT: i32>(
    codes: __m256i,
    acts: __m256i,
    lane_sums: *const i32,
) -> __m256 {
    let prod = _mm256_maddubs_epi16(codes, acts);
    let s = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
    let ls = _mm256_loadu_si256(lane_sums as *const __m256i);
    // The offset is a power of two (16 for Q5_0): a shift, not `VPMULLD` — which is
    // two micro-ops with ten cycles of latency, once per block per activation.
    _mm256_cvtepi32_ps(_mm256_sub_epi32(s, _mm256_slli_epi32(ls, SHIFT)))
}

/// `Σ codes·acts` as eight f32 partial sums, with no horizontal reduction.
#[inline(always)]
unsafe fn madd_ps(codes: __m256i, acts: __m256i) -> __m256 {
    let prod = _mm256_maddubs_epi16(codes, acts);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)))
}

#[cfg(test)]
#[target_feature(enable = "f16c")]
unsafe fn f16c_one(h: u16) -> f32 {
    f16_at(&h.to_le_bytes(), 0)
}

/// A block's f16 scale, converted by the F16C unit (`VCVTPH2PS`): two instructions
/// in place of `gguf::f16`'s branchy bit surgery, once per 32 weights in Q8_0/Q5_0.
/// The conversion is exact either way (every f16 is an f32), so the result is the
/// same bits — `f16c_matches_the_reference_conversion` checks every one of the 65,536.
#[inline(always)]
unsafe fn f16_at(b: &[u8], at: usize) -> f32 {
    debug_assert!(at + 2 <= b.len());
    let h = u16::from_le_bytes([*b.get_unchecked(at), *b.get_unchecked(at + 1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

/// `f16_at` for a scale with two more bytes of the same block after it: one 32-bit
/// load (`vmovd`) straight into the vector unit instead of two byte loads, a merge
/// and `vpinsrw` (a port-5 µop). Lane 0 is the same conversion of the same bits.
#[inline(always)]
unsafe fn f16_at4(p: *const u8) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
        (p as *const i32).read_unaligned(),
    )))
}

/// How far ahead of the block being decoded the kernels ask for weight bytes.
///
/// Decode streams every weight once per token, and the L2 streamer that would fetch
/// ahead stops at each 4 KB page: without a hint, every page of weights costs a full
/// DRAM round trip that the core spends waiting (round 13: the hot instructions in a
/// profile were the loads, and one core streamed 7.4 GB/s against llama.cpp's 10.7).
/// `SMILEY_CORE_PF` (bytes, 0 = off) is the A/B switch it was tuned with.
fn prefetch_distance() -> usize {
    static PF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PF.get_or_init(|| {
        std::env::var("SMILEY_CORE_PF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PF_DEFAULT)
    })
}
const PF_DEFAULT: usize = 1024;

/// Hint `lines` cache lines starting `pf` bytes past `p`. A prefetch never faults, so
/// running past the end of the tensor is harmless; `wrapping_add` keeps the pointer
/// arithmetic defined there.
#[inline(always)]
unsafe fn prefetch_ahead(p: *const u8, pf: usize, lines: usize) {
    if pf == 0 {
        return;
    }
    for l in 0..lines {
        _mm_prefetch(p.wrapping_add(pf + l * 64) as *const i8, _MM_HINT_T0);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q8_0<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let mut acc = [_mm256_setzero_ps(); T];
    for (b, blk) in w.chunks_exact(34).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let wq = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
        let absw = _mm256_sign_epi8(wq, wq);
        let dw = f16_at(blk, 0);
        for t in 0..T {
            let a = &acts[t];
            let sa = _mm256_sign_epi8(acts32(a, b), wq);
            acc[t] = _mm256_fmadd_ps(madd_ps(absw, sa), _mm256_set1_ps(dw * a.d[b]), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q5_0<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let lownib = _mm256_set1_epi8(0x0F);
    // Low lane keeps the bytes, high lane shifts each 64-bit half right by 4: after
    // the mask that is `qs & 0xF` in lanes 0..16 and `(qs >> 4) & 0xF` in 16..32 —
    // the `_mm256_set_m128i(hi, lo)` pair, from one broadcast load and no port-5
    // lane insert.
    let nib_shift = _mm256_set_epi64x(4, 4, 0, 0);
    let nblk = w.len() / 22;
    for a in &acts[..T] {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32 && a.lane_sums.len() >= nblk * 8);
    }
    let mut acc = [_mm256_setzero_ps(); T];
    for (b, blk) in w.chunks_exact(22).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qh = (blk.as_ptr().add(2) as *const u32).read_unaligned();
        let qs =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i));
        let nib = _mm256_and_si256(_mm256_srlv_epi64(qs, nib_shift), lownib);
        let codes = _mm256_or_si256(nib, q5_high_bits(u32::from_le(qh)));
        let dw = f16_at4(blk.as_ptr());
        for t in 0..T {
            let a = &acts[t];
            // Codes are [0,31] with offset 16, removed exactly inside the lanes.
            let v = madd_offset_ps::<4>(codes, acts32(a, b), a.lane_sums.as_ptr().add(b * 8));
            acc[t] = _mm256_fmadd_ps(v, _mm256_set1_ps(dw * *a.d.get_unchecked(b)), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q4_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); T];
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let qs = blk.as_ptr().add(16);
        let sm = unpack_q4k_scales(&blk[4..16]);
        let scv = q4k_scale_vecs(&sm);
        let mins16 = q4k_mins16(&sm[8..16]);
        let mut sumi = [_mm256_setzero_si256(); T];
        // Each 32 bytes of codes hold two sub-blocks: the low nibbles are sub-block
        // 2j, the high nibbles 2j+1. Both dot products are scaled by their 6-bit
        // scale in 16-bit lanes and summed in 32-bit lanes; nothing leaves the
        // integer domain until the super-block is done.
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(raw, lownib);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);
            let (scl, sch) = (scv[2 * j], scv[2 * j + 1]);
            for t in 0..T {
                let base = acts[t].qk.as_ptr().add(sb * 256 + j * 64);
                let q8l = _mm256_loadu_si256(base as *const __m256i);
                let q8h = _mm256_loadu_si256(base.add(32) as *const __m256i);
                let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q4l, q8l), scl);
                let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q4h, q8h), sch);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
            }
        }
        for t in 0..T {
            // w = d·sc·q − dmin·m: the min term is the run sums against the mins.
            let a = &acts[t];
            acc[t] = q4k_fold(
                acc[t],
                sumi[t],
                mins16,
                a.bsums.as_ptr().add(sb * 16),
                dv,
                dminv,
                a.dk[sb],
            );
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

/// The eight run-scale vectors of a Q6_K super-block, index `c·4 + g`: 16-bit lanes
/// 0..8 hold scale `c·8 + 2g`, lanes 8..16 scale `c·8 + 2g + 1` — the values
/// `_mm256_set_m128i(_mm_set1_epi16(..), _mm_set1_epi16(..))` produced, built with
/// one byte shuffle each instead of five port-5 µops through the integer registers
/// (which made the Q6_K kernel port-5 bound). `sc16` is the 16 scales as i16 lanes.
#[inline(always)]
unsafe fn q6k_scales(sc16: __m256i) -> [__m256i; 8] {
    // Both 128-bit halves of `half[c]` hold scales c·8 .. c·8+8 (16 bytes).
    let half = [
        _mm256_permute4x64_epi64::<0x44>(sc16),
        _mm256_permute4x64_epi64::<0xEE>(sc16),
    ];
    let ctl = Q6K_SPLIT16.as_ptr() as *const __m256i;
    // nafs-aql: spelled out for the same reason as `q4k_scale_vecs`.
    let t = |i: usize| _mm256_shuffle_epi8(half[i / 4], _mm256_loadu_si256(ctl.add(i % 4)));
    [t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)]
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q6_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm256_set1_epi8(0x0F);
    let two = _mm256_set1_epi8(0x03);
    let mut acc = [_mm256_setzero_ps(); T];
    for (sb, blk) in w.chunks_exact(210).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 4);
        let d = f16_at(blk, 208);
        // The 16 run scales as 16-bit lanes, for the offset term below.
        let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.as_ptr().add(192) as *const __m128i));
        let scales = q6k_scales(sc16);
        let mut sumi = [_mm256_setzero_si256(); T];
        for c in 0..2 {
            let ql = blk.as_ptr().add(c * 64);
            let qh = _mm256_loadu_si256(blk.as_ptr().add(128 + c * 32) as *const __m256i);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            for g in 0..4 {
                let (base, hbits) = match g {
                    0 => (ql0, _mm256_and_si256(qh, two)),
                    1 => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 2), two)),
                    2 => (ql0, _mm256_and_si256(_mm256_srli_epi16(qh, 4), two)),
                    _ => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 6), two)),
                };
                let low = if g < 2 {
                    _mm256_and_si256(base, lownib)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(base, 4), lownib)
                };
                let codes = _mm256_or_si256(low, _mm256_slli_epi16(hbits, 4));
                // Lanes 0..8 of the 16-bit products cover the first 16 weights,
                // lanes 8..16 the next 16; each run has its own scale.
                let scale = scales[c * 4 + g];
                let o = sb * 256 + c * 128 + g * 32;
                for t in 0..T {
                    let q8 = _mm256_loadu_si256(acts[t].qk.as_ptr().add(o) as *const __m256i);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, q8), scale);
                    sumi[t] = _mm256_add_epi32(sumi[t], p);
                }
            }
        }
        for t in 0..T {
            let a = &acts[t];
            // Offset 32 on every code: subtract 32·Σ(scale_run · Σqk_run), exact in integers.
            let bs = _mm256_loadu_si256(a.bsums.as_ptr().add(sb * 16) as *const __m256i);
            let off = _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs), 5);
            let v = _mm256_sub_epi32(sumi[t], off);
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(d * a.dk[sb]), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

/// `SMILEY_CORE_NO_WIDE=1` keeps the tiled kernels for every batch width — the A/B
/// switch the wide kernels were measured against (kbench, and a real prefill).
#[inline]
fn wide_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SMILEY_CORE_NO_WIDE")
            .map(|v| v != "1")
            .unwrap_or(true)
    })
}

/// Widest batch the wide kernels take in one call; `model::MAX_BATCH` must not exceed it.
pub const MAX_WIDE: usize = 128;

/// The per-activation half of a Q4_K super-block: `sumi[u] += Σ_j (q4l_j·a_{2j})·sc_{2j} + (q4h_j·a_{2j+1})·sc_{2j+1}`
/// for `T` activations against codes already unpacked by the caller.
#[inline(always)]
unsafe fn q4k_acts<const T: usize>(
    q4l: &[__m256i; 4],
    q4h: &[__m256i; 4],
    sc: &[u8; 8],
    acts: &[QuantAct],
    sb: usize,
    sumi: &mut [__m256i; 4],
) {
    for j in 0..4 {
        let scl = _mm256_set1_epi16(sc[2 * j] as i16);
        let sch = _mm256_set1_epi16(sc[2 * j + 1] as i16);
        for t in 0..T {
            let base = acts[t].qk.as_ptr().add(sb * 256 + j * 64);
            let q8l = _mm256_loadu_si256(base as *const __m256i);
            let q8h = _mm256_loadu_si256(base.add(32) as *const __m256i);
            let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q4l[j], q8l), scl);
            let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q4h[j], q8h), sch);
            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
        }
    }
}

/// Q4_K against a *wide* batch (a prompt): the super-block's scales, mins and
/// nibbles are unpacked once and run against every activation, where the tiled
/// kernel unpacks them again for each tile of four. The per-activation work is
/// the same eight integer ops; what this removes is the repeated row decoding,
/// which at a 51-token prompt was done thirteen times per super-block. The
/// accumulators live in a stack array (`MAX_WIDE` vectors), touched once per
/// super-block, so register pressure is the tile's, not the batch's.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q4_k(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= MAX_WIDE && acts.iter().all(|a| !a.qk.is_empty()));
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); MAX_WIDE];
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let qs = blk.as_ptr().add(16);
        let sm = unpack_q4k_scales(&blk[4..16]);
        let mut sc = [0u8; 8];
        sc.copy_from_slice(&sm[0..8]);
        let mins16 = q4k_mins16(&sm[8..16]);
        let mut q4l = [_mm256_setzero_si256(); 4];
        let mut q4h = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            q4l[j] = _mm256_and_si256(raw, lownib);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);
        }
        let mut t = 0;
        while t < n {
            let tw = (n - t).min(TILE);
            let mut sumi = [_mm256_setzero_si256(); 4];
            match tw {
                4 => q4k_acts::<4>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                3 => q4k_acts::<3>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                2 => q4k_acts::<2>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                _ => q4k_acts::<1>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
            }
            for u in 0..tw {
                let a = &acts[t + u];
                acc[t + u] = q4k_fold(
                    acc[t + u],
                    sumi[u],
                    mins16,
                    a.bsums.as_ptr().add(sb * 16),
                    dv,
                    dminv,
                    a.dk[sb],
                );
            }
            t += tw;
        }
    }
    for b in 0..n {
        out[b] = hsum_ps(acc[b]);
    }
}

/// The per-activation half of a Q6_K super-block against its eight unpacked code
/// vectors (`c·4 + g`), each run-scaled.
#[inline(always)]
unsafe fn q6k_acts<const T: usize>(
    codes: &[__m256i; 8],
    scales: &[__m256i; 8],
    acts: &[QuantAct],
    sb: usize,
    sumi: &mut [__m256i; 4],
) {
    for c in 0..2 {
        for g in 0..4 {
            let scale = scales[c * 4 + g];
            let o = sb * 256 + c * 128 + g * 32;
            for t in 0..T {
                let q8 = _mm256_loadu_si256(acts[t].qk.as_ptr().add(o) as *const __m256i);
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(codes[c * 4 + g], q8), scale);
                sumi[t] = _mm256_add_epi32(sumi[t], p);
            }
        }
    }
}

/// Q6_K against a wide batch; see `wide_q4_k`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q6_k(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= MAX_WIDE && acts.iter().all(|a| !a.qk.is_empty()));
    let lownib = _mm256_set1_epi8(0x0F);
    let two = _mm256_set1_epi8(0x03);
    let mut acc = [_mm256_setzero_ps(); MAX_WIDE];
    for (sb, blk) in w.chunks_exact(210).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 4);
        let d = f16_at(blk, 208);
        let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.as_ptr().add(192) as *const __m128i));
        // Built once per super-block, not once per tile of activations.
        let scales = q6k_scales(sc16);
        let mut codes = [_mm256_setzero_si256(); 8];
        for c in 0..2 {
            let ql = blk.as_ptr().add(c * 64);
            let qh = _mm256_loadu_si256(blk.as_ptr().add(128 + c * 32) as *const __m256i);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            for g in 0..4 {
                let (base, hbits) = match g {
                    0 => (ql0, _mm256_and_si256(qh, two)),
                    1 => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 2), two)),
                    2 => (ql0, _mm256_and_si256(_mm256_srli_epi16(qh, 4), two)),
                    _ => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 6), two)),
                };
                let low = if g < 2 {
                    _mm256_and_si256(base, lownib)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(base, 4), lownib)
                };
                codes[c * 4 + g] = _mm256_or_si256(low, _mm256_slli_epi16(hbits, 4));
            }
        }
        let mut t = 0;
        while t < n {
            let tw = (n - t).min(TILE);
            let mut sumi = [_mm256_setzero_si256(); 4];
            match tw {
                4 => q6k_acts::<4>(&codes, &scales, &acts[t..], sb, &mut sumi),
                3 => q6k_acts::<3>(&codes, &scales, &acts[t..], sb, &mut sumi),
                2 => q6k_acts::<2>(&codes, &scales, &acts[t..], sb, &mut sumi),
                _ => q6k_acts::<1>(&codes, &scales, &acts[t..], sb, &mut sumi),
            }
            for u in 0..tw {
                let a = &acts[t + u];
                let bs = _mm256_loadu_si256(a.bsums.as_ptr().add(sb * 16) as *const __m256i);
                let off = _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs), 5);
                let v = _mm256_sub_epi32(sumi[u], off);
                acc[t + u] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(v),
                    _mm256_set1_ps(d * a.dk[sb]),
                    acc[t + u],
                );
            }
            t += tw;
        }
    }
    for b in 0..n {
        out[b] = hsum_ps(acc[b]);
    }
}

/// One weight row against `acts.len()` activations, writing `out[b]` for each.
///
/// # Safety
/// Caller must have checked `have_batched()`. `w` must be exactly
/// `row_bytes(fmt, a.n)` long for every activation `a`, and `out.len() >= acts.len()`.
pub unsafe fn dot_batch(fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    macro_rules! run {
        ($f:ident) => {{
            let mut b = 0;
            while b + TILE <= acts.len() {
                $f::<4>(w, &acts[b..b + 4], &mut out[b..b + 4]);
                b += TILE;
            }
            match acts.len() - b {
                3 => $f::<3>(w, &acts[b..b + 3], &mut out[b..b + 3]),
                2 => $f::<2>(w, &acts[b..b + 2], &mut out[b..b + 2]),
                1 => $f::<1>(w, &acts[b..b + 1], &mut out[b..b + 1]),
                _ => {}
            }
        }};
    }
    match fmt {
        Format::Q8_0 => run!(tile_q8_0),
        Format::Q5_0 => run!(tile_q5_0),
        // A prompt-sized batch decodes the row once for all of it; decode and the
        // speculative block (≤ 4) keep the tiled kernels, so their figures are unchanged.
        Format::Q4_K if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q4_k(w, acts, out)
        }
        Format::Q6_K if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q6_k(w, acts, out)
        }
        Format::Q4_K => run!(tile_q4_k),
        Format::Q6_K => run!(tile_q6_k),
        Format::F32 | Format::F16 => {
            for (b, a) in acts.iter().enumerate() {
                out[b] = crate::quant::dot_scalar(fmt, w, a);
            }
        }
    }
}

/// `|x|` maximum over a slice, eight lanes at a time. `max` is exact and order-free
/// for finite values, so this equals the scalar fold.
///
/// # Safety
/// AVX2 must be available (`have_avx2`).
#[target_feature(enable = "avx2")]
pub unsafe fn absmax(x: &[f32]) -> f32 {
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= x.len() {
        m = _mm256_max_ps(
            m,
            _mm256_andnot_ps(sign, _mm256_loadu_ps(x.as_ptr().add(i))),
        );
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().fold(0.0f32, |a, &b| a.max(b));
    for &v in &x[i..] {
        r = r.max(v.abs());
    }
    r
}

/// `out[i] = clamp(round(x[i] · inv), ±127)` for 32 values — `QuantAct::fill`'s
/// inner loop. Rust's `round` goes half away from zero, which no single AVX rounding
/// mode does; it is rebuilt exactly as `trunc(|y|) + (frac ≥ 0.5)` (the fraction of a
/// float below 2²³ is exact), with the sign put back. Bit-identical to the scalar
/// loop for every finite input (`vector_quantisation_matches_the_scalar_loop`).
///
/// # Safety
/// AVX2 must be available; `x` must point at 32 readable floats and `out` at 32
/// writable bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn quantize32(x: *const f32, inv: f32, out: *mut i8) {
    let vinv = _mm256_set1_ps(inv);
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let hi = _mm256_set1_ps(127.0);
    let lo = _mm256_set1_ps(-127.0);
    let mut iv = [_mm256_setzero_si256(); 4];
    for (k, v) in iv.iter_mut().enumerate() {
        let y = _mm256_mul_ps(_mm256_loadu_ps(x.add(8 * k)), vinv);
        let ay = _mm256_andnot_ps(sign, y);
        let t = _mm256_round_ps(ay, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let up = _mm256_and_ps(_mm256_cmp_ps(_mm256_sub_ps(ay, t), half, _CMP_GE_OQ), one);
        let r = _mm256_or_ps(_mm256_add_ps(t, up), _mm256_and_ps(y, sign));
        *v = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(r, lo), hi));
    }
    // i32 → i16 → i8 packs interleave the 128-bit lanes; the permute restores order.
    let a = _mm256_packs_epi32(iv[0], iv[1]);
    let b = _mm256_packs_epi32(iv[2], iv[3]);
    let c = _mm256_packs_epi16(a, b);
    let c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(out as *mut __m256i, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_saturation_headroom_is_real() {
        // VPMADDUBSW sums two products into a saturating i16. The widest case in
        // this file is Q6_K: codes reach 63 and activations are clamped to ±127.
        // If `QuantAct::fill` ever allowed -128, this margin disappears and the
        // kernel would silently saturate on large activations.
        assert!(63 * 127 * 2 < i16::MAX as i32, "Q6_K would saturate");
        assert!(31 * 127 * 2 < i16::MAX as i32, "Q5_0 would saturate");
        assert!(15 * 127 * 2 < i16::MAX as i32, "Q4_K would saturate");
        // Q8_0 is the tight one. Its codes are signed, and `_mm256_sign_epi8` maps a
        // code of -128 to unsigned 128 (negation overflows back to itself), so the
        // unsigned operand reaches 128 and not 127.
        assert!(128 * 127 * 2 < i16::MAX as i32, "Q8_0 would saturate");
        // An activation of -128 against that same 128 would reach exactly 32768,
        // one past i16::MAX, and saturate. That is why `QuantAct::fill` clamps to
        // ±127 rather than to the full i8 range — the clamp is load-bearing here,
        // not defensive.
        assert!(
            128 * 128 * 2 > i16::MAX as i32,
            "the ±127 activation clamp is load-bearing"
        );
    }

    /// The word-wise unpacking and the vector min term against the byte-wise
    /// reference, on random scale bytes and run sums (including the i16 extremes a
    /// 256-block of ±127 can produce).
    #[test]
    fn q4k_scale_unpacking_and_min_term_match_the_reference() {
        if !have_batched() {
            return;
        }
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..20_000 {
            let q: Vec<u8> = (0..12).map(|_| (rnd() >> 32) as u8).collect();
            let sm = unpack_q4k_scales(&q);
            let mut bs = [0i16; 16];
            for v in bs.iter_mut() {
                *v = ((rnd() >> 40) as i32 % 2033 - 1016) as i16;
            }
            let mut want = 0i32;
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, &q);
                assert_eq!((sm[j], sm[8 + j]), (sc, m), "j={j} q={q:?}");
                want += m as i32 * (bs[2 * j] as i32 + bs[2 * j + 1] as i32);
            }
            let got = unsafe {
                let lanes = _mm256_madd_epi16(
                    q4k_mins16(&sm[8..16]),
                    _mm256_loadu_si256(bs.as_ptr() as *const __m256i),
                );
                hsum_i32(lanes)
            };
            assert_eq!(got, want);
        }
    }

    /// Every code and every sum of the vector `fill` against the scalar loop, on
    /// random activations plus the cases that break a naive rounding: exact halves,
    /// the float just below a half, values past the clamp, zeros and a zero block.
    #[test]
    fn vector_quantisation_matches_the_scalar_loop() {
        if !have_avx2() {
            return;
        }
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for trial in 0..400 {
            let n = if trial % 2 == 0 { 2560 } else { 960 };
            let mut x: Vec<f32> = (0..n)
                .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0)
                .collect();
            if trial % 5 == 0 {
                for v in x[..32].iter_mut() {
                    *v = 0.0;
                }
            }
            for (i, v) in x.iter_mut().enumerate().skip(40).step_by(97) {
                // Values that land exactly on, or one ulp under, a half after scaling.
                *v = match i % 4 {
                    0 => 0.5,
                    1 => -2.5,
                    2 => f32::from_bits(0.5f32.to_bits() - 1),
                    _ => -127.5,
                };
            }
            let mut a = QuantAct::with_capacity(n);
            a.fill(&x);
            let mut b = QuantAct::with_capacity(n);
            b.fill_scalar(&x);
            assert_eq!(a.q, b.q, "trial {trial}");
            assert_eq!(a.d, b.d);
            assert_eq!(
                (&a.sums, &a.sums16, &a.lane_sums),
                (&b.sums, &b.sums16, &b.lane_sums)
            );
            assert_eq!((&a.qk, &a.dk, &a.bsums), (&b.qk, &b.dk, &b.bsums));
        }
    }

    #[test]
    fn f16c_matches_the_reference_conversion() {
        if !have_batched() {
            return;
        }
        for h in 0..=u16::MAX {
            let want = f16(h);
            let got = unsafe { f16c_one(h) };
            // NaN payloads aside, the two must be the same bits.
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}");
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn q5_high_bit_expansion_maps_bit_i_to_byte_i() {
        if !have_avx2() {
            return;
        }
        for qh in [
            0u32,
            0xFFFF_FFFF,
            0x0000_0001,
            0x8000_0000,
            0xDEAD_BEEF,
            0x5555_5555,
        ] {
            let mut got = [0u8; 32];
            unsafe {
                _mm256_storeu_si256(got.as_mut_ptr() as *mut __m256i, q5_high_bits(qh));
            }
            for i in 0..32 {
                let want = if (qh >> i) & 1 == 1 { 0x10 } else { 0x00 };
                assert_eq!(got[i], want, "qh={qh:#010x} bit {i}");
            }
        }
    }
}
