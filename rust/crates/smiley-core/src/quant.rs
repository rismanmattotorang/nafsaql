//! Quantised weight kernels: fused dequantise-and-dot.
//!
//! The central rule (spec P1) is that a dequantised weight tensor is never
//! materialised. Decoding a Q4_K row into f32 and then running a float dot product
//! would move eight bytes for every four and a half the format actually stores,
//! which puts the engine below the bandwidth roof and makes every number it reports
//! a statement about our own buffer traffic rather than about the machine. So the
//! block decode happens inside the accumulation loop, in registers.
//!
//! # The integer identity this file is built on
//!
//! Activations are quantised to int8 in blocks of 32 (`QuantAct`), so the inner
//! product can be integer throughout. Every quantised format here stores its codes
//! *unsigned* and applies an offset at decode time — `−16` for Q5_0, `−32` for Q6_K,
//! an explicit `−dmin·m` for Q4_K. Reconstructing the signed weight just to multiply
//! it would throw away the one instruction that makes the integer path worth having
//! (`VPMADDUBSW` takes an unsigned operand and a signed one). Instead, for codes
//! `c_i` with offset `z` and activations `a_i`:
//!
//! ```text
//!   Σ (c_i − z)·a_i  =  Σ c_i·a_i  −  z·Σ a_i
//! ```
//!
//! exact in integers. `Σ a_i` is computed once per activation block when the
//! activation is quantised and reused by every output row, so the correction costs
//! O(1) per block rather than O(n). This is the paper's Theorem 1 generalised from
//! ternary weights to any offset format.
//!
//! # Why the scalar path is the specification
//!
//! Each kernel below mirrors llama.cpp's own reference `dequantize_row_*` exactly,
//! including the interleaved output orderings, which are the part everyone gets
//! wrong: a transposed nibble order still decodes to plausible-looking floats and
//! produces fluent, wrong text. `tests/oracle.rs` checks these against the real
//! weight file, and the AVX2 kernels in `simd.rs` are checked against these.

use crate::gguf::{f16, Format};

/// An activation vector quantised to int8 in blocks of 32.
///
/// Built once per matmul and shared across every output row — that sharing is what
/// makes the bias correction in the module docs O(1) rather than O(n).
#[derive(Default, Clone)]
pub struct QuantAct {
    pub n: usize,
    /// int8 codes, `n` of them.
    pub q: Vec<i8>,
    /// One scale per 32-block: `a_i ≈ q_i · d[i/32]`.
    pub d: Vec<f32>,
    /// `Σ q` over each 32-block — the `Σ a_i` of the bias identity, in code space.
    pub sums: Vec<i32>,
    /// `Σ q` over each 16-half. Q6_K changes scale every 16 weights, so it needs the
    /// finer partition; `sums` is derivable from this but kept separately because
    /// the 32-block kernels are the hot ones.
    pub sums16: Vec<i32>,
    /// `Σ q` over each run of four codes, eight per 32-block, in the lane order
    /// `VPMADDWD` produces. The batched kernels subtract an offset format's bias
    /// *inside the integer lanes* (`lane − offset·Σq`), which is exact; applying it in
    /// float after accumulation cancels two large, nearly equal numbers and lost three
    /// digits on small outputs (measured: −0.015595 against −0.015600).
    pub lane_sums: Vec<i32>,
    /// The same activation quantised with **one scale per 256** (llama.cpp's Q8_K),
    /// for the k-quant formats whose super-block is 256 weights. With one scale per
    /// super-block the eight sub-block dot products can be scaled and summed in
    /// integer lanes and converted to float once per 256 weights, instead of once per
    /// 32 — which is the difference between a kernel bound by vector-ALU latency and
    /// one bound by memory. Empty when `n` is not a multiple of 256.
    pub qk: Vec<i8>,
    /// One scale per 256-block: `a_i ≈ qk_i · dk[i/256]`.
    pub dk: Vec<f32>,
    /// `Σ qk` over each run of sixteen, sixteen per 256-block: the Q4_K mins and the
    /// Q6_K offset are applied against these, in integer lanes.
    pub bsums: Vec<i16>,
}

impl QuantAct {
    pub fn with_capacity(n: usize) -> QuantAct {
        QuantAct {
            n: 0,
            q: Vec::with_capacity(n),
            d: Vec::with_capacity(n / 32 + 1),
            sums: Vec::with_capacity(n / 32 + 1),
            sums16: Vec::with_capacity(n / 16 + 1),
            lane_sums: Vec::with_capacity(n / 4 + 1),
            qk: Vec::with_capacity(n),
            dk: Vec::with_capacity(n / 256 + 1),
            bsums: Vec::with_capacity(n / 16 + 1),
        }
    }

    /// Quantise `x` symmetrically, one absmax scale per 32 weights.
    ///
    /// Symmetric (no zero point) because every weight format here is either symmetric
    /// or carries its own min, so an activation zero point would have to be carried
    /// through the bias correction for no accuracy gain.
    pub fn fill(&mut self, x: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::have_avx2() {
                // SAFETY: AVX2 checked above.
                unsafe { self.fill_avx2(x) };
                return;
            }
        }
        self.fill_scalar(x);
    }

    /// The vector `fill`: the same scales from the same scalar arithmetic, the codes
    /// eight at a time. It runs four times per layer per token, serially, and was a
    /// tenth of a SmolLM2 decode step on four threads (round 13).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fill_avx2(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 32,
            0,
            "activation length {} is not a multiple of 32",
            x.len()
        );
        self.n = x.len();
        self.q.clear();
        self.d.clear();
        self.sums.clear();
        self.sums16.clear();
        self.lane_sums.clear();
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();
        let mut codes = [0i8; 32];
        if x.len().is_multiple_of(256) {
            for blk in x.chunks_exact(256) {
                let amax = crate::simd::absmax(blk);
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for half in blk.chunks_exact(32) {
                    crate::simd::quantize32(half.as_ptr(), inv, codes.as_mut_ptr());
                    self.qk.extend_from_slice(&codes);
                    for run in codes.chunks_exact(16) {
                        self.bsums.push(run.iter().map(|&q| q as i16).sum());
                    }
                }
                self.dk.push(d);
            }
        }
        for blk in x.chunks_exact(32) {
            let amax = crate::simd::absmax(blk);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            crate::simd::quantize32(blk.as_ptr(), inv, codes.as_mut_ptr());
            self.q.extend_from_slice(&codes);
            for j in 0..8 {
                self.lane_sums
                    .push(codes[4 * j..4 * j + 4].iter().map(|&v| v as i32).sum());
            }
            let s_lo: i32 = codes[..16].iter().map(|&v| v as i32).sum();
            let s_hi: i32 = codes[16..].iter().map(|&v| v as i32).sum();
            self.d.push(d);
            self.sums16.push(s_lo);
            self.sums16.push(s_hi);
            self.sums.push(s_lo + s_hi);
        }
    }

    /// The scalar specification of `fill`, which the vector form is tested against.
    pub fn fill_scalar(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 32,
            0,
            "activation length {} is not a multiple of 32",
            x.len()
        );
        self.n = x.len();
        self.q.clear();
        self.d.clear();
        self.sums.clear();
        self.sums16.clear();
        self.lane_sums.clear();
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();

        if x.len().is_multiple_of(256) {
            for blk in x.chunks_exact(256) {
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for run in blk.chunks_exact(16) {
                    let mut s = 0i16;
                    for &v in run {
                        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        self.qk.push(q);
                        s += q as i16;
                    }
                    self.bsums.push(s);
                }
                self.dk.push(d);
            }
        }

        for blk in x.chunks_exact(32) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // An all-zero block is real (a masked position, a dead channel). Scale 0
            // with codes 0 reconstructs exactly, and avoids a division by zero.
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                self.q.push(q);
                if i < 16 {
                    s_lo += q as i32;
                } else {
                    s_hi += q as i32;
                }
            }
            let base = self.q.len() - 32;
            for j in 0..8 {
                let l = &self.q[base + 4 * j..base + 4 * j + 4];
                self.lane_sums.push(l.iter().map(|&v| v as i32).sum());
            }
            self.d.push(d);
            self.sums16.push(s_lo);
            self.sums16.push(s_hi);
            self.sums.push(s_lo + s_hi);
        }
    }
}

/// `f32` → IEEE half, round-to-nearest-even, with overflow to infinity and gradual
/// underflow — the inverse of `gguf::f16`, for writing block scales.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal half: shift the full mantissa (with its implicit bit) into place.
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut q = m >> shift;
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is the correct rounding up
    }
    sign | h as u16
}

/// One row as Q8_0 blocks (34 bytes per 32 weights), appended to `out`. The same
/// arithmetic as llama.cpp's `quantize_row_q8_0_ref`: `d = amax/127` stored as f16,
/// codes `round(x / d)` from the unrounded `d`.
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(32),
        "Q8_0 rows are whole blocks of 32"
    );
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|v| (v * id).round() as i8 as u8));
    }
}

/// ggml's `nearest_int`: round to nearest, ties to even, by the float-add trick —
/// reproduced exactly so the quantiser below picks the same codes as llama.cpp's.
#[inline]
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    ((v.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// `make_qkx2_quants` from `ggml-quants.c`: a scale and min for one 32-weight
/// sub-block, searched over `nstep` candidate scales to minimise the weighted error.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    nmax: i32,
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> f32 {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.iter_mut().for_each(|v| *v = 0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let d = scale * q as f32 + min - x[i];
        best += weights[i] * d * d;
    }
    let mut laux = [0u8; 32];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            sl += w * q as f32;
            sl2 += w * (q * q) as f32;
            sxl += w * q as f32 * x[i];
        }
        let det = sum_w * sl2 - sl * sl;
        if det > 0.0 {
            let mut this_scale = (sum_w * sxl - sum_x * sl) / det;
            let mut this_min = (sl2 * sum_x - sl * sxl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sxl / sl2;
            }
            let mut err = 0.0f32;
            for i in 0..n {
                let d = this_scale * laux[i] as f32 + this_min - x[i];
                err += weights[i] * d * d;
            }
            if err < best {
                l[..n].copy_from_slice(&laux[..n]);
                best = err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// One row as Q4_K super-blocks (144 bytes per 256 weights), appended to `out` —
/// `quantize_row_q4_K_ref` from `ggml-quants.c`, step for step, so a row quantised
/// here is the row llama.cpp would write. Used for the DSpark drafter, whose bf16
/// checkpoint would otherwise run at 8.5 bits per weight.
pub fn quantize_row_q4_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(256),
        "Q4_K rows are whole super-blocks of 256"
    );
    for xb in x.chunks_exact(256) {
        let mut l = [0u8; 256];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
        for j in 0..8 {
            let sub = &xb[32 * j..32 * j + 32];
            let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
            let av_x = (sum_x2 / 32.0).sqrt();
            let mut w = [0.0f32; 32];
            for (wi, v) in w.iter_mut().zip(sub) {
                *wi = av_x + v.abs();
            }
            scales[j] = make_qkx2_quants(
                sub,
                15,
                &w,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                -1.0,
                0.1,
                20,
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; 12];
        for j in 0..8 {
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                sc[j] = ls;
                sc[j + 4] = lm;
            } else {
                sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                sc[j - 4] |= (ls >> 4) << 6;
                sc[j] |= (lm >> 4) << 6;
            }
        }
        let d16 = f32_to_f16(max_scale / 63.0);
        let dmin16 = f32_to_f16(max_min / 63.0);
        let (d, dmin) = (crate::gguf::f16(d16), crate::gguf::f16(dmin16));
        for j in 0..8 {
            let (s6, m6) = scale_min_k4(j, &sc);
            let dj = d * s6 as f32;
            if dj == 0.0 {
                continue;
            }
            let dm = dmin * m6 as f32;
            for ii in 0..32 {
                l[32 * j + ii] = nearest_int((xb[32 * j + ii] + dm) / dj).clamp(0, 15) as u8;
            }
        }
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&dmin16.to_le_bytes());
        out.extend_from_slice(&sc);
        for j in (0..256).step_by(64) {
            for i in 0..32 {
                out.push(l[j + i] | (l[j + i + 32] << 4));
            }
        }
    }
}

/// `get_scale_min_k4` from `ggml-quants.c`: unpack the 6-bit scale and min for
/// sub-block `j` of a k-quant super-block out of the 12 packed bytes.
#[inline(always)]
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

#[inline(always)]
fn h(b: &[u8], i: usize) -> f32 {
    f16(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Number of bytes one row of `n` weights occupies in `fmt`.
pub fn row_bytes(fmt: Format, n: usize) -> usize {
    debug_assert_eq!(n % fmt.block(), 0);
    n / fmt.block() * fmt.block_bytes()
}

// ---------------------------------------------------------------------------
// Reference decode. Mirrors `dequantize_row_*` in ggml-quants.c element for
// element, including output interleaving. Used for embedding lookup, and as the
// oracle the fused kernels are checked against.
// ---------------------------------------------------------------------------

// The Q6_K decoders below contain `qh[l] >> 0`. It is an identity, and it is kept:
// the four shifts form the series 0/2/4/6, and writing the first one as a bare
// `qh[l]` breaks the column alignment that lets a reader check this transcription
// line-for-line against `ggml-quants.c`. That checkability is the whole reason the
// scalar path is the specification.
#[allow(clippy::identity_op, clippy::needless_range_loop)]
pub fn dequant_row(fmt: Format, w: &[u8], y: &mut [f32]) {
    let n = y.len();
    debug_assert_eq!(w.len(), row_bytes(fmt, n));
    match fmt {
        Format::F32 => {
            for (o, c) in y.iter_mut().zip(w.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        Format::F16 => {
            for (o, c) in y.iter_mut().zip(w.chunks_exact(2)) {
                *o = f16(u16::from_le_bytes(c.try_into().unwrap()));
            }
        }
        Format::Q8_0 => {
            for (blk, out) in w.chunks_exact(34).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                for j in 0..32 {
                    out[j] = d * (blk[2 + j] as i8) as f32;
                }
            }
        }
        Format::Q5_0 => {
            for (blk, out) in w.chunks_exact(22).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..22];
                // The two halves are interleaved: low nibbles produce outputs 0..16,
                // high nibbles produce 16..32, and the fifth bit for output j lives
                // at bit j of qh (and bit j+16 for the upper half).
                for j in 0..16 {
                    let hi0 = ((qh >> j) << 4) as u8 & 0x10;
                    let hi1 = (qh >> (j + 12)) as u8 & 0x10;
                    out[j] = d * (((qs[j] & 0x0F) | hi0) as i32 - 16) as f32;
                    out[j + 16] = d * (((qs[j] >> 4) | hi1) as i32 - 16) as f32;
                }
            }
        }
        Format::Q4_K => {
            for (blk, out) in w.chunks_exact(144).zip(y.chunks_exact_mut(256)) {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    // Sub-blocks alternate between the low and high nibble of the
                    // same 32 bytes: is=0 low of qs[0..32], is=1 high of qs[0..32],
                    // is=2 low of qs[32..64], and so on.
                    let base = (is / 2) * 32;
                    let o = &mut out[is * 32..is * 32 + 32];
                    if is % 2 == 0 {
                        for l in 0..32 {
                            o[l] = d1 * (qs[base + l] & 0x0F) as f32 - m1;
                        }
                    } else {
                        for l in 0..32 {
                            o[l] = d1 * (qs[base + l] >> 4) as f32 - m1;
                        }
                    }
                }
            }
        }
        Format::Q6_K => {
            for (blk, out) in w.chunks_exact(210).zip(y.chunks_exact_mut(256)) {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc: &[i8] =
                    unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
                let d = h(blk, 208);
                for c in 0..2 {
                    let ql = &ql[c * 64..c * 64 + 64];
                    let qh = &qh[c * 32..c * 32 + 32];
                    let sc = &sc[c * 8..c * 8 + 8];
                    let out = &mut out[c * 128..c * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        out[l] = d * sc[is] as f32 * q1 as f32;
                        out[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                        out[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                        out[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused dequantise-and-dot, scalar. `Σ w_i · â_i` where `â` is the int8
// reconstruction of the activation. No allocation, no float weight buffer.
// ---------------------------------------------------------------------------

/// Fused dequantise-and-dot, taking whichever kernel this machine can run.
///
/// The scalar twin is the specification (spec P3); this dispatch exists only to make
/// it fast. `tests/simd.rs` asserts the two agree on the real weight file, so a
/// divergence is a test failure rather than a subtly wrong model.
#[inline]
pub fn dot(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_avx2() {
            // SAFETY: feature checked immediately above; lengths are the same ones
            // `dot_scalar` asserts on.
            return unsafe { crate::simd::dot(fmt, w, a) };
        }
    }
    dot_scalar(fmt, w, a)
}

/// One weight row against every activation in `acts`, written to `out[..acts.len()]`.
///
/// This is what `matmul` calls. On a machine with AVX2 and FMA the row is decoded
/// once per tile of four activations instead of once per activation (see the batched
/// kernels in `simd.rs`); otherwise it is the plain per-activation loop, so the
/// result never depends on which path ran by more than float rounding.
#[inline]
pub fn dot_batch(fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    debug_assert!(out.len() >= acts.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_batched() {
            // SAFETY: feature checked immediately above; the length preconditions are
            // the same ones `matmul` already guarantees for `dot`.
            return unsafe { crate::simd::dot_batch(fmt, w, acts, out) };
        }
    }
    for (b, a) in acts.iter().enumerate() {
        out[b] = dot(fmt, w, a);
    }
}

#[allow(clippy::identity_op, clippy::needless_range_loop)]
pub fn dot_scalar(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), row_bytes(fmt, a.n));
    match fmt {
        Format::Q8_0 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(34).enumerate() {
                let mut s = 0i32;
                let aq = &a.q[b * 32..b * 32 + 32];
                for j in 0..32 {
                    s += (blk[2 + j] as i8) as i32 * aq[j] as i32;
                }
                // Signed codes already, so no bias term.
                acc += h(blk, 0) * a.d[b] * s as f32;
            }
            acc
        }
        Format::Q5_0 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(22).enumerate() {
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..22];
                let aq = &a.q[b * 32..b * 32 + 32];
                let mut s = 0i32;
                for j in 0..16 {
                    let hi0 = ((qh >> j) << 4) as u8 & 0x10;
                    let hi1 = (qh >> (j + 12)) as u8 & 0x10;
                    s += ((qs[j] & 0x0F) | hi0) as i32 * aq[j] as i32;
                    s += ((qs[j] >> 4) | hi1) as i32 * aq[j + 16] as i32;
                }
                // Bias identity: codes are [0,31] with offset 16, so subtract
                // 16·Σa once for the whole block instead of per weight.
                acc += h(blk, 0) * a.d[b] * (s - 16 * a.sums[b]) as f32;
            }
            acc
        }
        Format::Q4_K => {
            // Against the per-256 activation (`qk`, `dk`, `bsums`): the eight
            // sub-block dot products are scaled and summed as integers and the
            // super-block's one float scale is applied once.
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(144).enumerate() {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let bs = &a.bsums[sb * 16..sb * 16 + 16];
                let mut sumi = 0i32;
                let mut summ = 0i32;
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let base = (is / 2) * 32;
                    let act = &aq[is * 32..is * 32 + 32];
                    let mut s = 0i32;
                    if is % 2 == 0 {
                        for l in 0..32 {
                            s += (qs[base + l] & 0x0F) as i32 * act[l] as i32;
                        }
                    } else {
                        for l in 0..32 {
                            s += (qs[base + l] >> 4) as i32 * act[l] as i32;
                        }
                    }
                    sumi += sc as i32 * s;
                    // w = d·sc·q − dmin·m, so the min contributes −dmin·m·Σa.
                    summ += m as i32 * (bs[2 * is] as i32 + bs[2 * is + 1] as i32);
                }
                acc += a.dk[sb] * (d * sumi as f32 - dmin * summ as f32);
            }
            acc
        }
        Format::Q6_K => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(210).enumerate() {
                let ql_all = &blk[0..128];
                let qh_all = &blk[128..192];
                let sc_all: &[i8] =
                    unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
                let d = h(blk, 208);
                let mut sumi = 0i32;
                for c in 0..2 {
                    let ql = &ql_all[c * 64..c * 64 + 64];
                    let qh = &qh_all[c * 32..c * 32 + 32];
                    let sc = &sc_all[c * 8..c * 8 + 8];
                    // Outputs of this 128-chunk start here, in activation terms.
                    let obase = sb * 256 + c * 128;
                    // Four interleaved groups of 32, each split into two 16-runs that
                    // carry different scales — hence sums16 rather than sums.
                    for g in 0..4 {
                        for hf in 0..2 {
                            let scale = sc[2 * g + hf] as i32;
                            let o = obase + g * 32 + hf * 16;
                            let aq = &a.qk[o..o + 16];
                            let mut s = 0i32;
                            for l in 0..16 {
                                let li = hf * 16 + l;
                                let code = match g {
                                    0 => (ql[li] & 0x0F) | (((qh[li] >> 0) & 3) << 4),
                                    1 => (ql[li + 32] & 0x0F) | (((qh[li] >> 2) & 3) << 4),
                                    2 => (ql[li] >> 4) | (((qh[li] >> 4) & 3) << 4),
                                    _ => (ql[li + 32] >> 4) | (((qh[li] >> 6) & 3) << 4),
                                };
                                s += code as i32 * aq[l] as i32;
                            }
                            // Offset 32, applied once per 16-run via bsums.
                            let bias = 32 * a.bsums[o / 16] as i32;
                            sumi += scale * (s - bias);
                        }
                    }
                }
                acc += d * a.dk[sb] * sumi as f32;
            }
            acc
        }
        Format::F32 | Format::F16 => {
            // Not a matmul path — these appear as norm vectors, which are applied
            // elementwise. Implemented for completeness and used by tests.
            let mut acc = 0.0f32;
            let step = fmt.block_bytes();
            for i in 0..a.n {
                let wv = if fmt == Format::F32 {
                    f32::from_le_bytes(w[i * 4..i * 4 + 4].try_into().unwrap())
                } else {
                    h(w, i * step)
                };
                acc += wv * a.q[i] as f32 * a.d[i / 32];
            }
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_round_trips_and_rounds_to_even() {
        for h in 0..=u16::MAX {
            let f = crate::gguf::f16(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x} ({f})");
        }
        // Halfway between 1.0 and the next half (1 + 2^-10): ties go to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    /// The oracle: a row that is already exactly representable in Q4_K — one decoded
    /// from a real Q4_K block — must quantise back to essentially the same values.
    /// And a smooth random row must come back with a small error.
    #[test]
    fn q4_k_quantisation_matches_ggml_on_representable_rows() {
        let mut s = 0x9E37_79B9u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let x: Vec<f32> = (0..1024)
            .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2)
            .collect();
        let mut q = Vec::new();
        quantize_row_q4_k(&x, &mut q);
        assert_eq!(q.len(), row_bytes(Format::Q4_K, 1024));
        let mut y = vec![0.0f32; 1024];
        dequant_row(Format::Q4_K, &q, &mut y);
        let rmse = (x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 1024.0)
            .sqrt();
        let rms = (x.iter().map(|a| a * a).sum::<f32>() / 1024.0).sqrt();
        assert!(rmse / rms < 0.12, "relative RMSE {}", rmse / rms);
        // Re-quantising the decoded row is (near) idempotent.
        let mut q2 = Vec::new();
        quantize_row_q4_k(&y, &mut q2);
        let mut z = vec![0.0f32; 1024];
        dequant_row(Format::Q4_K, &q2, &mut z);
        let drift = y
            .iter()
            .zip(&z)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 2e-3, "re-quantising moved a value by {drift}");
    }

    #[test]
    fn q8_0_quantisation_round_trips_within_half_a_step() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut q = Vec::new();
        quantize_row_q8_0(&x, &mut q);
        assert_eq!(q.len(), row_bytes(Format::Q8_0, 256));
        let mut y = vec![0.0f32; 256];
        dequant_row(Format::Q8_0, &q, &mut y);
        for (blk_x, blk_y) in x.chunks(32).zip(y.chunks(32)) {
            let step = blk_x.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (a, b) in blk_x.iter().zip(blk_y) {
                // Half a step from rounding the code, plus up to 127·d·2⁻¹¹ because
                // the stored scale is rounded to f16 while the codes used the exact one.
                assert!(
                    (a - b).abs() <= step * (0.5 + 127.0 / 2048.0) + 1e-6,
                    "{a} vs {b}"
                );
            }
        }
    }

    /// A deterministic pseudo-random byte stream. Real weight bytes are exercised by
    /// `tests/oracle.rs`; this is for the structural properties.
    fn prng(seed: u64, n: usize) -> Vec<u8> {
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

    /// Force every f16 block scale in a randomly generated row to a finite, sane
    /// value. Raw random bytes hit the half-precision Inf/NaN encodings about one
    /// block in 128, and a NaN scale propagates through both the kernel and the
    /// reference, so the comparison would pass vacuously (NaN == NaN is never
    /// asserted) or fail for a reason that has nothing to do with the kernel.
    /// Clearing the top exponent bits leaves a normal half in [0.125, 0.25).
    fn sane_scales(fmt: Format, w: &mut [u8]) {
        let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
        let sz = fmt.block_bytes();
        for blk in w.chunks_exact_mut(sz) {
            match fmt {
                Format::Q8_0 | Format::Q5_0 => fix(blk, 0),
                Format::Q4_K => {
                    fix(blk, 0);
                    fix(blk, 2);
                }
                Format::Q6_K => fix(blk, 208),
                Format::F32 | Format::F16 => {}
            }
        }
    }

    fn acts(n: usize, seed: u64) -> (Vec<f32>, QuantAct) {
        let raw = prng(seed, n * 2);
        let x: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| (u16::from_le_bytes(c.try_into().unwrap()) as f32 / 32768.0) - 1.0)
            .collect();
        let mut q = QuantAct::with_capacity(n);
        q.fill(&x);
        (x, q)
    }

    /// The central invariant of this file: the fused kernel must equal the reference
    /// decode followed by a plain dot product. If these disagree, the fusion is
    /// wrong — which is a class of bug that still produces fluent text.
    #[allow(clippy::needless_range_loop)]
    fn fused_equals_reference(fmt: Format, n: usize) {
        let mut w = prng(0xA5A5_1234, row_bytes(fmt, n));
        sane_scales(fmt, &mut w);
        let (_, a) = acts(n, 0x1357_9BDF);

        let mut deq = vec![0.0f32; n];
        dequant_row(fmt, &w, &mut deq);
        // Reference: dot of the decoded weights against the *reconstructed*
        // activation, which is what the fused kernel actually computes — the
        // per-256 representation for the k-quants, the per-32 one otherwise.
        let kq = matches!(fmt, Format::Q4_K | Format::Q6_K);
        let mut want = 0.0f64;
        for i in 0..n {
            let act = if kq {
                a.qk[i] as f32 * a.dk[i / 256]
            } else {
                a.q[i] as f32 * a.d[i / 32]
            };
            want += (deq[i] * act) as f64;
        }
        let got = dot_scalar(fmt, &w, &a) as f64;
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() / scale < 1e-5,
            "{}: fused {} vs reference {} (rel {:.2e})",
            fmt.label(),
            got,
            want,
            (got - want).abs() / scale
        );
    }

    /// The per-256 activation is a coarser quantisation of the same vector: it
    /// must reconstruct within its own step and carry exact run sums.
    #[test]
    fn per_256_activation_reconstructs_and_sums_exactly() {
        let (x, a) = acts(512, 0x2468_ACE0);
        assert_eq!(a.qk.len(), 512);
        assert_eq!(a.dk.len(), 2);
        assert_eq!(a.bsums.len(), 32);
        for (i, &v) in x.iter().enumerate() {
            let r = a.qk[i] as f32 * a.dk[i / 256];
            assert!(
                (r - v).abs() <= a.dk[i / 256] * 0.5 + 1e-6,
                "{i}: {r} vs {v}"
            );
        }
        for run in 0..32 {
            let s: i32 = a.qk[run * 16..run * 16 + 16]
                .iter()
                .map(|&q| q as i32)
                .sum();
            assert_eq!(s, a.bsums[run] as i32);
        }
        let (_, b) = acts(96, 1);
        assert!(
            b.qk.is_empty() && b.dk.is_empty(),
            "not a multiple of 256: no per-256 form"
        );
    }

    #[test]
    fn fused_dot_matches_reference_decode_q8_0() {
        fused_equals_reference(Format::Q8_0, 960);
    }

    #[test]
    fn fused_dot_matches_reference_decode_q5_0() {
        fused_equals_reference(Format::Q5_0, 960);
    }

    #[test]
    fn fused_dot_matches_reference_decode_q4_k() {
        fused_equals_reference(Format::Q4_K, 2560);
    }

    #[test]
    fn fused_dot_matches_reference_decode_q6_k() {
        fused_equals_reference(Format::Q6_K, 2560);
    }

    #[test]
    fn scale_min_k4_unpacks_six_bit_fields() {
        // Construct a byte pattern where every field is distinguishable, then check
        // the two branches agree with the packing rule in ggml-quants.c.
        let q: Vec<u8> = (0..12).map(|i| (i * 17 + 3) as u8).collect();
        for j in 0..4 {
            let (sc, m) = scale_min_k4(j, &q);
            assert_eq!(sc, q[j] & 63);
            assert_eq!(m, q[j + 4] & 63);
        }
        for j in 4..8 {
            let (sc, m) = scale_min_k4(j, &q);
            assert_eq!(sc, (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            assert_eq!(m, (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
            // Six-bit fields: nothing may leak into the top two bits.
            assert!(sc < 64 && m < 64, "j={j} sc={sc} m={m}");
        }
    }

    #[test]
    fn activation_quantisation_round_trips_within_one_step() {
        let (x, a) = acts(256, 0xFEED);
        for (i, &v) in x.iter().enumerate() {
            let back = a.q[i] as f32 * a.d[i / 32];
            // One quantisation step is amax/127; reconstruction must be within half.
            let step = a.d[i / 32];
            assert!((back - v).abs() <= step * 0.5 + 1e-9, "i={i} {v} -> {back}");
        }
        // And the cached sums must match what the codes actually are.
        for b in 0..a.n / 32 {
            let s: i32 = a.q[b * 32..b * 32 + 32].iter().map(|&q| q as i32).sum();
            assert_eq!(s, a.sums[b], "block {b} sum cache is stale");
            assert_eq!(a.sums16[2 * b] + a.sums16[2 * b + 1], a.sums[b]);
            assert_eq!(a.lane_sums[b * 8..b * 8 + 8].iter().sum::<i32>(), a.sums[b]);
            let lo: i32 = a.lane_sums[b * 8..b * 8 + 4].iter().sum();
            assert_eq!(lo, a.sums16[2 * b], "lanes 0..4 are the first 16 codes");
        }
    }

    #[test]
    fn an_all_zero_activation_block_does_not_divide_by_zero() {
        let mut a = QuantAct::with_capacity(32);
        a.fill(&[0.0f32; 32]);
        assert_eq!(a.d[0], 0.0);
        assert!(a.q.iter().all(|&q| q == 0));
        let mut w = prng(7, row_bytes(Format::Q5_0, 32));
        sane_scales(Format::Q5_0, &mut w);
        assert_eq!(dot_scalar(Format::Q5_0, &w, &a), 0.0);
    }
}
