//! The key–value cache, and the `κ(T)` term of the performance model.
//!
//! This is the one part of the weight traffic that grows with context. At short
//! prompts it is negligible against the 267 MB of weights this model reads per
//! token; by a few thousand positions it is a visible fraction, which is exactly
//! why the cache precision is a lever worth having and why `bytes_read_per_token`
//! is reported rather than assumed.
//!
//! # Why int8 and not int4
//!
//! The evidence on cache quantisation is asymmetric. int8 keys and values are close
//! to lossless across model families; int4 degrades measurably, and the degradation
//! concentrates in long contexts — which is precisely the regime that motivates
//! quantising the cache in the first place. Offering int4 would be offering a
//! setting that is worst exactly where it is advertised as best, so this cache
//! stops at int8 and the decision is recorded rather than left to a future reader.
//!
//! # Layout
//!
//! `[layer][kv_head][position][head_dim]`, so the positions of one head are
//! contiguous. Attention scans all positions of one head in order, and that is the
//! only access pattern the cache has.

use crate::gguf::{f16, to_f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrec {
    F16,
    I8,
}

impl KvPrec {
    pub fn label(self) -> &'static str {
        match self {
            KvPrec::F16 => "f16",
            KvPrec::I8 => "int8",
        }
    }
    /// Bytes per stored element, excluding the per-vector scale that I8 adds.
    const fn elem(self) -> usize {
        match self {
            KvPrec::F16 => 2,
            KvPrec::I8 => 1,
        }
    }
}

/// Half-precision encode with F16C, round-to-nearest-even — the same rounding as
/// `to_f16`, which the cache test pins.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn encode_f16_avx(x: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= x.len() {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(x.as_ptr().add(i)));
        _mm_storeu_si128(out.as_mut_ptr().add(i * 2) as *mut __m128i, h);
        i += 8;
    }
}

/// Whether the vector attention path may run. F16C is what widens the stored
/// half-precision keys; FMA and AVX2 are the arithmetic.
#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_path() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

pub struct KvCache {
    prec: KvPrec,
    n_layer: usize,
    n_kv_head: usize,
    head_dim: usize,
    max_ctx: usize,
    /// Encoded keys and values, one flat buffer each.
    k: Vec<u8>,
    v: Vec<u8>,
    /// One scale per (layer, head, position) vector, for `I8` only. Per-vector
    /// rather than per-tensor: coarse scaling is where the reported int8 accuracy
    /// loss comes from, and a scale per 64 values costs 4 bytes against 64.
    ks: Vec<f32>,
    vs: Vec<f32>,
    /// Positions currently filled.
    pub len: usize,
}

impl KvCache {
    pub fn new(
        prec: KvPrec,
        n_layer: usize,
        n_kv_head: usize,
        head_dim: usize,
        max_ctx: usize,
    ) -> KvCache {
        let vectors = n_layer * n_kv_head * max_ctx;
        let elems = vectors * head_dim;
        let scales = if prec == KvPrec::I8 { vectors } else { 0 };
        KvCache {
            prec,
            n_layer,
            n_kv_head,
            head_dim,
            max_ctx,
            k: vec![0u8; elems * prec.elem()],
            v: vec![0u8; elems * prec.elem()],
            ks: vec![0.0; scales],
            vs: vec![0.0; scales],
            len: 0,
        }
    }

    pub fn prec(&self) -> KvPrec {
        self.prec
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Drop everything from `pos` onward. Used when a speculative block is only
    /// partly accepted: the rejected positions were written before verification and
    /// must not be visible to the next step.
    pub fn truncate(&mut self, pos: usize) {
        self.len = self.len.min(pos);
    }

    #[inline]
    fn vec_index(&self, layer: usize, head: usize, pos: usize) -> usize {
        debug_assert!(layer < self.n_layer && head < self.n_kv_head && pos < self.max_ctx);
        (layer * self.n_kv_head + head) * self.max_ctx + pos
    }

    /// Bytes one layer holds for one position, K and V together. This is the unit
    /// the attention loop accumulates, because that loop runs once per layer — using
    /// the whole-model figure there overcounts cache traffic by `n_layer`.
    pub fn bytes_per_position_per_layer(&self) -> u64 {
        let per_vec =
            self.head_dim * self.prec.elem() + if self.prec == KvPrec::I8 { 4 } else { 0 };
        (self.n_kv_head * per_vec * 2) as u64 // ×2 for K and V
    }

    /// Total bytes the cache holds for one position across all layers — `κ` per
    /// position. Includes the I8 scales, because they are read too.
    pub fn bytes_per_position(&self) -> u64 {
        self.n_layer as u64 * self.bytes_per_position_per_layer()
    }

    /// `κ(T)`: bytes read from the cache to produce one token at the current length.
    pub fn bytes_read_at(&self, len: usize) -> u64 {
        self.bytes_per_position() * len as u64
    }

    fn encode_into(&mut self, is_k: bool, idx: usize, x: &[f32]) {
        let hd = self.head_dim;
        let off = idx * hd;
        match self.prec {
            KvPrec::F16 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if hd.is_multiple_of(8) && vector_path() {
                        let buf = if is_k { &mut self.k } else { &mut self.v };
                        // SAFETY: features checked; `off + hd` elements fit (vec_index).
                        unsafe { encode_f16_avx(x, &mut buf[off * 2..(off + hd) * 2]) };
                        return;
                    }
                }
                let buf = if is_k { &mut self.k } else { &mut self.v };
                for (i, &val) in x.iter().enumerate() {
                    let h = to_f16(val).to_le_bytes();
                    buf[(off + i) * 2] = h[0];
                    buf[(off + i) * 2 + 1] = h[1];
                }
            }
            KvPrec::I8 => {
                let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                let buf = if is_k { &mut self.k } else { &mut self.v };
                for (i, &val) in x.iter().enumerate() {
                    buf[off + i] = ((val * inv).round().clamp(-127.0, 127.0) as i8) as u8;
                }
                if is_k {
                    self.ks[idx] = d;
                } else {
                    self.vs[idx] = d;
                }
            }
        }
    }

    /// Append one position's key and value for one head. Callers must write every
    /// layer and head for a position before advancing `len`.
    pub fn put(&mut self, layer: usize, head: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.head_dim);
        debug_assert_eq!(v.len(), self.head_dim);
        let idx = self.vec_index(layer, head, pos);
        self.encode_into(true, idx, k);
        self.encode_into(false, idx, v);
    }

    pub fn advance(&mut self, to: usize) {
        self.len = to;
    }

    /// `q · k[pos]`, decoding the stored key in the loop rather than into a buffer —
    /// the same fusion rule the weight kernels follow.
    #[inline]
    pub fn dot_k(&self, layer: usize, head: usize, pos: usize, q: &[f32]) -> f32 {
        let idx = self.vec_index(layer, head, pos);
        let hd = self.head_dim;
        match self.prec {
            KvPrec::F16 => {
                let b = &self.k[idx * hd * 2..(idx + 1) * hd * 2];
                let mut s = 0.0f32;
                for i in 0..hd {
                    s += f16(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])) * q[i];
                }
                s
            }
            KvPrec::I8 => {
                let b = &self.k[idx * hd..(idx + 1) * hd];
                let mut s = 0.0f32;
                for i in 0..hd {
                    s += (b[i] as i8) as f32 * q[i];
                }
                s * self.ks[idx]
            }
        }
    }

    /// `acc += w · v[pos]`, decoded in the loop.
    #[inline]
    pub fn accum_v(&self, layer: usize, head: usize, pos: usize, w: f32, acc: &mut [f32]) {
        let idx = self.vec_index(layer, head, pos);
        let hd = self.head_dim;
        match self.prec {
            KvPrec::F16 => {
                let b = &self.v[idx * hd * 2..(idx + 1) * hd * 2];
                for i in 0..hd {
                    acc[i] += w * f16(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]));
                }
            }
            KvPrec::I8 => {
                let b = &self.v[idx * hd..(idx + 1) * hd];
                let s = w * self.vs[idx];
                for i in 0..hd {
                    acc[i] += s * (b[i] as i8) as f32;
                }
            }
        }
    }

    /// `out[p] = scale · q·k[p]` for every position `0..n_ctx` of one head.
    ///
    /// The whole scan in one call, so the vector path is entered once per head rather
    /// than once per position, and the half-precision keys are widened eight at a time
    /// by `VCVTPH2PS` instead of one at a time through a scalar bit-twiddling
    /// conversion. At a few hundred positions the scalar version was the largest
    /// single cost of attention; it is the term that grows with context.
    #[allow(clippy::needless_range_loop)] // index loops read as the kernel is written
    pub fn scores(
        &self,
        layer: usize,
        head: usize,
        n_ctx: usize,
        q: &[f32],
        scale: f32,
        out: &mut [f32],
    ) {
        debug_assert!(out.len() >= n_ctx && q.len() == self.head_dim);
        #[cfg(target_arch = "x86_64")]
        {
            if self.head_dim.is_multiple_of(8) && vector_path() {
                // SAFETY: features checked; `vec_index` bounds `idx0 + n_ctx` inside the
                // buffer because `n_ctx <= max_ctx` is enforced by `forward`.
                unsafe { return self.scores_avx2(layer, head, n_ctx, q, scale, out) };
            }
        }
        for p in 0..n_ctx {
            out[p] = self.dot_k(layer, head, p, q) * scale;
        }
    }

    /// `out += Σ_p w[p] · v[p]` for one head, decoded in the loop.
    #[allow(clippy::needless_range_loop)] // index loops read as the kernel is written
    pub fn mix(&self, layer: usize, head: usize, n_ctx: usize, w: &[f32], out: &mut [f32]) {
        debug_assert!(w.len() >= n_ctx && out.len() == self.head_dim);
        #[cfg(target_arch = "x86_64")]
        {
            if self.head_dim.is_multiple_of(8) && vector_path() {
                // SAFETY: as `scores`.
                unsafe { return self.mix_avx2(layer, head, n_ctx, w, out) };
            }
        }
        for p in 0..n_ctx {
            self.accum_v(layer, head, p, w[p], out);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)] // index loops read as the kernel is written
    unsafe fn scores_avx2(
        &self,
        layer: usize,
        head: usize,
        n_ctx: usize,
        q: &[f32],
        scale: f32,
        out: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        let hd = self.head_dim;
        let idx0 = self.vec_index(layer, head, 0);
        for p in 0..n_ctx {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            match self.prec {
                KvPrec::F16 => {
                    let kp = self.k.as_ptr().add((idx0 + p) * hd * 2);
                    let mut i = 0;
                    while i + 16 <= hd {
                        let a = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i * 2) as *const __m128i));
                        let b =
                            _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i * 2 + 16) as *const __m128i));
                        acc0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(q.as_ptr().add(i)), acc0);
                        acc1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(q.as_ptr().add(i + 8)), acc1);
                        i += 16;
                    }
                    if i < hd {
                        let a = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i * 2) as *const __m128i));
                        acc0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(q.as_ptr().add(i)), acc0);
                    }
                    out[p] = crate::simd::hsum_ps(_mm256_add_ps(acc0, acc1)) * scale;
                }
                KvPrec::I8 => {
                    let kp = self.k.as_ptr().add((idx0 + p) * hd);
                    let mut i = 0;
                    while i + 16 <= hd {
                        let a = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            kp.add(i) as *const __m128i,
                        )));
                        let b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            kp.add(i + 8) as *const __m128i,
                        )));
                        acc0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(q.as_ptr().add(i)), acc0);
                        acc1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(q.as_ptr().add(i + 8)), acc1);
                        i += 16;
                    }
                    if i < hd {
                        let a = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            kp.add(i) as *const __m128i,
                        )));
                        acc0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(q.as_ptr().add(i)), acc0);
                    }
                    out[p] =
                        crate::simd::hsum_ps(_mm256_add_ps(acc0, acc1)) * self.ks[idx0 + p] * scale;
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::needless_range_loop)] // index loops read as the kernel is written
    unsafe fn mix_avx2(&self, layer: usize, head: usize, n_ctx: usize, w: &[f32], out: &mut [f32]) {
        use std::arch::x86_64::*;
        let hd = self.head_dim;
        let idx0 = self.vec_index(layer, head, 0);
        // Sixty-four dimensions at a time: eight accumulators stay in registers for
        // the whole pass over the positions, instead of being loaded and stored once
        // per position.
        let mut d0 = 0;
        while d0 < hd {
            let cnt = (hd - d0).min(64);
            let lanes = cnt / 8;
            let mut acc = [_mm256_setzero_ps(); 8];
            for p in 0..n_ctx {
                match self.prec {
                    KvPrec::F16 => {
                        let vp = self.v.as_ptr().add((idx0 + p) * hd * 2 + d0 * 2);
                        let wv = _mm256_set1_ps(w[p]);
                        for j in 0..lanes {
                            let x =
                                _mm256_cvtph_ps(_mm_loadu_si128(vp.add(j * 16) as *const __m128i));
                            acc[j] = _mm256_fmadd_ps(wv, x, acc[j]);
                        }
                    }
                    KvPrec::I8 => {
                        let vp = self.v.as_ptr().add((idx0 + p) * hd + d0);
                        let wv = _mm256_set1_ps(w[p] * self.vs[idx0 + p]);
                        for j in 0..lanes {
                            let x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                                vp.add(j * 8) as *const __m128i,
                            )));
                            acc[j] = _mm256_fmadd_ps(wv, x, acc[j]);
                        }
                    }
                }
            }
            for j in 0..lanes {
                let o = out.as_mut_ptr().add(d0 + j * 8);
                _mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), acc[j]));
            }
            d0 += cnt;
        }
    }

    /// Resident bytes of the cache itself, for the memory report.
    pub fn resident_bytes(&self) -> u64 {
        (self.k.len() + self.v.len() + (self.ks.len() + self.vs.len()) * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) / n as f32 - 0.5) * scale)
            .collect()
    }

    #[test]
    fn f16_cache_round_trips_a_dot_product() {
        let hd = 64;
        let mut c = KvCache::new(KvPrec::F16, 2, 2, hd, 8);
        let k = ramp(hd, 3.0);
        let v = ramp(hd, 1.0);
        c.put(1, 1, 3, &k, &v);
        let q = ramp(hd, 2.0);

        let want: f32 = k.iter().zip(&q).map(|(a, b)| a * b).sum();
        let got = c.dot_k(1, 1, 3, &q);
        assert!(
            (got - want).abs() < 1e-2 * want.abs().max(1.0),
            "{got} vs {want}"
        );

        let mut acc = vec![0.0f32; hd];
        c.accum_v(1, 1, 3, 2.0, &mut acc);
        for i in 0..hd {
            assert!((acc[i] - 2.0 * v[i]).abs() < 1e-3, "i={i}");
        }
    }

    #[test]
    fn int8_cache_is_accurate_to_its_quantisation_step() {
        let hd = 64;
        let mut c = KvCache::new(KvPrec::I8, 1, 1, hd, 4);
        let k = ramp(hd, 3.0);
        let v = ramp(hd, 1.0);
        c.put(0, 0, 0, &k, &v);
        let q = ramp(hd, 2.0);
        let want: f32 = k.iter().zip(&q).map(|(a, b)| a * b).sum();
        let got = c.dot_k(0, 0, 0, &q);
        // Per-vector absmax int8: worst case error is half a step per element.
        let step = k.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 127.0;
        let bound: f32 = q.iter().map(|x| x.abs()).sum::<f32>() * step * 0.5;
        assert!(
            (got - want).abs() <= bound,
            "{got} vs {want}, bound {bound}"
        );
    }

    /// The whole reason the cache has a precision setting: int8 must actually halve
    /// the bytes read. A layout that stored scales per element would not.
    #[test]
    fn int8_moves_about_half_the_bytes_of_f16() {
        let f = KvCache::new(KvPrec::F16, 32, 5, 64, 2048);
        let i = KvCache::new(KvPrec::I8, 32, 5, 64, 2048);
        let (bf, bi) = (f.bytes_per_position(), i.bytes_per_position());
        assert_eq!(bf, 32 * 5 * 64 * 2 * 2);
        // And the per-layer unit must be exactly the whole divided by the layers.
        assert_eq!(f.bytes_per_position_per_layer() * 32, bf);
        // int8 adds 4 bytes of scale per 64-element vector, so the saving is just
        // under half — stating it exactly keeps the telemetry honest.
        assert_eq!(bi, 32 * 5 * (64 + 4) * 2);
        let ratio = bi as f64 / bf as f64;
        assert!((0.5..0.55).contains(&ratio), "ratio {ratio}");
    }

    /// The vector scan must be the scalar scan, for both precisions and for a head
    /// width that is not a multiple of sixteen (the single-register tail).
    #[test]
    fn vector_attention_matches_the_scalar_loops() {
        for prec in [KvPrec::F16, KvPrec::I8] {
            for hd in [64usize, 24, 128] {
                let n = 37;
                let mut c = KvCache::new(prec, 2, 2, hd, 64);
                for p in 0..n {
                    let k: Vec<f32> = ramp(hd, 2.0 + p as f32 * 0.01)
                        .iter()
                        .map(|v| v * (1.0 + p as f32 * 0.05))
                        .collect();
                    let v: Vec<f32> = ramp(hd, 1.5).iter().map(|x| x - p as f32 * 0.003).collect();
                    c.put(1, 1, p, &k, &v);
                }
                c.advance(n);
                let q = ramp(hd, 1.2);
                let mut got = vec![0.0f32; n];
                c.scores(1, 1, n, &q, 0.125, &mut got);
                for p in 0..n {
                    let want = c.dot_k(1, 1, p, &q) * 0.125;
                    assert!(
                        (got[p] - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "{prec:?} hd={hd} p={p}: {} vs {want}",
                        got[p]
                    );
                }
                let w: Vec<f32> = (0..n).map(|p| 1.0 / (1.0 + p as f32)).collect();
                let mut mixed = vec![0.25f32; hd];
                c.mix(1, 1, n, &w, &mut mixed);
                let mut refv = vec![0.25f32; hd];
                for p in 0..n {
                    c.accum_v(1, 1, p, w[p], &mut refv);
                }
                for i in 0..hd {
                    assert!(
                        (mixed[i] - refv[i]).abs() <= 1e-4 * refv[i].abs().max(1.0),
                        "{prec:?} hd={hd} i={i}: {} vs {}",
                        mixed[i],
                        refv[i]
                    );
                }
            }
        }
    }

    /// The F16C encoder must round exactly as the scalar `to_f16` does, including
    /// subnormals, overflow to infinity and the round-half-to-even ties.
    #[test]
    fn vector_f16_encode_matches_scalar_rounding() {
        let mut xs: Vec<f32> = vec![
            0.0, -0.0, 1.0, -2.5, 65504.0, 65520.0, 1e9, -1e9, 6.0e-8, 3.0e-8, 1.0e-5, -1.0e-7,
        ];
        // Ties at several exponents: 1 + 2^-11 sits exactly between two halves.
        for e in -14i32..15 {
            let base = 2f32.powi(e);
            xs.push(base * (1.0 + 2f32.powi(-11)));
            xs.push(base * (1.0 + 3.0 * 2f32.powi(-11)));
            xs.push(-base * 1.3333);
        }
        while xs.len() % 8 != 0 {
            xs.push(0.123);
        }
        let hd = xs.len();
        let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, 2);
        c.put(0, 0, 0, &xs, &xs);
        for (i, &x) in xs.iter().enumerate() {
            let got = u16::from_le_bytes([c.k[i * 2], c.k[i * 2 + 1]]);
            assert_eq!(got, to_f16(x), "x = {x:e} (index {i})");
        }
    }

    #[test]
    fn truncate_discards_rejected_speculative_positions() {
        let mut c = KvCache::new(KvPrec::F16, 1, 1, 8, 16);
        c.advance(10);
        c.truncate(4);
        assert_eq!(c.len, 4);
        // Truncating above the current length must not extend it.
        c.truncate(9);
        assert_eq!(c.len, 4);
    }

    #[test]
    fn kappa_grows_linearly_with_context() {
        let c = KvCache::new(KvPrec::F16, 32, 5, 64, 4096);
        assert_eq!(c.bytes_read_at(0), 0);
        assert_eq!(c.bytes_read_at(1000), c.bytes_per_position() * 1000);
        assert_eq!(c.bytes_read_at(2000), 2 * c.bytes_read_at(1000));
    }
}
