//! LittleBit: binary low-rank weight factorization (arXiv 2506.13771), on a CPU.
//!
//! A weight matrix `W (d_out×d_in)` is replaced by `S` stages of
//! `diag(h) · U_sign · diag(ℓ) · V_signᵀ · diag(g)` — two sign matrices at one bit
//! per entry and three small scale vectors per stage. The forward pass never
//! materialises the weight (the paper's Proposition 1, and this crate's rule P1):
//!
//! ```text
//!   y = Σ_stages  h ⊙ ( U_sign · ( ℓ ⊙ ( V_signᵀ · (g ⊙ x) ) ) )
//! ```
//!
//! which is `r(d_in + d_out)` *additions* per stage — no multiplies in the hot loop
//! at all — against `r(d_in + d_out)/8` bytes of weight traffic. On an engine whose
//! decode is bandwidth-bound (SMILEY-CORE-SPEC), that byte collapse is the whole
//! point: at 1 BPW the projection matrices move ~4× fewer bytes than Q4_K.
//!
//! Three parts live here, in file order:
//!
//! 1. **Kernels** — `lb_dot_batch`: one sign-bit row against a batch of float rows.
//!    The scalar form is the specification; the AVX2 form expands each bit-byte
//!    through a 256-entry mask table and flips float signs with `XOR` — load, xor,
//!    add, three µops per eight weights. Checked equal in tests, as `simd.rs` is.
//! 2. **The SLB container** — the mmap-first sidecar format (`model.gguf.slb`).
//!    Layout decisions and the RustFS assessment that shaped it are in
//!    `docs/LITTLEBIT-PLAN.md` §3: no storage daemon, but its good ideas kept —
//!    CRC32C on every section (a flipped sign bit is a *silently wrong* model),
//!    atomic publish (tmp + fsync + rename), and a source-identity stamp so a stale
//!    sidecar is detected rather than trusted.
//! 3. **The builder** — Dual-SVID (paper §3.2) by randomised subspace iteration,
//!    an S-stage residual cascade (§3.3 generalised), and alternating closed-form
//!    refinement of `h` and `g`, which is what can still be optimised without the
//!    paper's QAT. PTQ-only quality at aggressive BPW is *not* the paper's — the
//!    plan says so up front — but the representation, container and kernels are
//!    exact, tested, and fast.

use crate::gguf::{f16, to_f16, Gguf};
use crate::pool::{share, Pool};
use crate::quant::{dequant_row, row_bytes};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------ checksums --

/// CRC32C (Castagnoli) — cheap insurance against the one failure mode binary weights
/// make invisible: a flipped bit still generates fluent text. SSE4.2's `crc32`
/// instruction computes exactly this polynomial (~8 bytes a cycle, so a 100 MB
/// sidecar verifies in ~15 ms); the table is the specification and the fallback.
const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0x82F6_3B78 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}
static CRC_TABLE: [u32; 256] = crc_table();

pub fn crc32c(b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            // SAFETY: SSE4.2 checked above.
            return unsafe { crc32c_sse42(b) };
        }
    }
    crc32c_table(b)
}

pub fn crc32c_table(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for &x in b {
        c = CRC_TABLE[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(b: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    let mut c = !0u64;
    let mut words = b.chunks_exact(8);
    for w in &mut words {
        c = _mm_crc32_u64(c, u64::from_le_bytes(w.try_into().unwrap()));
    }
    let mut c = c as u32;
    for &x in words.remainder() {
        c = _mm_crc32_u8(c, x);
    }
    !c
}

/// FNV-1a 64 — the source-identity stamp. Not cryptographic and not meant to be:
/// it detects "this sidecar was built from a different file", not an adversary.
pub fn fnv1a64(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The stamp of a source GGUF: its length plus the FNV of its first 64 KiB (the
/// header region — metadata and tensor directory — which pins the model identity).
/// Reads only those 64 KiB: it runs on every sidecar load and status check, and
/// the file behind it can be gigabytes.
pub fn source_stamp(path: &Path) -> Result<(u64, u64), String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("slb: cannot read {path:?}: {e}"))?;
    let len = f
        .metadata()
        .map_err(|e| format!("slb: cannot stat {path:?}: {e}"))?
        .len();
    let mut head = Vec::with_capacity(64 * 1024);
    f.take(64 * 1024)
        .read_to_end(&mut head)
        .map_err(|e| format!("slb: cannot read {path:?}: {e}"))?;
    Ok((len, fnv1a64(&head)))
}

// -------------------------------------------------------------------- kernels --

/// One mask vector per possible bit-byte: lane `i` is `0` when bit `i` is set
/// (+1, leave the float alone) and `0x8000_0000` when clear (−1, flip its sign).
const fn sign_lut() -> [[u32; 8]; 256] {
    let mut t = [[0u32; 8]; 256];
    let mut b = 0;
    while b < 256 {
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 0 {
                t[b][i] = 0x8000_0000;
            }
            i += 1;
        }
        b += 1;
    }
    t
}
static SIGN_LUT: [[u32; 8]; 256] = sign_lut();

/// `Σ_i ±x_i` with the sign from bit `i` (LSB-first; 1 ⇒ +, 0 ⇒ −).
/// The specification the vector kernel is tested against.
pub fn lb_dot_scalar(bits: &[u8], n: usize, x: &[f32]) -> f32 {
    debug_assert!(n.is_multiple_of(8) && bits.len() >= n / 8 && x.len() >= n);
    let mut acc = 0.0f32;
    for (c, &byte) in bits[..n / 8].iter().enumerate() {
        let xs = &x[c * 8..c * 8 + 8];
        let mut s = 0.0f32;
        for (i, &v) in xs.iter().enumerate() {
            s += if (byte >> i) & 1 == 1 { v } else { -v };
        }
        acc += s;
    }
    acc
}

/// One sign row against `nb` activation rows (`xs[b*stride..][..n]`), written to
/// `out[..nb]`. The bit row is read once for the whole batch — the same discipline
/// as the dense `dot_batch`, and the reason prefill and verification stay cheap.
pub fn lb_dot_batch(bits: &[u8], n: usize, xs: &[f32], stride: usize, nb: usize, out: &mut [f32]) {
    debug_assert!(n.is_multiple_of(32), "lb rows are whole 32-blocks, got {n}");
    debug_assert!(xs.len() >= (nb - 1) * stride + n && out.len() >= nb);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            let mut b = 0;
            while b < nb {
                let t = (nb - b).min(4);
                // SAFETY: AVX2 checked above; slice bounds asserted above.
                unsafe {
                    lb_dot_avx2(
                        bits,
                        n,
                        xs.as_ptr().add(b * stride),
                        stride,
                        t,
                        &mut out[b..b + t],
                    )
                };
                b += t;
            }
            return;
        }
    }
    for b in 0..nb {
        out[b] = lb_dot_scalar(bits, n, &xs[b * stride..]);
    }
}

/// The AVX2 twin: each bit-byte indexes `SIGN_LUT`, whose row is one 256-bit mask;
/// `xor` flips the signs of eight floats at once and `add` accumulates. With one
/// activation row, four interleaved accumulator chains hide the add latency (the
/// FLOPS-probe lesson: one chain measures latency, not throughput); with a batch,
/// the chains are the rows themselves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lb_dot_avx2(
    bits: &[u8],
    n: usize,
    xs: *const f32,
    stride: usize,
    nb: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    unsafe fn mask(b: u8) -> __m256i {
        _mm256_loadu_si256(SIGN_LUT[b as usize].as_ptr() as *const __m256i)
    }
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }
    let nb4 = n / 32;
    if nb == 1 {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        for c in 0..nb4 {
            let bp = bits.as_ptr().add(c * 4);
            let x = xs.add(c * 32);
            a0 = _mm256_add_ps(
                a0,
                _mm256_xor_ps(_mm256_loadu_ps(x), _mm256_castsi256_ps(mask(*bp))),
            );
            a1 = _mm256_add_ps(
                a1,
                _mm256_xor_ps(
                    _mm256_loadu_ps(x.add(8)),
                    _mm256_castsi256_ps(mask(*bp.add(1))),
                ),
            );
            a2 = _mm256_add_ps(
                a2,
                _mm256_xor_ps(
                    _mm256_loadu_ps(x.add(16)),
                    _mm256_castsi256_ps(mask(*bp.add(2))),
                ),
            );
            a3 = _mm256_add_ps(
                a3,
                _mm256_xor_ps(
                    _mm256_loadu_ps(x.add(24)),
                    _mm256_castsi256_ps(mask(*bp.add(3))),
                ),
            );
        }
        out[0] = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        return;
    }
    let mut acc = [_mm256_setzero_ps(); 4];
    for (c, &byte) in bits[..n / 8].iter().enumerate() {
        let m = _mm256_castsi256_ps(mask(byte));
        for (b, a) in acc.iter_mut().enumerate().take(nb) {
            let x = _mm256_loadu_ps(xs.add(b * stride + c * 8));
            *a = _mm256_add_ps(*a, _mm256_xor_ps(x, m));
        }
    }
    for (b, o) in out.iter_mut().enumerate().take(nb) {
        *o = hsum(acc[b]);
    }
}

// ---------------------------------------------------------- int8-activation path --
//
// The f32 kernel above is *compute-bound* on the reference machine (measured, plan
// §8): eight weights per vector op against the dense Q4_K path's thirty-two. With
// the activation row quantised to int8 once, a sign dot product is
// `Σ_i ±q_i = 2·Σ_{bit set} q_i − Σ_i q_i`: the second term is known at quantisation
// time, and the first is `pshufb` (spread four bytes of bits over 32 lanes), `and`,
// `min` (→ 0/1 bytes) and one `maddubs` per 32 weights.
//
// Scales are per block of 256 weights (a shorter last block when `n` is not a
// multiple of 256), as in the dense Q8_K activations, not per row: the activation rows are `x ⊙ g` and `ℓ ⊙ t`, which carry outlier channels, and
// a single row scale would spend its 127 levels on the outlier — the reason the dense
// path quantises activations per block too.
//
// A quantised row is packed into the same `n` floats the f32 row would occupy, so
// the model's scratch buffers serve both paths: `n` int8 values, then `n/block`
// f32 scales, then one f32 correction `Σ_blk scale·Σq` (the `−Σ±` term above).

/// Weights per int8 scale block.
pub const I8_BLOCK: usize = 256;

/// Scale blocks in a row of `n`.
pub fn i8_blocks(n: usize) -> usize {
    n.div_ceil(I8_BLOCK)
}

/// Floats a packed int8 row needs — always ≤ `n` for the engine's `n ≥ 32`.
pub fn i8_row_floats(n: usize) -> usize {
    n / 4 + i8_blocks(n) + 1
}

/// Quantise `n` values given by `f` into the packed layout at the start of `row`.
///
/// `f` is evaluated once per element into a 256-float stack block, which the AVX2
/// `absmax`/`quantize32` pair (bit-identical to the scalar round-half-away loop)
/// then reduces: this runs serially between the pool phases of every LittleBit
/// group, and `f32::round` per element was a libm call.
pub fn quantize_i8_row(row: &mut [f32], n: usize, f: impl Fn(usize) -> f32) {
    debug_assert!(n.is_multiple_of(32) && row.len() >= i8_row_floats(n) && i8_row_floats(n) <= n);
    let nblk = i8_blocks(n);
    let (qf, rest) = row.split_at_mut(n / 4);
    // SAFETY: `qf` is `n/4` aligned f32s = `n` bytes; i8 has no invalid bit patterns.
    let q = unsafe { std::slice::from_raw_parts_mut(qf.as_mut_ptr() as *mut i8, n) };
    #[cfg(target_arch = "x86_64")]
    let simd = crate::simd::have_avx2();
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    let mut buf = [0.0f32; I8_BLOCK];
    let mut corr = 0.0f32;
    for k in 0..nblk {
        let (b0, b1) = (k * I8_BLOCK, ((k + 1) * I8_BLOCK).min(n));
        let x = &mut buf[..b1 - b0];
        for (j, v) in x.iter_mut().enumerate() {
            *v = f(b0 + j);
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 checked above.
        let amax = if simd {
            unsafe { crate::simd::absmax(x) }
        } else {
            x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
        };
        #[cfg(not(target_arch = "x86_64"))]
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let qb = &mut q[b0..b1];
        if amax == 0.0 {
            qb.fill(0);
            rest[k] = 0.0;
            continue;
        }
        let (inv, sc) = (127.0 / amax, amax / 127.0);
        if simd {
            #[cfg(target_arch = "x86_64")]
            for (xc, qc) in x.chunks_exact(32).zip(qb.chunks_exact_mut(32)) {
                // SAFETY: 32 readable floats and 32 writable bytes per chunk (block
                // lengths are multiples of 32); AVX2 checked above.
                unsafe { crate::simd::quantize32(xc.as_ptr(), inv, qc.as_mut_ptr()) };
            }
        } else {
            for (d, &v) in qb.iter_mut().zip(x.iter()) {
                *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let sum: i32 = qb.iter().map(|&v| v as i32).sum();
        rest[k] = sc;
        corr += sc * sum as f32;
    }
    rest[nblk] = corr;
}

/// The specification: `Σ_blk scale·(2·Σ_{set} q − Σq)` with integer block sums,
/// computed from the packed row exactly as the vector kernel reads it.
pub fn lb_dot_i8_scalar(bits: &[u8], n: usize, row: &[f32]) -> f32 {
    let q = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const i8, n) };
    let scales = &row[n / 4..];
    let nblk = i8_blocks(n);
    let mut acc = 0.0f32;
    for k in 0..nblk {
        let mut set = 0i32;
        for i in k * I8_BLOCK..((k + 1) * I8_BLOCK).min(n) {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                set += q[i] as i32;
            }
        }
        acc += scales[k] * (2 * set) as f32;
    }
    acc - scales[nblk]
}

/// One sign row against `nb` packed activation rows (`rows[b*stride..]`), written to
/// `out[..nb]` — the int8 twin of [`lb_dot_batch`], same batching discipline.
pub fn lb_dot_i8_batch(
    bits: &[u8],
    n: usize,
    rows: &[f32],
    stride: usize,
    nb: usize,
    out: &mut [f32],
) {
    debug_assert!(n.is_multiple_of(32) && bits.len() >= n / 8 && out.len() >= nb);
    debug_assert!(rows.len() >= (nb - 1) * stride + i8_row_floats(n));
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            let mut b = 0;
            while b < nb {
                let t = (nb - b).min(4);
                let r = &rows[b * stride..];
                // SAFETY: AVX2 checked above; slice bounds asserted above.
                unsafe {
                    match t {
                        1 => lb_i8_avx2::<1>(bits, n, r, stride, &mut out[b..]),
                        2 => lb_i8_avx2::<2>(bits, n, r, stride, &mut out[b..]),
                        3 => lb_i8_avx2::<3>(bits, n, r, stride, &mut out[b..]),
                        _ => lb_i8_avx2::<4>(bits, n, r, stride, &mut out[b..]),
                    }
                }
                b += t;
            }
            return;
        }
    }
    for b in 0..nb {
        out[b] = lb_dot_i8_scalar(bits, n, &rows[b * stride..]);
    }
}

/// 0: not decided yet (the environment decides on first use), 1: f32, 2: int8.
static I8_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the model path uses the int8 kernels. `set_i8_enabled` (the app's
/// setting) wins; otherwise `SMILEY_CORE_LB_I8=0` keeps f32. Read once per fused
/// group, so a change takes effect at the next projection — both paths compute the
/// same product to within int8 rounding.
pub fn i8_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match I8_MODE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("SMILEY_CORE_LB_I8")
                .map(|v| v != "0")
                .unwrap_or(true);
            let _ = I8_MODE.compare_exchange(0, if on { 2 } else { 1 }, Relaxed, Relaxed);
            I8_MODE.load(Relaxed) == 2
        }
    }
}

/// Choose the LittleBit kernel for every model in this process.
pub fn set_i8_enabled(on: bool) {
    I8_MODE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lb_i8_avx2<const NB: usize>(
    bits: &[u8],
    n: usize,
    rows: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    // Lane group k (bytes 8k..8k+8 of each 128-bit half) reads input byte k of the
    // broadcast dword; the bit tested in lane i is 1 << (i % 8).
    let spread = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let one_hot = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let (nchunks, nblk) = (n / 32, i8_blocks(n));
    let mut facc = [_mm256_setzero_ps(); NB];
    let qp: [*const __m256i; NB] =
        std::array::from_fn(|b| rows.as_ptr().add(b * stride) as *const __m256i);
    let sp: [*const f32; NB] = std::array::from_fn(|b| rows.as_ptr().add(b * stride + n / 4));
    for k in 0..nblk {
        let mut acc = [_mm256_setzero_si256(); NB];
        for c in k * (I8_BLOCK / 32)..((k + 1) * (I8_BLOCK / 32)).min(nchunks) {
            let w = _mm256_set1_epi32(std::ptr::read_unaligned(
                bits.as_ptr().add(c * 4) as *const i32
            ));
            let b01 = _mm256_min_epu8(
                _mm256_and_si256(_mm256_shuffle_epi8(w, spread), one_hot),
                ones8,
            );
            for b in 0..NB {
                let x = _mm256_loadu_si256(qp[b].add(c));
                // ≤ 2·127 per i16 lane per chunk, eight chunks per block: no overflow.
                acc[b] = _mm256_add_epi16(acc[b], _mm256_maddubs_epi16(b01, x));
            }
        }
        for b in 0..NB {
            let v = _mm256_cvtepi32_ps(_mm256_madd_epi16(acc[b], ones16));
            facc[b] = _mm256_add_ps(
                facc[b],
                _mm256_mul_ps(v, _mm256_set1_ps(2.0 * *sp[b].add(k))),
            );
        }
    }
    for b in 0..NB {
        let lo = _mm256_castps256_ps128(facc[b]);
        let hi = _mm256_extractf128_ps(facc[b], 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[b] = _mm_cvtss_f32(s) - *sp[b].add(nblk);
    }
}

// ------------------------------------------------------------- the container --

/// One stage of one tensor, as loaded: byte ranges of the two bit matrices in the
/// map, and the scale vectors decoded from f16 once at load — they are kilobytes,
/// read on every call, and the kernel should never convert.
pub struct SlbStage {
    u_off: usize,
    v_off: usize,
    pub h: Vec<f32>,
    pub l: Vec<f32>,
    pub g: Vec<f32>,
}

pub struct SlbTensor {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// True latent rank, and the stored rank padded to a multiple of 32. Padded
    /// latents are inert by construction: their `ℓ` is zero, so stage 1's extra
    /// sums are multiplied away and stage 2's extra bits multiply zeros.
    pub r: usize,
    pub r_pad: usize,
    pub stages: Vec<SlbStage>,
    /// Bytes one forward call moves for this tensor (bits + decoded f32 scales),
    /// independent of batch size — the batched kernel reads each row once.
    pub bytes_per_use: u64,
}

impl SlbTensor {
    /// Multiply-accumulate-equivalents one activation row costs: the additions of
    /// both binary products, per stage.
    pub fn macs_per_row(&self) -> u64 {
        (self.stages.len() * self.r_pad * (self.d_in + self.d_out)) as u64
    }
    /// Stored bits per source weight, counting everything (bits + f16 scales) —
    /// the honest `bpw` the spec's accounting wants.
    pub fn bpw(&self) -> f64 {
        let s = self.stages.len() as f64;
        let bits = s * (self.r_pad * (self.d_out + self.d_in)) as f64
            + 16.0 * s * (self.d_out + self.d_in + self.r_pad) as f64;
        bits / (self.d_out as f64 * self.d_in as f64)
    }
}

pub struct Slb {
    map: Mmap,
    pub tensors: Vec<SlbTensor>,
    pub by_name: HashMap<String, usize>,
    /// Total bytes of bit and scale sections (what the file stores, less headers).
    pub data_bytes: u64,
}

const MAGIC: &[u8; 4] = b"SLB1";

/// Where a model's sidecar would live: `<model>.slb` beside it, `SMILEY_CORE_LB`
/// overriding (a path) or disabling (`0`). `keep::open` keys its cache on this
/// file's identity too, so converting a model invalidates the kept dense load.
pub fn sidecar_path(gguf_path: &Path) -> Option<PathBuf> {
    match std::env::var("SMILEY_CORE_LB").ok().as_deref() {
        Some("0") => None,
        Some(p) if !p.is_empty() && p != "1" => Some(PathBuf::from(p)),
        _ => Some(default_sidecar_path(gguf_path)),
    }
}

/// `<model>.slb` beside the model — where the converter writes by default, and
/// what the app shows, converts into and removes, whatever the environment says.
pub fn default_sidecar_path(gguf_path: &Path) -> PathBuf {
    let mut os = gguf_path.as_os_str().to_owned();
    os.push(".slb");
    PathBuf::from(os)
}

/// Bounds-checked little-endian cursor, as `gguf.rs` uses: a truncated or hostile
/// sidecar must produce an error, never a panic.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("slb: length overflow")?;
        let s = self
            .b
            .get(self.p..end)
            .ok_or_else(|| format!("slb: truncated at {} (+{n})", self.p))?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

impl Slb {
    /// Open and fully verify a sidecar. Every section's CRC32C is checked here —
    /// the file is tens of MB and the table runs at ~1 GB/s, so this is cheap, and
    /// it converts "silently wrong model" into "refused at load".
    pub fn open(path: &Path) -> Result<Slb, String> {
        let (slb, _, _) = Self::open_raw(path)?;
        Ok(slb)
    }

    fn open_raw(path: &Path) -> Result<(Slb, u64, u64), String> {
        let file =
            std::fs::File::open(path).map_err(|e| format!("slb: cannot open {path:?}: {e}"))?;
        // SAFETY: read-only map; all accesses below are bounds-checked. The residual
        // truncation hazard is the same one gguf.rs documents and accepts.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("slb: mmap failed: {e}"))?;
        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != MAGIC {
            return Err("slb: bad magic — not an SLB sidecar".into());
        }
        let version = c.u32()?;
        if version != 1 {
            return Err(format!("slb: unsupported version {version}"));
        }
        let n_tensors = c.u32()? as usize;
        if n_tensors == 0 || n_tensors > 1 << 20 {
            return Err(format!("slb: implausible tensor count {n_tensors}"));
        }
        let _max_stages = c.u32()?;
        let src_size = c.u64()?;
        let src_hash = c.u64()?;
        let dir_bytes = c.u64()? as usize;
        let dir_start = c.p;
        let dir = c.take(dir_bytes)?;
        let want_crc = c.u32()?;
        if crc32c(dir) != want_crc {
            return Err("slb: directory checksum mismatch — corrupt sidecar".into());
        }

        let mut d = Cur {
            b: &map[dir_start..dir_start + dir_bytes],
            p: 0,
        };
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut by_name = HashMap::with_capacity(n_tensors);
        let mut data_bytes = 0u64;
        for _ in 0..n_tensors {
            let name_len = d.u32()? as usize;
            if name_len > 4096 {
                return Err("slb: implausible tensor name length".into());
            }
            let name = String::from_utf8_lossy(d.take(name_len)?).into_owned();
            let d_out = d.u32()? as usize;
            let d_in = d.u32()? as usize;
            let r = d.u32()? as usize;
            let r_pad = d.u32()? as usize;
            let n_stages = d.u32()? as usize;
            if d_out == 0
                || d_in == 0
                || r == 0
                || !r_pad.is_multiple_of(32)
                || r > r_pad
                || r_pad > d_in.min(d_out)
                || !d_in.is_multiple_of(32)
                || !d_out.is_multiple_of(32)
                || n_stages == 0
                || n_stages > 8
            {
                return Err(format!("slb: implausible geometry for {name}: {d_out}×{d_in} r {r}/{r_pad} stages {n_stages}"));
            }
            let (ub, vb) = (d_out * r_pad / 8, r_pad * d_in / 8);
            let mut stages = Vec::with_capacity(n_stages);
            for _ in 0..n_stages {
                let mut sect = |len: usize| -> Result<(usize, &[u8]), String> {
                    let off = d.u64()? as usize;
                    let crc = d.u32()?;
                    let end = off.checked_add(len).ok_or("slb: section overflow")?;
                    let bytes = map.get(off..end).ok_or_else(|| {
                        format!(
                            "slb: section of {name} runs to {end} but the file is {} bytes",
                            map.len()
                        )
                    })?;
                    if crc32c(bytes) != crc {
                        return Err(format!(
                            "slb: checksum mismatch in a section of {name} — corrupt sidecar"
                        ));
                    }
                    Ok((off, bytes))
                };
                let (u_off, _) = sect(ub)?;
                let (v_off, _) = sect(vb)?;
                let (_, hb) = sect(d_out * 2)?;
                let (_, lb) = sect(r_pad * 2)?;
                let (_, gb) = sect(d_in * 2)?;
                let dec = |b: &[u8]| -> Vec<f32> {
                    b.chunks_exact(2)
                        .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
                        .collect()
                };
                stages.push(SlbStage {
                    u_off,
                    v_off,
                    h: dec(hb),
                    l: dec(lb),
                    g: dec(gb),
                });
                data_bytes += (ub + vb + 2 * (d_out + r_pad + d_in)) as u64;
            }
            let bytes_per_use = stages.len() as u64 * (ub + vb) as u64
                + stages.len() as u64 * (4 * (d_out + r_pad + d_in)) as u64;
            let t = SlbTensor {
                name: name.clone(),
                d_out,
                d_in,
                r,
                r_pad,
                stages,
                bytes_per_use,
            };
            by_name.insert(name, tensors.len());
            tensors.push(t);
        }
        Ok((
            Slb {
                map,
                tensors,
                by_name,
                data_bytes,
            },
            src_size,
            src_hash,
        ))
    }

    /// The sidecar for a GGUF at `gguf_path`, when one exists and matches it.
    ///
    /// `SMILEY_CORE_LB=0` disables the pickup; any other value names an explicit
    /// sidecar path. A sidecar that exists but is corrupt or stale is an *error*,
    /// not a silent fall-through: running dense when the user built a sidecar is
    /// the kind of quiet divergence this engine refuses on principle.
    pub fn open_for(gguf_path: &Path) -> Result<Option<Slb>, String> {
        let Some(side) = sidecar_path(gguf_path) else {
            return Ok(None);
        };
        if !side.exists() {
            return Ok(None);
        }
        let (slb, src_size, src_hash) = Self::open_raw(&side)?;
        let (size, hash) = source_stamp(gguf_path)?;
        if (src_size, src_hash) != (size, hash) {
            return Err(format!(
                "slb: {side:?} was built from a different model file (stamp {src_size}/{src_hash:016x}, \
                 model is {size}/{hash:016x}) — reconvert or set SMILEY_CORE_LB=0"
            ));
        }
        Ok(Some(slb))
    }

    /// The whole U bit matrix of one stage: `d_out` rows of `r_pad/8` bytes.
    pub fn u_bits(&self, t: &SlbTensor, stage: usize) -> &[u8] {
        let s = &t.stages[stage];
        &self.map[s.u_off..s.u_off + t.d_out * t.r_pad / 8]
    }

    /// The whole Vᵀ bit matrix of one stage: `r_pad` rows of `d_in/8` bytes —
    /// latent-major, so stage 1 streams rows against an L1-resident activation.
    pub fn v_bits(&self, t: &SlbTensor, stage: usize) -> &[u8] {
        let s = &t.stages[stage];
        &self.map[s.v_off..s.v_off + t.r_pad * t.d_in / 8]
    }

    /// Reference forward of one tensor on one activation row, scalar, allocating.
    /// This is the specification `model.rs`'s pooled path is tested against.
    pub fn apply(&self, ti: usize, x: &[f32], out: &mut [f32]) {
        let t = &self.tensors[ti];
        assert!(x.len() >= t.d_in && out.len() >= t.d_out);
        out[..t.d_out].fill(0.0);
        let mut xp = vec![0.0f32; t.d_in];
        let mut lat = vec![0.0f32; t.r_pad];
        for (si, st) in t.stages.iter().enumerate() {
            for j in 0..t.d_in {
                xp[j] = x[j] * st.g[j];
            }
            let v = self.v_bits(t, si);
            let vrb = t.d_in / 8;
            for (j, l) in lat.iter_mut().enumerate() {
                *l = st.l[j] * lb_dot_scalar(&v[j * vrb..(j + 1) * vrb], t.d_in, &xp);
            }
            let u = self.u_bits(t, si);
            let urb = t.r_pad / 8;
            for (i, o) in out[..t.d_out].iter_mut().enumerate() {
                *o += st.h[i] * lb_dot_scalar(&u[i * urb..(i + 1) * urb], t.r_pad, &lat);
            }
        }
    }
}

// --------------------------------------------------------------------- writer --

/// One stage as built: packed bits and f32 scales (written as f16).
pub struct LbStageBuild {
    pub u_bits: Vec<u8>,
    pub v_bits: Vec<u8>,
    pub h: Vec<f32>,
    pub l: Vec<f32>,
    pub g: Vec<f32>,
}

pub struct LbBuild {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub r: usize,
    pub stages: Vec<LbStageBuild>,
    /// `‖W − Ŵ‖_F / ‖W‖_F` after all stages — reported, never hidden.
    pub rel_err: f64,
}

/// Write a sidecar atomically: `*.tmp`, fsync, rename. A crashed conversion can
/// never leave a half-file that loads (the RustFS idea, minus the daemon).
pub fn write_slb(path: &Path, src_size: u64, src_hash: u64, ts: &[LbBuild]) -> Result<(), String> {
    if ts.is_empty() {
        return Err("slb: nothing to write — no tensor was eligible".into());
    }
    let align = |p: usize| p.div_ceil(64) * 64;
    // Directory first, with offsets computed against the data region that follows it.
    let mut dir_len = 0usize;
    for t in ts {
        dir_len += 4 + t.name.len() + 5 * 4 + t.stages.len() * 5 * 12;
    }
    let data_start = align(4 + 4 + 4 + 4 + 8 + 8 + 8 + dir_len + 4);
    let mut dir = Vec::with_capacity(dir_len);
    let mut cursor = data_start;
    let mut sections: Vec<(usize, &[u8])> = Vec::new();
    let mut f16s: Vec<Vec<u8>> = Vec::new();
    // Two passes over the scales: materialise the f16 bytes first (the directory
    // stores their CRCs), then borrow them for the section list.
    for t in ts {
        for s in &t.stages {
            for v in [&s.h, &s.l, &s.g] {
                f16s.push(v.iter().flat_map(|&x| to_f16(x).to_le_bytes()).collect());
            }
        }
    }
    let mut fi = 0usize;
    for t in ts {
        let r_pad = t.stages[0].l.len();
        dir.extend_from_slice(&(t.name.len() as u32).to_le_bytes());
        dir.extend_from_slice(t.name.as_bytes());
        for v in [
            t.d_out as u32,
            t.d_in as u32,
            t.r as u32,
            r_pad as u32,
            t.stages.len() as u32,
        ] {
            dir.extend_from_slice(&v.to_le_bytes());
        }
        for s in &t.stages {
            debug_assert_eq!(s.u_bits.len(), t.d_out * r_pad / 8);
            debug_assert_eq!(s.v_bits.len(), r_pad * t.d_in / 8);
            for bytes in [
                &s.u_bits[..],
                &s.v_bits[..],
                &f16s[fi],
                &f16s[fi + 1],
                &f16s[fi + 2],
            ] {
                dir.extend_from_slice(&(cursor as u64).to_le_bytes());
                dir.extend_from_slice(&crc32c(bytes).to_le_bytes());
                sections.push((cursor, bytes));
                cursor = align(cursor + bytes.len());
            }
            fi += 3;
        }
    }
    debug_assert_eq!(dir.len(), dir_len);

    let mut out = vec![0u8; cursor];
    out[0..4].copy_from_slice(MAGIC);
    out[4..8].copy_from_slice(&1u32.to_le_bytes());
    out[8..12].copy_from_slice(&(ts.len() as u32).to_le_bytes());
    let max_stages = ts.iter().map(|t| t.stages.len()).max().unwrap_or(0) as u32;
    out[12..16].copy_from_slice(&max_stages.to_le_bytes());
    out[16..24].copy_from_slice(&src_size.to_le_bytes());
    out[24..32].copy_from_slice(&src_hash.to_le_bytes());
    out[32..40].copy_from_slice(&(dir_len as u64).to_le_bytes());
    out[40..40 + dir_len].copy_from_slice(&dir);
    let dc = crc32c(&dir);
    out[40 + dir_len..44 + dir_len].copy_from_slice(&dc.to_le_bytes());
    for (off, bytes) in sections {
        out[off..off + bytes.len()].copy_from_slice(bytes);
    }

    let tmp = path.with_extension("slb.tmp");
    {
        use std::io::Write;
        let mut f =
            std::fs::File::create(&tmp).map_err(|e| format!("slb: cannot create {tmp:?}: {e}"))?;
        f.write_all(&out)
            .map_err(|e| format!("slb: write failed: {e}"))?;
        f.sync_all()
            .map_err(|e| format!("slb: fsync failed: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("slb: rename failed: {e}"))?;
    Ok(())
}

// ------------------------------------------------------------------ the maths --

/// The rank that spends a `bpw` budget on `stages` stages of a `d_out×d_in` layer,
/// counting the f16 scale vectors (plan §1), floored to a whole 32-block. The
/// paper's own examples land within one block: 4096×4096 at 0.55 BPW → 544 here
/// against the paper's 546 (its two scale vectors are counted jointly at 32 bits).
pub fn rank_for(bpw: f64, stages: usize, d_out: usize, d_in: usize) -> usize {
    let s = stages as f64;
    let dd = (d_out + d_in) as f64;
    let num = bpw * d_out as f64 * d_in as f64 / s - 16.0 * dd;
    let den = dd + 16.0;
    let r = (num / den).max(0.0) as usize;
    r / 32 * 32
}

/// xorshift64* — deterministic sketches, seeded per tensor name.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Rademacher ±1 — all a range sketch needs.
    fn pm1(&mut self) -> f32 {
        if self.next() & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: every user writes only indices or ranges no other worker touches,
// keyed by the worker's share — the same contract model.rs documents.
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Ptr {
    /// Methods rather than field access on purpose: with edition 2021's disjoint
    /// closure capture, `p.0` inside a worker closure captures the raw `*mut f32`
    /// instead of the wrapper, and the closure then fails `Sync` (model.rs hit the
    /// same with `OutPtr`).
    ///
    /// # Safety
    /// `off..off+len` must be in bounds and used by exactly one worker at a time.
    #[inline(always)]
    unsafe fn slice<'a>(self, off: usize, len: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(off), len)
    }
    /// # Safety
    /// `i` must be in bounds and owned by the calling worker.
    #[inline(always)]
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// `Σ a_i b_i`, f64 accumulation (the builder's precision is cheap; its errors
/// become the model's).
fn dotf(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: features checked; equal lengths asserted by the caller's zip use.
            return unsafe { dotf_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotf_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0;
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            s0,
        );
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
            s3,
        );
        i += 32;
    }
    let s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), s);
    let mut acc: f64 = buf.iter().map(|&v| v as f64).sum();
    while i < n {
        acc += a[i] as f64 * b[i] as f64;
        i += 1;
    }
    acc
}

/// `y += s·x`.
fn axpy(s: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: features checked.
            unsafe { axpy_avx2(s, x, y) };
            return;
        }
    }
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += s * xi;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_avx2(s: f32, x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len().min(y.len());
    let sv = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(
            y.as_mut_ptr().add(i),
            _mm256_fmadd_ps(sv, _mm256_loadu_ps(x.as_ptr().add(i)), yv),
        );
        i += 8;
    }
    while i < n {
        y[i] += s * x[i];
        i += 1;
    }
}

/// `c (m×n) = a (m×p) · b (p×n)`, all row-major, pool-parallel over `m`.
///
/// Blocked so the `b` panel (`PB×NB` ≈ 512 KB) stays in the shared L3 across every
/// row block — without this, `b` would be streamed from DRAM once per four rows and
/// the converter would spend its life waiting on memory. The AVX2 microkernel does
/// four rows × eight columns with FMA; everything ragged falls to the scalar edge.
/// Offline-path code: the inference kernels above never call this.
pub fn gemm(m: usize, p: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], pool: &Pool) {
    assert!(a.len() >= m * p && b.len() >= p * n && c.len() >= m * n);
    c[..m * n].fill(0.0);
    const NB: usize = 512;
    const PB: usize = 256;
    let (ap, bp, cp) = (
        a.as_ptr() as usize,
        b.as_ptr() as usize,
        Ptr(c.as_mut_ptr()),
    );
    #[cfg(target_arch = "x86_64")]
    let fast =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let fast = false;
    pool.run(|wk, nw| {
        let (lo, hi) = share(m, wk, nw);
        let a = unsafe { std::slice::from_raw_parts(ap as *const f32, m * p) };
        let b = unsafe { std::slice::from_raw_parts(bp as *const f32, p * n) };
        for j0 in (0..n).step_by(NB) {
            let j1 = (j0 + NB).min(n);
            for p0 in (0..p).step_by(PB) {
                let p1 = (p0 + PB).min(p);
                let mut i = lo;
                while i < hi {
                    let rows = (hi - i).min(4);
                    // SAFETY: rows `i..i+rows` of `c` belong to this worker's share.
                    unsafe {
                        if fast && rows == 4 {
                            #[cfg(target_arch = "x86_64")]
                            gemm_micro4(a, b, cp, i, p0, p1, j0, j1, p, n);
                            #[cfg(not(target_arch = "x86_64"))]
                            gemm_edge(a, b, cp, i, rows, p0, p1, j0, j1, p, n);
                        } else {
                            gemm_edge(a, b, cp, i, rows, p0, p1, j0, j1, p, n);
                        }
                    }
                    i += rows;
                }
            }
        }
    });
}

/// # Safety
/// Rows `i0..i0+rows` of `c` must belong to the calling worker.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_edge(
    a: &[f32],
    b: &[f32],
    c: Ptr,
    i0: usize,
    rows: usize,
    p0: usize,
    p1: usize,
    j0: usize,
    j1: usize,
    p: usize,
    n: usize,
) {
    for r in 0..rows {
        let crow = std::slice::from_raw_parts_mut(c.0.add((i0 + r) * n + j0), j1 - j0);
        for pp in p0..p1 {
            let av = a[(i0 + r) * p + pp];
            if av != 0.0 {
                axpy(av, &b[pp * n + j0..pp * n + j1], crow);
            }
        }
    }
}

/// # Safety
/// AVX2+FMA must be available; rows `i0..i0+4` of `c` must belong to the caller.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_micro4(
    a: &[f32],
    b: &[f32],
    c: Ptr,
    i0: usize,
    p0: usize,
    p1: usize,
    j0: usize,
    j1: usize,
    p: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let mut j = j0;
    while j + 8 <= j1 {
        let mut acc = [_mm256_setzero_ps(); 4];
        for pp in p0..p1 {
            let bv = _mm256_loadu_ps(b.as_ptr().add(pp * n + j));
            for (r, ac) in acc.iter_mut().enumerate() {
                let av = _mm256_broadcast_ss(&a[(i0 + r) * p + pp]);
                *ac = _mm256_fmadd_ps(av, bv, *ac);
            }
        }
        for (r, ac) in acc.iter().enumerate() {
            let cp = c.0.add((i0 + r) * n + j);
            _mm256_storeu_ps(cp, _mm256_add_ps(_mm256_loadu_ps(cp), *ac));
        }
        j += 8;
    }
    if j < j1 {
        gemm_edge(a, b, c, i0, 4, p0, p1, j, j1, p, n);
    }
}

/// Blocked transpose: `b (n×m) = aᵀ` for `a (m×n)`, both row-major.
pub fn transpose(m: usize, n: usize, a: &[f32]) -> Vec<f32> {
    let mut b = vec![0.0f32; m * n];
    const T: usize = 32;
    for i0 in (0..m).step_by(T) {
        for j0 in (0..n).step_by(T) {
            for i in i0..(i0 + T).min(m) {
                for j in j0..(j0 + T).min(n) {
                    b[j * m + i] = a[i * n + j];
                }
            }
        }
    }
    b
}

/// Modified Gram–Schmidt on the rows of `a (k×d)`, in place, pool-parallel over
/// the trailing rows, with selective re-orthogonalization: when the right-looking
/// pass has cancelled most of a row's mass, what remains is rounding noise that is
/// *not* orthogonal to the earlier basis (measured: a near-dependent row kept a
/// 0.79 overlap), so such a row is projected once more against everything before
/// it. A row that is numerically dependent is zeroed rather than normalised — the
/// sketch oversamples, so a lost direction costs slack, not correctness.
fn mgs_rows(k: usize, d: usize, a: &mut [f32], pool: &Pool) {
    let norm0: Vec<f64> = (0..k)
        .map(|i| dotf(&a[i * d..(i + 1) * d], &a[i * d..(i + 1) * d]).sqrt())
        .collect();
    #[allow(clippy::needless_range_loop)] // `i` also names the split point and the row
    for i in 0..k {
        let (head, tail) = a.split_at_mut((i + 1) * d);
        let (prev, qi) = head.split_at_mut(i * d);
        let mut norm = dotf(qi, qi).sqrt();
        if norm < 0.5 * norm0[i] && i > 0 {
            for j in 0..i {
                let qj = &prev[j * d..(j + 1) * d];
                let r = dotf(qj, qi) as f32;
                if r != 0.0 {
                    axpy(-r, qj, qi);
                }
            }
            norm = dotf(qi, qi).sqrt();
        }
        if norm <= 1e-7 * norm0[i].max(1e-30) || norm < 1e-30 {
            qi.fill(0.0);
            continue;
        }
        let inv = (1.0 / norm) as f32;
        for v in qi.iter_mut() {
            *v *= inv;
        }
        let qi: &[f32] = qi;
        let rem = k - i - 1;
        if rem == 0 {
            continue;
        }
        let tp = Ptr(tail.as_mut_ptr());
        pool.run(|wk, nw| {
            let (lo, hi) = share(rem, wk, nw);
            for jj in lo..hi {
                // SAFETY: row `jj` of the tail belongs to this worker's share alone.
                let row = unsafe { tp.slice(jj * d, d) };
                let r = dotf(qi, row) as f32;
                if r != 0.0 {
                    axpy(-r, qi, row);
                }
            }
        });
    }
}

/// Householder reduction of a symmetric matrix to tridiagonal form, with the
/// orthogonal transform accumulated in `z` — the classic EISPACK `tred2`,
/// transcribed in f64 because the builder can afford exactness.
fn tred2(nk: usize, z: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    for i in (1..nk).rev() {
        let l = i - 1;
        let mut h = 0.0f64;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| z[i * nk + k].abs()).sum();
            if scale == 0.0 {
                e[i] = z[i * nk + l];
            } else {
                for k in 0..=l {
                    z[i * nk + k] /= scale;
                    h += z[i * nk + k] * z[i * nk + k];
                }
                let mut f = z[i * nk + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z[i * nk + l] = f - g;
                let mut tau = 0.0f64;
                for j in 0..=l {
                    z[j * nk + i] = z[i * nk + j] / h;
                    let mut g = 0.0f64;
                    for k in 0..=j {
                        g += z[j * nk + k] * z[i * nk + k];
                    }
                    for k in j + 1..=l {
                        g += z[k * nk + j] * z[i * nk + k];
                    }
                    e[j] = g / h;
                    tau += e[j] * z[i * nk + j];
                }
                let hh = tau / (h + h);
                for j in 0..=l {
                    f = z[i * nk + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        z[j * nk + k] -= f * e[k] + g * z[i * nk + k];
                    }
                }
            }
        } else {
            e[i] = z[i * nk + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for i in 0..nk {
        let l = i;
        if d[i] != 0.0 {
            for j in 0..l {
                let mut g = 0.0f64;
                for k in 0..l {
                    g += z[i * nk + k] * z[k * nk + j];
                }
                for k in 0..l {
                    z[k * nk + j] -= g * z[k * nk + i];
                }
            }
        }
        d[i] = z[i * nk + i];
        z[i * nk + i] = 1.0;
        for j in 0..i {
            z[j * nk + i] = 0.0;
            z[i * nk + j] = 0.0;
        }
    }
}

/// QL with implicit shifts on the tridiagonal form, eigenvectors accumulated —
/// EISPACK `tql2`. Returns eigenvalues in `d` (ascending not guaranteed; the
/// caller sorts) with eigenvector `j` in column `j` of `z`.
fn tql2(nk: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64]) -> Result<(), String> {
    for i in 1..nk {
        e[i - 1] = e[i];
    }
    e[nk - 1] = 0.0;
    for l in 0..nk {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < nk {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 50 {
                return Err("slb: eigensolver failed to converge".into());
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c) = (1.0f64, 1.0f64);
            let mut p = 0.0f64;
            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..nk {
                    f = z[k * nk + i + 1];
                    z[k * nk + i + 1] = s * z[k * nk + i] + c * f;
                    z[k * nk + i] = c * z[k * nk + i] - s * f;
                }
            }
            if r == 0.0 && m > l {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}

/// Best nonnegative rank-1 fit `M ≈ col·rowᵀ` of a nonnegative `d×r` matrix, by
/// alternating least squares — the `|U′| ≈ h₀ ℓᵀ` step of Dual-SVID.
fn rank1_als(m: &[f32], d: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let mut col = vec![0.0f32; d];
    let mut row = vec![1.0f32; r];
    for i in 0..d {
        let s: f64 = m[i * r..(i + 1) * r].iter().map(|&v| v as f64).sum();
        col[i] = (s / r as f64) as f32;
    }
    for _ in 0..8 {
        let cs: f64 = col.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if cs > 0.0 {
            // Row-major sweep (each `num[k]` still sums over `i` in order, so the
            // result is unchanged); column-strided over a multi-MB `m` it was a
            // cache miss per element.
            let mut num = vec![0.0f64; r];
            for i in 0..d {
                let c = col[i] as f64;
                for (nk, &v) in num.iter_mut().zip(&m[i * r..(i + 1) * r]) {
                    *nk += v as f64 * c;
                }
            }
            for (rk, nk) in row.iter_mut().zip(&num) {
                *rk = (nk / cs).max(0.0) as f32;
            }
        }
        let rs: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if rs > 0.0 {
            for (i, c) in col.iter_mut().enumerate() {
                let num: f64 = m[i * r..(i + 1) * r]
                    .iter()
                    .zip(&row)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                *c = (num / rs).max(0.0) as f32;
            }
        }
    }
    (col, row)
}

/// Truncated SVD factors `W ≈ U′V′ᵀ` (singular values split `√Σ` to each side) by
/// randomised subspace iteration: Rademacher sketch, two power passes with MGS
/// re-orthonormalisation, then an exact eigensolve of the small Gram matrix.
#[allow(clippy::too_many_arguments)]
fn svid_factor(
    w: &[f32],
    wt: &[f32],
    d_out: usize,
    d_in: usize,
    r: usize,
    pool: &Pool,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<f32>) {
    let k = (r + 16).min(d_out.min(d_in));
    // Sketch, transposed: Ωᵀ (k×d_in); Y₀ᵀ = (W·Ω)ᵀ = Ωᵀ·Wᵀ (k×d_out).
    let mut sk = vec![0.0f32; k * d_in];
    for v in sk.iter_mut() {
        *v = rng.pm1();
    }
    let mut qt = vec![0.0f32; k * d_out];
    gemm(k, d_in, d_out, &sk, wt, &mut qt, pool);
    mgs_rows(k, d_out, &mut qt, pool);
    for _ in 0..2 {
        let mut zt = vec![0.0f32; k * d_in];
        gemm(k, d_out, d_in, &qt, w, &mut zt, pool); // Zᵀ = QᵀW
        mgs_rows(k, d_in, &mut zt, pool);
        gemm(k, d_in, d_out, &zt, wt, &mut qt, pool); // Yᵀ = ZᵀWᵀ
        mgs_rows(k, d_out, &mut qt, pool);
    }
    // B = QᵀW (k×d_in); eigensolve of B·Bᵀ gives the singular structure.
    let mut b = vec![0.0f32; k * d_in];
    gemm(k, d_out, d_in, &qt, w, &mut b, pool);
    let bt = transpose(k, d_in, &b);
    let mut gram = vec![0.0f32; k * k];
    gemm(k, d_in, k, &b, &bt, &mut gram, pool);
    let mut z: Vec<f64> = gram.iter().map(|&v| v as f64).collect();
    // Symmetrise: the gemm's two triangles differ by rounding, and tred2 assumes
    // exact symmetry.
    for i in 0..k {
        for j in 0..i {
            let m = 0.5 * (z[i * k + j] + z[j * k + i]);
            z[i * k + j] = m;
            z[j * k + i] = m;
        }
    }
    let mut d = vec![0.0f64; k];
    let mut e = vec![0.0f64; k];
    tred2(k, &mut z, &mut d, &mut e);
    if tql2(k, &mut d, &mut e, &mut z).is_err() {
        // Fall back to the unrotated basis: U′ = Q·√diag, degraded but defined.
        // In practice tql2 converges; this guards a pathological tensor.
        for v in z.iter_mut() {
            *v = 0.0;
        }
        for i in 0..k {
            z[i * k + i] = 1.0;
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &bb| {
        d[bb]
            .partial_cmp(&d[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // E_r columns scaled two ways: by √σ (for U′ = Q·E·√Σ... see below) and by
    // 1/σ·√σ = 1/√σ (for V′ = Bᵀ·E·Σ^{-1}·√Σ = Bᵀ·E·Σ^{-1/2}).
    let mut e_u = vec![0.0f32; k * r]; // E_r · Σ^{+1/2}... applied after the basis product
    let mut e_v = vec![0.0f32; k * r];
    for (c, &oi) in order.iter().take(r).enumerate() {
        let sigma = d[oi].max(0.0).sqrt(); // eigenvalue of BBᵀ = σ²
        let su = sigma.sqrt();
        let sv = if sigma > 1e-20 { su / sigma } else { 0.0 }; // σ^{-1/2}
        for row in 0..k {
            e_u[row * r + c] = (z[row * k + oi] * su) as f32;
            e_v[row * r + c] = (z[row * k + oi] * sv) as f32;
        }
    }
    // U′ (d_out×r) = Qᵀᵀ·E_u; V′ (d_in×r) = Bᵀ·E_v.
    let q = transpose(k, d_out, &qt);
    let mut u = vec![0.0f32; d_out * r];
    gemm(d_out, k, r, &q, &e_u, &mut u, pool);
    let mut v = vec![0.0f32; d_in * r];
    gemm(d_in, k, r, &bt, &e_v, &mut v, pool);
    (u, v)
}

/// Pack a `d×r` sign matrix row-major into `d` rows of `r_pad` bits (LSB-first,
/// bit = 1 ⇔ value ≥ 0; padding bits set, inert through ℓ's zero padding).
fn pack_rows(m: &[f32], d: usize, r: usize, r_pad: usize) -> Vec<u8> {
    let rb = r_pad / 8;
    let mut out = vec![0u8; d * rb];
    for i in 0..d {
        for k in 0..r_pad {
            let bit = k >= r || m[i * r + k] >= 0.0;
            if bit {
                out[i * rb + k / 8] |= 1 << (k % 8);
            }
        }
    }
    out
}

/// Pack `V′ (d_in×r)` latent-major: `r_pad` rows of `d_in` bits each.
fn pack_latent(v: &[f32], d_in: usize, r: usize, r_pad: usize) -> Vec<u8> {
    let rb = d_in / 8;
    let mut out = vec![0u8; r_pad * rb];
    for k in 0..r_pad {
        for j in 0..d_in {
            let bit = k >= r || v[j * r + k] >= 0.0;
            if bit {
                out[k * rb + j / 8] |= 1 << (j % 8);
            }
        }
    }
    out
}

/// One Dual-SVID stage of `w`, plus `refine` alternating passes on `h` and `g`
/// (each coordinate update is the exact least-squares solution given the rest, so
/// `‖W−Ŵ‖_F` never increases). Returns the stage and `Ŵ` for the residual cascade.
#[allow(clippy::too_many_arguments)]
fn build_stage(
    w: &[f32],
    wt: &[f32],
    d_out: usize,
    d_in: usize,
    r: usize,
    r_pad: usize,
    refine: usize,
    pool: &Pool,
    rng: &mut Rng,
) -> (LbStageBuild, Vec<f32>) {
    let (u, v) = svid_factor(w, wt, d_out, d_in, r, pool, rng);
    let uabs: Vec<f32> = u.iter().map(|x| x.abs()).collect();
    let vabs: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    let (mut h, lu) = rank1_als(&uabs, d_out, r);
    let (mut g, lv) = rank1_als(&vabs, d_in, r);
    let mut l: Vec<f32> = lu.iter().zip(&lv).map(|(a, b)| a * b).collect();
    // Normalise ℓ's scale into h: keeps every vector in a sane f16 range.
    let lmax = l.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if lmax > 0.0 {
        for x in l.iter_mut() {
            *x /= lmax;
        }
        for x in h.iter_mut() {
            *x *= lmax;
        }
    }
    let us: Vec<f32> = u
        .iter()
        .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    // vst (r×d_in): vst[k][j] = sign(v[j][k]).
    let mut vst = vec![0.0f32; r * d_in];
    for j in 0..d_in {
        for k in 0..r {
            vst[k * d_in + j] = if v[j * r + k] >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    // A = U_s · diag(ℓ) · V_sᵀ, materialised once (its signs and ℓ are fixed; the
    // refinement only moves h and g, which scale A's rows and columns).
    let mut usl = vec![0.0f32; d_out * r];
    let mut a = vec![0.0f32; d_out * d_in];
    for pass in 0..=refine {
        if pass == 0 {
            for i in 0..d_out {
                for k in 0..r {
                    usl[i * r + k] = us[i * r + k] * l[k];
                }
            }
            gemm(d_out, r, d_in, &usl, &vst, &mut a, pool);
        }
        if pass == refine {
            break;
        }
        // h_i ← Σ_j g_j A_ij W_ij / Σ_j (g_j A_ij)², rows pool-parallel.
        let hp = Ptr(h.as_mut_ptr());
        let (ga, gw) = (&g, &a);
        pool.run(|wk, nw| {
            let (lo, hi) = share(d_out, wk, nw);
            for i in lo..hi {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for j in 0..d_in {
                    let t = (ga[j] * gw[i * d_in + j]) as f64;
                    num += t * w[i * d_in + j] as f64;
                    den += t * t;
                }
                if den > 0.0 {
                    // SAFETY: index `i` is in this worker's share.
                    unsafe { hp.set(i, (num / den) as f32) };
                }
            }
        });
        // g_j ← Σ_i h_i A_ij W_ij / Σ_i (h_i A_ij)², with the h just computed.
        let mut num = vec![0.0f64; d_in];
        let mut den = vec![0.0f64; d_in];
        for i in 0..d_out {
            let hi = h[i] as f64;
            for j in 0..d_in {
                let t = hi * a[i * d_in + j] as f64;
                num[j] += t * w[i * d_in + j] as f64;
                den[j] += t * t;
            }
        }
        for j in 0..d_in {
            if den[j] > 0.0 {
                g[j] = (num[j] / den[j]) as f32;
            }
        }
    }
    // Ŵ = diag(h)·A·diag(g), reusing A's buffer.
    for i in 0..d_out {
        for j in 0..d_in {
            a[i * d_in + j] *= h[i] * g[j];
        }
    }
    let mut l_pad = l.clone();
    l_pad.resize(r_pad, 0.0);
    let stage = LbStageBuild {
        u_bits: pack_rows(&us, d_out, r, r_pad),
        v_bits: pack_latent(&v, d_in, r, r_pad),
        h,
        l: l_pad,
        g,
    };
    (stage, a)
}

/// Build every stage of one tensor. `w` is consumed as the running residual.
#[allow(clippy::too_many_arguments)] // each is a separate axis of the conversion
pub fn build_tensor(
    name: &str,
    mut w: Vec<f32>,
    d_out: usize,
    d_in: usize,
    r: usize,
    stages: usize,
    refine: usize,
    pool: &Pool,
) -> LbBuild {
    assert!(r > 0 && r.is_multiple_of(32) && d_in.is_multiple_of(32) && d_out.is_multiple_of(32));
    let r_pad = r;
    let norm0 = dotf(&w, &w).sqrt().max(1e-30);
    let mut rng = Rng(fnv1a64(name.as_bytes()) | 1);
    let mut built = Vec::with_capacity(stages);
    for _ in 0..stages {
        let wt = transpose(d_out, d_in, &w);
        let (stage, what) = build_stage(&w, &wt, d_out, d_in, r, r_pad, refine, pool, &mut rng);
        for (wi, ai) in w.iter_mut().zip(&what) {
            *wi -= ai;
        }
        built.push(stage);
    }
    let rel_err = dotf(&w, &w).sqrt() / norm0;
    LbBuild {
        name: name.into(),
        d_out,
        d_in,
        r,
        stages: built,
        rel_err,
    }
}

// ------------------------------------------------------------------ converter --

/// The seven per-layer projections — the matrices worth converting. Embeddings and
/// norms never qualify; the output head only on request (it is usually tied).
fn eligible(name: &str) -> bool {
    name.starts_with("blk.")
        && [
            ".attn_q.weight",
            ".attn_k.weight",
            ".attn_v.weight",
            ".attn_output.weight",
            ".ffn_gate.weight",
            ".ffn_up.weight",
            ".ffn_down.weight",
        ]
        .iter()
        .any(|s| name.ends_with(s))
}

/// One converter step, for a progress bar: a tensor finished (or was skipped).
#[derive(Clone, Debug)]
pub struct ConvertEvent {
    pub done: usize,
    pub total: usize,
    /// Weights covered so far and in all — tensors differ ~3× in size, so this,
    /// not `done`, is what an ETA should extrapolate.
    pub done_params: u64,
    pub total_params: u64,
    pub name: String,
    /// 0 for a skipped tensor.
    pub rank: usize,
    pub rel_err: f64,
    pub secs: f64,
}

/// A progress callback; a newtype so `ConvertOpts` stays `Clone + Debug`.
#[derive(Clone)]
pub struct Progress(pub std::sync::Arc<dyn Fn(&ConvertEvent) + Send + Sync>);

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Progress(..)")
    }
}

#[derive(Clone, Debug)]
pub struct ConvertOpts {
    /// Target bits per weight across bit matrices and f16 scales (plan §1).
    pub bpw: f64,
    /// Residual cascade depth. 2 is the paper's primary + residual.
    pub stages: usize,
    /// Alternating h/g refinement passes per stage.
    pub refine: usize,
    /// Rank multiplier for the K and V projections (the paper's GQA finding).
    pub kv_mult: usize,
    /// Convert `output.weight` too, when the model has one (most tie it).
    pub head: bool,
    /// Fixed rank for every tensor, overriding the budget — for tests.
    pub rank_override: Option<usize>,
    pub threads: usize,
    pub verbose: bool,
    /// Output path; `None` writes `<model>.slb` beside the source.
    pub out: Option<PathBuf>,
    /// Called after every tensor.
    pub progress: Option<Progress>,
    /// Checked between tensors; a set flag stops the conversion with nothing written.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        ConvertOpts {
            bpw: 1.0,
            stages: 2,
            refine: 1,
            kv_mult: 1,
            head: false,
            rank_override: None,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            verbose: false,
            out: None,
            progress: None,
            cancel: None,
        }
    }
}

pub struct ConvertReport {
    pub out: PathBuf,
    pub converted: Vec<(String, usize, f64)>,
    pub skipped: Vec<String>,
    pub data_bytes: u64,
    pub source_bytes: u64,
    pub bpw: f64,
}

impl ConvertReport {
    /// Mean per-tensor relative reconstruction error `‖W−Ŵ‖/‖W‖` (Q4_K is ≈0.1).
    pub fn mean_err(&self) -> f64 {
        self.converted.iter().map(|c| c.2).sum::<f64>() / self.converted.len().max(1) as f64
    }
    pub fn worst(&self) -> Option<&(String, usize, f64)> {
        self.converted.iter().max_by(|a, b| a.2.total_cmp(&b.2))
    }
}

/// The rank one tensor gets under `opts`, or `None` when it stays dense.
fn tensor_rank(name: &str, d_out: usize, d_in: usize, opts: &ConvertOpts) -> Option<usize> {
    let base = match opts.rank_override {
        Some(r) => r,
        None => {
            let mut r = rank_for(opts.bpw, opts.stages, d_out, d_in);
            if opts.kv_mult > 1
                && (name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight"))
            {
                r *= opts.kv_mult;
            }
            r
        }
    };
    let r = base.min(d_out.min(d_in) * 3 / 4 / 32 * 32);
    (r >= 32 && d_in.is_multiple_of(32) && d_out.is_multiple_of(32)).then_some(r)
}

/// Sidecar bytes one tensor takes at rank `r` over `stages`: both bit matrices and
/// three f16 scale vectors per stage — counted as `Slb::data_bytes` counts them
/// (section payloads, not the alignment between them).
fn tensor_bytes(d_out: usize, d_in: usize, r: usize, stages: usize) -> u64 {
    (stages * (d_out * r / 8 + r * d_in / 8 + 2 * (d_out + r + d_in))) as u64
}

/// What a conversion would produce, without doing it: the app shows this before
/// the user commits minutes of CPU.
#[derive(Clone, Debug)]
pub struct ConvertPlan {
    /// (name, d_out, d_in, rank) of every tensor that would be converted.
    pub tensors: Vec<(String, usize, usize, usize)>,
    pub skipped: Vec<String>,
    pub data_bytes: u64,
    /// The dense bytes those tensors take in the GGUF now.
    pub source_bytes: u64,
    pub params: u64,
    /// Conversion work in the units its time follows (see `plan_gguf`): seconds are
    /// `work / rate`, the rate measured per machine.
    pub work: f64,
}

impl ConvertPlan {
    pub fn bpw(&self) -> f64 {
        8.0 * self.data_bytes as f64 / self.params.max(1) as f64
    }
}

fn candidates<'a>(g: &'a Gguf, opts: &ConvertOpts) -> Vec<&'a String> {
    let mut names: Vec<&String> = g
        .tensors
        .keys()
        .filter(|n| eligible(n) || (opts.head && n.as_str() == "output.weight"))
        .collect();
    names.sort();
    names
}

pub fn plan_gguf(path: &Path, opts: &ConvertOpts) -> Result<ConvertPlan, String> {
    let g = Gguf::open(path)?;
    let mut p = ConvertPlan {
        tensors: Vec::new(),
        skipped: Vec::new(),
        data_bytes: 0,
        source_bytes: 0,
        params: 0,
        work: 0.0,
    };
    for name in candidates(&g, opts) {
        let t = &g.tensors[name];
        let rank = (t.dims.len() == 2)
            .then(|| tensor_rank(name, t.dims[1] as usize, t.dims[0] as usize, opts))
            .flatten();
        let Some(r) = rank else {
            p.skipped.push(name.clone());
            continue;
        };
        let (d_in, d_out) = (t.dims[0] as usize, t.dims[1] as usize);
        p.data_bytes += tensor_bytes(d_out, d_in, r, opts.stages);
        p.source_bytes += t.bytes();
        p.params += (d_out * d_in) as u64;
        // What conversion time actually follows (fitted on SmolLM2 at 1.0 and 3.0
        // bpw: 53 s and 321 s, a 6× ratio a d_out·d_in·r model put at 2.3×): the
        // orthonormalisation sweeps, k²·(d_in + d_out), and the eigensolve, k³, over
        // the sketch width k = r + 16. The GEMMs' d_out·d_in·k term fitted to ~0.
        let k = (r + 16).min(d_out.min(d_in)) as f64;
        p.work += opts.stages as f64 * (k * k * (d_out + d_in) as f64 + k * k * k);
        p.tensors.push((name.clone(), d_out, d_in, r));
    }
    if p.tensors.is_empty() {
        return Err("slb: no eligible tensor — nothing to convert".into());
    }
    Ok(p)
}

/// What a model's sidecar is, for the app: present or not, usable or why not.
#[derive(Clone, Debug)]
pub struct SidecarInfo {
    pub path: PathBuf,
    pub file_bytes: u64,
    /// `None` when usable; otherwise why the engine would refuse it.
    pub problem: Option<String>,
    pub tensors: usize,
    pub stages: usize,
    pub rank_min: usize,
    pub rank_max: usize,
    /// Over the converted weights, like `ConvertReport::bpw`.
    pub bpw: f64,
}

/// Inspect `<model>.slb` (never the environment's override: this is what the app
/// manages). `None` when there is no sidecar.
pub fn sidecar_info(gguf_path: &Path) -> Option<SidecarInfo> {
    let path = default_sidecar_path(gguf_path);
    let file_bytes = std::fs::metadata(&path).ok()?.len();
    let mut info = SidecarInfo {
        path: path.clone(),
        file_bytes,
        problem: None,
        tensors: 0,
        stages: 0,
        rank_min: 0,
        rank_max: 0,
        bpw: 0.0,
    };
    match Slb::open_raw(&path) {
        Err(e) => info.problem = Some(e),
        Ok((slb, size, hash)) => {
            info.tensors = slb.tensors.len();
            info.stages = slb
                .tensors
                .iter()
                .map(|t| t.stages.len())
                .max()
                .unwrap_or(0);
            info.rank_min = slb.tensors.iter().map(|t| t.r).min().unwrap_or(0);
            info.rank_max = slb.tensors.iter().map(|t| t.r).max().unwrap_or(0);
            let params: u64 = slb.tensors.iter().map(|t| (t.d_out * t.d_in) as u64).sum();
            info.bpw = 8.0 * slb.data_bytes as f64 / params.max(1) as f64;
            match source_stamp(gguf_path) {
                Ok(st) if st == (size, hash) => {}
                Ok(_) => {
                    info.problem = Some("built from a different model file — reconvert".into())
                }
                Err(e) => info.problem = Some(e),
            }
        }
    }
    Some(info)
}

/// Convert a GGUF's projection matrices into a LittleBit sidecar.
pub fn convert_gguf(path: &Path, opts: &ConvertOpts) -> Result<ConvertReport, String> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let g = Gguf::open(path)?;
    let (src_size, src_hash) = source_stamp(path)?;
    let names = candidates(&g, opts);
    let total = names.len();
    let numel = |n: &str| g.tensors[n].dims.iter().product::<u64>();
    let total_params: u64 = names.iter().map(|n| numel(n)).sum();
    // Tensors are converted in parallel, each on a single-threaded pool: the SVD's
    // eigensolve, the MGS sweeps and the scale fits are serial or barrier-bound
    // within one tensor, but tensors are independent. A tensor's working set is
    // about six copies of its f32 matrix, so the width is capped by an 8 GB budget
    // for the largest one. (One tensor per worker also makes the output independent
    // of the thread count.)
    let largest = names.iter().map(|n| numel(n)).max().unwrap_or(0).max(1);
    let workers = opts
        .threads
        .max(1)
        .min((8_000_000_000 / (24 * largest)).max(1) as usize)
        .min(total.max(1));

    struct Done {
        rank: usize,
        rel_err: f64,
        build: Option<LbBuild>,
    }
    let results: Mutex<Vec<Option<Done>>> = Mutex::new((0..total).map(|_| None).collect());
    let progress = Mutex::new((0usize, 0u64)); // (done, done_params), in completion order
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let cancelled = || {
        opts.cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    };
    let work = |pool: &Pool| loop {
        let k = next.fetch_add(1, Ordering::Relaxed);
        if k >= total || stop.load(Ordering::Relaxed) {
            return;
        }
        if cancelled() {
            stop.store(true, Ordering::Relaxed);
            return;
        }
        let name = names[k];
        let t = &g.tensors[name];
        let t0 = std::time::Instant::now();
        let rank = (t.dims.len() == 2)
            .then(|| tensor_rank(name, t.dims[1] as usize, t.dims[0] as usize, opts))
            .flatten();
        let done = match rank {
            None => Done {
                rank: 0,
                rel_err: 0.0,
                build: None,
            },
            Some(r) => {
                let (d_in, d_out) = (t.dims[0] as usize, t.dims[1] as usize);
                let mut w = vec![0.0f32; d_out * d_in];
                let rb = row_bytes(t.kind, d_in);
                let bytes = g.bytes(t);
                for i in 0..d_out {
                    dequant_row(
                        t.kind,
                        &bytes[i * rb..(i + 1) * rb],
                        &mut w[i * d_in..(i + 1) * d_in],
                    );
                }
                let b = build_tensor(name, w, d_out, d_in, r, opts.stages, opts.refine, pool);
                if opts.verbose {
                    eprintln!(
                        "  {name}  {d_out}×{d_in}  r {r}  rel err {:.4}  ({:.1}s)",
                        b.rel_err,
                        t0.elapsed().as_secs_f64()
                    );
                }
                Done {
                    rank: r,
                    rel_err: b.rel_err,
                    build: Some(b),
                }
            }
        };
        let (rank, rel_err) = (done.rank, done.rel_err);
        results.lock().unwrap_or_else(|e| e.into_inner())[k] = Some(done);
        let mut pg = progress.lock().unwrap_or_else(|e| e.into_inner());
        pg.0 += 1;
        pg.1 += t.dims.iter().product::<u64>();
        if let Some(p) = &opts.progress {
            (p.0)(&ConvertEvent {
                done: pg.0,
                total,
                done_params: pg.1,
                total_params,
                name: name.clone(),
                rank,
                rel_err,
                secs: t0.elapsed().as_secs_f64(),
            });
        }
    };
    if workers == 1 {
        work(&Pool::new(opts.threads.max(1)));
    } else {
        std::thread::scope(|sc| {
            for _ in 0..workers {
                sc.spawn(|| work(&Pool::new(1)));
            }
        });
    }
    if cancelled() || stop.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }

    let mut builds = Vec::new();
    let mut skipped = Vec::new();
    let mut converted = Vec::new();
    let (mut src_bytes, mut params) = (0u64, 0u64);
    for (name, done) in names
        .iter()
        .zip(results.into_inner().unwrap_or_else(|e| e.into_inner()))
    {
        let Some(done) = done else {
            return Err(format!("slb: {name} was not converted"));
        };
        match done.build {
            None => skipped.push((*name).clone()),
            Some(b) => {
                let t = &g.tensors[*name];
                src_bytes += t.bytes();
                params += t.dims.iter().product::<u64>();
                converted.push(((*name).clone(), done.rank, done.rel_err));
                builds.push(b);
            }
        }
    }
    if builds.is_empty() {
        return Err("slb: no eligible tensor — nothing to convert".into());
    }
    let out = opts
        .out
        .clone()
        .unwrap_or_else(|| default_sidecar_path(path));
    write_slb(&out, src_size, src_hash, &builds)?;
    let slb = Slb::open(&out)?; // read back: the writer is only trusted verified
    let bpw = 8.0 * slb.data_bytes as f64 / params.max(1) as f64;
    Ok(ConvertReport {
        out,
        converted,
        skipped,
        data_bytes: slb.data_bytes,
        source_bytes: src_bytes,
        bpw,
    })
}

// ---------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_answer() {
        // The RFC 3720 check value for "123456789".
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c_table(b"123456789"), 0xE306_9283);
        // The instruction path equals the table at every length and alignment.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let buf: Vec<u8> = (0..4099)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        for (a, n) in [(0, 0), (0, 7), (1, 8), (3, 9), (5, 1000), (0, 4099)] {
            assert_eq!(
                crc32c(&buf[a..a + n - a.min(n)]),
                crc32c_table(&buf[a..a + n - a.min(n)]),
                "at {a}+{n}"
            );
        }
    }

    #[test]
    fn the_source_stamp_reads_the_header_and_the_length() {
        let p = std::env::temp_dir().join(format!("smiley-stamp-{}.bin", std::process::id()));
        let mut data = vec![7u8; 100_000];
        std::fs::write(&p, &data).unwrap();
        let (len, h) = source_stamp(&p).unwrap();
        assert_eq!((len, h), (100_000, fnv1a64(&data[..65536])));
        // A change past the header region keeps the stamp (by design); one inside moves it.
        data[90_000] = 0;
        std::fs::write(&p, &data).unwrap();
        assert_eq!(source_stamp(&p).unwrap().1, h);
        data[10] = 0;
        std::fs::write(&p, &data).unwrap();
        assert_ne!(source_stamp(&p).unwrap().1, h);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rank_matches_the_papers_examples_within_a_block() {
        // Paper §A.4: 4096×4096 at 0.55 BPW → r = 546; 4096×11008 at 0.1 → 133.
        // Our f16-per-stage accounting lands within one 32-block of each.
        let r1 = rank_for(0.55, 2, 4096, 4096);
        assert!((512..=576).contains(&r1), "r1 = {r1}");
        let r2 = rank_for(0.1, 2, 4096, 11008);
        assert!((96..=160).contains(&r2), "r2 = {r2}");
        // A budget too small for even the scales must not underflow.
        assert_eq!(rank_for(0.0001, 2, 64, 64), 0);
    }

    #[test]
    fn scalar_dot_follows_the_sign_bits() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 + 1.0).collect();
        // All bits set: plain sum. None set: negated sum.
        assert_eq!(lb_dot_scalar(&[0xFF; 4], 32, &x), x.iter().sum::<f32>());
        assert_eq!(lb_dot_scalar(&[0x00; 4], 32, &x), -x.iter().sum::<f32>());
        // Bit 0 of byte 0 controls x[0] (LSB-first).
        let got = lb_dot_scalar(&[0xFE, 0xFF, 0xFF, 0xFF], 32, &x);
        assert_eq!(got, x.iter().sum::<f32>() - 2.0 * x[0]);
    }

    #[test]
    fn batched_and_simd_dots_match_the_scalar_specification() {
        let mut s = 0x1234_5678_9ABC_DEFFu64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n = 96;
        let bits: Vec<u8> = (0..n / 8).map(|_| (rnd() >> 32) as u8).collect();
        let stride = n + 32;
        for nb in 1..=5 {
            let xs: Vec<f32> = (0..nb * stride)
                .map(|_| ((rnd() >> 40) as f32 / (1u64 << 23) as f32) - 1.0)
                .collect();
            let mut out = vec![0.0f32; nb];
            lb_dot_batch(&bits, n, &xs, stride, nb, &mut out);
            for b in 0..nb {
                let want = lb_dot_scalar(&bits, n, &xs[b * stride..]);
                assert!(
                    (out[b] - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "nb {nb} b {b}: {} vs {want}",
                    out[b]
                );
            }
        }
    }

    #[test]
    fn int8_sign_dot_matches_its_spec_and_tracks_the_f32_kernel() {
        let mut s = 0x0DDB_1A5E_5BAD_5EEDu64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in [32usize, 96, 960, 4224] {
            let bits: Vec<u8> = (0..n / 8).map(|_| (rnd() >> 32) as u8).collect();
            let stride = n + 8;
            for nb in [1usize, 2, 3, 4, 5] {
                // Heavy-tailed activations: a few outlier channels, as in real models.
                let xs: Vec<f32> = (0..nb * n)
                    .map(|i| {
                        let v = ((rnd() >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
                        if i % 97 == 0 {
                            v * 60.0
                        } else {
                            v
                        }
                    })
                    .collect();
                let mut rows = vec![0.0f32; (nb - 1) * stride + n];
                for b in 0..nb {
                    quantize_i8_row(&mut rows[b * stride..], n, |i| xs[b * n + i]);
                }
                let mut out = vec![0.0f32; nb];
                lb_dot_i8_batch(&bits, n, &rows, stride, nb, &mut out);
                for b in 0..nb {
                    let spec = lb_dot_i8_scalar(&bits, n, &rows[b * stride..]);
                    assert!(
                        (out[b] - spec).abs() <= 1e-4 * spec.abs().max(1.0),
                        "n {n} nb {nb} b {b}: {} vs spec {spec}",
                        out[b]
                    );
                    let want = lb_dot_scalar(&bits, n, &xs[b * n..]);
                    // Per-block quantisation noise: well under the block's own scale.
                    let smax = rows[b * stride + n / 4..][..i8_blocks(n)]
                        .iter()
                        .fold(0.0f32, |m, v| m.max(*v));
                    let tol = 4.0 * (n as f32).sqrt() * smax / 12f32.sqrt() + 1e-3;
                    assert!(
                        (out[b] - want).abs() <= tol,
                        "n {n} nb {nb} b {b}: {} vs f32 {want} (tol {tol})",
                        out[b]
                    );
                }
                // Negative control: one flipped sign bit must move the result.
                let top = (0..n)
                    .max_by(|&i, &j| xs[i].abs().total_cmp(&xs[j].abs()))
                    .unwrap();
                let mut bad = bits.clone();
                bad[top / 8] ^= 1 << (top % 8);
                let mut o2 = vec![0.0f32; nb];
                lb_dot_i8_batch(&bad, n, &rows, stride, nb, &mut o2);
                assert!(
                    o2.iter().zip(&out).any(|(a, b)| a != b),
                    "n {n} nb {nb}: flip undetected"
                );
            }
        }
    }

    #[test]
    fn the_vector_quantiser_equals_the_scalar_one_bit_for_bit() {
        let mut s = 0xA076_1D64_78BD_642Fu64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in [32usize, 96, 320, 704, 960, 2560] {
            let x: Vec<f32> = (0..n)
                .map(|i| {
                    let v = ((rnd() >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
                    if i % 61 == 0 {
                        v * 80.0
                    } else if i % 7 == 0 {
                        0.0
                    } else {
                        v
                    }
                })
                .collect();
            let mut row = vec![0.0f32; n];
            quantize_i8_row(&mut row, n, |i| x[i]);
            // The scalar reference this replaced, verbatim in effect.
            let q = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const i8, n) };
            let mut corr = 0.0f32;
            for k in 0..i8_blocks(n) {
                let r = k * I8_BLOCK..((k + 1) * I8_BLOCK).min(n);
                let amax = r.clone().fold(0.0f32, |m, i| m.max(x[i].abs()));
                let inv = 127.0 / amax;
                let mut sum = 0i32;
                for i in r {
                    let want = (x[i] * inv).round() as i32;
                    assert_eq!(q[i] as i32, want, "n {n} element {i}");
                    sum += want;
                }
                assert_eq!(row[n / 4 + k].to_bits(), (amax / 127.0).to_bits());
                corr += (amax / 127.0) * sum as f32;
            }
            assert_eq!(row[n / 4 + i8_blocks(n)].to_bits(), corr.to_bits());
        }
    }

    #[test]
    fn gemm_matches_a_naive_product_on_ragged_shapes() {
        let pool = Pool::new(2);
        let (m, p, n) = (7, 13, 19);
        let a: Vec<f32> = (0..m * p)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.1)
            .collect();
        let b: Vec<f32> = (0..p * n)
            .map(|i| ((i * 29 % 31) as f32 - 15.0) * 0.1)
            .collect();
        let mut c = vec![0.0f32; m * n];
        gemm(m, p, n, &a, &b, &mut c, &pool);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..p).map(|k| a[i * p + k] * b[k * n + j]).sum();
                assert!(
                    (c[i * n + j] - want).abs() < 1e-4,
                    "({i},{j}): {} vs {want}",
                    c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn mgs_produces_an_orthonormal_basis() {
        let pool = Pool::new(2);
        let (k, d) = (6, 40);
        let mut a: Vec<f32> = (0..k * d)
            .map(|i| ((i as f32) * 0.613).sin() + ((i as f32) * 0.237).cos())
            .collect();
        mgs_rows(k, d, &mut a, &pool);
        for i in 0..k {
            for j in 0..=i {
                let dot = dotf(&a[i * d..(i + 1) * d], &a[j * d..(j + 1) * d]);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "({i},{j}): {dot}");
            }
        }
        // A near-dependent set (windows of one periodic sawtooth): every surviving
        // pair must still be orthonormal; a dependent row may be zeroed, never kept
        // askew. This data held a 0.79 overlap before selective re-orthogonalization.
        let mut b: Vec<f32> = (0..k * d)
            .map(|i| ((i * 131 % 97) as f32 - 48.0) * 0.07)
            .collect();
        mgs_rows(k, d, &mut b, &pool);
        for i in 0..k {
            let ni = dotf(&b[i * d..(i + 1) * d], &b[i * d..(i + 1) * d]);
            assert!(
                ni.abs() < 1e-9 || (ni - 1.0).abs() < 1e-5,
                "row {i} norm² {ni}"
            );
            for j in 0..i {
                let dot = dotf(&b[i * d..(i + 1) * d], &b[j * d..(j + 1) * d]);
                assert!(dot.abs() < 1e-4, "({i},{j}): {dot}");
            }
        }
    }

    #[test]
    fn eigensolver_reproduces_the_matrix_it_was_given() {
        // A·v = λ·v for every eigenpair, on a random symmetric matrix — the property
        // that matters, independent of eigenvalue ordering conventions.
        let k = 12;
        let mut m = vec![0.0f64; k * k];
        let mut s = 7u64;
        for i in 0..k {
            for j in 0..=i {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let v = ((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
                m[i * k + j] = v;
                m[j * k + i] = v;
            }
        }
        let mut z = m.clone();
        let mut d = vec![0.0f64; k];
        let mut e = vec![0.0f64; k];
        tred2(k, &mut z, &mut d, &mut e);
        tql2(k, &mut d, &mut e, &mut z).unwrap();
        for c in 0..k {
            for i in 0..k {
                let av: f64 = (0..k).map(|j| m[i * k + j] * z[j * k + c]).sum();
                assert!((av - d[c] * z[i * k + c]).abs() < 1e-9, "col {c} row {i}");
            }
        }
    }

    #[test]
    fn rank1_als_recovers_an_exact_rank1_nonnegative_matrix() {
        let (d, r) = (20, 6);
        let col: Vec<f32> = (0..d).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let row: Vec<f32> = (0..r).map(|k| 1.0 + (k as f32) * 0.3).collect();
        let m: Vec<f32> = (0..d * r).map(|i| col[i / r] * row[i % r]).collect();
        let (c2, r2) = rank1_als(&m, d, r);
        for i in 0..d {
            for k in 0..r {
                let got = c2[i] * r2[k];
                let want = m[i * r + k];
                assert!(
                    (got - want).abs() < 1e-3 * want,
                    "({i},{k}): {got} vs {want}"
                );
            }
        }
    }

    /// Build a small synthetic tensor, write it, read it back, and check the
    /// reference `apply` against an explicit f64 reconstruction of Ŵ — the format
    /// and the forward pass agree about what the bits mean. Then flip one stored
    /// bit on disk and demand the loader refuse the file (the bitrot control).
    #[test]
    fn container_roundtrips_and_detects_bitrot() {
        let dir = std::env::temp_dir().join(format!("smiley-lbq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d_out, d_in, r) = (64, 96, 32);
        let w: Vec<f32> = (0..d_out * d_in)
            .map(|i| (((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0) * 0.05)
            .collect();
        let pool = Pool::new(2);
        let b = build_tensor("t.weight", w.clone(), d_out, d_in, r, 2, 1, &pool);
        assert!(
            b.rel_err < 1.0,
            "an approximation, not noise: {}",
            b.rel_err
        );
        let path = dir.join("round.slb");
        write_slb(&path, 123, 0xABCD, std::slice::from_ref(&b)).unwrap();
        let slb = Slb::open(&path).unwrap();
        assert_eq!(slb.tensors.len(), 1);
        let t = &slb.tensors[0];
        assert_eq!((t.d_out, t.d_in, t.r, t.stages.len()), (d_out, d_in, r, 2));

        // Reconstruct Ŵ from the *loaded* bits and scales, in f64.
        let mut what = vec![0.0f64; d_out * d_in];
        for (si, st) in t.stages.iter().enumerate() {
            let (ub, vb) = (slb.u_bits(t, si), slb.v_bits(t, si));
            for i in 0..d_out {
                for j in 0..d_in {
                    let mut s = 0.0f64;
                    for k in 0..t.r_pad {
                        let us = if ub[i * t.r_pad / 8 + k / 8] >> (k % 8) & 1 == 1 {
                            1.0
                        } else {
                            -1.0
                        };
                        let vs = if vb[k * d_in / 8 + j / 8] >> (j % 8) & 1 == 1 {
                            1.0
                        } else {
                            -1.0
                        };
                        s += us * vs * st.l[k] as f64;
                    }
                    what[i * d_in + j] += st.h[i] as f64 * st.g[j] as f64 * s;
                }
            }
        }
        let x: Vec<f32> = (0..d_in).map(|j| ((j as f32) * 0.37).sin()).collect();
        let mut got = vec![0.0f32; d_out];
        slb.apply(0, &x, &mut got);
        let mut worst = 0.0f64;
        for i in 0..d_out {
            let want: f64 = (0..d_in).map(|j| what[i * d_in + j] * x[j] as f64).sum();
            worst = worst.max((got[i] as f64 - want).abs() / want.abs().max(1.0));
        }
        assert!(
            worst < 1e-4,
            "forward disagrees with reconstruction: {worst}"
        );

        // The negative control: one flipped bit in a sign matrix must be *detected*.
        let mut bytes = std::fs::read(&path).unwrap();
        let off = t.stages[0].u_off;
        bytes[off] ^= 0x10;
        let bad = dir.join("rot.slb");
        std::fs::write(&bad, &bytes).unwrap();
        match Slb::open(&bad) {
            Err(e) => assert!(e.contains("checksum"), "{e}"),
            Ok(_) => panic!("accepted a sidecar with a flipped sign bit"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_refuses_garbage_and_truncation() {
        let dir = std::env::temp_dir().join(format!("smiley-lbq-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("not.slb");
        std::fs::write(&p, b"GGUFnope").unwrap();
        match Slb::open(&p) {
            Err(e) => assert!(e.contains("bad magic"), "{e}"),
            Ok(_) => panic!("accepted a non-SLB file"),
        }
        // A real file truncated mid-directory must refuse, not panic.
        let (d_out, d_in, r) = (32, 32, 32);
        let pool = Pool::new(1);
        let b = build_tensor("x", vec![0.01; d_out * d_in], d_out, d_in, r, 1, 0, &pool);
        let full = dir.join("full.slb");
        write_slb(&full, 1, 2, &[b]).unwrap();
        let bytes = std::fs::read(&full).unwrap();
        let cut = dir.join("cut.slb");
        std::fs::write(&cut, &bytes[..60]).unwrap();
        assert!(Slb::open(&cut).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stages_and_refinement_both_reduce_reconstruction_error() {
        let (d_out, d_in, r) = (64, 64, 32);
        let w: Vec<f32> = (0..d_out * d_in)
            .map(|i| ((((i * 48271) % 65536) as f32 / 32768.0) - 1.0) * 0.1)
            .collect();
        let pool = Pool::new(2);
        let e1 = build_tensor("a", w.clone(), d_out, d_in, r, 1, 0, &pool).rel_err;
        let e2 = build_tensor("a", w.clone(), d_out, d_in, r, 2, 0, &pool).rel_err;
        let e1r = build_tensor("a", w.clone(), d_out, d_in, r, 1, 2, &pool).rel_err;
        assert!(e2 < e1, "a residual stage must help: {e2} vs {e1}");
        assert!(e1r <= e1 + 1e-9, "refinement must not hurt: {e1r} vs {e1}");
    }
}
