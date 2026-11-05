//! GGUF container reader: header, metadata, tensor directory, and a memory map.
//!
//! The file is mapped rather than read. That is not a micro-optimisation — it is the
//! residency policy the whole engine is built on. Weights are never copied into owned
//! buffers and never dequantised into a float tensor (spec P1), so the page cache *is*
//! the weight cache and resident set is shared between processes. Every byte this
//! engine reports as "moved" is a byte read straight out of this map.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Weight encodings this engine can read.
///
/// Named for the GGML types they mirror rather than renamed to satisfy Rust casing:
/// these strings appear in model cards, in llama.cpp's source and in Smiley's own
/// catalogue, and a reader checking a layout against `ggml-common.h` should not have
/// to translate. The set is deliberately closed — an unsupported type is a load
/// error, never a silent fallback to a format with different numerics.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q4_K,
    Q6_K,
}

impl Format {
    /// GGML type id as stored in the tensor directory.
    pub fn from_id(id: u32) -> Option<Format> {
        Some(match id {
            0 => Format::F32,
            1 => Format::F16,
            6 => Format::Q5_0,
            8 => Format::Q8_0,
            12 => Format::Q4_K,
            14 => Format::Q6_K,
            _ => return None,
        })
    }

    /// Weights per block.
    pub const fn block(self) -> usize {
        match self {
            Format::F32 | Format::F16 => 1,
            Format::Q5_0 | Format::Q8_0 => 32,
            Format::Q4_K | Format::Q6_K => 256,
        }
    }

    /// Bytes per block. These are the `static_assert`ed sizes in `ggml-common.h`.
    pub const fn block_bytes(self) -> usize {
        match self {
            Format::F32 => 4,
            Format::F16 => 2,
            Format::Q5_0 => 22,  // f16 d + u8 qh[4] + u8 qs[16]
            Format::Q8_0 => 34,  // f16 d + i8 qs[32]
            Format::Q4_K => 144, // f16 d + f16 dmin + u8 scales[12] + u8 qs[128]
            Format::Q6_K => 210, // u8 ql[128] + u8 qh[64] + i8 scales[16] + f16 d
        }
    }

    /// Bits per weight as actually stored, including the block's own scale metadata.
    /// This is the `bpw` of the spec's Equation (3), so it must count the overhead:
    /// quoting 4.0 for Q4_K when the block costs 4.5 would understate the floor.
    pub fn bpw(self) -> f64 {
        (self.block_bytes() * 8) as f64 / self.block() as f64
    }

    pub fn label(self) -> &'static str {
        match self {
            Format::F32 => "F32",
            Format::F16 => "F16",
            Format::Q5_0 => "Q5_0",
            Format::Q8_0 => "Q8_0",
            Format::Q4_K => "Q4_K",
            Format::Q6_K => "Q6_K",
        }
    }
}

/// A metadata value. GGUF's type system is small enough to model exactly; arrays are
/// split by element type because the two that matter here (the 49k-entry token list
/// and the token-type list) have very different shapes.
#[derive(Debug, Clone)]
pub enum Value {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    /// An array whose element type we do not need; kept so the file still parses.
    ArrOther,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF stores dimensions fastest-varying first. For a weight matrix this is
    /// `[in_features, out_features]`, i.e. one row of `dims[0]` contiguous weights
    /// per output — which is exactly the layout the row-parallel matmul wants.
    pub dims: Vec<u64>,
    pub kind: Format,
    /// Offset from the start of the tensor data section, not from the file start.
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Exact on-disk byte length. This is the number telemetry reports as moved when
    /// the tensor is touched, so it is computed from the block geometry rather than
    /// estimated from the element count.
    pub fn bytes(&self) -> u64 {
        let n = self.numel();
        debug_assert_eq!(n % self.kind.block() as u64, 0);
        n / self.kind.block() as u64 * self.kind.block_bytes() as u64
    }
}

pub struct Gguf {
    map: Mmap,
    data_start: usize,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

/// Cursor over the header region. Every read is bounds-checked and returns a
/// `Result`: a truncated or hostile file must produce an error, never a panic in the
/// middle of the app.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("gguf: length overflow")?;
        let s = self
            .b
            .get(self.p..end)
            .ok_or_else(|| format!("gguf: truncated at {} (+{})", self.p, n))?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        // Model-supplied bytes: tokenizer vocabularies legitimately contain
        // fragments that are not valid UTF-8 on their own, so this must not be
        // `from_utf8().unwrap()`.
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// Size in bytes of a scalar metadata element, by GGUF type id.
fn scalar_width(t: u32) -> Option<usize> {
    Some(match t {
        0 | 1 | 7 => 1, // u8, i8, bool
        2 | 3 => 2,     // u16, i16
        4..=6 => 4,     // u32, i32, f32
        10..=12 => 8,   // u64, i64, f64
        _ => return None,
    })
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| format!("gguf: cannot open {path:?}: {e}"))?;
        // SAFETY: the mapping is read-only and the engine treats its contents as
        // untrusted input (every access is bounds-checked against `TensorInfo`).
        // The residual hazard is another process truncating the file while it is
        // mapped, which no Rust API can rule out; the model directory is Smiley's own.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("gguf: mmap failed: {e}"))?;

        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != b"GGUF" {
            return Err("gguf: bad magic — not a GGUF file".into());
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("gguf: unsupported container version {version}"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        // The counts come from the file. Capacity is a hint, never a promise to a
        // corrupt header: a failed allocation aborts the process (no unwind for
        // `catch_unwind` to catch), so cap it by what the file could possibly hold.
        let plausible = map.len() / 16;
        if n_tensors > plausible || n_kv > plausible {
            return Err(format!(
                "gguf: implausible header ({n_tensors} tensors, {n_kv} metadata keys)"
            ));
        }

        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.str()?;
            let v = read_value(&mut c)?;
            meta.insert(key, v);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.str()?;
            let nd = c.u32()? as usize;
            if nd > 4 {
                return Err(format!("gguf: tensor {name} has {nd} dimensions"));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let tid = c.u32()?;
            let kind = Format::from_id(tid).ok_or_else(|| {
                format!(
                    "gguf: tensor {name} uses ggml type {tid}, which Smiley Core does not \
                     implement. Supported: F32, F16, Q5_0, Q8_0, Q4_K, Q6_K."
                )
            })?;
            let offset = c.u64()?;
            // `numel`/`bytes` multiply these freely afterwards; refuse a directory
            // whose products would wrap instead of trusting wrapped arithmetic.
            if dims
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .is_none_or(|n| n > 1 << 48)
            {
                return Err(format!(
                    "gguf: tensor {name} has implausible dimensions {dims:?}"
                ));
            }
            let ti = TensorInfo {
                name: name.clone(),
                dims,
                kind,
                offset,
            };
            if !ti.numel().is_multiple_of(kind.block() as u64) {
                return Err(format!(
                    "gguf: tensor {name} has {} elements, not a multiple of the {} block size",
                    ti.numel(),
                    kind.label()
                ));
            }
            tensors.insert(name, ti);
        }

        // Tensor data begins at the next `alignment` boundary after the directory.
        let alignment = match meta.get("general.alignment") {
            Some(Value::U32(a)) => *a as usize,
            _ => 32,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("gguf: implausible alignment {alignment}"));
        }
        let data_start = c.p.div_ceil(alignment) * alignment;
        if data_start > map.len() {
            return Err("gguf: tensor data starts past end of file".into());
        }

        let g = Gguf {
            map,
            data_start,
            meta,
            tensors,
        };
        // Fail at load rather than at the first dot product: a tensor whose bytes run
        // past the end of the map would otherwise surface as a panic mid-generation.
        for t in g.tensors.values() {
            let end = (g.data_start as u64)
                .checked_add(t.offset)
                .and_then(|e| e.checked_add(t.bytes()))
                .unwrap_or(u64::MAX);
            if end > g.map.len() as u64 {
                return Err(format!(
                    "gguf: tensor {} runs to {} but the file is {} bytes — truncated download?",
                    t.name,
                    end,
                    g.map.len()
                ));
            }
        }
        Ok(g)
    }

    /// The raw block bytes of a tensor, straight out of the map. No copy.
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let s = self.data_start + t.offset as usize;
        &self.map[s..s + t.bytes() as usize]
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("gguf: model has no tensor {name:?}"))
    }

    pub fn total_weight_bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.bytes()).sum()
    }

    /// A boolean metadata value, or `None` when the key is absent or not a bool.
    pub fn bool(&self, k: &str) -> Option<bool> {
        match self.meta.get(k) {
            Some(Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    pub fn u32(&self, k: &str) -> Result<u32, String> {
        match self.meta.get(k) {
            Some(Value::U32(v)) => Ok(*v),
            Some(Value::I32(v)) if *v >= 0 => Ok(*v as u32),
            Some(Value::U64(v)) => Ok(*v as u32),
            _ => Err(format!("gguf: missing or non-integer metadata {k:?}")),
        }
    }

    pub fn f32(&self, k: &str) -> Result<f32, String> {
        match self.meta.get(k) {
            Some(Value::F32(v)) => Ok(*v),
            _ => Err(format!("gguf: missing or non-float metadata {k:?}")),
        }
    }

    pub fn str(&self, k: &str) -> Result<&str, String> {
        match self.meta.get(k) {
            Some(Value::Str(s)) => Ok(s),
            _ => Err(format!("gguf: missing or non-string metadata {k:?}")),
        }
    }

    pub fn arr_str(&self, k: &str) -> Result<&[String], String> {
        match self.meta.get(k) {
            Some(Value::ArrStr(v)) => Ok(v),
            _ => Err(format!("gguf: missing or non-string-array metadata {k:?}")),
        }
    }

    pub fn arr_i32(&self, k: &str) -> Result<&[i32], String> {
        match self.meta.get(k) {
            Some(Value::ArrI32(v)) => Ok(v),
            _ => Err(format!("gguf: missing or non-i32-array metadata {k:?}")),
        }
    }
}

fn read_value(c: &mut Cur) -> Result<Value, String> {
    let t = c.u32()?;
    Ok(match t {
        0 => Value::U32(c.take(1)?[0] as u32),
        1 => Value::I32(c.take(1)?[0] as i8 as i32),
        2 => Value::U32(u16::from_le_bytes(c.take(2)?.try_into().unwrap()) as u32),
        3 => Value::I32(i16::from_le_bytes(c.take(2)?.try_into().unwrap()) as i32),
        4 => Value::U32(c.u32()?),
        5 => Value::I32(c.i32()?),
        6 => Value::F32(c.f32()?),
        7 => Value::Bool(c.take(1)?[0] != 0),
        8 => Value::Str(c.str()?),
        9 => {
            let et = c.u32()?;
            let n = c.u64()? as usize;
            match et {
                8 => {
                    let mut v = Vec::with_capacity(n.min(1 << 20));
                    for _ in 0..n {
                        v.push(c.str()?);
                    }
                    Value::ArrStr(v)
                }
                5 => {
                    let raw = c.take(n.checked_mul(4).ok_or("gguf: array too large")?)?;
                    Value::ArrI32(
                        raw.chunks_exact(4)
                            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                            .collect(),
                    )
                }
                6 => {
                    let raw = c.take(n.checked_mul(4).ok_or("gguf: array too large")?)?;
                    Value::ArrF32(
                        raw.chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect(),
                    )
                }
                other => {
                    // Skip it, but only if we know its width — guessing would
                    // desynchronise the cursor and corrupt every later key.
                    let w = scalar_width(other)
                        .ok_or_else(|| format!("gguf: array of unknown type {other}"))?;
                    c.take(n.checked_mul(w).ok_or("gguf: array too large")?)?;
                    Value::ArrOther
                }
            }
        }
        10 => Value::U64(c.u64()?),
        11 => Value::I32(c.u64()? as i32),
        12 => Value::F32(f64::from_le_bytes(c.take(8)?.try_into().unwrap()) as f32),
        other => return Err(format!("gguf: unknown metadata type {other}")),
    })
}

/// IEEE-754 binary16 to binary32.
///
/// Written out rather than pulled from a crate because it is on the hot path for
/// every block scale in the model and the conversion is twenty lines. Handles
/// subnormals and infinities: a subnormal block scale is rare but a model that
/// contains one must not decode to zero.
#[inline(always)]
pub fn f16(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: value is `man · 2^-24`. Let `k` be the index of the highest
            // set bit of `man`, so the normalised exponent is `k - 24` and the
            // mantissa field is the remainder shifted to bit 23. With
            // `shift = 10 - k` (what it takes to bring bit k up to bit 10):
            //   exponent field = (k - 24) + 127 = 113 - shift
            //   mantissa field = man << (23 - k) = man << (13 + shift), masked
            // The mask drops the implicit bit that `man`'s own top bit becomes.
            let shift = man.leading_zeros() - 21; // man < 2^10, so this is 0..=9
            sign | ((113 - shift) << 23) | ((man << (shift + 13)) & 0x007f_ffff)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// binary32 to binary16, round-to-nearest-even.
///
/// Needed by the f16 KV cache, which is the only place this engine *writes* halves.
/// Rounding rather than truncating matters there: the cache is read back every step
/// for every position, so a systematic downward bias would compound across a long
/// context in a way that looks like the model degrading with length.
#[inline]
pub fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;

    if e == 0xff {
        // Inf, or NaN with its payload collapsed to a quiet NaN.
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflows the half range
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // rounds to zero
        }
        // Subnormal: restore the implicit bit and shift into place.
        let m = man | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1u32 << (shift - 1);
        // Round half to even.
        let up = rem > half || (rem == half && (h & 1) == 1);
        return sign | ((h as u16) + u16::from(up));
    }
    let h = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1);
    sign | ((h as u16) + u16::from(up))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every representable half must survive a round trip. This is 65536 cases and
    /// it is the only way to be sure the subnormal and rounding branches are right;
    /// spot checks pass happily with an off-by-one in the subnormal shift.
    #[test]
    fn f16_round_trips_through_f32_for_every_encoding() {
        for bits in 0u32..=0xffff {
            let h = bits as u16;
            let f = f16(h);
            if f.is_nan() {
                continue; // NaN payloads are not preserved, by design
            }
            let back = to_f16(f);
            assert_eq!(back, h, "0x{h:04x} -> {f} -> 0x{back:04x}");
        }
    }

    #[test]
    fn to_f16_rounds_to_nearest_even_and_saturates() {
        assert_eq!(to_f16(0.0), 0x0000);
        assert_eq!(to_f16(-0.0), 0x8000);
        assert_eq!(to_f16(1.0), 0x3c00);
        assert_eq!(to_f16(65504.0), 0x7bff);
        // Beyond the half range, saturate to infinity rather than wrapping.
        assert_eq!(to_f16(1e30), 0x7c00);
        assert_eq!(to_f16(-1e30), 0xfc00);
        assert!(f16(to_f16(f32::NAN)).is_nan());
        // Exactly halfway between two halves rounds to the even neighbour.
        let lo = f16(0x3c00); // 1.0
        let hi = f16(0x3c01); // next representable
        assert_eq!(to_f16((lo + hi) / 2.0), 0x3c00, "tie must go to even");
        // Far below the subnormal range, round to zero rather than to a denormal.
        assert_eq!(to_f16(1e-12), 0x0000);
    }

    // Several constants below are written to their full exact decimal expansion —
    // `0.333251953125` really is the exact value of half `0x3555`. Rounding them to
    // f32's printed precision would turn an exact-value assertion into an
    // approximate one, which is the opposite of what these tests are for.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn f16_matches_known_values() {
        assert_eq!(f16(0x0000), 0.0);
        assert_eq!(f16(0x8000), -0.0);
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xbc00), -1.0);
        assert_eq!(f16(0x4000), 2.0);
        assert_eq!(f16(0x3555), 0.333251953125);
        assert_eq!(f16(0x7bff), 65504.0); // largest finite half
        assert!(f16(0x7c00).is_infinite() && f16(0x7c00) > 0.0);
        assert!(f16(0xfc00).is_infinite() && f16(0xfc00) < 0.0);
        assert!(f16(0x7e00).is_nan());
        // Subnormals: the smallest positive half is 2^-24. A naive implementation
        // that ignores the exp==0 branch returns 0.0 here and silently zeroes
        // whichever block scale happens to be tiny.
        assert_eq!(f16(0x0001), 5.960_464_5e-8);
        assert_eq!(f16(0x03ff), 6.097_555e-5); // largest subnormal
        assert_eq!(f16(0x0400), 6.103_515_6e-5); // smallest normal
    }

    #[test]
    fn block_geometry_matches_ggml() {
        // The sizes ggml static_asserts. A wrong one here misreads every tensor.
        assert_eq!((Format::Q5_0.block(), Format::Q5_0.block_bytes()), (32, 22));
        assert_eq!((Format::Q8_0.block(), Format::Q8_0.block_bytes()), (32, 34));
        assert_eq!(
            (Format::Q4_K.block(), Format::Q4_K.block_bytes()),
            (256, 144)
        );
        assert_eq!(
            (Format::Q6_K.block(), Format::Q6_K.block_bytes()),
            (256, 210)
        );
        // And the bpw figures the spec quotes.
        assert_eq!(Format::Q4_K.bpw(), 4.5);
        assert_eq!(Format::Q5_0.bpw(), 5.5);
        assert_eq!(Format::Q8_0.bpw(), 8.5);
        assert!((Format::Q6_K.bpw() - 6.5625).abs() < 1e-9);
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        let p = std::env::temp_dir().join("smiley-core-notgguf.bin");
        std::fs::write(&p, b"not a gguf file at all").unwrap();
        // Matched rather than `unwrap_err`: that would require `Debug` on `Gguf`,
        // which owns an `Mmap` and has nothing useful to print.
        match Gguf::open(&p) {
            Err(e) => assert!(e.contains("bad magic"), "{e}"),
            Ok(_) => panic!("accepted a file that is not GGUF"),
        }
        let _ = std::fs::remove_file(&p);
    }
}
