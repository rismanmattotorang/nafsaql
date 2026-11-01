//! A safetensors reader: the header, the tensor directory, and rows as `f32`.
//!
//! Exists for one reason — DeepSeek's DSpark drafters ship only as safetensors
//! (bf16), and this crate's rule is no dependency beyond `memmap2`. The format is a
//! little-endian `u64` header length, a JSON object mapping each tensor name to
//! `{"dtype", "shape", "data_offsets": [begin, end]}` (offsets relative to the end of
//! the header), an optional `"__metadata__"` entry, then the raw bytes. The JSON
//! here is machine-written and flat, so a small recursive-descent parser covers it;
//! anything it does not understand is an error, never a guess.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
}

impl DType {
    fn parse(s: &str) -> Result<DType, String> {
        Ok(match s {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            other => return Err(format!("safetensors dtype {other} is not supported")),
        })
    }
    pub fn size(self) -> usize {
        match self {
            DType::BF16 | DType::F16 => 2,
            DType::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct STensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    begin: usize,
    end: usize,
}

impl STensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    pub tensors: HashMap<String, STensor>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only map of a file this process does not modify. A concurrent
        // truncation by another process is the usual mmap caveat (SIGBUS), accepted
        // for weights exactly as the GGUF reader accepts it.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        if map.len() < 8 {
            return Err("safetensors file is shorter than its header length".into());
        }
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if n > 100 << 20 || 8 + n > map.len() {
            return Err(format!("implausible safetensors header length {n}"));
        }
        let text = std::str::from_utf8(&map[8..8 + n]).map_err(|_| "header is not UTF-8")?;
        let root = Json::parse(text)?;
        let Json::Obj(entries) = root else {
            return Err("header is not a JSON object".into());
        };
        let data_start = 8 + n;
        let data_len = map.len() - data_start;
        let mut tensors = HashMap::new();
        for (name, v) in entries {
            if name == "__metadata__" {
                continue;
            }
            let dtype = DType::parse(
                v.get("dtype")
                    .and_then(Json::as_str)
                    .ok_or(format!("{name}: no dtype"))?,
            )?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Json::as_arr)
                .ok_or(format!("{name}: no shape"))?
                .iter()
                .map(|d| d.as_num().map(|x| x as usize))
                .collect::<Option<_>>()
                .ok_or(format!("{name}: bad shape"))?;
            let off = v
                .get("data_offsets")
                .and_then(Json::as_arr)
                .ok_or(format!("{name}: no data_offsets"))?;
            let (begin, end) = match off.as_slice() {
                [a, b] => (
                    a.as_num().ok_or("bad offset")? as usize,
                    b.as_num().ok_or("bad offset")? as usize,
                ),
                _ => return Err(format!("{name}: data_offsets is not a pair")),
            };
            let t = STensor {
                dtype,
                shape,
                begin,
                end,
            };
            if end < begin || end > data_len || end - begin != t.numel() * dtype.size() {
                return Err(format!(
                    "{name}: offsets {begin}..{end} disagree with its shape and dtype"
                ));
            }
            tensors.insert(name, t);
        }
        Ok(SafeTensors {
            map,
            data_start,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&STensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} is not in the file"))
    }

    pub fn bytes(&self, t: &STensor) -> &[u8] {
        &self.map[self.data_start + t.begin..self.data_start + t.end]
    }

    /// Row `r` of a 2-D tensor (or the whole of a 1-D one with `r = 0`) as `f32`.
    pub fn row_f32(&self, t: &STensor, r: usize, out: &mut [f32]) {
        let cols = *t.shape.last().unwrap_or(&1);
        debug_assert_eq!(out.len(), cols);
        let b = self.bytes(t);
        let sz = t.dtype.size();
        let row = &b[r * cols * sz..(r + 1) * cols * sz];
        for (o, c) in out.iter_mut().zip(row.chunks_exact(sz)) {
            *o = match t.dtype {
                DType::BF16 => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
                DType::F16 => crate::gguf::f16(u16::from_le_bytes([c[0], c[1]])),
                DType::F32 => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            };
        }
    }

    /// A whole tensor as `f32` (for small tensors: norms, biases, the confidence head).
    pub fn vec_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.get(name)?;
        let cols = *t.shape.last().unwrap_or(&1);
        let rows = t.numel() / cols.max(1);
        let mut v = vec![0.0f32; t.numel()];
        for r in 0..rows {
            self.row_f32(t, r, &mut v[r * cols..(r + 1) * cols]);
        }
        Ok(v)
    }
}

/// The JSON subset safetensors headers use: objects, arrays, strings, numbers.
#[derive(Debug, Clone)]
pub enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let b = s.as_bytes();
        let mut i = 0;
        let v = value(b, &mut i)?;
        skip_ws(b, &mut i);
        if i != b.len() {
            return Err(format!("trailing characters in JSON at byte {i}"));
        }
        Ok(v)
    }
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, x)| x),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_arr(&self) -> Option<&Vec<Json>> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_num(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    match b.get(*i) {
        Some(b'{') => {
            *i += 1;
            let mut out = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(out));
            }
            loop {
                skip_ws(b, i);
                let Json::Str(k) = string(b, i)? else {
                    unreachable!()
                };
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return Err(format!("expected ':' at byte {i}"));
                }
                *i += 1;
                out.push((k, value(b, i)?));
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(out));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {i}")),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut out = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(out));
            }
            loop {
                out.push(value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(out));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {i}")),
                }
            }
        }
        Some(b'"') => string(b, i),
        Some(b't') if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let s = *i;
            while *i < b.len()
                && (b[*i].is_ascii_digit() || matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                *i += 1;
            }
            std::str::from_utf8(&b[s..*i])
                .ok()
                .and_then(|t| t.parse().ok())
                .map(Json::Num)
                .ok_or_else(|| format!("bad number at byte {s}"))
        }
        _ => Err(format!("unexpected JSON at byte {i}")),
    }
}

fn string(b: &[u8], i: &mut usize) -> Result<Json, String> {
    if b.get(*i) != Some(&b'"') {
        return Err(format!("expected a string at byte {i}"));
    }
    *i += 1;
    let mut out = String::new();
    while let Some(&c) = b.get(*i) {
        *i += 1;
        match c {
            b'"' => return Ok(Json::Str(out)),
            b'\\' => {
                let e = *b.get(*i).ok_or("dangling escape")?;
                *i += 1;
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i..*i + 4).ok_or("short \\u escape")?)
                            .map_err(|_| "bad \\u")?;
                        let cp = u32::from_str_radix(h, 16).map_err(|_| "bad \\u escape")?;
                        out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        *i += 4;
                    }
                    other => out.push(other as char),
                }
            }
            _ => {
                // Multi-byte UTF-8 passes through byte by byte.
                let start = *i - 1;
                let len = match c {
                    0x00..=0x7f => 1,
                    0xc0..=0xdf => 2,
                    0xe0..=0xef => 3,
                    _ => 4,
                };
                let end = (start + len).min(b.len());
                out.push_str(
                    std::str::from_utf8(&b[start..end]).map_err(|_| "bad UTF-8 in string")?,
                );
                *i = end;
            }
        }
    }
    Err("unterminated string".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_header_shaped_object() {
        let j = Json::parse(r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"x":[1,-2.5e1,true,null,"é\"q"]}"#).unwrap();
        assert_eq!(
            j.get("a.weight")
                .and_then(|t| t.get("dtype"))
                .and_then(Json::as_str),
            Some("BF16")
        );
        let x = j.get("x").and_then(Json::as_arr).unwrap();
        assert_eq!(x[1].as_num(), Some(-25.0));
        assert_eq!(x[4].as_str(), Some("é\"q"));
        assert!(Json::parse("{\"a\":1,}").is_err());
    }

    #[test]
    fn reads_bf16_rows_from_a_written_file() {
        let dir = std::env::temp_dir().join(format!("st_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let vals = [1.0f32, -2.0, 0.5, 3.25, 0.0, -0.125];
        let data: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let hdr = r#"{"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut f = (hdr.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(hdr.as_bytes());
        f.extend_from_slice(&data);
        std::fs::write(&p, f).unwrap();
        let st = SafeTensors::open(&p).unwrap();
        let t = st.get("w").unwrap();
        let mut row = [0.0f32; 3];
        st.row_f32(t, 1, &mut row);
        assert_eq!(row, [3.25, 0.0, -0.125]);
        assert_eq!(st.vec_f32("w").unwrap(), vals);
        std::fs::remove_dir_all(&dir).ok();
    }
}
