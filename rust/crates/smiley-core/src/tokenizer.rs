//! Byte-level BPE tokenizer, built from the vocabulary embedded in the GGUF file.
//!
//! This is the second place (after the block layouts) where being *nearly* right
//! produces fluent, wrong output rather than an error. A tokenizer that splits
//! `"2026"` into one token where the model was trained on four still generates
//! confident English — it is just answering a different prompt than the user typed.
//! So this module is checked against `llama-tokenize`'s own output on real strings
//! (`tests/tokenize.rs`), not against expectation.
//!
//! # Pipeline
//!
//! ```text
//!   text ──▶ split out special tokens (matched atomically, longest first)
//!        ──▶ regex 1: every \p{N} becomes its own piece
//!        ──▶ regex 2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//!        ──▶ bytes ──▶ GPT-2 byte-to-char alphabet
//!        ──▶ BPE merges by rank
//!        ──▶ ids
//! ```
//!
//! The two regexes are the ones llama.cpp uses for `LLAMA_VOCAB_PRE_TYPE_SMOLLM`,
//! transcribed from `llama-vocab.cpp`. They are hand-scanned rather than run through
//! a regex engine: the crate has one dependency and this pattern is a fixed,
//! six-way alternation that a scanner expresses more directly than a regex would.
//!
//! Two more pre-tokenizers are implemented the same way, because the catalogue's
//! Llama 3.2 and Qwen2.5 files use them:
//!
//! ```text
//!   llama-bpe: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//!              | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   qwen2:     the same with \p{N} in place of \p{N}{1,3}
//! ```
//!
//! `split_llama3` mirrors `unicode_regex_split_custom_llama3` in llama.cpp's
//! `unicode.cpp` branch for branch, since that function — not a regex engine — is
//! what produced the golden fixtures. A model whose `tokenizer.ggml.pre` names
//! anything else is refused at load, never approximated.
//!
//! # Known divergence
//!
//! `\p{L}` is approximated by `char::is_alphabetic()`, which is the Unicode
//! *Alphabetic* property: `\p{L}` plus `Nl` (letter-like numerals such as `Ⅻ`) and
//! `Other_Alphabetic` (some combining marks). For those few code points this
//! tokenizer takes the `\p{L}+` branch where llama.cpp takes `\p{N}+`. `\p{N}` maps
//! exactly onto `char::is_numeric()`. Rust's standard library does not expose
//! General_Category, and shipping a category table to close a gap this narrow would
//! cost more than it buys — but it is a real difference and it is recorded here
//! rather than discovered later.

use crate::gguf::Gguf;
use std::collections::HashMap;

/// GGUF token types. Only the distinction between "normal" and "matched atomically"
/// matters here.
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

/// Which pre-tokenizer regex set the vocabulary was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `smollm`: digits one at a time, then the GPT-2 alternation.
    SmolLm,
    /// `llama-bpe`: Llama 3's pattern, digits in groups of up to three.
    Llama3,
    /// `qwen2`: Llama 3's pattern with digits one at a time.
    Qwen2,
}

impl PreType {
    pub fn from_name(name: &str) -> Option<PreType> {
        Some(match name {
            "smollm" => PreType::SmolLm,
            "llama-bpe" => PreType::Llama3,
            "qwen2" => PreType::Qwen2,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            PreType::SmolLm => "smollm",
            PreType::Llama3 => "llama-bpe",
            PreType::Qwen2 => "qwen2",
        }
    }
}

/// Pre-tokenizer names this crate implements, for the compatibility probe's message.
pub const PRE_TYPES: &[&str] = &["smollm", "llama-bpe", "qwen2"];

pub struct Tokenizer {
    /// id → stored piece, in the byte-to-char alphabet (so a leading space is `Ġ`).
    pieces: Vec<String>,
    ids: HashMap<String, u32>,
    /// (left, right) → merge rank. Lower merges first.
    ranks: HashMap<(String, String), u32>,
    /// Literal text → id for tokens matched atomically, longest first.
    specials: Vec<(String, u32)>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub bos: u32,
    pub eos: u32,
    /// `tokenizer.ggml.add_bos_token`: whether a prompt starts with `bos`. Llama 3
    /// says yes and its template writes the token itself; SmolLM and Qwen say no.
    pub add_bos: bool,
    pub pre: PreType,
    pub n_vocab: usize,
}

/// The GPT-2 byte alphabet: every byte maps to a printable code point so that BPE
/// can operate on text while remaining byte-exact and total over arbitrary input.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut enc = ['\0'; 256];
    let mut taken = [false; 256];
    for (a, b) in [(0x21u32, 0x7Eu32), (0xA1, 0xAC), (0xAE, 0xFF)] {
        for c in a..=b {
            enc[c as usize] = char::from_u32(c).unwrap();
            taken[c as usize] = true;
        }
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !taken[b] {
            enc[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let dec = enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
    (enc, dec)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(format!(
                "tokenizer model {model:?} is not implemented — Smiley Core reads gpt2 \
                 byte-level BPE vocabularies"
            ));
        }
        let pre_name = g.str("tokenizer.ggml.pre").unwrap_or("smollm");
        let pre = PreType::from_name(pre_name).ok_or_else(|| {
            format!(
                "pre-tokenizer {pre_name:?} is not implemented — Smiley Core reads {}",
                PRE_TYPES.join(", ")
            )
        })?;
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        let types = g.arr_i32("tokenizer.ggml.token_type")?;
        if toks.len() != types.len() {
            return Err(format!(
                "vocab has {} tokens but {} token types",
                toks.len(),
                types.len()
            ));
        }
        let merges = g.arr_str("tokenizer.ggml.merges")?;

        let pieces: Vec<String> = toks.to_vec();
        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            // First id wins: a duplicate piece in the vocabulary is a model-file
            // problem, and preferring the lower id matches llama.cpp.
            ids.entry(p.clone()).or_insert(i as u32);
        }

        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (l, r) = m
                .split_once(' ')
                .ok_or_else(|| format!("merge {rank} is not a pair: {m:?}"))?;
            ranks.insert((l.to_string(), r.to_string()), rank as u32);
        }

        // Control and user-defined tokens are matched against the raw text before
        // any splitting, so that `<|im_start|>` stays one token instead of being
        // shredded into punctuation by the regex.
        let mut specials: Vec<(String, u32)> = pieces
            .iter()
            .zip(types.iter())
            .enumerate()
            .filter(|(_, (_, &t))| t == TYPE_CONTROL || t == TYPE_USER_DEFINED)
            .map(|(i, (p, _))| (p.clone(), i as u32))
            .collect();
        // Longest first, so `<|im_start|>` is preferred over a shorter prefix.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let (byte_enc, byte_dec) = byte_alphabet();
        let n_vocab = pieces.len();
        Ok(Tokenizer {
            pieces,
            ids,
            ranks,
            specials,
            byte_enc,
            byte_dec,
            bos: g.u32("tokenizer.ggml.bos_token_id").unwrap_or(1),
            eos: g.u32("tokenizer.ggml.eos_token_id").unwrap_or(2),
            // llama.cpp turns `add_bos` on for `llama-bpe` whatever the file says
            // (`llama-vocab.cpp`, the LLAMA3 pre-type branch), and the Llama 3
            // template writes the token itself; the flag is reported, not acted on.
            add_bos: g
                .bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(pre == PreType::Llama3),
            pre,
            n_vocab,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        let mut p = 0usize;
        while p < text.len() {
            // Atomic special tokens first.
            let mut hit = None;
            for (s, id) in &self.specials {
                if !s.is_empty() && text[p..].starts_with(s.as_str()) {
                    hit = Some((s.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = hit {
                out.push(id);
                p += len;
                continue;
            }
            // Otherwise accumulate a plain run up to the next special, and split it.
            let mut q = p;
            while q < text.len() {
                if text.is_char_boundary(q)
                    && self
                        .specials
                        .iter()
                        .any(|(s, _)| !s.is_empty() && text[q..].starts_with(s.as_str()))
                {
                    break;
                }
                q += 1;
                while q < text.len() && !text.is_char_boundary(q) {
                    q += 1;
                }
            }
            self.encode_plain(&text[p..q], out);
            p = q;
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        match self.pre {
            PreType::SmolLm => {
                for chunk in split_numbers(text) {
                    let mut p = 0usize;
                    while p < chunk.len() {
                        let take = match_piece(&chunk[p..]).unwrap_or_else(|| {
                            // No alternative matched: emit one character, as llama.cpp's
                            // splitter does for text its patterns do not cover.
                            chunk[p..].chars().next().map(|c| c.len_utf8()).unwrap_or(1)
                        });
                        self.bpe(&chunk[p..p + take], out);
                        p += take;
                    }
                }
            }
            PreType::Llama3 => {
                for piece in split_llama3(text, 3) {
                    self.bpe(piece, out);
                }
            }
            PreType::Qwen2 => {
                for piece in split_llama3(text, 1) {
                    self.bpe(piece, out);
                }
            }
        }
    }

    /// Byte-level BPE over one pre-tokenized piece.
    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        // Into the byte alphabet: one symbol per *byte*, not per character.
        let mut syms: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_enc[b as usize].to_string())
            .collect();

        // `ignore_merges` (llama.cpp, Llama 3 only): a piece that is itself a
        // vocabulary entry is emitted whole. Llama 3's tiktoken vocabulary holds
        // tokens its merge list cannot reach, so running the merges would split
        // "Hello" into pieces the model never saw as a word.
        if self.pre == PreType::Llama3 {
            let whole: String = syms.concat();
            if let Some(&id) = self.ids.get(&whole) {
                out.push(id);
                return;
            }
        }

        // Repeatedly merge the lowest-ranked adjacent pair. Pieces are short (a word
        // at most), so the quadratic scan is cheaper than maintaining a heap.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none_or(|(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }

        for s in syms {
            match self.ids.get(&s) {
                Some(&id) => out.push(id),
                None => {
                    // Should be unreachable for a byte-level vocabulary, since every
                    // single-byte symbol is in it. Emit the bytes individually rather
                    // than dropping text, and never panic on user input.
                    for ch in s.chars() {
                        if let Some(&id) = self.ids.get(&ch.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Raw bytes of one token, before UTF-8 validation.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match self.pieces.get(id as usize) {
            Some(p) => p
                .chars()
                .filter_map(|c| self.byte_dec.get(&c).copied())
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.specials.iter().any(|(_, i)| *i == id)
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Look up a token by its literal text, e.g. `"<|im_end|>"`.
    pub fn id_of(&self, literal: &str) -> Option<u32> {
        self.ids.get(literal).copied()
    }

    pub fn n_byte_tokens(&self) -> usize {
        self.byte_dec.len()
    }
}

/// Streaming UTF-8 assembler.
///
/// A token is a run of *bytes*, and a multi-byte character is routinely split across
/// two tokens — emoji and accented Indonesian text do it constantly. Decoding each
/// token independently would emit a replacement character for the first half and
/// then another for the second, so the UI would show `��` where the model produced
/// `é`. This holds the incomplete tail back until its continuation arrives.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Feed one token's bytes; returns whatever is now a complete UTF-8 prefix.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let good = e.valid_up_to();
                // An *invalid* sequence will never complete, so flush it lossily
                // rather than blocking the stream forever. Only a truncated tail
                // (error_len() == None) is worth holding.
                let keep = match e.error_len() {
                    None => good,
                    Some(_) => self.pending.len(),
                };
                let out = String::from_utf8_lossy(&self.pending[..keep]).into_owned();
                self.pending.drain(..keep);
                out
            }
        }
    }

    /// Flush anything left at end of generation, lossily.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Regex 1: `\p{N}` — every numeric character becomes its own piece.
fn split_numbers(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        if ch.is_numeric() {
            if i > start {
                out.push(&s[start..i]);
            }
            out.push(&s[i..i + ch.len_utf8()]);
            start = i + ch.len_utf8();
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Regex 2, as a scanner. Returns the byte length of the match at the start of
/// `rest`, trying the alternatives in the order the pattern lists them.
fn match_piece(rest: &str) -> Option<usize> {
    for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(c) {
            return Some(c.len());
        }
    }
    // The three ` ?X+` alternatives share an optional single leading space.
    let sp = usize::from(rest.starts_with(' '));
    let after = &rest[sp..];

    let run = |pred: fn(char) -> bool| -> usize {
        let mut n = 0;
        for ch in after.chars() {
            if pred(ch) {
                n += ch.len_utf8();
            } else {
                break;
            }
        }
        n
    };
    for pred in [
        (|c: char| c.is_alphabetic()) as fn(char) -> bool,
        |c: char| c.is_numeric(),
        |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric(),
    ] {
        let n = run(pred);
        if n > 0 {
            return Some(sp + n);
        }
    }

    // `\s+(?!\S)`. Greedy `\s+` with a negative lookahead backtracks by exactly one
    // whitespace character when a non-space follows — that character is left for the
    // next match, where it becomes the leading space of ` ?\p{L}+`. If the run
    // reaches the end of the piece there is nothing to give back.
    let mut ws = 0usize;
    for ch in rest.chars() {
        if ch.is_whitespace() {
            ws += ch.len_utf8();
        } else {
            break;
        }
    }
    if ws > 0 {
        if ws == rest.len() {
            return Some(ws);
        }
        let last = rest[..ws].chars().next_back().unwrap().len_utf8();
        if ws > last {
            return Some(ws - last);
        }
    }
    None
}

/// The Llama 3 family splitter, branch for branch after llama.cpp's
/// `unicode_regex_split_custom_llama3`. `max_digits` is 3 for `llama-bpe`
/// (`\p{N}{1,3}`) and 1 for `qwen2` (`\p{N}`); nothing else differs.
///
/// `\p{L}` and `\p{N}` are `char::is_alphabetic` / `char::is_numeric`, and `\s` is
/// `char::is_whitespace` — the same approximations as the SmolLM scanner, recorded
/// in the module docs.
pub fn split_llama3(s: &str, max_digits: usize) -> Vec<&str> {
    let cps: Vec<(usize, char)> = s.char_indices().collect();
    let n = cps.len();
    let at = |i: usize| -> Option<char> { cps.get(i).map(|x| x.1) };
    let letter = |i: usize| at(i).is_some_and(char::is_alphabetic);
    let number = |i: usize| at(i).is_some_and(char::is_numeric);
    let white = |i: usize| at(i).is_some_and(char::is_whitespace);
    let newline = |i: usize| matches!(at(i), Some('\r') | Some('\n'));
    let byte = |i: usize| if i < n { cps[i].0 } else { s.len() };

    let mut out = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;
    macro_rules! add {
        ($end:expr) => {{
            let end: usize = $end;
            if end > prev_end {
                out.push(&s[byte(prev_end)..byte(end)]);
            }
            prev_end = end;
        }};
    }

    while pos < n {
        let c = cps[pos].1;

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' && pos + 1 < n {
            let c1 = at(pos + 1).map(|x| x.to_ascii_lowercase());
            if matches!(c1, Some('s') | Some('t') | Some('m') | Some('d')) {
                pos += 2;
                add!(pos);
                continue;
            }
            if pos + 2 < n {
                let c2 = at(pos + 2).map(|x| x.to_ascii_lowercase());
                if matches!(
                    (c1, c2),
                    (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l'))
                ) {
                    pos += 3;
                    add!(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if !(c == '\r' || c == '\n' || number(pos)) && (letter(pos) || letter(pos + 1)) {
            pos += 1;
            while letter(pos) {
                pos += 1;
            }
            add!(pos);
            continue;
        }

        // \p{N}{1,max}
        if number(pos) {
            let mut ini = pos;
            while number(pos) {
                pos += 1;
                if pos - ini >= max_digits {
                    add!(pos);
                    ini = pos;
                }
            }
            add!(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        let f2 = if c == ' ' { pos + 1 } else { pos };
        if at(f2).is_some() && !(white(f2) || letter(f2) || number(f2)) {
            pos = f2;
            while at(pos).is_some() && !(white(pos) || letter(pos) || number(pos)) {
                pos += 1;
            }
            while newline(pos) {
                pos += 1;
            }
            add!(pos);
            continue;
        }

        let mut num_ws = 0usize;
        let mut last_end_nl = 0usize;
        while white(pos + num_ws) {
            if newline(pos + num_ws) {
                last_end_nl = pos + num_ws + 1;
            }
            num_ws += 1;
        }
        // \s*[\r\n]+
        if last_end_nl > 0 {
            pos = last_end_nl;
            add!(pos);
            continue;
        }
        // \s+(?!\S)
        if num_ws > 1 && at(pos + num_ws).is_some() {
            pos += num_ws - 1;
            add!(pos);
            continue;
        }
        // \s+
        if num_ws > 0 {
            pos += num_ws;
            add!(pos);
            continue;
        }
        // no match: one character
        pos += 1;
        add!(pos);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_splitter_follows_llama_cpp_branch_for_branch() {
        // Digits: groups of three for Llama 3, singles for Qwen2.
        assert_eq!(split_llama3("12345", 3), vec!["123", "45"]);
        assert_eq!(split_llama3("12345", 1), vec!["1", "2", "3", "4", "5"]);
        // A leading space attaches to a letter run; before a digit it stands alone.
        assert_eq!(split_llama3("a b", 3), vec!["a", " b"]);
        assert_eq!(split_llama3("a 1", 3), vec!["a", " ", "1"]);
        // Contractions are case-insensitive and win over the letter run.
        assert_eq!(
            split_llama3("I'LL don't", 3),
            vec!["I", "'LL", " don", "'t"]
        );
        // Punctuation takes one optional space and trailing newlines.
        assert_eq!(
            split_llama3(
                "x !?

y", 3
            ),
            vec![
                "x", " !?

", "y"
            ]
        );
        // Whitespace ending in a newline is one piece; a trailing space goes on.
        assert_eq!(
            split_llama3(
                "a

  
 b", 3
            ),
            vec![
                "a", "

  
", " b"
            ]
        );
        // Run of spaces before a non-space gives back exactly one.
        assert_eq!(split_llama3("a    b", 3), vec!["a", "   ", " b"]);
        // Trailing whitespace is kept whole.
        assert_eq!(split_llama3("a   ", 3), vec!["a", "   "]);
        // Any non-newline, non-letter, non-digit char may prefix a letter run.
        assert_eq!(split_llama3("(abc", 3), vec!["(abc"]);
        assert_eq!(
            split_llama3(
                "
abc", 3
            ),
            vec![
                "
", "abc"
            ]
        );
        assert_eq!(split_llama3("", 3), Vec::<&str>::new());
    }

    #[test]
    fn pre_types_round_trip_their_names() {
        for n in PRE_TYPES {
            assert_eq!(PreType::from_name(n).unwrap().name(), *n);
        }
        assert!(PreType::from_name("gpt2").is_none());
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (enc, dec) = byte_alphabet();
        assert_eq!(
            dec.len(),
            256,
            "every byte must map to a distinct character"
        );
        for b in 0..256usize {
            assert_eq!(dec[&enc[b]], b as u8);
        }
        // The GPT-2 constants everyone recognises: space is Ġ, newline is Ċ.
        assert_eq!(enc[b' ' as usize], 'Ġ');
        assert_eq!(enc[b'\n' as usize], 'Ċ');
        assert_eq!(enc[b'!' as usize], '!');
    }

    #[test]
    fn digits_are_split_individually() {
        assert_eq!(split_numbers("abc"), vec!["abc"]);
        assert_eq!(split_numbers("a1b"), vec!["a", "1", "b"]);
        assert_eq!(split_numbers("2026"), vec!["2", "0", "2", "6"]);
        assert_eq!(split_numbers(" 5%"), vec![" ", "5", "%"]);
    }

    #[test]
    fn the_whitespace_lookahead_gives_back_exactly_one_character() {
        // "  abc": the greedy run backtracks so the final space joins "abc".
        assert_eq!(match_piece("  abc"), Some(1));
        // At end of piece there is nothing to give back.
        assert_eq!(match_piece("  "), Some(2));
        // A single space before a word cannot give anything back, so this
        // alternative fails and ` ?\p{L}+` takes it instead.
        assert_eq!(match_piece(" abc"), Some(4));
        // A lone newline followed by a word matches nothing and is emitted raw.
        assert_eq!(match_piece("\nline"), None);
        // Two newlines: one is given back.
        assert_eq!(match_piece("\n\nline"), Some(1));
    }

    #[test]
    fn contractions_win_over_the_letter_run() {
        assert_eq!(match_piece("'s rest"), Some(2));
        assert_eq!(match_piece("'ll go"), Some(3));
        // Not a listed contraction: falls through to the punctuation run.
        assert_eq!(match_piece("'x"), Some(1));
    }

    #[test]
    fn utf8_stream_holds_a_split_character_until_it_completes() {
        let mut s = Utf8Stream::default();
        // "é" is 0xC3 0xA9, routinely split across two tokens.
        assert_eq!(s.push(&[0xC3]), "");
        assert_eq!(s.push(&[0xA9]), "é");
        assert_eq!(s.push("ok".as_bytes()), "ok");
        // Genuinely invalid input must not block the stream forever.
        let mut s2 = Utf8Stream::default();
        assert_eq!(s2.push(&[0xFF]), "\u{fffd}");
        assert_eq!(s2.push(b"a"), "a");
    }

    #[test]
    fn utf8_stream_flushes_a_truncated_tail_at_the_end() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[0xE2, 0x82]), ""); // two thirds of "€"
        assert_eq!(s.finish(), "\u{fffd}");
        assert_eq!(s.finish(), "");
    }
}
