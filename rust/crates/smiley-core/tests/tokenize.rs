// nafs-aql: upstream test kept as it is; the workspace lints with -D warnings.
#![allow(clippy::ptr_arg)]
//! The tokenizer must agree with llama.cpp token for token.
//!
//! Golden ids produced by the same binary that runs the model:
//!
//! ```text
//! llama-tokenize -m SmolLM2-360M-Instruct.Q4_K_M.gguf -f case.txt --no-bos --ids
//! ```
//!
//! This is checked rather than assumed because the failure mode is not an error. A
//! tokenizer that splits one string differently than the model was trained on still
//! produces fluent text; it is simply answering a prompt the user did not type, and
//! nothing downstream can tell.

use smiley_core::gguf::Gguf;
use smiley_core::tokenizer::Tokenizer;
use std::path::PathBuf;

fn model_path() -> PathBuf {
    std::env::var_os("SMILEY_CORE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// The three vocabularies the catalogue ships, each with goldens from the same
/// pinned `llama-tokenize` (b11146) on the same twelve strings.
fn oracle_models() -> Vec<(PathBuf, &'static str, usize, &'static str)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let m = |f: &str| PathBuf::from(&home).join("models").join(f);
    vec![
        (
            model_path(),
            include_str!("fixtures/tokenize-golden.tsv"),
            49152,
            "smollm",
        ),
        (
            m("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
            include_str!("fixtures/tokenize-golden-qwen2.tsv"),
            151936,
            "qwen2",
        ),
        (
            m("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            include_str!("fixtures/tokenize-golden-llama3.tsv"),
            128256,
            "llama-bpe",
        ),
    ]
}

#[test]
fn encoding_matches_llama_cpp() {
    let mut checked = 0;
    for (path, tsv, n_vocab, pre) in oracle_models() {
        if !path.exists() {
            eprintln!("\n  SKIPPED: tokenizer oracle needs the model at {path:?}\n");
            continue;
        }
        encoding_matches(&path, tsv, n_vocab, pre);
        checked += 1;
    }
    assert!(
        checked >= 1 || !model_path().exists(),
        "no oracle model present"
    );
}

fn encoding_matches(path: &PathBuf, tsv: &str, n_vocab: usize, pre: &str) {
    let g = Gguf::open(path).expect("open model");
    let tk = Tokenizer::from_gguf(&g).expect("build tokenizer");

    assert_eq!(tk.n_vocab, n_vocab);
    assert_eq!(tk.pre.name(), pre);
    assert_eq!(tk.n_byte_tokens(), 256);

    let mut cases = 0;
    let mut failures = Vec::new();
    for line in tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let (text, ids) = line.split_once('\t').expect("malformed fixture line");
        let text = unescape(text);
        let want: Vec<u32> = ids.split(',').map(|s| s.trim().parse().unwrap()).collect();
        let got = tk.encode(&text);
        if got != want {
            failures.push(format!(
                "  {:?}\n    llama.cpp   {:?}\n    smiley-core {:?}",
                text, want, got
            ));
        }
        cases += 1;
    }
    assert!(cases >= 10, "fixture is too thin: {cases} cases");
    assert!(
        failures.is_empty(),
        "{pre}: {} of {} tokenisations differ from llama.cpp:\n{}",
        failures.len(),
        cases,
        failures.join("\n")
    );
    eprintln!("tokenizer ({pre}): {cases} strings match llama.cpp exactly");
}

#[test]
fn decoding_inverts_encoding() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    let tk = Tokenizer::from_gguf(&g).expect("build tokenizer");

    // Byte-level BPE is lossless by construction; if a round trip loses text, the
    // byte alphabet is wrong somewhere and generated output will be subtly corrupt.
    for s in [
        "Hello world",
        "Normalisasi katalog produk kecantikan",
        "café naïve — “quoted” über",
        "INCI: Aqua, Glycerin, Niacinamide 5%",
        "line one\nline two\n\nline four",
        "  leading and trailing  ",
        "\t\ttabs\t",
        "emoji 🙂 and 日本語",
    ] {
        let round = tk.decode(&tk.encode(s));
        assert_eq!(round, s, "round trip lost text");
    }
}

#[test]
fn special_tokens_are_matched_atomically() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    let tk = Tokenizer::from_gguf(&g).expect("build tokenizer");

    // The chat template depends on these being single tokens. If the regex shredded
    // `<|im_start|>` into punctuation the model would never see a turn boundary and
    // would ramble past the end of its reply.
    for lit in ["<|im_start|>", "<|im_end|>"] {
        let id = tk
            .id_of(lit)
            .unwrap_or_else(|| panic!("{lit} missing from vocab"));
        assert_eq!(tk.encode(lit), vec![id], "{lit} was not matched atomically");
        assert!(tk.is_special(id), "{lit} should be flagged special");
    }
    let ids = tk.encode("<|im_start|>user\nhi<|im_end|>");
    assert_eq!(ids[0], tk.id_of("<|im_start|>").unwrap());
    assert_eq!(*ids.last().unwrap(), tk.id_of("<|im_end|>").unwrap());
}
