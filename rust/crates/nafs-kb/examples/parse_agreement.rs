//! A self-check of the readers (Phase 25, FR-K74): the same document read from two formats must yield the same words.
//! No public score exists for the PDF reader's reading order or for the TeX stripper, so this compares what the two
//! readers make of one paper: the share of the distinct terms and of the numbers of one reading found in the other.
//!
//!     cargo run --release -p nafs-kb --example parse_agreement -- docs/report/nafs_aql_report.pdf \
//!         docs/report/nafs_aql_report.tex docs/report/sections/*.tex
//!
//! The first file is one reading; the remaining files together are the other.

use std::collections::HashSet;
use std::time::Instant;

use nafs_kb::index::Analyzer;
use nafs_kb::{parse, Kind};

fn read(path: &str) -> Result<(String, f64), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let kind = Kind::from_filename(path).ok_or(format!("{path}: unknown kind"))?;
    let t = Instant::now();
    let parsed = parse(kind, &bytes).map_err(|e| format!("{path}: {e}"))?;
    Ok((parsed.text, t.elapsed().as_secs_f64() * 1000.0))
}

fn terms(an: &Analyzer, text: &str) -> (HashSet<String>, HashSet<String>) {
    let (mut words, mut numbers) = (HashSet::new(), HashSet::new());
    an.terms(text, |t| {
        if t.chars().all(|c| c.is_ascii_digit()) {
            // One- and two-digit numbers are page numbers, list numbers and table cells as often as facts.
            if t.len() >= 3 {
                numbers.insert(t);
            }
        } else if t.chars().all(char::is_alphabetic) && t.chars().count() >= 4 {
            words.insert(t);
        }
    });
    (words, numbers)
}

fn share(of: &HashSet<String>, found_in: &HashSet<String>) -> f64 {
    if of.is_empty() {
        return 1.0;
    }
    of.iter().filter(|t| found_in.contains(*t)).count() as f64 / of.len() as f64
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (first, rest) = args
        .split_first()
        .ok_or("usage: parse_agreement <file> <file>...")?;
    let an = Analyzer::default();
    let (a_text, a_ms) = read(first)?;
    let mut b_text = String::new();
    let mut b_ms = 0.0;
    for path in rest {
        let (t, ms) = read(path)?;
        b_text.push_str(&t);
        b_text.push('\n');
        b_ms += ms;
    }
    let (aw, an_) = terms(&an, &a_text);
    let (bw, bn) = terms(&an, &b_text);
    println!(
        "{first}: {} characters, {} distinct words, {} distinct numbers, read in {a_ms:.0} ms",
        a_text.len(),
        aw.len(),
        an_.len()
    );
    println!(
        "{} other file(s): {} characters, {} distinct words, {} distinct numbers, read in {b_ms:.0} ms",
        rest.len(),
        b_text.len(),
        bw.len(),
        bn.len()
    );
    println!(
        "words of the others found in the first: {:.3}; words of the first found in the others: {:.3}",
        share(&bw, &aw),
        share(&aw, &bw)
    );
    println!(
        "numbers of the others found in the first: {:.3}; numbers of the first found in the others: {:.3}",
        share(&bn, &an_),
        share(&an_, &bn)
    );
    let mut missing: Vec<&String> = bw.iter().filter(|t| !aw.contains(*t)).collect();
    missing.sort();
    println!(
        "examples of words only in the others: {}",
        missing
            .iter()
            .take(25)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}
