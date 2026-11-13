//! A score of the PDF reader against the TeX source of the same document (Phase 28). `parse_agreement` counts distinct
//! words, which a reader that interleaves two columns or breaks every line still gets right; this one also scores the
//! order and the structure:
//!
//! - `order`: the share of the source's runs of five words that the PDF reading also holds as five consecutive words
//!   (interleaved columns, a header inside a sentence, a hyphen left in a word each break the runs they touch);
//! - `junk`: the share of the PDF reading's runs of five words found nowhere in the source (running headers, page
//!   numbers and column text pushed into sentences make these);
//! - `headings`: the share of the source's `\section` and `\subsection` titles that the PDF reading marks as a heading.
//!
//! Usage:
//!
//!     cargo run --release -p nafs-kb --example parse_score -- \
//!         docs/report/nafs_aql_report.pdf=docs/report/nafs_aql_report.tex,docs/report/sections/*.tex ...
//!
//! Each argument is `pdf=source[,source...]`; globs are expanded by the shell before the comma split, so pass the
//! sources of one document comma-separated. `--json <file>` writes the rows.

use std::collections::HashSet;
use std::time::Instant;

use nafs_kb::{parse, Kind};
use serde_json::json;

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3 && w.chars().all(char::is_alphabetic))
        .map(str::to_lowercase)
        .collect()
}

fn grams(w: &[String], n: usize) -> HashSet<String> {
    w.windows(n).map(|g| g.join(" ")).collect()
}

fn share(of: &HashSet<String>, within: &HashSet<String>) -> f64 {
    if of.is_empty() {
        return 1.0;
    }
    of.iter().filter(|g| within.contains(*g)).count() as f64 / of.len() as f64
}

/// `\section{Title}` and `\subsection*{Title}` titles of a TeX source, as words.
fn tex_titles(src: &str) -> Vec<Vec<String>> {
    let re = regex::Regex::new(r"\\(?:sub)?section\*?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
        .expect("valid regex");
    re.captures_iter(src)
        .map(|c| words(&c[1]))
        .filter(|w| !w.is_empty())
        .collect()
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = None;
    if let Some(i) = args.iter().position(|a| a == "--json") {
        out = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
    }
    let mut rows = Vec::new();
    for arg in &args {
        let (pdf, srcs) = arg
            .split_once('=')
            .ok_or("argument is pdf=source[,source]")?;
        let bytes = std::fs::read(pdf).map_err(|e| format!("{pdf}: {e}"))?;
        let t = Instant::now();
        let parsed = parse(Kind::Pdf, &bytes).map_err(|e| format!("{pdf}: {e}"))?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut tex = String::new();
        let mut raw = String::new();
        for s in srcs.split(',') {
            let src = std::fs::read_to_string(s).map_err(|e| format!("{s}: {e}"))?;
            tex.push_str(&nafs_kb::parse::parse_tex(&src).text);
            tex.push('\n');
            raw.push_str(&src);
            raw.push('\n');
        }
        let (pw, tw) = (words(&parsed.text), words(&tex));
        let (pg, tg) = (grams(&pw, 5), grams(&tw, 5));
        let order = share(&tg, &pg);
        let junk = 1.0 - share(&pg, &tg);
        let heading_lines: Vec<HashSet<String>> = parsed
            .text
            .lines()
            .filter(|l| l.trim_start().starts_with('#'))
            .map(|l| words(l).into_iter().collect())
            .collect();
        let titles = tex_titles(&raw);
        let found = titles
            .iter()
            .filter(|t| {
                heading_lines
                    .iter()
                    .any(|h| t.iter().all(|w| h.contains(w)))
            })
            .count();
        let headings = if titles.is_empty() {
            1.0
        } else {
            found as f64 / titles.len() as f64
        };
        let pages = parsed.pages.unwrap_or(1).max(1);
        println!(
            "{pdf}: pages {pages}, order {order:.3}, junk {junk:.3}, headings {found}/{} = {headings:.3}, \
             {} heading lines, {ms:.0} ms ({:.1} ms/page)",
            titles.len(),
            heading_lines.len(),
            ms / pages as f64
        );
        rows.push(
            json!({"pdf": pdf, "pages": pages, "order": order, "junk": junk, "headings": headings,
            "titles": titles.len(), "heading_lines": heading_lines.len(), "ms": ms}),
        );
    }
    let n = rows.len().max(1) as f64;
    let mean = |k: &str| {
        rows.iter()
            .map(|r| r[k].as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n
    };
    println!(
        "mean: order {:.3}, junk {:.3}, headings {:.3}",
        mean("order"),
        mean("junk"),
        mean("headings")
    );
    if let Some(path) = out {
        let doc = json!({"rows": rows, "mean": {"order": mean("order"), "junk": mean("junk"), "headings": mean("headings")}});
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap())
            .map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}
