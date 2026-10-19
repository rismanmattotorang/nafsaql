//! Answer extraction and normalisation shared by the verifiers and the gate's agreement feature.
use regex::Regex;
use std::sync::OnceLock;

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).unwrap())
}

/// Make common LaTeX plain: \frac{a}{b} -> (a)/(b), \sqrt{x} -> sqrt(x), \times -> *, Unicode minus -> '-'.
pub fn latex_to_plain(s: &str) -> String {
    static FRAC: OnceLock<Regex> = OnceLock::new();
    static SQRT: OnceLock<Regex> = OnceLock::new();
    static TEXT: OnceLock<Regex> = OnceLock::new();
    let mut x = s
        .replace('\u{2212}', "-")
        .replace("\\!", "")
        .replace("\\,", "")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\%", "%")
        .replace("\\$", "$");
    // Innermost-first: each pass rewrites macros whose arguments contain no braces, so nesting such as
    // \frac{3\sqrt{2}}{2} resolves over successive passes.
    for _ in 0..6 {
        let before = x.clone();
        x = re(&SQRT, r"\\sqrt\{([^{}]*)\}")
            .replace_all(&x, "sqrt($1)")
            .to_string();
        x = re(
            &TEXT,
            r"\\(?:text|mathrm|mathbf|textbf|operatorname|boxed)\{([^{}]*)\}",
        )
        .replace_all(&x, "$1")
        .to_string();
        x = re(&FRAC, r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
            .replace_all(&x, "($1)/($2)")
            .to_string();
        if x == before {
            break;
        }
    }
    x.replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace(['×', '·'], "*")
        .replace("\\pi", "pi")
}

/// Code to verify. Prefers the last fenced block that defines `target` (if given); otherwise the
/// last python block with a `def`; otherwise all python blocks concatenated; otherwise the last block.
pub fn extract_code(text: &str, target: Option<&str>) -> Option<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    let r = re(&R, r"(?s)```([a-zA-Z0-9_+-]*)[^\n]*\n(.*?)```");
    let blocks: Vec<(String, String)> = r
        .captures_iter(text)
        .map(|c| (c[1].to_lowercase(), c[2].to_string()))
        .collect();
    if blocks.is_empty() {
        return None;
    }
    let is_py = |l: &str| l == "python" || l == "py" || l.is_empty();
    if let Some(t) = target {
        let def = format!("def {t}(");
        if let Some((_, b)) = blocks.iter().rev().find(|(_, b)| b.contains(&def)) {
            // keep helper/import-only python blocks that precede it
            let helpers: Vec<&str> = blocks
                .iter()
                .filter(|(l, x)| {
                    is_py(l)
                        && !x.contains(&def)
                        && !x.contains("assert ")
                        && !x.contains("def test_")
                })
                .map(|(_, x)| x.as_str())
                .collect();
            return Some(if helpers.is_empty() {
                b.clone()
            } else {
                format!("{}\n{}", helpers.join("\n"), b)
            });
        }
    }
    if let Some((_, b)) = blocks
        .iter()
        .rev()
        .find(|(l, b)| is_py(l) && b.contains("def ") && !b.trim_start().starts_with("def test_"))
    {
        return Some(b.clone());
    }
    let py: Vec<&str> = blocks
        .iter()
        .filter(|(l, _)| is_py(l))
        .map(|(_, b)| b.as_str())
        .collect();
    if !py.is_empty() {
        return Some(py.join("\n"));
    }
    blocks.last().map(|(_, b)| b.clone())
}

/// JSON value from a ```json block, else the first balanced object/array that parses, else the whole text.
pub fn extract_json(text: &str) -> Option<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    for c in re(&R, r"(?s)```(?:json)?[^\n]*\n(.*?)```").captures_iter(text) {
        let s = c[1].trim();
        if serde_json::from_str::<serde_json::Value>(s).is_ok() {
            return Some(s.to_string());
        }
    }
    // Otherwise the largest balanced object/array that parses (so a citation like "[1]" never wins).
    let chars: Vec<char> = text.chars().collect();
    let mut best: Option<String> = None;
    let mut start = 0;
    while let Some(off) = chars[start..].iter().position(|&ch| ch == '{' || ch == '[') {
        let s0 = start + off;
        if let Some(end) = balanced_end(&chars, s0) {
            let cand: String = chars[s0..=end].iter().collect();
            if serde_json::from_str::<serde_json::Value>(&cand).is_ok()
                && best.as_ref().map(|b| cand.len() > b.len()).unwrap_or(true)
            {
                best = Some(cand);
            }
        }
        start = s0 + 1;
    }
    if best.is_some() {
        return best;
    }
    let t = text.trim();
    serde_json::from_str::<serde_json::Value>(t)
        .ok()
        .map(|_| t.to_string())
}

fn balanced_end(chars: &[char], start: usize) -> Option<usize> {
    let (open, close) = if chars[start] == '{' {
        ('{', '}')
    } else {
        ('[', ']')
    };
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, &ch) in chars.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false
            } else if ch == '\\' {
                esc = true
            } else if ch == '"' {
                in_str = false
            }
            continue;
        }
        if ch == '"' {
            in_str = true
        } else if ch == open {
            depth += 1
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Numbers in text. A '-' counts as a minus sign only at the start or after a non-alphanumeric
/// character, so ranges (10-12) and dates (2024-05-01) are not read as negatives.
pub fn numbers(text: &str) -> Vec<String> {
    static NUM: OnceLock<Regex> = OnceLock::new();
    let t = text.replace('\u{2212}', "-");
    re(&NUM, r"-?\$?\d[\d,]*(?:\.\d+)?")
        .find_iter(&t)
        .map(|m| {
            let s = m.as_str();
            let prev = t[..m.start()].chars().last();
            let s = if s.starts_with('-')
                && prev
                    .map(|c| c.is_alphanumeric() || c == ')' || c == '.')
                    .unwrap_or(false)
            {
                &s[1..]
            } else {
                s
            };
            s.trim_end_matches(',').to_string()
        })
        .collect()
}

/// Final answer: \boxed{...}, "#### x", "Answer: ..." (value after the last '='), or the last number.
pub fn extract_final_answer(text: &str) -> Option<String> {
    static BOX: OnceLock<Regex> = OnceLock::new();
    static ANS: OnceLock<Regex> = OnceLock::new();
    static HASH: OnceLock<Regex> = OnceLock::new();
    if let Some(c) = re(&BOX, r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
        .captures_iter(text)
        .last()
    {
        return Some(latex_to_plain(c[1].trim()));
    }
    if let Some(c) = re(&HASH, r"####\s*([^\n]+)").captures_iter(text).last() {
        return Some(c[1].trim().to_string());
    }
    if let Some(c) = re(
        &ANS,
        r"(?i)(?:final answer|the answer is|answer\s*:)\s*[:=]?\s*\**\s*([^\n]+)",
    )
    .captures_iter(text)
    .last()
    {
        let line = latex_to_plain(c[1].trim().trim_end_matches(['.', '*']));
        let part = line.rsplit('=').next().unwrap_or(&line).trim().to_string();
        // Keep symbolic expressions whole (3*sqrt(2)/2, 2^10, (1)/(3)); never reduce them to a digit.
        static EXPR: OnceLock<Regex> = OnceLock::new();
        if re(
            &EXPR,
            r"sqrt|\bpi\b|\^|\*\*|\blog|\bln\b|[\d)]\s*[/*]\s*[\d(a-z]|\d\s*sqrt|\d\s*pi",
        )
        .is_match(&part)
            && part.len() <= 80
        {
            return Some(part);
        }
        if let Some(n) = numbers(&part).into_iter().next() {
            return Some(n);
        }
        if let Some(n) = numbers(&line).into_iter().last() {
            return Some(n);
        }
        return Some(part);
    }
    numbers(&latex_to_plain(text)).into_iter().last()
}

/// Normalised answer key used for agreement among samples (gate feature) and consensus.
pub fn normalize_answer(text: &str) -> String {
    if let Some(n) = extract_final_answer(text)
        .as_deref()
        .and_then(crate::parse_number)
    {
        if text.len() < 4000 && looks_numeric_answer(text) {
            return format!("num:{}", (n * 1e6).round() / 1e6);
        }
    }
    let lowered: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    format!(
        "txt:{}",
        lowered
            .split_whitespace()
            .take(24)
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// Agreement between two answers in [0, 1]: exact match of numeric final answers; otherwise the
/// Jaccard overlap of content words (so paraphrases of the same answer agree).
pub fn answer_similarity(a: &str, b: &str) -> f32 {
    let (ka, kb) = (normalize_answer(a), normalize_answer(b));
    if ka.starts_with("num:") || kb.starts_with("num:") {
        return (ka == kb) as i32 as f32;
    }
    const STOP: &[&str] = &[
        "the", "and", "for", "are", "but", "not", "you", "with", "this", "that", "from", "have",
        "has", "was", "were", "its", "can", "will", "into", "than", "then", "they", "their",
        "there", "which", "what", "when", "also", "such", "use", "one", "may", "more", "most",
    ];
    let words = |t: &str| -> std::collections::HashSet<String> {
        t.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !STOP.contains(w))
            .take(80)
            .map(String::from)
            .collect()
    };
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() && wb.is_empty() {
        return 1.0;
    }
    let union = wa.union(&wb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        wa.intersection(&wb).count() as f32 / union
    }
}

fn looks_numeric_answer(text: &str) -> bool {
    text.contains("\\boxed")
        || text.contains("####")
        || text.to_lowercase().contains("answer")
        || text.trim().len() < 60
}
