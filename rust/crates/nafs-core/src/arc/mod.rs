//! Reasoning cache (Phase 38), after CPU-ARC (Belcak et al., NVIDIA, arXiv 2611.09891).
//!
//! Once System 2 has reasoned a question through, the same reasoning answers every variant that differs only in
//! wording or numbers. The cache keeps that reasoning as **code**: System 2 (the builder) turns its own answer into
//! parameterised Python *semantic functions* that name their inputs, outputs and callees by description. On a later
//! question the cache is tried before any model:
//!
//! 1. **Entry search** ([`ArcStore::lookup`]): the stored template as a pattern (no model), then the question's
//!    shape with its numbers masked (no model), then the meaning vector from the embedder already running.
//! 2. **Parameter extraction**: the template's captures, or the numbers by position, or System 1 reading the values
//!    out of the question (one short JSON reply). Every value must occur in the question, and every number in the
//!    question must be used, or the request is a miss.
//! 3. **Dispatch** ([`ArcStore::invoke`]): the entry's code runs in the verifier sandbox on CPU, in about a tenth of
//!    a second.
//! 4. On a miss, an error, or a disagreement with a verifier, the request falls through to the models.
//!
//! An entry is served only after it has agreed with the models' own answer `trust_after` times (shadow runs on
//! live traffic, after vCache and Krites); a disagreement on a checked answer quarantines it. A mature cache
//! compiles into one standard-library Python module ([`ArcStore::compile`]) that answers template matches with no
//! model at all.

use crate::backend::{Backend, CancelFlag, GenRequest};
use crate::config::ArcCfg;
use nafs_types::Message;
use nafs_types::{now_utc, Rfc3339Ext};
use nafs_verify::extract::{extract_final_answer, latex_to_plain};
use nafs_verify::plain_numeric;
use nafs_verify::sandbox::SandboxLimits;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

pub const RUNTIME: &str = include_str!("runtime.py");
pub const TOOLS: &str = include_str!("compile.py");
pub const BUILDER_PROMPT: &str = include_str!("builder_prompt.txt");

/// The one Python file the sandbox runs: the runtime, the describe/compile tools, and the call to `main`.
pub fn module_source() -> String {
    format!("{RUNTIME}\n\n{TOOLS}\n\nmain()\n")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Param {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub description: String,
    /// The value in the question the entry was built from.
    pub value: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Built and tested, not yet agreed with live answers: run beside the models, never served.
    Candidate,
    /// Served on a hit.
    Trusted,
    /// Disagreed with a checked answer: kept for inspection, never run.
    Quarantined,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Stats {
    /// Lookups that found this entry.
    pub hits: u64,
    /// Hits answered from the cache.
    pub served: u64,
    /// Hits run beside the models to compare.
    pub shadow_runs: u64,
    pub agreements: u64,
    pub disagreements: u64,
    /// Runs that failed (conversion, exception, missing value): each fell through to the models.
    pub errors: u64,
    pub last_hit: Option<String>,
    /// Lowest meaning-vector similarity at which this entry agreed with the models (vCache-style floor).
    pub min_agreed: Option<f32>,
    /// Highest similarity at which it disagreed: the floor for the meaning tier stays above it.
    pub max_disagreed: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FnInfo {
    pub name: String,
    pub description: String,
    pub root: bool,
    pub loads: Vec<String>,
    pub calls: Vec<String>,
    pub returns: Vec<String>,
    pub narrates: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub created_at: String,
    pub domain: String,
    /// The question the entry was built from, q(o).
    pub question: String,
    /// The question with `{name}` in place of each parameter, q(.).
    pub template: String,
    pub params: Vec<Param>,
    /// The answer the code had to reproduce at build time.
    pub expected: String,
    pub code: String,
    pub functions: Vec<FnInfo>,
    /// Name of the root function in `code`.
    pub root: String,
    /// `do()` descriptions resolved at registration: description -> "entry:function".
    #[serde(default)]
    pub links: BTreeMap<String, String>,
    pub status: Status,
    #[serde(default)]
    pub stats: Stats,
    /// Build record: rounds, time, tokens, checks.
    #[serde(default)]
    pub build: Value,
    /// The narration of the run on the original question.
    #[serde(default)]
    pub narration: Vec<String>,
    #[serde(default)]
    pub source_reply: Option<String>,
    /// Meaning vectors of the entry points (the original question). Not part of the readable record.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vectors: Vec<Vec<f32>>,
    /// Meaning vectors of the function descriptions, in `functions` order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fn_vectors: Vec<Vec<f32>>,
}

impl Entry {
    /// The readable summary the list endpoint returns.
    pub fn summary(&self) -> Value {
        json!({
            "id": self.id, "created_at": self.created_at, "domain": self.domain, "question": self.question,
            "template": self.template, "params": self.params, "status": self.status, "stats": self.stats,
            "functions": self.functions.iter().map(|f| json!({"name": f.name, "description": f.description, "root": f.root})).collect::<Vec<_>>(),
            "reused": self.links.values().filter(|v| !v.starts_with(&format!("{}:", self.id))).count(),
        })
    }
    pub fn root_id(&self) -> String {
        format!("{}:{}", self.id, self.root)
    }
}

/// A question waiting for the builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildJob {
    pub question: String,
    /// The answer System 2 gave (its reply text).
    pub answer: String,
    #[serde(default)]
    pub reasoning: String,
    /// A reference answer when the request carried one (it wins over the reply's own).
    #[serde(default)]
    pub expected: Option<String>,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub source_reply: Option<String>,
    #[serde(default)]
    pub attempts: u32,
}

/// How an entry was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// The template matched as a pattern: values are its captures.
    Template,
    /// Same words, other numbers: values by position.
    Shape,
    /// Meaning vector, then System 1 extraction.
    Meaning,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub entry: Entry,
    pub tier: Tier,
    pub similarity: f32,
    /// Values by parameter description, when the tier provides them.
    pub params: Option<BTreeMap<String, String>>,
}

/// Result of running an entry.
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub answer: String,
    pub summary: Option<String>,
    pub outputs: Value,
    pub narration: Vec<String>,
    pub ms: u64,
}

pub struct ArcStore {
    path: PathBuf,
    entries: RwLock<Vec<Entry>>,
    queue: Mutex<VecDeque<BuildJob>>,
    /// Serialises writes of the file.
    write: Mutex<()>,
    pub limits: SandboxLimits,
}

fn now() -> String {
    now_utc().to_rfc3339()
}

// ---------------------------------------------------------------- text shape (no model)

fn num_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?|[-+]?\.\d+").unwrap()
    })
}

/// Whitespace collapsed, ends trimmed.
pub fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The numbers written in `s`, in order (thousands separators kept as written).
pub fn numbers(s: &str) -> Vec<String> {
    num_re()
        .find_iter(s)
        .map(|m| m.as_str().trim().to_string())
        .collect()
}

fn num_value(s: &str) -> Option<f64> {
    let t = s.replace([',', ' '], "");
    if let Some((a, b)) = t.split_once('/') {
        let (a, b) = (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?);
        return (b != 0.0).then(|| a / b);
    }
    t.parse::<f64>().ok()
}

/// The question's shape: lower case, numbers masked, punctuation at the end dropped.
pub fn shape(s: &str) -> String {
    let n = normalize(s).to_lowercase();
    let masked = num_re().replace_all(&n, "#");
    masked
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace())
        .to_string()
}

fn is_numeric_type(t: &str) -> bool {
    matches!(t, "int" | "float" | "Fraction")
}

/// The template as an anchored pattern: literal text with flexible spaces, one capture per `{name}`.
pub fn template_regex(template: &str, params: &[Param]) -> Option<Regex> {
    static PH: OnceLock<Regex> = OnceLock::new();
    let ph = PH.get_or_init(|| Regex::new(r"\{(\w+)\}").unwrap());
    let t = normalize(template);
    let t = t.trim_end_matches(['?', '.', '!']);
    let mut out = String::from(r"(?is)^\s*");
    let mut last = 0;
    let mut seen = 0;
    for c in ph.captures_iter(t) {
        let m = c.get(0).unwrap();
        let lit = &t[last..m.start()];
        out.push_str(&literal(lit));
        let name = &c[1];
        let ty = params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.ty.as_str())?;
        out.push_str(if is_numeric_type(ty) {
            r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?|[-+]?\.\d+)"
        } else {
            r"(.+?)"
        });
        last = m.end();
        seen += 1;
    }
    if seen == 0 {
        return None;
    }
    out.push_str(&literal(&t[last..]));
    out.push_str(r"\s*[?.!]?\s*$");
    Regex::new(&out).ok()
}

fn literal(lit: &str) -> String {
    let mut s = String::new();
    let mut first = true;
    for (i, w) in lit.split(' ').enumerate() {
        if i > 0 {
            s.push_str(r"\s+");
        }
        if w.is_empty() {
            continue;
        }
        first = false;
        s.push_str(&regex::escape(w));
    }
    let _ = first;
    s
}

/// Values of `e`'s parameters read from `q` by its template, by description.
pub fn match_template(e: &Entry, q: &str) -> Option<BTreeMap<String, String>> {
    if !template_is_specific(&e.template) {
        return None;
    }
    let re = template_regex(&e.template, &e.params)?;
    let qn = normalize(q);
    let qn = qn.trim_end_matches(['?', '.', '!']);
    let caps = re.captures(qn)?;
    static PH: OnceLock<Regex> = OnceLock::new();
    let ph = PH.get_or_init(|| Regex::new(r"\{(\w+)\}").unwrap());
    let names: Vec<String> = ph
        .captures_iter(&e.template)
        .map(|c| c[1].to_string())
        .collect();
    let mut by_name: BTreeMap<String, String> = BTreeMap::new();
    for (i, n) in names.iter().enumerate() {
        let v = caps.get(i + 1)?.as_str().trim().to_string();
        // A name used twice must capture the same text twice.
        if let Some(prev) = by_name.get(n) {
            if prev != &v {
                return None;
            }
        }
        by_name.insert(n.clone(), v);
    }
    let mut out = BTreeMap::new();
    for p in &e.params {
        out.insert(
            p.description.clone(),
            by_name
                .get(&p.name)
                .cloned()
                .unwrap_or_else(|| p.value.clone()),
        );
    }
    Some(out)
}

/// Same words, other numbers: numeric parameters take the new question's numbers by their position in the original.
pub fn match_shape(e: &Entry, q: &str) -> Option<BTreeMap<String, String>> {
    if shape(q) != shape(&e.question) {
        return None;
    }
    let old = numbers(&e.question);
    let new = numbers(q);
    if old.len() != new.len() {
        return None;
    }
    let mut out = BTreeMap::new();
    for p in &e.params {
        let v = if is_numeric_type(&p.ty) {
            let pos = old.iter().position(|x| {
                x == &p.value
                    || num_value(x)
                        .zip(num_value(&p.value))
                        .is_some_and(|(a, b)| a == b)
            })?;
            new[pos].clone()
        } else {
            p.value.clone()
        };
        out.insert(p.description.clone(), v);
    }
    Some(out)
}

/// A value is in the question when its text is (case and spacing ignored) or, for a number, when the question
/// writes that number.
pub fn verbatim(q: &str, value: &str) -> bool {
    let v = normalize(value).to_lowercase();
    if v.is_empty() {
        return false;
    }
    let qn = normalize(q).to_lowercase();
    if qn.contains(&v) {
        return true;
    }
    match num_value(&v) {
        Some(x) => numbers(q)
            .iter()
            .any(|n| num_value(n).is_some_and(|y| (x - y).abs() <= 1e-12 * x.abs().max(1.0))),
        None => false,
    }
}

/// Numbers written in the question that no parameter value carries and that the template does not write itself: the
/// question holds information the code would ignore, so it is not the same problem.
pub fn unused_numbers(e: &Entry, q: &str, values: &BTreeMap<String, String>) -> Vec<String> {
    let fixed: Vec<f64> = numbers(&e.template)
        .iter()
        .filter_map(|n| num_value(n))
        .collect();
    let used: Vec<f64> = values
        .values()
        .flat_map(|v| numbers(v))
        .filter_map(|n| num_value(&n))
        .collect();
    numbers(q)
        .into_iter()
        .filter(|n| {
            let x = match num_value(n) {
                Some(x) => x,
                None => return false,
            };
            !used
                .iter()
                .chain(fixed.iter())
                .any(|y| (x - y).abs() <= 1e-12 * x.abs().max(1.0))
        })
        .collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// The final answer of a reply: the boxed or "Answer:" value, else the text.
pub fn answer_of(text: &str) -> String {
    extract_final_answer(text).unwrap_or_else(|| text.trim().to_string())
}

fn decimals(s: &str) -> u32 {
    let t = latex_to_plain(s);
    let n = numbers(&t);
    n.last()
        .and_then(|x| x.split_once('.'))
        .map(|(_, d)| d.chars().take_while(|c| c.is_ascii_digit()).count() as u32)
        .unwrap_or(0)
}

/// Whether the cache's answer and a model's answer agree: numbers equal to the precision the less precise one was
/// written with (a model writes 5.13 where the code computes 5.1321), otherwise the same text.
pub fn same_answer(cache: &str, reference: &str) -> bool {
    let want = answer_of(reference);
    let got = latex_to_plain(cache.trim());
    let wantp = latex_to_plain(&want);
    match (plain_numeric(&got), plain_numeric(&wantp)) {
        (Some(a), Some(b)) => {
            let d = decimals(&got).min(decimals(&wantp));
            let unit = 0.5 * 10f64.powi(-(d as i32));
            let tol = (1e-6 * b.abs().max(1.0)).max(if decimals(&got) == decimals(&wantp) {
                1e-9
            } else {
                unit
            });
            (a - b).abs() <= tol
        }
        _ => {
            let canon = |x: &str| {
                x.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || matches!(c, '/' | '-' | '.' | ','))
                    .collect::<String>()
                    .trim_end_matches('.')
                    .to_string()
            };
            !got.trim().is_empty() && canon(&got) == canon(&wantp)
        }
    }
}

// ---------------------------------------------------------------- builder output

/// The three sections of the builder's reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    pub template: String,
    pub params: Vec<Param>,
    pub code: String,
}

pub fn parse_draft(text: &str) -> Result<Draft, String> {
    let code = {
        static CODE: OnceLock<Regex> = OnceLock::new();
        let re = CODE.get_or_init(|| Regex::new(r"(?s)```(?:python|py)?\s*\n(.*?)```").unwrap());
        re.captures_iter(text)
            .map(|c| c[1].to_string())
            .filter(|c| c.contains("@semantic"))
            .last()
            .ok_or("the reply has no ```python block with @semantic functions")?
    };
    let head = text.split("```").next().unwrap_or(text);
    let section = |name: &str| -> Option<String> {
        let up = head.to_string();
        let i = up.rfind(&format!("{name}:"))?;
        let rest = &up[i + name.len() + 1..];
        let end = ["TEMPLATE:", "PARAMS:", "CODE:"]
            .iter()
            .filter_map(|s| rest.find(s))
            .min()
            .unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    };
    let template = section("TEMPLATE")
        .map(|t| {
            t.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|t| !t.is_empty())
        .ok_or("the reply has no TEMPLATE section")?;
    let ptext = section("PARAMS").ok_or("the reply has no PARAMS section")?;
    let mut params = vec![];
    for line in ptext.lines() {
        let l = line.trim().trim_start_matches(['-', '*', ' ']);
        let parts: Vec<&str> = l.split('|').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        let name: String = parts[0]
            .trim_matches(['`', '{', '}'])
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() || name.eq_ignore_ascii_case("name") {
            continue;
        }
        let ty = parts[1].trim_matches('`').to_string();
        let ty = match ty.to_lowercase().as_str() {
            "int" | "integer" => "int",
            "float" | "number" | "real" | "decimal" => "float",
            "fraction" => "Fraction",
            "bool" | "boolean" => "bool",
            "list" => "list",
            _ => "str",
        }
        .to_string();
        let value = parts[3..]
            .join("|")
            .trim()
            .trim_matches(['`', '"'])
            .to_string();
        params.push(Param {
            name,
            ty,
            description: parts[2].trim_matches('"').to_string(),
            value,
        });
    }
    if params.is_empty() {
        return Err("PARAMS lists no input as name | type | description | value".into());
    }
    Ok(Draft {
        template,
        params,
        code,
    })
}

/// Whether a template is mostly fixed text: at least three words and half its characters outside the placeholders. A
/// template that is all placeholders (a text input holding the whole question) would match any question.
pub fn template_is_specific(template: &str) -> bool {
    static PH: OnceLock<Regex> = OnceLock::new();
    let ph = PH.get_or_init(|| Regex::new(r"\{(\w+)\}").unwrap());
    let t = normalize(template);
    let fixed = ph.replace_all(&t, " ");
    let fixed = normalize(&fixed);
    let words = fixed
        .split(' ')
        .filter(|w| w.chars().any(char::is_alphabetic))
        .count();
    words >= 3 && fixed.chars().count() * 2 >= t.chars().count()
}

/// The template filled with the declared values must be the question (spacing and end punctuation aside), every value
/// must be in the question, and every placeholder must be declared.
pub fn check_template(question: &str, d: &Draft) -> Result<(), String> {
    if !template_is_specific(&d.template) {
        return Err("the template is mostly placeholders: keep the question's own words and replace only the values".into());
    }
    let mut filled = d.template.clone();
    for p in &d.params {
        if !d.template.contains(&format!("{{{}}}", p.name)) {
            return Err(format!(
                "the template has no {{{}}} although PARAMS declares it",
                p.name
            ));
        }
        if !verbatim(question, &p.value) {
            return Err(format!(
                "the value \"{}\" of {} is not written in the question; copy values exactly",
                p.value, p.name
            ));
        }
        filled = filled.replace(&format!("{{{}}}", p.name), &p.value);
    }
    static PH: OnceLock<Regex> = OnceLock::new();
    let ph = PH.get_or_init(|| Regex::new(r"\{(\w+)\}").unwrap());
    if let Some(c) = ph.captures(&filled) {
        return Err(format!(
            "{{{}}} is in the template but not in PARAMS",
            &c[1]
        ));
    }
    let a = shape(&filled);
    let b = shape(question);
    if a != b
        || normalize(&filled)
            .to_lowercase()
            .trim_end_matches(['?', '.', '!'])
            != normalize(question)
                .to_lowercase()
                .trim_end_matches(['?', '.', '!'])
    {
        return Err(format!(
            "the template filled with the values is not the question.\nFilled: {}\nQuestion: {}",
            normalize(&filled),
            normalize(question)
        ));
    }
    Ok(())
}

/// A template built from the question itself: each declared value replaced by its placeholder, where the value is
/// written exactly once (numbers as whole numbers, not inside another number). `None` when any value is missing or
/// ambiguous. The builder often rewords its TEMPLATE; this keeps the question's own words.
pub fn synthesize_template(question: &str, params: &[Param]) -> Option<String> {
    let q = normalize(question);
    let mut spans: Vec<(usize, usize, String)> = vec![];
    for p in params {
        let hits: Vec<(usize, usize)> = if is_numeric_type(&p.ty) {
            let want = num_value(&p.value)?;
            num_re()
                .find_iter(&q)
                .filter(|m| {
                    num_value(m.as_str())
                        .is_some_and(|x| (x - want).abs() <= 1e-12 * want.abs().max(1.0))
                })
                .map(|m| (m.start(), m.end()))
                .collect()
        } else {
            let v = normalize(&p.value);
            if v.is_empty() {
                return None;
            }
            let lq = q.to_lowercase();
            let lv = v.to_lowercase();
            lq.match_indices(&lv)
                .map(|(i, _)| (i, i + lv.len()))
                .collect()
        };
        if hits.len() != 1 {
            return None;
        }
        spans.push((hits[0].0, hits[0].1, p.name.clone()));
    }
    spans.sort();
    if spans.windows(2).any(|w| w[0].1 > w[1].0) {
        return None;
    }
    let mut out = String::new();
    let mut last = 0;
    for (a, b, n) in spans {
        out.push_str(&q[last..a]);
        out.push_str(&format!("{{{n}}}"));
        last = b;
    }
    out.push_str(&q[last..]);
    Some(out)
}

// ---------------------------------------------------------------- extraction with System 1

pub fn extraction_messages(e: &Entry, q: &str) -> Vec<Message> {
    let mut inputs = String::new();
    let mut example = serde_json::Map::new();
    example.insert("same".into(), json!(true));
    for p in &e.params {
        inputs.push_str(&format!("- {} ({}): {}\n", p.name, p.ty, p.description));
        example.insert(p.name.clone(), json!(p.value));
    }
    let user = format!(
        "A program answers questions like the example below. Its inputs:\n{inputs}\nExample question: {}\nExample reply: {}\n\nRead the inputs out of the new question, copying each value exactly as it is written there. Set \"same\" to false when the new question asks for something else, or needs information these inputs do not hold.\n\nNew question: {}\nReply:",
        normalize(&e.question),
        Value::Object(example),
        normalize(q)
    );
    vec![
        Message {
            role: "system".into(),
            content:
                "You read values out of a question. Reply with one JSON object and nothing else."
                    .into(),
        },
        Message {
            role: "user".into(),
            content: user,
        },
    ]
}

/// System 1's reply as values by description; `None` when it says the question is another one or is unreadable.
pub fn parse_extraction(e: &Entry, text: &str) -> Result<BTreeMap<String, String>, String> {
    let body =
        nafs_verify::extract::extract_json(text).ok_or("System 1 did not reply with JSON")?;
    let v: Value = serde_json::from_str(&body).map_err(|e| format!("System 1's JSON: {e}"))?;
    if v.get("same").and_then(Value::as_bool) == Some(false) {
        return Err("System 1 read it as a different question".into());
    }
    let mut out = BTreeMap::new();
    for p in &e.params {
        let x = v
            .get(&p.name)
            .or_else(|| v.get(&p.description))
            .ok_or_else(|| format!("System 1 gave no value for {}", p.name))?;
        let s = match x {
            Value::String(s) => s.clone(),
            Value::Null => return Err(format!("System 1 gave no value for {}", p.name)),
            other => other.to_string(),
        };
        out.insert(p.description.clone(), s);
    }
    Ok(out)
}

/// `preferred` when it is Python 3.9 or newer, else the first of the usual interpreters that is; `preferred` when none
/// is.
pub fn python_at_least_39(preferred: &str) -> String {
    let ok = |p: &str| {
        std::process::Command::new(p)
            .args([
                "-I",
                "-c",
                "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    // Resolved to a full path: the Linux sandbox starts the interpreter by path inside its own mount namespace, where
    // a bare `python3` is not found (seen on the Ubuntu CI runner).
    let preferred = nafs_verify::sandbox::resolve_python(preferred);
    let preferred = preferred.as_str();
    let mut candidates: Vec<String> = vec![preferred.to_string()];
    for c in [
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.9",
    ] {
        candidates.push(nafs_verify::sandbox::resolve_python(c));
    }
    candidates
        .into_iter()
        .find(|c| ok(c))
        .unwrap_or_else(|| preferred.to_string())
}

// ---------------------------------------------------------------- store

impl ArcStore {
    pub fn open(dir: &str, mut limits: SandboxLimits) -> Self {
        // Compilation needs Python 3.9 (`ast.unparse`); the first `python3` on a PATH can be older (a conda base).
        limits.python = python_at_least_39(&limits.python);
        // The runtime answers with one JSON object (a compiled module can be large); the verifier's excerpt limit
        // would cut it.
        limits.max_output_bytes = limits.max_output_bytes.max(4 << 20);
        limits.hard_output_bytes = limits.hard_output_bytes.max(8 << 20);
        let path = PathBuf::from(dir).join("cache.json");
        let entries: Vec<Entry> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let queue: VecDeque<BuildJob> =
            std::fs::read_to_string(PathBuf::from(dir).join("queue.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
        Self {
            path,
            entries: RwLock::new(entries),
            queue: Mutex::new(queue),
            write: Mutex::new(()),
            limits,
        }
    }

    fn dir(&self) -> PathBuf {
        self.path.parent().map(PathBuf::from).unwrap_or_default()
    }

    fn persist(&self) {
        let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = self
            .entries
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let dir = self.dir();
        let _ = std::fs::create_dir_all(&dir);
        let tmp = self.path.with_extension("json.tmp");
        match serde_json::to_vec(&snapshot) {
            Ok(b) => {
                if std::fs::write(&tmp, b).is_ok() {
                    let _ = std::fs::rename(&tmp, &self.path);
                }
            }
            Err(e) => tracing::warn!(error = %e, "reasoning cache not saved"),
        }
        let q = self.queue.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let _ = std::fs::write(
            dir.join("queue.json"),
            serde_json::to_vec(&q).unwrap_or_default(),
        );
    }

    pub fn list(&self) -> Vec<Entry> {
        self.entries
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, id: &str) -> Option<Entry> {
        self.entries
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|e| e.id == id)
            .cloned()
    }
    pub fn remove(&self, id: &str) -> bool {
        let found = {
            let mut g = self.entries.write().unwrap_or_else(|e| e.into_inner());
            let n = g.len();
            g.retain(|e| e.id != id);
            n != g.len()
        };
        if found {
            self.persist();
        }
        found
    }
    pub fn update<F: FnOnce(&mut Entry)>(&self, id: &str, f: F) -> Option<Entry> {
        let out = {
            let mut g = self.entries.write().unwrap_or_else(|e| e.into_inner());
            g.iter_mut().find(|e| e.id == id).map(|e| {
                f(e);
                e.clone()
            })
        };
        if out.is_some() {
            self.persist();
        }
        out
    }
    /// Adds an entry; past `max_entries` the least used one goes (LFU, which a 2026 study found never far from the
    /// best policy for LLM caches).
    pub fn insert(&self, e: Entry, max_entries: usize) {
        {
            let mut g = self.entries.write().unwrap_or_else(|e| e.into_inner());
            g.retain(|x| x.id != e.id);
            g.push(e);
            while max_entries > 0 && g.len() > max_entries {
                // Entries another entry calls are kept.
                let used: BTreeSet<String> = g
                    .iter()
                    .flat_map(|x| {
                        x.links
                            .values()
                            .filter_map(|l| l.split(':').next().map(str::to_string))
                            .filter(|id| id != &x.id)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let victim = g
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| !used.contains(&x.id))
                    .min_by_key(|(_, x)| (x.stats.hits, x.created_at.clone()))
                    .map(|(i, _)| i);
                match victim {
                    Some(i) => {
                        g.remove(i);
                    }
                    None => break,
                }
            }
        }
        self.persist();
    }

    pub fn enqueue(&self, job: BuildJob, cap: usize) -> bool {
        let added = {
            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            if q.iter().any(|j| shape(&j.question) == shape(&job.question)) || q.len() >= cap.max(1)
            {
                false
            } else {
                q.push_back(job);
                true
            }
        };
        if added {
            self.persist();
        }
        added
    }
    pub fn next_job(&self) -> Option<BuildJob> {
        let j = self
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        if j.is_some() {
            self.persist();
        }
        j
    }
    pub fn requeue(&self, job: BuildJob) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_front(job);
        self.persist();
    }
    pub fn queued(&self) -> Vec<BuildJob> {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// The model-free tiers: template pattern, then shape. Quarantined entries are skipped.
    pub fn lookup_text(&self, q: &str) -> Option<Hit> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let live = || g.iter().filter(|e| e.status != Status::Quarantined);
        for e in live() {
            if let Some(p) = match_template(e, q) {
                return Some(Hit {
                    entry: e.clone(),
                    tier: Tier::Template,
                    similarity: 1.0,
                    params: Some(p),
                });
            }
        }
        for e in live() {
            if let Some(p) = match_shape(e, q) {
                return Some(Hit {
                    entry: e.clone(),
                    tier: Tier::Shape,
                    similarity: 1.0,
                    params: Some(p),
                });
            }
        }
        None
    }

    /// The meaning tier: the closest entry by the question's vector, if it clears `floor` and the entry's own floor.
    /// The closest live entry by meaning vector and its similarity, whatever the floor.
    pub fn nearest(&self, qv: &[f32]) -> Option<(String, f32)> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        g.iter()
            .filter(|e| e.status != Status::Quarantined)
            .flat_map(|e| e.vectors.iter().map(move |v| (e.id.clone(), cosine(qv, v))))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn lookup_vector(&self, qv: &[f32], cfg: &ArcCfg) -> Option<Hit> {
        self.lookup_vector_at(qv, cfg, None)
    }

    /// As [`Self::lookup_vector`], with one floor for every entry in place of their learned ones (measurement only).
    pub fn lookup_vector_at(&self, qv: &[f32], cfg: &ArcCfg, floor: Option<f32>) -> Option<Hit> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let mut best: Option<(f32, &Entry)> = None;
        for e in g.iter().filter(|e| e.status != Status::Quarantined) {
            for v in &e.vectors {
                let s = cosine(qv, v);
                if best.map(|(b, _)| s > b).unwrap_or(true) {
                    best = Some((s, e));
                }
            }
        }
        let (s, e) = best?;
        (s >= floor.unwrap_or_else(|| entry_floor(e, cfg))).then(|| Hit {
            entry: e.clone(),
            tier: Tier::Meaning,
            similarity: s,
            params: None,
        })
    }

    /// Runs `e` on `params` (values by description) in the sandbox.
    pub async fn invoke(
        &self,
        e: &Entry,
        params: &BTreeMap<String, String>,
    ) -> Result<Outcome, String> {
        let t = Instant::now();
        let code = self.closure(e);
        let req = json!({"root": e.root_id(), "params": params, "links": e.links, "code": code});
        let v = self.run_tool(&req).await?;
        if v["ok"].as_bool() != Some(true) {
            return Err(v["error"]
                .as_str()
                .unwrap_or("the entry failed")
                .to_string());
        }
        let answer = match &v["answer"] {
            Value::String(s) => s.clone(),
            Value::Null => return Err("the entry returned no answer".into()),
            other => other.to_string(),
        };
        Ok(Outcome {
            answer,
            summary: v["outputs"]["summary"].as_str().map(str::to_string),
            outputs: v["outputs"].clone(),
            narration: v["narration"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            ms: t.elapsed().as_millis() as u64,
        })
    }

    /// The code of `e` and of every entry it calls, as (entry id, code) pairs.
    fn closure(&self, e: &Entry) -> Vec<(String, String)> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let mut out = vec![(e.id.clone(), e.code.clone())];
        let mut seen: BTreeSet<String> = BTreeSet::from([e.id.clone()]);
        let mut todo: Vec<String> = e
            .links
            .values()
            .filter_map(|l| l.split(':').next().map(str::to_string))
            .collect();
        while let Some(id) = todo.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(x) = g.iter().find(|x| x.id == id) {
                out.push((x.id.clone(), x.code.clone()));
                todo.extend(
                    x.links
                        .values()
                        .filter_map(|l| l.split(':').next().map(str::to_string)),
                );
            }
        }
        out
    }

    async fn run_tool(&self, req: &Value) -> Result<Value, String> {
        let src = module_source();
        let stdin = req.to_string();
        let out = nafs_verify::sandbox::run(
            &[("arc.py", src.as_str())],
            "arc.py",
            Some(&stdin),
            &self.limits.python,
            &self.limits,
        )
        .await?;
        if let Some(k) = &out.killed {
            return Err(format!("the sandbox stopped the run ({k})"));
        }
        serde_json::from_str(out.stdout.trim()).map_err(|_| {
            format!(
                "the cache runtime gave no result: {}",
                out.excerpt(400).trim()
            )
        })
    }

    pub async fn describe(&self, code: &str) -> Result<(Vec<FnInfo>, Vec<String>), String> {
        let v = self
            .run_tool(&json!({"mode": "describe", "code": code}))
            .await?;
        if v["ok"].as_bool() != Some(true) {
            return Err(v["error"].as_str().unwrap_or("describe failed").to_string());
        }
        let fns: Vec<FnInfo> = serde_json::from_value(v["functions"].clone()).unwrap_or_default();
        let problems: Vec<String> =
            serde_json::from_value(v["problems"].clone()).unwrap_or_default();
        Ok((fns, problems))
    }

    /// One standard-library module holding every served entry (or every live one with `all`), descriptions resolved.
    /// Returns the module, the number of entries in it, and the entries left out with the reason.
    pub async fn compile(&self, all: bool) -> Result<(String, usize, Value), String> {
        let entries: Vec<Entry> = self
            .list()
            .into_iter()
            .filter(|e| e.status == Status::Trusted || (all && e.status == Status::Candidate))
            .collect();
        if entries.is_empty() {
            return Err(
                "no entry to compile (only trusted entries are compiled unless all is set)".into(),
            );
        }
        let req = json!({"mode": "compile", "runtime": RUNTIME, "entries": entries.iter().map(|e| json!({
            "id": e.id, "code": e.code, "template": e.template, "params": e.params, "links": e.links, "expected": e.expected})).collect::<Vec<_>>()});
        let v = self.run_tool(&req).await?;
        if v["ok"].as_bool() != Some(true) {
            return Err(v["error"].as_str().unwrap_or("compile failed").to_string());
        }
        let skipped = v["skipped"].clone();
        let n = entries.len() - skipped.as_object().map(|o| o.len()).unwrap_or(0);
        Ok((
            v["module"].as_str().unwrap_or_default().to_string(),
            n,
            skipped,
        ))
    }

    /// What a live comparison with the models' answer says about the entry (shadow validation).
    pub fn record_comparison(
        &self,
        id: &str,
        agree: bool,
        checked: bool,
        tier: Tier,
        similarity: f32,
        cfg: &ArcCfg,
    ) -> Option<Entry> {
        self.update(id, |e| {
            if agree {
                e.stats.agreements += 1;
                if tier == Tier::Meaning {
                    e.stats.min_agreed =
                        Some(e.stats.min_agreed.map_or(similarity, |m| m.min(similarity)));
                }
                if e.status == Status::Candidate
                    && e.stats.agreements >= cfg.trust_after as u64
                    && e.stats.disagreements == 0
                {
                    e.status = Status::Trusted;
                }
            } else {
                e.stats.disagreements += 1;
                if tier == Tier::Meaning {
                    e.stats.max_disagreed = Some(
                        e.stats
                            .max_disagreed
                            .map_or(similarity, |m| m.max(similarity)),
                    );
                }
                // A disagreement found by the template or the shape is the code's fault; one from the meaning
                // tier may be a different question, which the raised floor already handles.
                let code_fault = checked && tier != Tier::Meaning;
                let net = e.stats.disagreements >= 2 && e.stats.disagreements > e.stats.agreements;
                if code_fault || net {
                    e.status = Status::Quarantined;
                }
            }
        })
    }

    pub fn note_hit(&self, id: &str, served: bool, error: bool) {
        self.update(id, |e| {
            e.stats.hits += 1;
            e.stats.last_hit = Some(now());
            if error {
                e.stats.errors += 1;
            } else if served {
                e.stats.served += 1;
            } else {
                e.stats.shadow_runs += 1;
            }
        });
    }

    pub fn set_status(&self, id: &str, s: Status) -> Option<Entry> {
        self.update(id, |e| e.status = s)
    }

    /// Existing functions whose descriptions are closest to `qv` (for the builder to reuse), as (id, description).
    pub fn similar_functions(
        &self,
        qv: Option<&[f32]>,
        question: &str,
        k: usize,
    ) -> Vec<(String, String)> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let mut scored: Vec<(f32, String, String)> = vec![];
        let qwords = word_set(question);
        for e in g.iter().filter(|e| e.status != Status::Quarantined) {
            for (i, f) in e.functions.iter().enumerate() {
                let s = match (qv, e.fn_vectors.get(i)) {
                    (Some(q), Some(v)) => cosine(q, v),
                    _ => dice(&qwords, &word_set(&f.description)),
                };
                scored.push((s, format!("{}:{}", e.id, f.name), f.description.clone()));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(_, id, d)| (id, d))
            .collect()
    }

    /// Function id for a `do()` description, by meaning vector (when given) or word overlap, among all live entries.
    pub fn resolve_call(
        &self,
        desc: &str,
        dv: Option<&[f32]>,
        exclude_entry: &str,
        min_cos: f32,
    ) -> Option<(String, f32)> {
        let g = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let dw = word_set(desc);
        let mut best: Option<(f32, String)> = None;
        for e in g
            .iter()
            .filter(|e| e.status != Status::Quarantined && e.id != exclude_entry)
        {
            for (i, f) in e.functions.iter().enumerate() {
                let (s, floor) = match (dv, e.fn_vectors.get(i)) {
                    (Some(q), Some(v)) => (cosine(q, v), min_cos),
                    _ => (dice(&dw, &word_set(&f.description)), 0.6),
                };
                if s >= floor && best.as_ref().map(|(b, _)| s > *b).unwrap_or(true) {
                    best = Some((s, format!("{}:{}", e.id, f.name)));
                }
            }
        }
        best.map(|(s, id)| (id, s))
    }
}

/// The similarity a meaning-tier hit needs for this entry: the configured serving floor, lowered to the lowest
/// similarity it has agreed at (never below `min_similarity`), and kept above the highest it disagreed at.
pub fn entry_floor(e: &Entry, cfg: &ArcCfg) -> f32 {
    let mut f = cfg.serve_similarity as f32;
    if let Some(m) = e.stats.min_agreed {
        f = f.min(m);
    }
    f = f.max(cfg.min_similarity as f32);
    if let Some(d) = e.stats.max_disagreed {
        f = f.max(d + 0.005);
    }
    f
}

fn word_set(s: &str) -> BTreeSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(|w| w.trim_end_matches('s').to_string())
        .collect()
}
fn dice(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    2.0 * a.intersection(b).count() as f32 / (a.len() + b.len()) as f32
}

// ---------------------------------------------------------------- the builder

/// What a build did, for the record and the log.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BuildReport {
    pub ok: bool,
    pub entry: Option<String>,
    pub rounds: u32,
    pub ms: u64,
    pub builder_tokens: u32,
    pub reasons: Vec<String>,
}

fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..10].to_string()
}

/// Turns `job` into an entry with System 2 as the builder. `embed` gives meaning vectors (None: word overlap
/// only). The entry starts as a candidate unless `trust_after` is 0.
pub async fn build(
    store: &ArcStore,
    s2: &Backend,
    s1: Option<&Backend>,
    embed: &(dyn Fn(String) -> futures::future::BoxFuture<'static, Option<Vec<f32>>> + Send + Sync),
    job: &BuildJob,
    cfg: &ArcCfg,
    cancel: Option<CancelFlag>,
) -> BuildReport {
    let t0 = Instant::now();
    let mut rep = BuildReport::default();
    let expected_text = job
        .expected
        .clone()
        .unwrap_or_else(|| answer_of(&job.answer));
    let qv = embed(normalize(&job.question)).await;
    let reuse = store.similar_functions(qv.as_deref(), &job.question, 4);
    let reuse_text = if reuse.is_empty() {
        "none yet".to_string()
    } else {
        reuse
            .iter()
            .map(|(_, d)| format!("- {d}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let reasoning: String = {
        let r = job.reasoning.trim();
        let n = r.chars().count();
        r.chars()
            .skip(n.saturating_sub(cfg.build_reasoning_chars))
            .collect()
    };
    let mut messages = vec![
        Message {
            role: "system".into(),
            content: BUILDER_PROMPT.to_string(),
        },
        Message {
            role: "user".into(),
            content: format!(
                "Question:\n{}\n\nReasoning that solved it (excerpt):\n{}\n\nFinal answer:\n{}\n\nExpected final answer: {}\n\nExisting functions you may call with do() instead of rewriting them:\n{}",
                job.question.trim(),
                if reasoning.is_empty() { "(not kept)" } else { reasoning.as_str() },
                job.answer.trim().chars().take(1500).collect::<String>(),
                expected_text,
                reuse_text
            ),
        },
    ];
    for round in 1..=cfg.build_rounds.max(1) {
        rep.rounds = round;
        let req = GenRequest {
            messages: messages.clone(),
            max_tokens: cfg.build_max_tokens,
            temperature: 0.6,
            top_p: 0.95,
            thinking: Some(true),
            thinking_budget: Some(cfg.build_thinking_budget),
            cancel: cancel.clone(),
            ..Default::default()
        };
        // Streamed: a build reasons for minutes, longer than the total timeout of a non-streamed call; the stream's
        // idle timeout still catches a server that stops answering.
        let mut sink = |_: &str, _: bool| {};
        let out = match s2.stream(&req, &mut sink).await {
            Ok(o) => o,
            Err(e) => {
                rep.reasons
                    .push(format!("round {round}: System 2 failed: {e}"));
                break;
            }
        };
        rep.builder_tokens += out.completion_tokens;
        let feedback = match try_draft(
            store,
            s1,
            &embed,
            job,
            &expected_text,
            &out.content,
            qv.clone(),
            cfg,
        )
        .await
        {
            Ok(mut e) => {
                let checks = e.build.clone();
                e.build = json!({"rounds": round, "ms": t0.elapsed().as_millis() as u64,
                    "builder_tokens": rep.builder_tokens, "reasons": rep.reasons, "offered_for_reuse": reuse.len(),
                    "checks": checks});
                rep.ok = true;
                rep.entry = Some(e.id.clone());
                store.insert(e, cfg.max_entries);
                break;
            }
            Err(f) => f,
        };
        rep.reasons.push(format!(
            "round {round}: {}",
            feedback.lines().next().unwrap_or("")
        ));
        messages.push(Message {
            role: "assistant".into(),
            content: out.content,
        });
        messages.push(Message {
            role: "user".into(),
            content: format!("{feedback}\n\nReply again with the three sections TEMPLATE, PARAMS and CODE, corrected."),
        });
    }
    rep.ms = t0.elapsed().as_millis() as u64;
    rep
}

/// Checks one builder reply; the error is the feedback for the next round.
#[allow(clippy::too_many_arguments)]
async fn try_draft(
    store: &ArcStore,
    s1: Option<&Backend>,
    embed: &(dyn Fn(String) -> futures::future::BoxFuture<'static, Option<Vec<f32>>> + Send + Sync),
    job: &BuildJob,
    expected: &str,
    reply: &str,
    qv: Option<Vec<f32>>,
    cfg: &ArcCfg,
) -> Result<Entry, String> {
    let mut d = parse_draft(reply)?;
    let mut template_note = None;
    if let Err(why) = check_template(&job.question, &d) {
        // The values are right but the words were changed: rebuild the template from the question's own words.
        let fixed = synthesize_template(&job.question, &d.params).map(|t| Draft {
            template: t,
            ..d.clone()
        });
        match fixed.filter(|f| check_template(&job.question, f).is_ok()) {
            Some(f) => {
                d = f;
                template_note = Some("template rebuilt from the question");
            }
            None => return Err(why),
        }
    }
    let (functions, problems) = store.describe(&d.code).await?;
    if !problems.is_empty() {
        return Err(format!("The code has problems: {}.", problems.join("; ")));
    }
    let root = functions
        .iter()
        .find(|f| f.root)
        .ok_or("no root function")?
        .clone();
    let id = new_id();
    // Each do() description: one of this entry's functions, else another entry's (reuse), else unresolved.
    let mut links = BTreeMap::new();
    for f in &functions {
        for c in &f.calls {
            let own = functions
                .iter()
                .filter(|g| g.name != f.name)
                .map(|g| (dice(&word_set(c), &word_set(&g.description)), g))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            match own {
                Some((s, g)) if s >= 0.5 => {
                    links.insert(c.clone(), format!("{id}:{}", g.name));
                }
                _ => {
                    let dv = embed(c.clone()).await;
                    match store.resolve_call(c, dv.as_deref(), &id, cfg.link_similarity as f32) {
                        Some((target, _)) => {
                            links.insert(c.clone(), target);
                        }
                        None => {
                            if let Some((_, g)) = own {
                                links.insert(c.clone(), format!("{id}:{}", g.name));
                            } else {
                                return Err(format!(
                                    "do(ctx, \"{c}\", ...) matches no function: define it with @semantic(\"{c}\")."
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    let mut entry = Entry {
        id: id.clone(),
        created_at: now(),
        domain: job.domain.clone(),
        question: normalize(&job.question),
        template: normalize(&d.template),
        params: d.params.clone(),
        expected: expected.to_string(),
        code: d.code.clone(),
        functions: functions.clone(),
        root: root.name.clone(),
        links,
        status: if cfg.trust_after == 0 {
            Status::Trusted
        } else {
            Status::Candidate
        },
        stats: Stats::default(),
        build: Value::Null,
        narration: vec![],
        source_reply: job.source_reply.clone(),
        vectors: vec![],
        fn_vectors: vec![],
    };
    let params: BTreeMap<String, String> = d
        .params
        .iter()
        .map(|p| (p.description.clone(), p.value.clone()))
        .collect();
    // Phase 5 of CPU-ARC: the program on the original parameters must give the expected answer.
    let out = store.invoke(&entry, &params).await.map_err(|e| {
        format!("Running the root function on the question's own values failed: {e}")
    })?;
    if !same_answer(&out.answer, expected) {
        return Err(format!(
            "The program returned {} but the answer to the question is {}. Fix the computation; do not write the answer in.",
            out.answer, expected
        ));
    }
    if root.narrates == 0 {
        return Err("The root function records no step with narrate().".into());
    }
    // Robustness: the program must run on other values of its numeric inputs (it computes, it does not recall).
    let mut varied = 0;
    let mut changed = 0;
    for p in d.params.iter().filter(|p| is_numeric_type(&p.ty)) {
        for alt in perturb(&p.value, &p.ty) {
            let mut q = params.clone();
            q.insert(p.description.clone(), alt.clone());
            varied += 1;
            match store.invoke(&entry, &q).await {
                Ok(o) => {
                    if !same_answer(&o.answer, &out.answer) {
                        changed += 1;
                    }
                }
                Err(e) if e.contains("ZeroDivision") || e.contains("not a whole number") => {}
                Err(e) => {
                    return Err(format!(
                        "With {} = {alt} the program failed: {e}. Compute from the inputs so any valid value works.",
                        p.name
                    ))
                }
            }
        }
    }
    // The extraction round trip (CPU-ARC's second end-to-end stage): the question itself must hit this entry with
    // its own values through the tiers a later request would take.
    let mut trip = match_template(&entry, &job.question);
    if trip.is_none() {
        if let Some(s1) = s1 {
            trip = extract_with(s1, &entry, &job.question, None).await.ok();
        }
    }
    let trip_ok = trip.as_ref().is_some_and(|t| {
        t.iter().all(|(k, v)| {
            params.get(k).is_some_and(|o| {
                o == v || num_value(o).zip(num_value(v)).is_some_and(|(a, b)| a == b)
            })
        })
    });
    if !trip_ok {
        return Err("The question does not match its own TEMPLATE with these values: copy the question exactly into the template and replace only the values.".into());
    }
    entry.narration = out.narration.clone();
    if let Some(v) = qv {
        entry.vectors.push(v);
    }
    for f in &functions {
        entry
            .fn_vectors
            .push(embed(f.description.clone()).await.unwrap_or_default());
    }
    if entry.fn_vectors.iter().all(|v| v.is_empty()) {
        entry.fn_vectors.clear();
    }
    entry.build = json!({"perturbations": varied, "perturbations_changed": changed, "template_note": template_note});
    Ok(entry)
}

/// Other values for a numeric input, for the robustness run.
pub fn perturb(value: &str, ty: &str) -> Vec<String> {
    let Some(x) = num_value(value) else {
        return vec![];
    };
    match ty {
        "int" => {
            let n = x as i64;
            let mut v = vec![n + 1, n * 2];
            v.retain(|y| *y != n);
            v.into_iter().map(|y| y.to_string()).collect()
        }
        _ => vec![format!("{}", x * 1.5), format!("{}", x + 1.0)],
    }
}

/// Pairs of words that turn one question into another while keeping its topic and its numbers (an area for a perimeter,
/// the even numbers for the odd ones). A pair conflicts when the new question uses one side and not the other while the
/// entry's question does the opposite; the words are compared after a light stem. Returns (entry's word, new word).
pub fn contrast_conflict(original: &str, q: &str) -> Option<(String, String)> {
    const PAIRS: &[(&str, &str)] = &[
        ("odd", "even"),
        ("area", "perimeter"),
        ("simple", "compound"),
        ("least", "greatest"),
        ("smallest", "largest"),
        ("minimum", "maximum"),
        ("sum", "product"),
        ("sum", "difference"),
        ("product", "quotient"),
        ("multiple", "divisor"),
        ("lcm", "gcd"),
        ("increase", "decrease"),
        ("profit", "loss"),
        ("before", "after"),
        ("more", "less"),
        ("saved", "price"),
        ("discount", "price"),
        ("average", "total"),
        ("speed", "distance"),
        ("speed", "time"),
        ("distance", "time"),
        ("radius", "diameter"),
        ("volume", "area"),
        ("prime", "composite"),
        ("permutation", "combination"),
        ("left", "read"),
        ("interest", "amount"),
        ("workers", "days"),
        ("remaining", "removed"),
        ("added", "removed"),
    ];
    let stem = |s: &str| -> BTreeSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| {
                let w = w.trim_end_matches('s');
                w.to_string()
            })
            .collect()
    };
    let (a, b) = (stem(original), stem(q));
    let has = |set: &BTreeSet<String>, w: &str| set.contains(w.trim_end_matches('s'));
    for (x, y) in PAIRS {
        for (u, v) in [(x, y), (y, x)] {
            // The entry's question has u and not v; the new one has v and not u.
            if has(&a, u) && !has(&a, v) && has(&b, v) && !has(&b, u) {
                return Some((u.to_string(), v.to_string()));
            }
        }
    }
    None
}

/// System 1 asked directly whether `q` asks for the same quantity as the entry's question (measured on the Phase 38
/// item set: 21 of 23 meaning-tier matches judged right, 5 of 6 different questions refused).
pub async fn same_question(
    s1: &Backend,
    original: &str,
    q: &str,
    cancel: Option<CancelFlag>,
) -> Result<bool, String> {
    let req = GenRequest {
        messages: vec![Message {
            role: "user".into(),
            content: format!(
                "Question A: {}\nQuestion B: {}\n\nDoes question B ask for exactly the same quantity as question A, computed the same way, only with other wording or other numbers? Reply with one word: yes or no.",
                normalize(original),
                normalize(q)
            ),
        }],
        max_tokens: 8,
        temperature: 0.0,
        top_p: 1.0,
        thinking: Some(false),
        cancel,
        ..Default::default()
    };
    let out = s1.complete(&req).await.map_err(|e| e.to_string())?;
    Ok(out.content.trim().to_lowercase().starts_with("yes"))
}

/// System 1 reads the entry's inputs out of `q`; every value must be in the question.
pub async fn extract_with(
    s1: &Backend,
    e: &Entry,
    q: &str,
    cancel: Option<CancelFlag>,
) -> Result<BTreeMap<String, String>, String> {
    let req = GenRequest {
        messages: extraction_messages(e, q),
        max_tokens: 160 + 40 * e.params.len() as u32,
        temperature: 0.0,
        top_p: 1.0,
        thinking: Some(false),
        cancel,
        ..Default::default()
    };
    let out = s1.complete(&req).await.map_err(|e| e.to_string())?;
    let vals = parse_extraction(e, &out.content)?;
    for (d, v) in &vals {
        if !verbatim(q, v) {
            return Err(format!(
                "the value \"{v}\" for \"{d}\" is not in the question"
            ));
        }
    }
    Ok(vals)
}

pub mod serve;

#[cfg(test)]
mod tests;
