//! Memory files (spec §5): Markdown with front matter, one file per remembered thing, whose identity is its key. A
//! repeated fact updates its file instead of adding a near-duplicate; every write is a diff row that can be undone.
//! Extraction from a conversation is by rule here; a model pass, when on, produces the same `Candidate`s.

use chrono::{DateTime, NaiveDate, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::uri::{slug, Uri, MEMORY_KINDS};

/// Largest memory file body, in bytes.
pub const MAX_BODY: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryFile {
    pub kind: String,
    pub key: String,
    pub title: String,
    pub body: String,
    pub updated: DateTime<Utc>,
    /// The sessions (or "manual") a file was written from.
    #[serde(default)]
    pub sources: Vec<String>,
}

impl MemoryFile {
    pub fn uri(&self) -> Uri {
        Uri::memory(&self.kind, &self.key)
    }

    /// The file as Markdown with front matter.
    pub fn to_markdown(&self) -> String {
        let sources = self
            .sources
            .iter()
            .map(|s| s.replace([',', ']', '\n'], " "))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "---\nkind: {}\nkey: {}\ntitle: {}\nupdated: {}\nsources: [{}]\n---\n{}\n",
            self.kind,
            self.key,
            self.title.replace('\n', " "),
            self.updated.to_rfc3339(),
            sources,
            self.body.trim_end()
        )
    }

    /// Parse a file written by a person in the editor: front matter optional; `kind` and `key` come from the URI.
    pub fn from_markdown(
        uri_kind: &str,
        uri_key: &str,
        text: &str,
        now: DateTime<Utc>,
    ) -> Result<MemoryFile, String> {
        if !MEMORY_KINDS.contains(&uri_kind) {
            return Err(format!("unknown memory kind \"{uri_kind}\""));
        }
        if text.len() > MAX_BODY {
            return Err(format!(
                "a memory file holds at most {} KB",
                MAX_BODY / 1024
            ));
        }
        let (front, body) = split_front_matter(text);
        // A first line "# Title" is the title, not part of what is remembered.
        let (heading, body) = match body.trim_start().strip_prefix("# ") {
            Some(rest) => {
                let (h, b) = rest.split_once('\n').unwrap_or((rest, ""));
                (Some(h.trim().to_string()), b)
            }
            None => (None, body),
        };
        let title = front
            .iter()
            .find(|(k, _)| k == "title")
            .map(|(_, v)| v.clone())
            .filter(|t| !t.is_empty())
            .or(heading)
            .unwrap_or_else(|| title_from(body));
        let sources = front
            .iter()
            .find(|(k, _)| k == "sources")
            .map(|(_, v)| {
                v.trim_matches(['[', ']'])
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        Ok(MemoryFile {
            kind: uri_kind.to_string(),
            key: uri_key.to_string(),
            title,
            body: body.trim().to_string(),
            updated: now,
            sources,
        })
    }

    /// L0: the title and the first line of the body.
    pub fn abstract_line(&self) -> String {
        let first = self
            .body
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        let mut s = format!("{}: {}", self.title, first.trim());
        if s.chars().count() > 300 {
            s = s.chars().take(297).collect::<String>() + "...";
        }
        s
    }
}

fn split_front_matter(text: &str) -> (Vec<(String, String)>, &str) {
    let t = text.trim_start_matches('\u{feff}');
    if let Some(rest) = t.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            let head = &rest[..end];
            let body = rest[end + 4..].trim_start_matches('\n');
            let pairs = head
                .lines()
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            return (pairs, body);
        }
    }
    (Vec::new(), t)
}

fn title_from(body: &str) -> String {
    let first = body
        .lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .unwrap_or("Untitled");
    let words: Vec<&str> = first.split_whitespace().take(8).collect();
    words.join(" ")
}

/// Something a conversation said that is worth remembering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub kind: String,
    pub title: String,
    pub body: String,
    /// For an event: its date, when the text names one.
    #[serde(default)]
    pub date: Option<NaiveDate>,
}

impl Candidate {
    /// The key this candidate is filed under (spec §5): the identity of the thing, not of the sentence.
    pub fn key(&self) -> String {
        match self.kind.as_str() {
            "profile" => "profile".into(),
            "events" => {
                let d = self
                    .date
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "undated".into());
                format!("{d}-{}", slug(&self.title))
            }
            _ => slug(&self.title),
        }
    }
}

/// What a write did (a diff row's operation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Add,
    Update,
    Delete,
    Skip,
}

/// The merge of a candidate into the file that holds its key, if any. Returns the new body and the operation, or
/// `Skip` with the reason when nothing would change.
pub fn merge(existing: Option<&MemoryFile>, c: &Candidate) -> (Op, String, &'static str) {
    let new = c.body.trim();
    let Some(old) = existing else {
        return (Op::Add, new.to_string(), "new");
    };
    let had = old.body.trim();
    let same = |a: &str, b: &str| normal(a) == normal(b);
    match c.kind.as_str() {
        // Field lines: "name: value"; a new value for a field replaces the old one, other fields stay.
        "profile" => {
            let mut lines: Vec<String> = had.lines().map(str::to_string).collect();
            let mut changed = false;
            for l in new.lines().filter(|l| !l.trim().is_empty()) {
                let field = l.split_once(':').map(|(f, _)| normal(f));
                match field.and_then(|f| {
                    lines
                        .iter()
                        .position(|x| x.split_once(':').is_some_and(|(g, _)| normal(g) == f))
                }) {
                    Some(i) if same(&lines[i], l) => {}
                    Some(i) => {
                        lines[i] = l.trim().to_string();
                        changed = true;
                    }
                    None => {
                        lines.push(l.trim().to_string());
                        changed = true;
                    }
                }
            }
            if changed {
                (Op::Update, lines.join("\n"), "field set")
            } else {
                (Op::Skip, had.to_string(), "already known")
            }
        }
        // A preference is one statement: the newest replaces the old.
        "preferences" => {
            if same(had, new) {
                (Op::Skip, had.to_string(), "already known")
            } else {
                (Op::Update, new.to_string(), "replaced by a newer statement")
            }
        }
        // Events are never merged: the caller files a repeated key under a new suffix. The same text is a duplicate.
        "events" => {
            if same(had, new) {
                (Op::Skip, had.to_string(), "already known")
            } else {
                (Op::Add, new.to_string(), "new event")
            }
        }
        // Entities and notes gather lines.
        _ => {
            let known: Vec<String> = had.lines().map(normal).collect();
            let fresh: Vec<&str> = new
                .lines()
                .filter(|l| !l.trim().is_empty() && !known.contains(&normal(l)))
                .collect();
            if fresh.is_empty() {
                (Op::Skip, had.to_string(), "already known")
            } else {
                let mut body = had.to_string();
                for l in fresh {
                    body.push('\n');
                    body.push_str(l.trim());
                }
                (Op::Update, body, "lines added")
            }
        }
    }
}

fn normal(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Score multiplier of an event remembered `age_days` ago: full for a week, then halving every 30 days (spec §5).
pub fn event_decay(age_days: f64) -> f64 {
    const PROTECT: f64 = 7.0;
    const HALF_LIFE: f64 = 30.0;
    if age_days <= PROTECT {
        1.0
    } else {
        0.5f64.powf((age_days - PROTECT) / HALF_LIFE)
    }
}

// ------------------------------------------------------------------------------------------ extraction by rule

struct Rule {
    re: Regex,
    kind: &'static str,
}

fn rules() -> &'static [Rule] {
    static R: OnceLock<Vec<Rule>> = OnceLock::new();
    R.get_or_init(|| {
        let r = |p: &str, kind| Rule {
            re: Regex::new(p).expect("valid rule"),
            kind,
        };
        vec![
            // "Remember that the deploy key rotates monthly." / "Don't forget that ..." / "For future reference, ..."
            // -> a note. "Remember when / how / what ..." recalls the past; it is not a note.
            r(r"(?i)^(?:please\s+)?(?:remember|note|keep in mind|don'?t forget|do not forget|for future reference|fyi)(?:\s+that)?[:,]?\s+(?P<x>.{6,400})$", "notes"),
            // "Call me Ris." / "You can call me Pak Ris." / "My name is Ris." / "I'm Ris, by the way." -> profile.
            r(r"(?i)^(?:please\s+|you can\s+)?call me\s+(?P<name>[\p{L}][\p{L} .'-]{0,40})$", "profile"),
            r(r"(?i)^my name is\s+(?P<name>[\p{L}][\p{L} .'-]{0,40})$", "profile"),
            r(r"^(?:I'?m|I am)\s+(?P<name>\p{Lu}[\p{L}'-]{1,20}(?:\s+\p{Lu}[\p{L}'-]{1,20})?)(?:,?\s+by the way)?$", "profile"),
            // "I prefer ..." / "I'd rather get the code first" / "From now on, answer in English." / "Please always
            // cite the page." / "Never use bullet points." -> preferences. An instruction needs a verb of doing after
            // always / never / from now on, so "Never mind" and "Always the same error" are not preferences.
            r(r"(?i)^i\s*(?P<verb>prefer|like|love|want|don'?t like|do not like|dislike|hate|always want|never want|'d rather| would rather|would rather)\s+(?P<x>.{3,300})$", "preferences"),
            r(r"(?i)^(?:please\s+)?(?P<verb>from now on|always|never|don'?t ever|do not ever)[,]?\s+(?:please\s+)?(?P<x>(?:answer|use|reply|write|cite|give|show|include|put|explain|start|keep|call|send|format|respond|add|avoid|mention|speak|say|ask|list|sort)\b.{2,300})$", "preferences"),
            // "My manager is Dewi." / "My wife's name is Sari." / "Our CTO is Andi." / "I work at Paracorp" /
            // "I live in Depok." -> entities (filed under the thing named).
            r(r"(?i)^my\s+(?P<thing>[\p{L}][\p{L} -]{1,40}?)'s name is\s+(?P<x>.{2,100})$", "entities"),
            r(r"(?i)^(?:my|our)\s+(?P<thing>[\p{L}][\p{L} -]{1,40}?)\s+(?:is|are|was)\s+(?P<x>.{2,300})$", "entities"),
            r(r"(?i)^i\s+(?P<thing>work|live|study)\s+(?P<x>(?:at|in|for|as|with)\b.{2,300})$", "entities"),
            // "I moved to Bandung on 2026-10-01." / "We launched the beta on 3 November 2026." / "Yesterday we migrated
            // the database." / "Last week I passed the exam." -> events.
            r(r"(?i)^(?:i|we)\s+(?P<x>(?:moved|started|joined|left|launched|released|shipped|married|graduated|finished|began|met)\b.{2,300})$", "events"),
            r(r"(?i)^(?:yesterday|today|this morning|earlier today|last (?:week|month|year|night)|on \d{4}-\d{2}-\d{2}|in (?:january|february|march|april|may|june|july|august|september|october|november|december)(?: \d{4})?),?\s+(?:i|we)\s+(?P<x>\p{L}+ed\b.{2,300})$", "events"),
        ]
    })
}

/// Words that make a want, like or love about how to answer (a preference), not about what to do now.
const STYLE: &[&str] = &[
    "answer",
    "answers",
    "reply",
    "replies",
    "response",
    "responses",
    "explanation",
    "explanations",
    "format",
    "language",
    "english",
    "indonesian",
    "tone",
    "style",
    "summary",
    "summaries",
    "bullet",
    "bullets",
    "units",
    "spelling",
    "examples",
    "short",
    "long",
    "brief",
    "detailed",
    "formal",
    "informal",
    "sources",
    "citations",
];

fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut start = 0;
        let b = line.as_bytes();
        for i in 0..b.len() {
            if matches!(b[i], b'.' | b'!' | b'?') && (i + 1 == b.len() || b[i + 1] == b' ') {
                // A question keeps its mark, so it is recognised as one and never remembered.
                let end = if b[i] == b'?' { i + 1 } else { i };
                let s = line[start..end].trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
                start = i + 1;
            }
        }
        let s = line[start..].trim();
        if !s.is_empty() {
            out.push(s.to_string());
        }
    }
    out
}

fn month(m: &str) -> Option<u32> {
    let m = m.to_ascii_lowercase();
    [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    .iter()
    .position(|x| m.starts_with(x))
    .map(|i| i as u32 + 1)
}

/// A date named in the text: ISO (`2026-10-01`) or `3 November 2026` / `November 3, 2026`.
pub fn find_date(text: &str) -> Option<NaiveDate> {
    static ISO: OnceLock<Regex> = OnceLock::new();
    static DMY: OnceLock<Regex> = OnceLock::new();
    static MDY: OnceLock<Regex> = OnceLock::new();
    let iso = ISO.get_or_init(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").expect("valid"));
    let dmy = DMY
        .get_or_init(|| Regex::new(r"(?i)\b(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})\b").expect("valid"));
    let mdy = MDY.get_or_init(|| {
        Regex::new(r"(?i)\b([a-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b").expect("valid")
    });
    if let Some(c) = iso.captures(text) {
        return NaiveDate::from_ymd_opt(c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?);
    }
    if let Some(c) = dmy.captures(text) {
        if let Some(m) = month(&c[2]) {
            return NaiveDate::from_ymd_opt(c[3].parse().ok()?, m, c[1].parse().ok()?);
        }
    }
    if let Some(c) = mdy.captures(text) {
        if let Some(m) = month(&c[1]) {
            return NaiveDate::from_ymd_opt(c[3].parse().ok()?, m, c[2].parse().ok()?);
        }
    }
    None
}

fn sentence_case(s: &str) -> String {
    let mut c = s.trim().chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Memories a person's own message states, by rule (spec §6.2). Questions are never memories.
pub fn extract(user_message: &str) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    for s in sentences(user_message) {
        if s.ends_with('?') || s.chars().count() > 400 {
            continue;
        }
        for rule in rules() {
            let Some(c) = rule.re.captures(&s) else {
                continue;
            };
            let cand = match rule.kind {
                "notes" => {
                    let x = c["x"].trim();
                    // "Remember when we ..." recalls the past; it is not a note.
                    let first = x.split_whitespace().next().unwrap_or("").to_lowercase();
                    if ["when", "how", "what", "if", "where", "who", "why"]
                        .contains(&first.as_str())
                    {
                        break;
                    }
                    Candidate {
                        kind: "notes".into(),
                        title: x.split_whitespace().take(6).collect::<Vec<_>>().join(" "),
                        body: format!("{}.", sentence_case(x).trim_end_matches('.')),
                        date: None,
                    }
                }
                "profile" => Candidate {
                    kind: "profile".into(),
                    title: "Profile".into(),
                    body: format!("name: {}", c["name"].trim()),
                    date: None,
                },
                "preferences" => {
                    // "I'd rather not say" declines; it states no preference.
                    let object = c["x"].trim().to_lowercase();
                    if object.starts_with("not") {
                        break;
                    }
                    // "I want a script that parses CSV" asks for something now; it is not how to answer in general.
                    // A want, like or love is a preference only when it is about the answers themselves.
                    let verb0 = c["verb"].to_lowercase();
                    let task = [
                        "this ",
                        "the following",
                        "for this",
                        "these ",
                        "that file",
                        "it ",
                    ]
                    .iter()
                    .any(|w| object.starts_with(w) || object.contains(&format!(" {w}")));
                    if task {
                        break;
                    }
                    if ["want", "like", "love"].contains(&verb0.as_str())
                        && !STYLE.iter().any(|w| {
                            object
                                .split(|ch: char| !ch.is_alphanumeric())
                                .any(|x| x == *w)
                        })
                    {
                        break;
                    }
                    let verb = c["verb"]
                        .to_lowercase()
                        .replace("'d rather", "would rather");
                    let verb = if [
                        "from now on",
                        "always",
                        "never",
                        "don't ever",
                        "do not ever",
                    ]
                    .contains(&verb.as_str())
                    {
                        format!("want you to {verb}")
                    } else {
                        verb
                    };
                    let x = c["x"].trim().trim_end_matches('.');
                    // The topic is what the preference is about: its first content words.
                    let topic: Vec<&str> = x
                        .split_whitespace()
                        .filter(|w| {
                            !["a", "an", "the", "my", "to", "be", "it", "that"]
                                .contains(&w.to_lowercase().as_str())
                        })
                        .take(3)
                        .collect();
                    Candidate {
                        kind: "preferences".into(),
                        title: sentence_case(&topic.join(" ")),
                        body: format!("I {verb} {x}."),
                        date: None,
                    }
                }
                "entities" => {
                    let thing = c["thing"].trim().to_lowercase();
                    let lead = s.split_whitespace().next().unwrap_or("").to_lowercase();
                    if ["work", "live", "study"].contains(&thing.as_str()) && lead == "i" {
                        let x = c["x"].trim().trim_end_matches('.');
                        let what = match thing.as_str() {
                            "work" => "where I work",
                            "live" => "where I live",
                            _ => "where I study",
                        };
                        let cand = Candidate {
                            kind: "entities".into(),
                            title: sentence_case(what),
                            body: format!("I {thing} {x}."),
                            date: None,
                        };
                        if !out
                            .iter()
                            .any(|o| o.kind == cand.kind && o.key() == cand.key())
                        {
                            out.push(cand);
                        }
                        break;
                    }
                    let owner = if lead == "our" { "our" } else { "my" };
                    let is = if s.to_lowercase().contains("'s name is") {
                        "'s name is"
                    } else {
                        " is"
                    };
                    // "my question is ..." and the like are not things.
                    if [
                        "question", "point", "guess", "problem", "idea", "answer", "concern",
                        "request",
                    ]
                    .contains(&thing.as_str())
                    {
                        continue;
                    }
                    Candidate {
                        kind: "entities".into(),
                        title: sentence_case(&format!("{owner} {thing}")),
                        body: format!(
                            "{} {thing}{is} {}.",
                            sentence_case(owner),
                            c["x"].trim().trim_end_matches('.')
                        ),
                        date: None,
                    }
                }
                _ => {
                    let x = c["x"].trim().trim_end_matches('.');
                    Candidate {
                        kind: "events".into(),
                        title: x.split_whitespace().take(5).collect::<Vec<_>>().join(" "),
                        body: format!("{}.", sentence_case(&s).trim_end_matches('.')),
                        date: find_date(&s),
                    }
                }
            };
            if !out
                .iter()
                .any(|o| o.kind == cand.kind && o.key() == cand.key())
            {
                out.push(cand);
            }
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        "2026-11-13T07:00:00Z".parse().unwrap()
    }

    #[test]
    fn rules_find_notes_preferences_profile_entities_events() {
        let got = extract(
            "Call me Ris. I prefer answers in Indonesian. Remember that the deploy key rotates monthly. \
             My manager is Dewi Lestari. We launched the beta on 3 November 2026. What is BM25?",
        );
        let kinds: Vec<&str> = got.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(
            kinds,
            ["profile", "preferences", "notes", "entities", "events"],
            "{got:?}"
        );
        assert_eq!(got[0].body, "name: Ris");
        assert_eq!(got[1].key(), "answers-in-indonesian");
        assert_eq!(got[3].key(), "my-manager");
        assert_eq!(got[4].date, NaiveDate::from_ymd_opt(2026, 11, 3));
        assert!(got[4].key().starts_with("2026-11-03-"));
        assert!(extract("What is my manager's name? Is it Dewi?").is_empty());
        assert!(extract("My manager is who?").is_empty());
        assert!(extract("I want a Python script that parses CSV.").is_empty());
        assert!(extract("I like this approach.").is_empty());
        assert_eq!(extract("I want short answers.")[0].kind, "preferences");
        assert!(extract("My question is about BM25.").is_empty());
    }

    #[test]
    fn merges_by_kind() {
        let file = |kind: &str, body: &str| MemoryFile {
            kind: kind.into(),
            key: "k".into(),
            title: "t".into(),
            body: body.into(),
            updated: now(),
            sources: vec![],
        };
        let c = |kind: &str, body: &str| Candidate {
            kind: kind.into(),
            title: "t".into(),
            body: body.into(),
            date: None,
        };
        assert_eq!(merge(None, &c("notes", "x")).0, Op::Add);
        let p = file("profile", "name: Ris\ncity: Jakarta");
        let (op, body, _) = merge(Some(&p), &c("profile", "city: Bandung"));
        assert_eq!(
            (op, body.as_str()),
            (Op::Update, "name: Ris\ncity: Bandung")
        );
        assert_eq!(merge(Some(&p), &c("profile", "Name: Ris")).0, Op::Skip);
        let pref = file("preferences", "I prefer short answers.");
        assert_eq!(
            merge(Some(&pref), &c("preferences", "i prefer SHORT answers")).0,
            Op::Skip
        );
        assert_eq!(
            merge(Some(&pref), &c("preferences", "I prefer long answers.")).1,
            "I prefer long answers."
        );
        let ent = file("entities", "My manager is Dewi.");
        let (op, body, _) = merge(
            Some(&ent),
            &c(
                "entities",
                "My manager is Dewi.\nShe leads the search team.",
            ),
        );
        assert_eq!(op, Op::Update);
        assert_eq!(body.lines().count(), 2);
    }

    #[test]
    fn markdown_round_trip_and_editor_input() {
        let f = MemoryFile {
            kind: "preferences".into(),
            key: "answer-language".into(),
            title: "Answer language".into(),
            body: "Answer in Indonesian.".into(),
            updated: now(),
            sources: vec!["nafs://sessions/abc".into()],
        };
        let md = f.to_markdown();
        let back = MemoryFile::from_markdown("preferences", "answer-language", &md, now()).unwrap();
        assert_eq!(back, f);
        let typed = MemoryFile::from_markdown(
            "notes",
            "wifi",
            "# Office wifi\nThe password is on the fridge.",
            now(),
        )
        .unwrap();
        assert_eq!(typed.title, "Office wifi");
        assert!(MemoryFile::from_markdown("secrets", "x", "y", now()).is_err());
        assert_eq!(f.abstract_line(), "Answer language: Answer in Indonesian.");
    }

    #[test]
    fn events_decay_after_a_week() {
        assert_eq!(event_decay(3.0), 1.0);
        assert!((event_decay(37.0) - 0.5).abs() < 1e-9);
        assert!(event_decay(400.0) < 0.001);
    }
}
