//! The in-process index of a knowledge base (Phase 25, FR-K67 to FR-K74).
//!
//! SurrealDB stays the source of truth. What a search needs is also held here, per base and per document: the passages,
//! an inverted index over their words (BM25F: title, heading path, context line and body as separate fields), their
//! vectors (unit length, scanned exactly) and a graph of the terms they share, built from the text with no model call.
//! A search then makes no round trip to the store: measured on a 443-passage base under load, the store took 1.2 s for
//! the words and 5.3 s for a hybrid search at the median, almost all of it waiting on round trips.
//!
//! A base is loaded on first use, kept per document (`Part`) so an upload, a re-read or a deletion touches one
//! document's rows, and checked against the store's own counts in the background. A base that would not fit in
//! `[knowledge] index_memory_mb` keeps using the store path, as does any search asked for with `engine = "store"`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use rust_stemmers::{Algorithm, Stemmer};
use serde_json::{json, Value};

use crate::store::{Hit, Store, FUNCTION_WORDS};
use crate::vcache::dot;
use crate::{KbError, SearchMode};

/// Fields of a passage the words are indexed in, each with its own length normalisation and weight.
pub const FIELDS: usize = 4;
const TITLE: usize = 0;
const HEADING: usize = 1;
const CONTEXT: usize = 2;
const BODY: usize = 3;

/// BM25F (Robertson, Zaragoza and Taylor 2004): the term frequencies of the fields are normalised by field length,
/// weighted and summed before the saturation, so a word in a heading is not saturated separately from the body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25 {
    pub k1: f32,
    pub b: f32,
    /// Weights of title, heading path, context line, body.
    pub weights: [f32; FIELDS],
}

impl Default for Bm25 {
    fn default() -> Self {
        Bm25 {
            k1: 1.2,
            // Measured on 256 questions (Phase 25): 0.5 put the answering passage first for 53 of 60 quoted questions
            // against 50 at the textbook 0.75; passages here are all about one size, so length says little.
            b: 0.5,
            weights: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// How the words' ranking and the vectors' ranking become one (FR-K70).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fusion {
    /// Reciprocal-rank fusion, the vector arm counting `vector_weight` against the words' 1.0 (FR-K63).
    Rrf { vector_weight: f64 },
    /// A convex combination of the two scores, each min-max normalised over its own candidates (Bruch et al. 2022):
    /// `alpha` is the share of the vector arm.
    Convex { alpha: f64 },
}

/// Pseudo-relevance feedback without a model (RM3-like): the best terms of the first passages join the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prf {
    pub passages: usize,
    pub terms: usize,
    pub weight: f32,
}

/// Personalized PageRank over the passage-and-term graph as a third arm (FR-K72).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ppr {
    /// How much the graph's ranking counts against the fused search's (rank fusion); 0 turns the arm off.
    pub weight: f64,
    /// Probability of following an edge rather than returning to the seeds.
    pub damping: f32,
    /// Passages of the fused search that seed the walk.
    pub seeds: usize,
    /// Share of the seed mass given to terms named in the question (the rest goes to the seed passages).
    pub entity_share: f32,
}

impl Default for Ppr {
    fn default() -> Self {
        Ppr {
            weight: 0.5,
            damping: 0.5,
            seeds: 10,
            entity_share: 0.5,
        }
    }
}

/// Search settings of the in-process index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub bm25: Bm25,
    pub fusion: Fusion,
    pub prf: Option<Prf>,
    pub ppr: Option<Ppr>,
    /// Phase 26: a question word the base does not hold is replaced by the nearest word it does hold (one edit away,
    /// two for a long word), so a typing slip does not cost the passage.
    pub fuzzy: bool,
    /// Phase 28: how much a passage's best sentence window counts in its meaning score (0 = the passage's own
    /// vector only; see `Base::similarities_mixed`).
    pub piece_mix: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            bm25: Bm25::default(),
            fusion: Fusion::Rrf { vector_weight: 0.1 },
            prf: None,
            ppr: None,
            fuzzy: true,
            piece_mix: 0.5,
        }
    }
}

/// The weight of a corrected word against a word found as typed.
const FUZZY_WEIGHT: f32 = 0.7;

/// Edit distance with adjacent transpositions (optimal string alignment), given up beyond `max`.
fn edit_distance(a: &[char], b: &[char], max: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let mut prev2: Vec<usize> = Vec::new();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut row = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            row[j] = (prev[j] + 1).min(row[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                row[j] = row[j].min(prev2[j - 2] + 1);
            }
        }
        if row.iter().min().is_some_and(|m| *m > max) {
            return None;
        }
        prev2 = std::mem::replace(&mut prev, row);
    }
    (prev[b.len()] <= max).then_some(prev[b.len()])
}

/// `[knowledge]` settings of the index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexConfig {
    /// The most memory (MB) the indexes of all bases may use; 0 turns the index off (every search uses the store).
    pub memory_mb: u64,
    pub params: Params,
    /// Phase 28: target size (characters) of the sentence windows that get vectors of their own; 0 = none.
    pub piece_chars: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
            memory_mb: 512,
            params: Params::default(),
            piece_chars: 0,
        }
    }
}

// ---------------------------------------------------------------------------------------------- analyzer

/// The analyzer of the words index: lowercase, ASCII folding, English Snowball stems (what the store's analyzer did),
/// and identifiers kept whole beside their parts, so `vector_weight`, `Qwen3-1.7B` and `1.7` are terms of their own.
pub struct Analyzer {
    stem: Stemmer,
}

impl Default for Analyzer {
    fn default() -> Self {
        Analyzer {
            stem: Stemmer::create(Algorithm::English),
        }
    }
}

fn fold(c: char) -> char {
    let c = c.to_lowercase().next().unwrap_or(c);
    match c {
        'à'..='å' => 'a',
        'ç' => 'c',
        'è'..='ë' => 'e',
        'ì'..='ï' => 'i',
        'ñ' => 'n',
        'ò'..='ö' | 'ø' => 'o',
        'ù'..='ü' => 'u',
        'ý' | 'ÿ' => 'y',
        _ => c,
    }
}

fn is_joiner(c: char) -> bool {
    matches!(c, '_' | '-' | '.' | '\'' | '’')
}

impl Analyzer {
    /// Every term of `text`, in order, with repeats.
    pub fn terms(&self, text: &str, mut emit: impl FnMut(String)) {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if !chars[i].is_alphanumeric() {
                i += 1;
                continue;
            }
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphanumeric()
                    || (is_joiner(chars[i])
                        && i + 1 < chars.len()
                        && chars[i + 1].is_alphanumeric()))
            {
                i += 1;
            }
            self.word(&chars[start..i], &mut emit);
        }
    }

    fn word(&self, mut w: &[char], emit: &mut impl FnMut(String)) {
        // A possessive is the word.
        if w.len() > 2
            && matches!(w[w.len() - 2], '\'' | '’')
            && matches!(w[w.len() - 1], 's' | 'S')
        {
            w = &w[..w.len() - 2];
        }
        let mut parts = 0;
        let mut part = String::new();
        let mut prev: Option<char> = None;
        let flush = |part: &mut String, parts: &mut usize, emit: &mut dyn FnMut(String)| {
            if part.is_empty() {
                return;
            }
            *parts += 1;
            let p = std::mem::take(part);
            if p.chars().all(|c| c.is_ascii_digit()) {
                emit(p);
            } else if p.chars().count() > 1 && !FUNCTION_WORDS.contains(&p.as_str()) {
                if p.is_ascii() {
                    emit(self.stem.stem(&p).into_owned());
                } else {
                    emit(p);
                }
            }
        };
        for &c in w {
            if is_joiner(c) {
                flush(&mut part, &mut parts, emit);
                prev = None;
                continue;
            }
            if let Some(p) = prev {
                let class_change = p.is_alphabetic() != c.is_alphabetic();
                let camel = p.is_lowercase() && c.is_uppercase();
                if class_change || camel {
                    flush(&mut part, &mut parts, emit);
                }
            }
            part.push(fold(c));
            prev = Some(c);
        }
        flush(&mut part, &mut parts, emit);
        if parts > 1 && w.len() <= 48 {
            emit(w.iter().map(|&c| fold(c)).collect());
        }
    }

    /// The distinct terms of a question, in order.
    pub fn query(&self, text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        self.terms(text, |t| {
            if !out.contains(&t) {
                out.push(t);
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------------------------- terms of the graph

/// A term a passage names: a capitalised phrase, an acronym or an identifier (FR-K71).
#[derive(Debug, Clone, PartialEq)]
pub struct Cand {
    pub key: Box<str>,
    pub surface: Box<str>,
    /// Seen where capitals are not owed to the start of a sentence, or an acronym, identifier or phrase.
    pub strong: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Ident,
    Acronym,
    Cap,
    Num,
    Other,
}

fn classify(w: &str) -> Class {
    let cs: Vec<char> = w.chars().collect();
    if cs.is_empty() {
        return Class::Other;
    }
    if cs.iter().all(|c| c.is_ascii_digit() || *c == '.') {
        return Class::Num;
    }
    let upper = cs.iter().filter(|c| c.is_uppercase()).count();
    let has_digit = cs.iter().any(|c| c.is_ascii_digit());
    if cs.len() >= 2 && upper >= 2 && cs.iter().all(|c| c.is_uppercase() || c.is_ascii_digit()) {
        return Class::Acronym;
    }
    if cs[0].is_alphabetic() && cs.len() >= 3 {
        let camel = cs
            .windows(2)
            .any(|p| p[0].is_lowercase() && p[1].is_uppercase());
        let dotted = cs
            .windows(3)
            .any(|p| p[1] == '.' && p[0].is_alphabetic() && p[2].is_alphabetic())
            && w.split('.').all(|s| s.chars().count() >= 2);
        let joined = cs.iter().any(|c| matches!(c, '-' | '.')) && (upper > 0 || has_digit);
        if cs.contains(&'_') || has_digit || camel || dotted || joined {
            return Class::Ident;
        }
        if cs[0].is_uppercase() && cs[1..].iter().all(|c| c.is_lowercase() || *c == '-') {
            return Class::Cap;
        }
    }
    Class::Other
}

/// Words that open a sentence with a capital and name nothing.
const OPENERS: &[&str] = &[
    "all", "after", "before", "both", "each", "every", "here", "however", "most", "more", "note",
    "one", "see", "some", "thus", "two", "when", "while", "with", "without", "then", "there",
];

fn key_of(surface: &str) -> String {
    surface.to_lowercase()
}

/// The terms of one text. `heading` keeps only acronyms and identifiers (a heading is in title case throughout).
pub fn candidates(text: &str, heading: bool, out: &mut Vec<Cand>) {
    let mut run: Vec<(&str, Class, bool)> = Vec::new();
    let mut sentence_start = true;
    let push = |surface: String, strong: bool, out: &mut Vec<Cand>| {
        let key = key_of(&surface);
        match out.iter_mut().find(|c| *c.key == *key) {
            Some(c) => c.strong |= strong,
            None => out.push(Cand {
                key: key.into(),
                surface: surface.into(),
                strong,
            }),
        }
    };
    let flush = |run: &mut Vec<(&str, Class, bool)>, out: &mut Vec<Cand>| {
        // A sentence opener is not part of a name.
        while let Some((w, c, start)) = run.first().copied() {
            let lw = w.to_lowercase();
            if c == Class::Cap
                && start
                && (FUNCTION_WORDS.contains(&lw.as_str()) || OPENERS.contains(&lw.as_str()))
            {
                run.remove(0);
            } else {
                break;
            }
        }
        while matches!(run.last(), Some((_, Class::Num, _))) && run.len() == 1 {
            run.pop();
        }
        for (w, c, _) in run.iter() {
            if matches!(c, Class::Ident | Class::Acronym) {
                push(w.to_string(), true, out);
            }
        }
        if !heading {
            let named = run.iter().filter(|(_, c, _)| *c != Class::Num).count();
            if run.len() >= 2 && named >= 1 {
                let phrase = run.iter().map(|(w, _, _)| *w).collect::<Vec<_>>().join(" ");
                push(phrase, true, out);
            } else if let [(w, Class::Cap, start)] = run.as_slice() {
                push(w.to_string(), !*start, out);
            }
        }
        run.clear();
    };
    for raw in text.split_whitespace() {
        let core = raw.trim_matches(|c: char| !c.is_alphanumeric());
        let core = core
            .strip_suffix("'s")
            .or_else(|| core.strip_suffix("’s"))
            .unwrap_or(core);
        if core.is_empty() {
            flush(&mut run, out);
            sentence_start = true;
            continue;
        }
        let ends = raw
            .trim_end_matches(['"', '\'', ')', ']', '”', '’', '*', '_'])
            .ends_with(['.', '!', '?', ':', ';', ',']);
        let class = classify(core);
        match class {
            Class::Ident | Class::Acronym | Class::Cap => run.push((core, class, sentence_start)),
            Class::Num if !run.is_empty() => run.push((core, class, false)),
            _ => flush(&mut run, out),
        }
        if ends || run.len() >= 4 {
            flush(&mut run, out);
        }
        sentence_start = raw
            .trim_end_matches(['"', '\'', ')', ']', '”', '’', '*', '_'])
            .ends_with(['.', '!', '?', ':']);
    }
    flush(&mut run, out);
}

// ---------------------------------------------------------------------------------------------- one document

/// A passage as the store holds it, with its vector when it has one.
#[derive(Debug, Clone, Default)]
pub struct Row {
    pub id: String,
    pub ord: u64,
    pub heading: String,
    pub page: Option<u64>,
    pub context: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    /// Phase 28: vectors of the passage's sentence windows (`crate::chunk::pieces`), empty when it has none.
    pub pieces: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
struct Posting {
    /// Passage, as its position in the document.
    p: u32,
    tf: [u8; FIELDS],
}

/// One document's share of the index.
#[derive(Clone)]
pub struct Part {
    pub doc_id: String,
    pub title: String,
    rows: Vec<Row>,
    postings: HashMap<Box<str>, Vec<Posting>>,
    lens: Vec<[u32; FIELDS]>,
    len_sum: [u64; FIELDS],
    dims: usize,
    /// One row per passage, unit length; zeros where the passage has no vector yet.
    vecs: Vec<f32>,
    has_vec: Vec<bool>,
    /// Sentence-window vectors, unit length, one row each, and the passage (position in the document) each belongs to.
    pvecs: Vec<f32>,
    powner: Vec<u32>,
    cands: Vec<Vec<Cand>>,
    bytes: usize,
}

impl Part {
    pub fn build(
        doc_id: &str,
        title: &str,
        mut rows: Vec<Row>,
        dims: usize,
        an: &Analyzer,
    ) -> Part {
        rows.sort_by_key(|r| r.ord);
        let n = rows.len();
        let mut postings: HashMap<Box<str>, Vec<Posting>> = HashMap::new();
        let mut lens = vec![[0u32; FIELDS]; n];
        let mut len_sum = [0u64; FIELDS];
        let mut vecs = vec![0f32; n * dims];
        let mut has_vec = vec![false; n];
        let (mut pvecs, mut powner) = (Vec::new(), Vec::new());
        let mut cands = Vec::with_capacity(n);
        let mut bytes = 0usize;
        // The title is the same for every passage of the document: analysed once.
        let mut title_terms = Vec::new();
        an.terms(title, |t| title_terms.push(t));
        for (i, r) in rows.iter_mut().enumerate() {
            let mut tf: HashMap<String, [u8; FIELDS]> = HashMap::new();
            for t in &title_terms {
                let e = tf.entry(t.clone()).or_default();
                e[TITLE] = e[TITLE].saturating_add(1);
            }
            lens[i][TITLE] = title_terms.len() as u32;
            for (f, text) in [
                (HEADING, &r.heading),
                (CONTEXT, &r.context),
                (BODY, &r.text),
            ] {
                let mut count = 0u32;
                an.terms(text, |t| {
                    count += 1;
                    let e = tf.entry(t).or_default();
                    e[f] = e[f].saturating_add(1);
                });
                lens[i][f] = count;
            }
            for (sum, len) in len_sum.iter_mut().zip(&lens[i]) {
                *sum += *len as u64;
            }
            for (t, tf) in tf {
                bytes += 8;
                postings
                    .entry(t.into_boxed_str())
                    .or_default()
                    .push(Posting { p: i as u32, tf });
            }
            if let Some(mut v) = r.embedding.take() {
                if v.len() == dims && dims > 0 {
                    normalise(&mut v);
                    vecs[i * dims..(i + 1) * dims].copy_from_slice(&v);
                    has_vec[i] = true;
                }
            }
            for mut v in std::mem::take(&mut r.pieces) {
                if v.len() == dims && dims > 0 {
                    normalise(&mut v);
                    pvecs.extend_from_slice(&v);
                    powner.push(i as u32);
                }
            }
            let mut c = Vec::new();
            candidates(&r.heading, true, &mut c);
            candidates(&r.text, false, &mut c);
            bytes += r.text.len() + r.heading.len() + r.context.len() + r.id.len() + c.len() * 48;
            cands.push(c);
        }
        bytes +=
            postings.keys().map(|k| k.len() + 48).sum::<usize>() + vecs.len() * 4 + pvecs.len() * 4;
        Part {
            doc_id: doc_id.to_string(),
            title: title.to_string(),
            rows,
            postings,
            lens,
            len_sum,
            dims,
            vecs,
            has_vec,
            pvecs,
            powner,
            cands,
            bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn vectors(&self) -> usize {
        self.has_vec.iter().filter(|x| **x).count()
    }

    /// Give passages their vectors; `false` when one of the ids is not a passage of this document.
    fn set_vectors(&mut self, rows: &[(String, Vec<f32>)]) -> bool {
        for (id, v) in rows {
            let Some(i) = self.rows.iter().position(|r| &r.id == id) else {
                return false;
            };
            if v.len() != self.dims {
                return false;
            }
            let mut v = v.clone();
            normalise(&mut v);
            self.vecs[i * self.dims..(i + 1) * self.dims].copy_from_slice(&v);
            self.has_vec[i] = true;
        }
        true
    }
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

// ---------------------------------------------------------------------------------------------- the graph

/// The passage-and-term graph (FR-K71): a term is a node when at least two passages name it; a passage is linked to
/// the terms it names. Built from the text alone, in milliseconds, so every base has one from its first upload.
pub struct FastGraph {
    pub names: Vec<String>,
    pub df: Vec<u32>,
    idf: Vec<f32>,
    by_key: HashMap<Box<str>, u32>,
    /// Terms of each passage (by global passage number), and passages of each term.
    p2e: Vec<Vec<u32>>,
    e2p: Vec<Vec<u32>>,
}

impl FastGraph {
    fn build(parts: &[Arc<Part>], offsets: &[u32], n: usize) -> FastGraph {
        let mut stat: HashMap<&str, (u32, bool, &str)> = HashMap::new();
        for part in parts {
            for c in part.cands.iter().flatten() {
                let e = stat.entry(&c.key).or_insert((0, false, &c.surface));
                e.0 += 1;
                e.1 |= c.strong;
            }
        }
        // A term nearly every passage names links everything to everything and says nothing.
        let max_df = (n as u32 / 5).max(20);
        let mut kept: Vec<(&str, u32, &str)> = stat
            .into_iter()
            .filter(|(_, (df, strong, _))| *strong && *df >= 2 && *df <= max_df)
            .map(|(k, (df, _, s))| (k, df, s))
            .collect();
        kept.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let by_key: HashMap<Box<str>, u32> = kept
            .iter()
            .enumerate()
            .map(|(i, (k, _, _))| ((*k).into(), i as u32))
            .collect();
        let mut p2e = vec![Vec::new(); n];
        let mut e2p = vec![Vec::new(); kept.len()];
        for (pi, part) in parts.iter().enumerate() {
            for (i, cs) in part.cands.iter().enumerate() {
                let g = offsets[pi] + i as u32;
                for c in cs {
                    if let Some(&e) = by_key.get(&c.key) {
                        p2e[g as usize].push(e);
                        e2p[e as usize].push(g);
                    }
                }
            }
        }
        FastGraph {
            idf: kept
                .iter()
                .map(|(_, df, _)| (1.0 + n as f32 / *df as f32).ln())
                .collect(),
            names: kept.iter().map(|(_, _, s)| s.to_string()).collect(),
            df: kept.iter().map(|(_, df, _)| *df).collect(),
            by_key,
            p2e,
            e2p,
        }
    }

    pub fn terms(&self) -> usize {
        self.names.len()
    }

    pub fn links(&self) -> usize {
        self.p2e.iter().map(Vec::len).sum()
    }

    /// Terms of the graph a question names, longest phrase first: (term, weight).
    pub fn query_terms(&self, query: &str) -> Vec<(u32, f32)> {
        let words: Vec<String> = query
            .split_whitespace()
            .map(|w| {
                let w = w.trim_matches(|c: char| !c.is_alphanumeric());
                w.strip_suffix("'s").unwrap_or(w).to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();
        let mut out: Vec<(u32, f32)> = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let mut step = 1;
            for len in (1..=4.min(words.len() - i)).rev() {
                let phrase = words[i..i + len].join(" ");
                if let Some(&e) = self.by_key.get(phrase.as_str()) {
                    if !out.iter().any(|x| x.0 == e) {
                        out.push((e, self.idf[e as usize]));
                    }
                    step = len;
                    break;
                }
            }
            i += step;
        }
        out
    }

    /// Personalized PageRank (the retrieval step of HippoRAG 2 and LinearRAG) by power iteration: the walk starts at
    /// the seed passages and seed terms, follows passage-term links (a rarer term carries more), and returns to the
    /// seeds with probability `1 - damping`. Returns the visiting probability of every passage.
    pub fn ppr(
        &self,
        seed_passages: &[(u32, f32)],
        seed_terms: &[(u32, f32)],
        damping: f32,
        iterations: usize,
    ) -> Vec<f32> {
        let (n, m) = (self.p2e.len(), self.e2p.len());
        let mut sp = vec![0f32; n];
        let mut se = vec![0f32; m];
        for &(p, w) in seed_passages {
            sp[p as usize] += w.max(0.0);
        }
        for &(e, w) in seed_terms {
            se[e as usize] += w.max(0.0);
        }
        let total: f32 = sp.iter().sum::<f32>() + se.iter().sum::<f32>();
        if total <= 0.0 {
            return vec![0.0; n];
        }
        sp.iter_mut().for_each(|x| *x /= total);
        se.iter_mut().for_each(|x| *x /= total);
        let out_p: Vec<f32> = self
            .p2e
            .iter()
            .map(|es| es.iter().map(|&e| self.idf[e as usize]).sum())
            .collect();
        let (mut rp, mut re) = (sp.clone(), se.clone());
        let (mut np, mut ne) = (vec![0f32; n], vec![0f32; m]);
        for _ in 0..iterations {
            np.iter_mut().for_each(|x| *x = 0.0);
            ne.iter_mut().for_each(|x| *x = 0.0);
            // Mass of passages with no term has nowhere to go: it returns to the seeds.
            let mut lost = 0f32;
            for (p, es) in self.p2e.iter().enumerate() {
                if rp[p] == 0.0 {
                    continue;
                }
                if es.is_empty() {
                    lost += rp[p];
                    continue;
                }
                let share = rp[p] / out_p[p];
                for &e in es {
                    ne[e as usize] += share * self.idf[e as usize];
                }
            }
            for (e, ps) in self.e2p.iter().enumerate() {
                if re[e] == 0.0 {
                    continue;
                }
                let share = re[e] / ps.len() as f32;
                for &p in ps {
                    np[p as usize] += share;
                }
            }
            let back = 1.0 - damping + damping * lost;
            for p in 0..n {
                rp[p] = back * sp[p] + damping * np[p];
            }
            for e in 0..m {
                re[e] = back * se[e] + damping * ne[e];
            }
        }
        rp
    }
}

// ---------------------------------------------------------------------------------------------- one base

/// A base's index: its documents' parts, immutable once made (a change makes a new `Base` sharing the other parts).
pub struct Base {
    parts: Vec<Arc<Part>>,
    offsets: Vec<u32>,
    n: usize,
    n_vec: usize,
    dims: usize,
    len_sum: [u64; FIELDS],
    bytes: usize,
    graph: OnceLock<FastGraph>,
}

impl Base {
    pub fn new(mut parts: Vec<Arc<Part>>) -> Base {
        parts.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        let mut offsets = Vec::with_capacity(parts.len());
        let (mut n, mut n_vec, mut bytes) = (0usize, 0usize, 0usize);
        let mut len_sum = [0u64; FIELDS];
        for p in &parts {
            offsets.push(n as u32);
            n += p.len();
            n_vec += p.vectors();
            bytes += p.bytes;
            for (sum, len) in len_sum.iter_mut().zip(&p.len_sum) {
                *sum += len;
            }
        }
        Base {
            dims: parts.iter().map(|p| p.dims).max().unwrap_or(0),
            parts,
            offsets,
            n,
            n_vec,
            len_sum,
            bytes,
            graph: OnceLock::new(),
        }
    }

    pub fn passages(&self) -> usize {
        self.n
    }

    pub fn vectors(&self) -> usize {
        self.n_vec
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn documents(&self) -> usize {
        self.parts.len()
    }

    /// (document, passages, vectors) of every part, to compare with the store's own counts.
    pub fn counts(&self) -> Vec<(String, usize, usize)> {
        self.parts
            .iter()
            .map(|p| (p.doc_id.clone(), p.len(), p.vectors()))
            .collect()
    }

    pub fn graph(&self) -> &FastGraph {
        self.graph
            .get_or_init(|| FastGraph::build(&self.parts, &self.offsets, self.n))
    }

    fn locate(&self, g: u32) -> (&Part, usize) {
        let pi = self.offsets.partition_point(|&o| o <= g) - 1;
        (&self.parts[pi], (g - self.offsets[pi]) as usize)
    }

    /// The global number of a passage id.
    pub fn number_of(&self, chunk_id: &str) -> Option<u32> {
        self.parts.iter().zip(&self.offsets).find_map(|(p, o)| {
            p.rows
                .iter()
                .position(|r| r.id == chunk_id)
                .map(|i| o + i as u32)
        })
    }

    pub fn text_of(&self, g: u32) -> &str {
        let (p, i) = self.locate(g);
        &p.rows[i].text
    }

    pub fn id_of(&self, g: u32) -> &str {
        let (p, i) = self.locate(g);
        &p.rows[i].id
    }

    fn hit(&self, g: u32, score: f64, lex: Option<f32>, vec: Option<f32>, rank: usize) -> Hit {
        let (p, i) = self.locate(g);
        let r = &p.rows[i];
        Hit {
            chunk_id: r.id.clone(),
            doc_id: p.doc_id.clone(),
            doc_title: p.title.clone(),
            ord: r.ord,
            heading: r.heading.clone(),
            page: r.page,
            text: r.text.clone(),
            score,
            rank,
            lexical_score: lex.filter(|s| *s > 0.0).map(|s| s as f64),
            vector_score: vec.filter(|s| s.is_finite()).map(|s| s as f64),
        }
    }

    /// Passage `g` as a hit with no scores (its text, heading, document and page).
    pub fn passage(&self, g: u32) -> Hit {
        self.hit(g, 0.0, None, None, 0)
    }

    /// The BM25F score of every passage for weighted terms (0 where no term matches).
    pub fn lexical_scores(&self, terms: &[(String, f32)], p: &Bm25) -> Vec<f32> {
        let mut score = vec![0f32; self.n];
        if self.n == 0 {
            return score;
        }
        let n = self.n as f32;
        let avg: [f32; FIELDS] = std::array::from_fn(|f| self.len_sum[f] as f32 / n);
        for (term, qw) in terms {
            let lists: Vec<(usize, &Vec<Posting>)> = self
                .parts
                .iter()
                .enumerate()
                .filter_map(|(pi, part)| part.postings.get(term.as_str()).map(|l| (pi, l)))
                .collect();
            let df: usize = lists.iter().map(|(_, l)| l.len()).sum();
            if df == 0 {
                continue;
            }
            let idf = (1.0 + (n - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
            for (pi, list) in lists {
                let part = &self.parts[pi];
                let base = self.offsets[pi] as usize;
                for post in list {
                    let lens = &part.lens[post.p as usize];
                    let mut tf = 0f32;
                    for f in 0..FIELDS {
                        if post.tf[f] > 0 && avg[f] > 0.0 {
                            tf += p.weights[f] * post.tf[f] as f32
                                / (1.0 - p.b + p.b * lens[f] as f32 / avg[f]);
                        }
                    }
                    score[base + post.p as usize] += qw * idf * tf * (p.k1 + 1.0) / (p.k1 + tf);
                }
            }
        }
        score
    }

    /// Cosine similarity of every passage to `q` (negative infinity where a passage has no vector).
    pub fn similarities(&self, q: &[f32]) -> Option<Vec<f32>> {
        if q.len() != self.dims || self.n_vec == 0 {
            return None;
        }
        let mut q = q.to_vec();
        normalise(&mut q);
        let mut out = Vec::with_capacity(self.n);
        for part in &self.parts {
            for (i, has) in part.has_vec.iter().enumerate() {
                out.push(if *has {
                    dot(&part.vecs[i * part.dims..(i + 1) * part.dims], &q)
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        Some(out)
    }

    /// Phase 28: the meaning score of every passage with its sentence windows counted: `(1 - mix) * own +
    /// mix * max(own, best window)`. With `mix = 0`, or no windows, the passage's own similarity.
    pub fn similarities_mixed(&self, q: &[f32], mix: f32) -> Option<Vec<f32>> {
        let mut out = self.similarities(q)?;
        if mix <= 0.0 || self.parts.iter().all(|p| p.powner.is_empty()) {
            return Some(out);
        }
        let mut q = q.to_vec();
        normalise(&mut q);
        let mut at = 0;
        for part in &self.parts {
            let mut best = vec![f32::NEG_INFINITY; part.rows.len()];
            for (k, &owner) in part.powner.iter().enumerate() {
                let s = dot(&part.pvecs[k * part.dims..(k + 1) * part.dims], &q);
                let b = &mut best[owner as usize];
                *b = b.max(s);
            }
            for (i, b) in best.into_iter().enumerate() {
                let own = &mut out[at + i];
                if own.is_finite() && b.is_finite() {
                    *own = (1.0 - mix) * *own + mix * own.max(b);
                }
            }
            at += part.rows.len();
        }
        Some(out)
    }

    /// The word of the base nearest to `term`, when the base does not hold `term` itself: same first letter, one edit
    /// away (two for a word of eight letters or more), the commonest of the nearest. Only plain words of five letters
    /// or more are corrected; identifiers and numbers are taken as typed.
    pub fn nearest_term(&self, term: &str) -> Option<String> {
        let t: Vec<char> = term.chars().collect();
        if t.len() < 5 || !t.iter().all(|c| c.is_alphabetic()) {
            return None;
        }
        if self.parts.iter().any(|p| p.postings.contains_key(term)) {
            return None;
        }
        let max = if t.len() >= 8 { 2 } else { 1 };
        let mut found: HashMap<&str, (usize, usize)> = HashMap::new();
        let mut cand: Vec<char> = Vec::new();
        for part in &self.parts {
            for (word, list) in &part.postings {
                if let Some(e) = found.get_mut(&**word) {
                    e.1 += list.len();
                    continue;
                }
                if word.len().abs_diff(term.len()) > max * 4 || !word.starts_with(t[0]) {
                    continue;
                }
                cand.clear();
                cand.extend(word.chars());
                if !cand.iter().all(|c| c.is_alphabetic()) {
                    continue;
                }
                if let Some(d) = edit_distance(&t, &cand, max) {
                    found.insert(word, (d, list.len()));
                }
            }
        }
        found
            .into_iter()
            .min_by(|a, b| {
                (a.1 .0, std::cmp::Reverse(a.1 .1), a.0).cmp(&(
                    b.1 .0,
                    std::cmp::Reverse(b.1 .1),
                    b.0,
                ))
            })
            .map(|(w, _)| w.to_string())
    }

    /// The query's terms, with the feedback terms of the first passages when `prf` is set.
    fn weighted_terms(&self, an: &Analyzer, query: &str, p: &Params) -> Vec<(String, f32)> {
        let mut terms: Vec<(String, f32)> = an.query(query).into_iter().map(|t| (t, 1.0)).collect();
        if p.fuzzy {
            for t in terms.iter_mut() {
                if let Some(near) = self.nearest_term(&t.0) {
                    *t = (near, FUZZY_WEIGHT);
                }
            }
            let mut seen: Vec<&str> = Vec::new();
            let keep: Vec<bool> = terms
                .iter()
                .map(|t| {
                    let new = !seen.contains(&t.0.as_str());
                    seen.push(&t.0);
                    new
                })
                .collect();
            let mut keep = keep.into_iter();
            terms.retain(|_| keep.next().unwrap_or(true));
        }
        let Some(prf) = p.prf else { return terms };
        let first = top(&self.lexical_scores(&terms, &p.bm25), prf.passages, 0.0);
        if first.is_empty() {
            return terms;
        }
        let n = self.n as f32;
        let mut weight: HashMap<String, f32> = HashMap::new();
        for (g, _) in &first {
            an.terms(self.text_of(*g), |t| *weight.entry(t).or_insert(0.0) += 1.0);
        }
        let mut extra: Vec<(String, f32)> = weight
            .into_iter()
            .filter(|(t, _)| !terms.iter().any(|x| &x.0 == t))
            .map(|(t, tf)| {
                let df: usize = self
                    .parts
                    .iter()
                    .map(|part| part.postings.get(t.as_str()).map_or(0, Vec::len))
                    .sum();
                let idf = (1.0 + (n - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
                (t, tf * idf)
            })
            .collect();
        extra.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        extra.truncate(prf.terms);
        let top_w = extra.first().map_or(1.0, |x| x.1).max(f32::MIN_POSITIVE);
        terms.extend(extra.into_iter().map(|(t, w)| (t, prf.weight * w / top_w)));
        terms
    }

    /// Search (FR-K07) with no round trip to the store. `qv` is the question's vector when the mode uses one.
    pub fn search(
        &self,
        an: &Analyzer,
        query: &str,
        qv: Option<&[f32]>,
        mode: SearchMode,
        k: usize,
        p: &Params,
    ) -> Vec<Hit> {
        self.search_in(an, query, qv, mode, k, p, None)
    }

    /// Phase 29 (scoped search): the same search ranked only among the passages of the documents in `scope`. The
    /// passages outside are masked before ranking, so both arms rank within the scope (a filter after ranking would
    /// leave a scoped search with fewer hits than asked for).
    #[allow(clippy::too_many_arguments)]
    pub fn search_in(
        &self,
        an: &Analyzer,
        query: &str,
        qv: Option<&[f32]>,
        mode: SearchMode,
        k: usize,
        p: &Params,
        scope: Option<&std::collections::HashSet<String>>,
    ) -> Vec<Hit> {
        let depth = (k * 3).max(20);
        let mut lex = (mode != SearchMode::Vector)
            .then(|| self.lexical_scores(&self.weighted_terms(an, query, p), &p.bm25));
        let mut sim = match mode {
            SearchMode::Lexical => None,
            _ => qv.and_then(|q| self.similarities_mixed(q, p.piece_mix)),
        };
        let inside: Option<Vec<bool>> = scope.map(|docs| {
            let mut m = vec![false; self.n];
            for (part, &o) in self.parts.iter().zip(&self.offsets) {
                if docs.contains(&part.doc_id) {
                    m[o as usize..o as usize + part.len()].fill(true);
                }
            }
            m
        });
        if let Some(m) = &inside {
            for (i, keep) in m.iter().enumerate() {
                if !keep {
                    if let Some(l) = lex.as_mut() {
                        l[i] = 0.0;
                    }
                    if let Some(v) = sim.as_mut() {
                        v[i] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let lex_top = lex.as_ref().map(|s| top(s, depth, 0.0)).unwrap_or_default();
        let vec_top = sim
            .as_ref()
            .map(|s| top(s, depth, f32::NEG_INFINITY))
            .unwrap_or_default();
        let mut fused: Vec<(u32, f64)> = match (mode, sim.is_some()) {
            (SearchMode::Hybrid, true) => fuse(&lex_top, &vec_top, p.fusion),
            (SearchMode::Vector, true) => vec_top.iter().map(|x| (x.0, x.1 as f64)).collect(),
            _ => lex_top.iter().map(|x| (x.0, x.1 as f64)).collect(),
        };
        if let (Some(ppr), SearchMode::Hybrid) = (p.ppr, mode) {
            fused = self.with_graph(query, fused, &ppr, depth);
        }
        if let Some(m) = &inside {
            fused.retain(|(g, _)| m[*g as usize]);
        }
        fused
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(i, (g, s))| {
                self.hit(
                    g,
                    s,
                    lex.as_ref().map(|l| l[g as usize]),
                    sim.as_ref().map(|v| v[g as usize]),
                    i + 1,
                )
            })
            .collect()
    }

    /// The graph arm: Personalized PageRank from the search's best passages and the terms the question names, fused
    /// with the search by rank. The graph supplies passages the search missed and never overrides its first places
    /// alone (the rule FR-K50 found by measurement for the model-built graph).
    fn with_graph(
        &self,
        query: &str,
        fused: Vec<(u32, f64)>,
        ppr: &Ppr,
        depth: usize,
    ) -> Vec<(u32, f64)> {
        let graph = self.graph();
        if ppr.weight <= 0.0 || graph.terms() == 0 || fused.is_empty() {
            return fused;
        }
        let seeds: Vec<(u32, f32)> = {
            let head = &fused[..fused.len().min(ppr.seeds.max(1))];
            // Rank-based mass, so the seeds do not depend on the scale of whichever fusion made the list.
            head.iter()
                .enumerate()
                .map(|(i, (g, _))| (*g, 1.0 / (i as f32 + 1.0)))
                .collect()
        };
        let named = graph.query_terms(query);
        let passage_mass: f32 = seeds.iter().map(|x| x.1).sum();
        let term_mass: f32 = named.iter().map(|x| x.1).sum();
        let share = if named.is_empty() {
            0.0
        } else {
            ppr.entity_share.clamp(0.0, 1.0)
        };
        let seeds: Vec<(u32, f32)> = seeds
            .into_iter()
            .map(|(g, w)| (g, (1.0 - share) * w / passage_mass))
            .collect();
        let named: Vec<(u32, f32)> = named
            .into_iter()
            .map(|(e, w)| (e, share * w / term_mass.max(f32::MIN_POSITIVE)))
            .collect();
        let walk = top(&graph.ppr(&seeds, &named, ppr.damping, 20), depth, 0.0);
        let mut score: HashMap<u32, f64> = HashMap::new();
        for (rank, (g, _)) in fused.iter().enumerate() {
            *score.entry(*g).or_insert(0.0) += 1.0 / (60.0 + rank as f64 + 1.0);
        }
        for (rank, (g, _)) in walk.iter().enumerate() {
            *score.entry(*g).or_insert(0.0) += ppr.weight / (60.0 + rank as f64 + 1.0);
        }
        let mut out: Vec<(u32, f64)> = score.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// The Graph view of the passage-and-term graph: the most named terms (or one term and the terms it appears
    /// with), linked when passages name both. Same shape as the model-built graph's view.
    pub fn graph_view(&self, center: Option<&str>, limit: usize) -> Value {
        let g = self.graph();
        let limit = limit.clamp(5, 300);
        let mut ids: Vec<u32> = match center.and_then(|c| g.by_key.get(c.to_lowercase().as_str())) {
            Some(&c) => {
                let mut with: HashMap<u32, u32> = HashMap::new();
                for &p in &g.e2p[c as usize] {
                    for &e in &g.p2e[p as usize] {
                        if e != c {
                            *with.entry(e).or_insert(0) += 1;
                        }
                    }
                }
                let mut with: Vec<(u32, u32)> = with.into_iter().collect();
                with.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                std::iter::once(c)
                    .chain(with.into_iter().map(|x| x.0))
                    .take(limit)
                    .collect()
            }
            None => (0..g.terms().min(limit.min(40)) as u32).collect(),
        };
        ids.sort_unstable();
        let mut pairs: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for &e in &ids {
            for &p in &g.e2p[e as usize] {
                for &o in &g.p2e[p as usize] {
                    if o > e && ids.binary_search(&o).is_ok() {
                        let x = pairs.entry((e, o)).or_insert((0, p));
                        x.0 += 1;
                    }
                }
            }
        }
        let mut edges: Vec<((u32, u32), (u32, u32))> = pairs.into_iter().collect();
        edges.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(&b.0)));
        edges.truncate(limit * 3);
        let key = |e: u32| g.names[e as usize].to_lowercase();
        json!({
            "source": "text",
            "terms": g.terms(),
            "nodes": ids.iter().map(|&e| json!({
                "id": key(e), "name": g.names[e as usize], "type": "term",
                "mentions": g.df[e as usize], "community": Value::Null,
            })).collect::<Vec<_>>(),
            "edges": edges.iter().map(|((a, b), (w, p))| json!({
                "source": key(*a), "target": key(*b), "predicate": "appears with", "weight": w,
                "evidence": self.id_of(*p),
            })).collect::<Vec<_>>(),
        })
    }
}

/// The `k` best of a score array, best first, leaving out scores at or below `floor`.
pub fn top(scores: &[f32], k: usize, floor: f32) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| **s > floor)
        .map(|(i, s)| (i as u32, *s))
        .collect();
    let by = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if k < v.len() {
        v.select_nth_unstable_by(k, by);
        v.truncate(k);
    }
    v.sort_by(by);
    v
}

/// Fuse the words' and the vectors' candidates (best first each) into one list, best first.
pub fn fuse(lex: &[(u32, f32)], vec: &[(u32, f32)], fusion: Fusion) -> Vec<(u32, f64)> {
    let mut score: HashMap<u32, f64> = HashMap::new();
    match fusion {
        Fusion::Rrf { vector_weight } => {
            for (list, w) in [(lex, 1.0), (vec, vector_weight)] {
                for (rank, (g, _)) in list.iter().enumerate() {
                    *score.entry(*g).or_insert(0.0) += w / (60.0 + rank as f64 + 1.0);
                }
            }
        }
        Fusion::Convex { alpha } => {
            let alpha = alpha.clamp(0.0, 1.0);
            for (list, w) in [(lex, 1.0 - alpha), (vec, alpha)] {
                let (Some(hi), Some(lo)) = (list.first().map(|x| x.1), list.last().map(|x| x.1))
                else {
                    continue;
                };
                let span = (hi - lo).max(f32::MIN_POSITIVE);
                for (g, s) in list {
                    let norm = if hi == lo {
                        1.0
                    } else {
                        ((s - lo) / span) as f64
                    };
                    *score.entry(*g).or_insert(0.0) += w * norm;
                }
            }
        }
    }
    let mut out: Vec<(u32, f64)> = score.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

// ---------------------------------------------------------------------------------------------- the service's indexes

struct Entry {
    base: RwLock<Option<Arc<Base>>>,
    /// Held while the base is read from the store, so it is read once.
    loading: tokio::sync::Mutex<()>,
    /// Raised by every change; a load that started before a change does not install what it read.
    generation: AtomicU64,
    used: AtomicU64,
    checked: Mutex<Instant>,
}

/// What a lookup found.
pub enum Lookup {
    Ready(Arc<Base>),
    /// Being read from the store in the background; use the store for now.
    Loading,
    /// Off, or the base does not fit in the memory allowed.
    Unavailable,
}

/// Bases under this many passages are read while the first search waits (about what one store search costs); larger
/// ones are read in the background while searches use the store.
const INLINE_BELOW: u64 = 5_000;
const CHECK_EVERY: Duration = Duration::from_secs(30);

/// The indexes of the bases in use, within a memory limit; the least recently used base goes first.
pub struct Index {
    cap_bytes: u64,
    pub analyzer: Analyzer,
    map: Mutex<HashMap<String, Arc<Entry>>>,
    clock: AtomicU64,
}

impl Index {
    pub fn new(memory_mb: u64) -> Arc<Index> {
        Arc::new(Index {
            cap_bytes: memory_mb.saturating_mul(1024 * 1024),
            analyzer: Analyzer::default(),
            map: Mutex::new(HashMap::new()),
            clock: AtomicU64::new(1),
        })
    }

    pub fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    fn entry(&self, kb: &str) -> Arc<Entry> {
        self.map
            .lock()
            .expect("index map")
            .entry(kb.to_string())
            .or_insert_with(|| {
                Arc::new(Entry {
                    base: RwLock::new(None),
                    loading: tokio::sync::Mutex::new(()),
                    generation: AtomicU64::new(0),
                    used: AtomicU64::new(0),
                    checked: Mutex::new(Instant::now()),
                })
            })
            .clone()
    }

    fn loaded(&self, kb: &str) -> Option<(Arc<Entry>, Arc<Base>)> {
        let e = self.map.lock().expect("index map").get(kb).cloned()?;
        let b = e.base.read().expect("index base").clone()?;
        Some((e, b))
    }

    /// (bases held, passages, bytes) for the status page.
    pub fn stats(&self) -> (usize, usize, usize) {
        let entries: Vec<Arc<Entry>> = self
            .map
            .lock()
            .expect("index map")
            .values()
            .cloned()
            .collect();
        let mut out = (0, 0, 0);
        for e in entries {
            if let Some(b) = e.base.read().expect("index base").as_ref() {
                out = (out.0 + 1, out.1 + b.passages(), out.2 + b.bytes());
            }
        }
        out
    }

    /// The index of a base, read from the store on first use.
    pub async fn get(
        self: &Arc<Self>,
        store: &Store,
        kb: &str,
        dims: usize,
    ) -> Result<Lookup, KbError> {
        if !self.enabled() {
            return Ok(Lookup::Unavailable);
        }
        let entry = self.entry(kb);
        entry.used.store(
            self.clock.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        if let Some(b) = entry.base.read().expect("index base").clone() {
            self.check_later(store, kb, &entry, dims);
            return Ok(Lookup::Ready(b));
        }
        let Ok(guard) = entry.loading.try_lock() else {
            return Ok(Lookup::Loading);
        };
        let summary = match store.get_kb(kb).await {
            Ok(s) => s,
            Err(e) => {
                drop(guard);
                self.map.lock().expect("index map").remove(kb);
                return Err(e);
            }
        };
        // Text, postings and one vector per passage.
        let estimate = summary.chunks * (dims as u64 * 4 + 4_000);
        if estimate > self.cap_bytes {
            return Ok(Lookup::Unavailable);
        }
        if summary.chunks < INLINE_BELOW {
            self.load(store, kb, &entry, dims, summary.chunks).await?;
            drop(guard);
            return Ok(match entry.base.read().expect("index base").clone() {
                Some(b) => Lookup::Ready(b),
                None => Lookup::Loading,
            });
        }
        drop(guard);
        let (me, store, kb) = (self.clone(), store.clone(), kb.to_string());
        tokio::spawn(async move {
            let entry = me.entry(&kb);
            let Ok(_guard) = entry.loading.try_lock() else {
                return;
            };
            if let Err(e) = me.load(&store, &kb, &entry, dims, summary.chunks).await {
                tracing::warn!(kb = %kb, error = %e, "the knowledge index could not be loaded; searches use the store");
            }
        });
        Ok(Lookup::Loading)
    }

    async fn load(
        self: &Arc<Self>,
        store: &Store,
        kb: &str,
        entry: &Arc<Entry>,
        dims: usize,
        chunks: u64,
    ) -> Result<(), KbError> {
        let t0 = Instant::now();
        for _ in 0..3 {
            let generation = entry.generation.load(Ordering::Acquire);
            let docs = store.list_documents(kb).await?;
            let mut by_doc: HashMap<String, Vec<Row>> = HashMap::new();
            if chunks < INLINE_BELOW {
                for (doc, row) in store.index_rows(kb, None).await? {
                    by_doc.entry(doc).or_default().push(row);
                }
            } else {
                for d in &docs {
                    let rows = store.index_rows(kb, Some(&d.id)).await?;
                    by_doc.insert(d.id.clone(), rows.into_iter().map(|x| x.1).collect());
                }
            }
            let me = self.clone();
            let titles: Vec<(String, String)> = docs.into_iter().map(|d| (d.id, d.title)).collect();
            let base = tokio::task::spawn_blocking(move || {
                let parts = titles
                    .iter()
                    .filter_map(|(id, title)| {
                        let rows = by_doc.remove(id)?;
                        Some(Arc::new(Part::build(id, title, rows, dims, &me.analyzer)))
                    })
                    .collect();
                let base = Base::new(parts);
                base.graph();
                base
            })
            .await
            .map_err(|e| KbError::Store(format!("building the index stopped: {e}")))?;
            if entry.generation.load(Ordering::Acquire) != generation {
                continue; // the base changed while it was being read: read it again
            }
            tracing::info!(
                kb,
                passages = base.passages(),
                vectors = base.vectors(),
                terms = base.graph().terms(),
                mb = base.bytes() / 1_048_576,
                ms = t0.elapsed().as_millis() as u64,
                "knowledge index loaded"
            );
            *entry.base.write().expect("index base") = Some(Arc::new(base));
            *entry.checked.lock().expect("index check") = Instant::now();
            self.evict(kb);
            return Ok(());
        }
        Ok(())
    }

    /// Drop the least recently used bases until the indexes fit, never the one just loaded.
    fn evict(&self, keep: &str) {
        loop {
            let entries: Vec<(String, Arc<Entry>)> = self
                .map
                .lock()
                .expect("index map")
                .iter()
                .map(|(k, e)| (k.clone(), e.clone()))
                .collect();
            let mut held: Vec<(String, u64, u64)> = entries
                .iter()
                .filter_map(|(k, e)| {
                    let b = e.base.read().expect("index base").clone()?;
                    Some((k.clone(), e.used.load(Ordering::Relaxed), b.bytes() as u64))
                })
                .collect();
            if held.iter().map(|x| x.2).sum::<u64>() <= self.cap_bytes {
                return;
            }
            held.retain(|x| x.0 != keep);
            let Some(oldest) = held.into_iter().min_by_key(|x| x.1) else {
                return;
            };
            self.drop_kb(&oldest.0);
        }
    }

    /// Forget a base's index (it is read again on next use).
    pub fn drop_kb(&self, kb: &str) {
        if let Some(e) = self.map.lock().expect("index map").get(kb) {
            e.generation.fetch_add(1, Ordering::AcqRel);
            *e.base.write().expect("index base") = None;
        }
    }

    /// Forget a base altogether (it was deleted).
    pub fn remove_kb(&self, kb: &str) {
        self.drop_kb(kb);
        self.map.lock().expect("index map").remove(kb);
    }

    fn swap(&self, kb: &str, change: impl FnOnce(&mut Vec<Arc<Part>>) -> bool) -> bool {
        let Some(e) = self.map.lock().expect("index map").get(kb).cloned() else {
            return true;
        };
        e.generation.fetch_add(1, Ordering::AcqRel);
        let mut slot = e.base.write().expect("index base");
        let Some(old) = slot.as_ref() else {
            return true;
        };
        let mut parts = old.parts.clone();
        if change(&mut parts) {
            *slot = Some(Arc::new(Base::new(parts)));
            true
        } else {
            false
        }
    }

    /// A document was read, re-read or annotated: take its rows from the store again. Nothing to do for a base that
    /// is not loaded.
    pub async fn refresh_doc(&self, store: &Store, kb: &str, doc: &str, dims: usize) {
        if self.loaded(kb).is_none() {
            // A load in flight must not install rows read before this change.
            if let Some(e) = self.map.lock().expect("index map").get(kb) {
                e.generation.fetch_add(1, Ordering::AcqRel);
            }
            return;
        }
        let fresh = async {
            let d = store.get_document(kb, doc).await?;
            let rows = store.index_rows(kb, Some(doc)).await?;
            Ok::<_, KbError>((d.title, rows))
        };
        match fresh.await {
            Ok((title, rows)) => {
                let part = Arc::new(Part::build(
                    doc,
                    &title,
                    rows.into_iter().map(|x| x.1).collect(),
                    dims,
                    &self.analyzer,
                ));
                self.swap(kb, |parts| {
                    parts.retain(|p| p.doc_id != doc);
                    parts.push(part);
                    true
                });
            }
            Err(KbError::NotFound(_)) => self.remove_doc(kb, doc),
            // What the index holds may be stale now: read the base again on next use.
            Err(_) => self.drop_kb(kb),
        }
    }

    pub fn remove_doc(&self, kb: &str, doc: &str) {
        self.swap(kb, |parts| {
            parts.retain(|p| p.doc_id != doc);
            true
        });
    }

    /// Vectors were written for passages of a document; `false` when the index does not know one of them (the caller
    /// then refreshes the document).
    pub fn apply_vectors(&self, kb: &str, doc: &str, rows: &[(String, Vec<f32>)]) -> bool {
        self.swap(kb, |parts| {
            let Some(i) = parts.iter().position(|p| p.doc_id == doc) else {
                return false;
            };
            let mut part = (*parts[i]).clone();
            if !part.set_vectors(rows) {
                return false;
            }
            parts[i] = Arc::new(part);
            true
        })
    }

    /// Now and then, off the search's path, compare what the index holds with the store's counts per document and
    /// refresh the documents that differ (a write made by another process, or a change this service missed).
    fn check_later(self: &Arc<Self>, store: &Store, kb: &str, entry: &Arc<Entry>, dims: usize) {
        {
            let mut at = entry.checked.lock().expect("index check");
            if at.elapsed() < CHECK_EVERY {
                return;
            }
            *at = Instant::now();
        }
        let (me, store, kb) = (self.clone(), store.clone(), kb.to_string());
        tokio::spawn(async move {
            let Ok(docs) = store.list_documents(&kb).await else {
                return;
            };
            let Some((_, base)) = me.loaded(&kb) else {
                return;
            };
            let held: HashMap<String, (usize, usize)> = base
                .counts()
                .into_iter()
                .map(|(d, n, v)| (d, (n, v)))
                .collect();
            for d in &docs {
                let busy = !matches!(d.status.as_str(), "ready" | "failed");
                let same = held.get(&d.id).is_some_and(|(n, v)| {
                    *n as u64 == d.n_chunks && (busy || *v as u64 == d.n_embedded)
                });
                if !(same || busy && d.n_chunks == 0) {
                    me.refresh_doc(&store, &kb, &d.id, dims).await;
                }
            }
            for id in held.keys() {
                if !docs.iter().any(|d| &d.id == id) {
                    me.remove_doc(&kb, id);
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------------------------- layers (Phase 29)

/// A document's layers (spec `context_fs_spec.md` §3), made from what the index holds with no model call.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DocLayers {
    pub doc_id: String,
    pub title: String,
    /// L0: at most 300 characters.
    pub l0: String,
    /// L1: at most 2,000 characters of Markdown.
    pub l1: String,
    /// The most distinctive terms, best first, in their commonest written form.
    pub terms: Vec<String>,
    /// (heading path, ord of the first passage under it), in document order.
    pub sections: Vec<(String, u64)>,
    pub passages: usize,
}

const L0_MAX: usize = 300;
/// Words that name a part of a document, not what it is about.
const STRUCTURAL: &[&str] = &[
    "contents",
    "references",
    "abstract",
    "introduction",
    "conclusion",
    "conclusions",
    "appendix",
    "section",
    "sections",
    "chapter",
    "table",
    "figure",
    "results",
    "discussion",
    "background",
    "overview",
    "summary",
    "phase",
    "allowbreak",
    "textbf",
    "emph",
    "cite",
    "label",
    "ref",
    "https",
    "http",
    "available",
    "online",
    "arxiv",
    "preprint",
    "proceedings",
    "conference",
    "journal",
    "vol",
    "doi",
];

/// A passage under a references or bibliography heading: its words name other works.
fn is_references(heading: &str) -> bool {
    let h = heading
        .rsplit(" > ")
        .next()
        .unwrap_or(heading)
        .to_lowercase();
    [
        "references",
        "bibliography",
        "works cited",
        "literature cited",
    ]
    .iter()
    .any(|r| {
        h.trim_start_matches(|c: char| !c.is_alphabetic())
            .starts_with(r)
    })
}
const L1_MAX: usize = 2000;

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max.saturating_sub(3)).collect();
    // At a sentence or word boundary when one is near.
    let at = cut
        .rfind(". ")
        .filter(|&i| i > max / 2)
        .map(|i| i + 1)
        .or_else(|| cut.rfind(' '))
        .unwrap_or(cut.len());
    format!("{}...", cut[..at].trim_end())
}

/// The first paragraph of prose in a passage: not a heading, a table row or a list of numbers.
fn first_paragraph(text: &str) -> Option<String> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|b| !b.starts_with('#') && !b.starts_with('|') && !b.starts_with("```"))
        .map(|b| {
            b.lines()
                .filter(|l| !l.trim_start().starts_with('#'))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .find(|b| b.chars().filter(|c| c.is_alphabetic()).count() >= 60)
}

fn first_sentence(p: &str) -> String {
    let b = p.as_bytes();
    for i in 30..b.len().saturating_sub(1) {
        if matches!(b[i], b'.' | b'!' | b'?') && b[i + 1] == b' ' {
            return p[..=i].to_string();
        }
    }
    p.to_string()
}

fn captions(text: &str, out: &mut Vec<String>) {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"^(?:Figure|Fig\.|FIGURE|Table|TABLE)\s+[0-9IVX]+[.:](?:\s|$)")
            .expect("valid regex")
    });
    for l in text.lines().map(str::trim) {
        if out.len() < 6 && re.is_match(l) && !out.iter().any(|c| c == l) {
            out.push(clip(l, 160));
        }
    }
}

impl Base {
    /// (document id, title) of every document the index holds.
    pub fn documents_list(&self) -> Vec<(String, String)> {
        self.parts
            .iter()
            .map(|p| (p.doc_id.clone(), p.title.clone()))
            .collect()
    }

    /// Passage `ord` of a document, as a hit.
    pub fn passage_at(&self, doc: &str, ord: u64) -> Option<Hit> {
        let (pi, part) = self
            .parts
            .iter()
            .enumerate()
            .find(|(_, p)| p.doc_id == doc)?;
        let i = part.rows.iter().position(|r| r.ord == ord)?;
        Some(self.passage(self.offsets[pi] + i as u32))
    }

    /// The passages of a document, in order, as hits.
    pub fn passages_of(&self, doc: &str) -> Vec<Hit> {
        let Some((pi, part)) = self.parts.iter().enumerate().find(|(_, p)| p.doc_id == doc) else {
            return Vec::new();
        };
        (0..part.len() as u32)
            .map(|i| self.passage(self.offsets[pi] + i))
            .collect()
    }

    /// How many passages hold `term` (a stem as the analyzer makes it).
    fn df(&self, term: &str) -> usize {
        self.parts
            .iter()
            .map(|p| p.postings.get(term).map_or(0, Vec::len))
            .sum()
    }

    /// Terms of `texts` ranked by frequency times inverse passage frequency in this base, shown in their commonest
    /// written form. Numbers, short terms and terms in more than half of the base's passages are left out.
    fn distinctive(&self, an: &Analyzer, texts: &[&str], k: usize) -> Vec<String> {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut surface: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for t in texts {
            for w in t.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_')) {
                let w = w.trim_matches('-');
                let digits = w.chars().filter(char::is_ascii_digit).count();
                let roman = w.len() <= 6 && w.chars().all(|c| "IVXLC".contains(c));
                if w.chars().count() < 4
                    || digits * 2 >= w.len()
                    || roman
                    || STRUCTURAL.contains(&w.to_lowercase().as_str())
                {
                    continue;
                }
                let mut stems = Vec::new();
                an.terms(w, |s| stems.push(s));
                // A compound ("Qwen3-1.7B") is its own term beside its parts; the whole is the last one emitted.
                if let Some(stem) = stems.pop() {
                    *tf.entry(stem.clone()).or_default() += 1;
                    *surface
                        .entry(stem)
                        .or_default()
                        .entry(w.to_string())
                        .or_default() += 1;
                }
            }
        }
        let n = self.n.max(1) as f64;
        let mut scored: Vec<(String, f64)> = tf
            .into_iter()
            .filter_map(|(t, f)| {
                let df = self.df(&t);
                if df == 0 || df as f64 > n / 2.0 || f < 2 {
                    return None;
                }
                let idf = (1.0 + (n - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
                Some((t, f as f64 * idf))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
            .into_iter()
            .take(k)
            .filter_map(|(t, _)| {
                surface.get(&t).and_then(|m| {
                    m.iter()
                        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                        .map(|(w, _)| w.clone())
                })
            })
            .collect()
    }

    /// L0 and L1 of a document (spec §3).
    pub fn doc_layers(&self, an: &Analyzer, doc: &str) -> Option<DocLayers> {
        let part = self.parts.iter().find(|p| p.doc_id == doc)?;
        let mut sections: Vec<(String, u64)> = Vec::new();
        for r in &part.rows {
            // The first passage under each heading path, once.
            if !r.heading.is_empty() && !sections.iter().any(|s| s.0 == r.heading) {
                sections.push((r.heading.clone(), r.ord));
            }
        }
        let texts: Vec<&str> = part
            .rows
            .iter()
            .filter(|r| !is_references(&r.heading))
            .map(|r| r.text.as_str())
            .collect();
        let terms = self.distinctive(an, &texts, 12);
        let para = part
            .rows
            .iter()
            .find_map(|r| first_paragraph(&r.text))
            .unwrap_or_default();
        let mut caps = Vec::new();
        for r in &part.rows {
            captions(&r.text, &mut caps);
        }
        let title = part.title.clone();
        // The topics are kept whole; the sentence takes what is left of the 300 characters.
        let topics = if terms.is_empty() {
            String::new()
        } else {
            format!(
                " Topics: {}.",
                terms.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
            )
        };
        let mut l0 = clip(&title, 150);
        if !para.is_empty() {
            let room = L0_MAX.saturating_sub(l0.chars().count() + topics.chars().count() + 3);
            if room >= 40 {
                l0.push_str(" — ");
                l0.push_str(&clip(&first_sentence(&para), room));
            }
        }
        l0.push_str(&topics);
        let mut l1 = format!("# {title}\n\n{}\n", clip(&para, 400));
        if !sections.is_empty() {
            l1.push_str("\n## Contents\n\n");
            for (h, _) in sections.iter().take(24) {
                let depth = h.matches(" > ").count();
                let leaf = h.rsplit(" > ").next().unwrap_or(h);
                l1.push_str(&format!("{}- {leaf}\n", "  ".repeat(depth)));
            }
            if sections.len() > 24 {
                l1.push_str(&format!(
                    "- ... and {} more sections\n",
                    sections.len() - 24
                ));
            }
        }
        if !caps.is_empty() {
            l1.push_str("\n## Tables and figures\n\n");
            for c in &caps {
                l1.push_str(&format!("- {c}\n"));
            }
        }
        if !terms.is_empty() {
            l1.push_str(&format!("\n## Key terms\n\n{}\n", terms.join(", ")));
        }
        Some(DocLayers {
            doc_id: doc.to_string(),
            title,
            l0: clip(&l0, L0_MAX),
            l1: clip(&l1, L1_MAX),
            terms,
            sections,
            passages: part.len(),
        })
    }

    /// L0 and L1 of the whole base (spec §3): one line per document with its L0 (32 evenly spaced documents above
    /// that), and the base's own distinctive terms.
    pub fn base_layers(&self, an: &Analyzer, name: &str, description: &str) -> (String, String) {
        let mut docs: Vec<DocLayers> = self
            .parts
            .iter()
            .filter_map(|p| self.doc_layers(an, &p.doc_id))
            .collect();
        docs.sort_by_key(|d| d.title.to_lowercase());
        let shown: Vec<&DocLayers> = if docs.len() <= 32 {
            docs.iter().collect()
        } else {
            (0..32).map(|i| &docs[i * (docs.len() - 1) / 31]).collect()
        };
        let texts: Vec<&str> = self
            .parts
            .iter()
            .flat_map(|p| p.rows.iter().map(|r| r.heading.as_str()))
            .collect();
        let terms = self.distinctive(an, &texts, 8);
        let mut l0 = format!("{name}: {} documents, {} passages", docs.len(), self.n);
        if !terms.is_empty() {
            l0.push_str(&format!(". Topics: {}.", terms.join(", ")));
        }
        let mut l1 = format!("# {name}\n\n");
        if !description.trim().is_empty() {
            l1.push_str(description.trim());
            l1.push_str("\n\n");
        }
        l1.push_str("## Documents\n\n");
        for d in &shown {
            l1.push_str(&format!("- {}\n", d.l0));
        }
        if docs.len() > shown.len() {
            l1.push_str(&format!(
                "- ... {} more documents not shown\n",
                docs.len() - shown.len()
            ));
        }
        (clip(&l0, L0_MAX), clip(&l1, 4 * L1_MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, ord: u64, heading: &str, text: &str, v: Option<Vec<f32>>) -> Row {
        Row {
            id: id.into(),
            ord,
            heading: heading.into(),
            page: None,
            context: String::new(),
            text: text.into(),
            embedding: v,
            pieces: Vec::new(),
        }
    }

    fn base() -> Base {
        let an = Analyzer::default();
        let a = Part::build(
            "a",
            "Storage notes",
            vec![
                row("a0", 0, "Store", "SurrealDB listens on port 8300 and keeps the passages. The store is SurrealDB.", Some(vec![1.0, 0.0])),
                row("a1", 1, "Store > Index", "The vector_weight setting is 0.1 for hybrid searches over the HNSW index.", Some(vec![0.0, 1.0])),
                row("a2", 2, "Chunks", "Passages are 1,200 characters long and overlap by 150.", None),
            ],
            2,
            &an,
        );
        let b = Part::build(
            "b",
            "Models",
            vec![
                row(
                    "b0",
                    0,
                    "Embedding",
                    "Vectors come from Qwen3-Embedding-0.6B, served beside SurrealDB.",
                    Some(vec![0.6, 0.8]),
                ),
                row(
                    "b1",
                    1,
                    "Quick model",
                    "The quick model is Qwen3-1.7B. It reads passages for the HNSW index notes.",
                    Some(vec![0.8, 0.6]),
                ),
            ],
            2,
            &an,
        );
        Base::new(vec![Arc::new(a), Arc::new(b)])
    }

    #[test]
    fn a_scoped_search_ranks_only_inside_its_documents() {
        let b = base();
        let an = Analyzer::default();
        let p = Params::default();
        let all = b.search(&an, "SurrealDB passages", None, SearchMode::Lexical, 5, &p);
        assert!(all.iter().any(|h| h.doc_id == "a") && all.iter().any(|h| h.doc_id == "b"));
        let scope: std::collections::HashSet<String> = ["b".to_string()].into();
        let only = b.search_in(
            &an,
            "SurrealDB passages",
            Some(&[0.0, 1.0]),
            SearchMode::Hybrid,
            5,
            &p,
            Some(&scope),
        );
        assert!(
            !only.is_empty() && only.iter().all(|h| h.doc_id == "b"),
            "{only:?}"
        );
    }

    #[test]
    fn layers_come_from_headings_text_and_terms() {
        let b = base();
        let an = Analyzer::default();
        let l = b.doc_layers(&an, "a").unwrap();
        assert_eq!(
            l.sections.iter().map(|s| s.0.as_str()).collect::<Vec<_>>(),
            ["Store", "Store > Index", "Chunks"]
        );
        assert!(
            l.l1.contains("## Contents") && l.l1.contains("  - Index"),
            "{}",
            l.l1
        );
        assert!(
            l.l0.starts_with("Storage notes") && l.l0.chars().count() <= 300,
            "{}",
            l.l0
        );
        assert_eq!(l.passages, 3);
        let (l0, l1) = b.base_layers(&an, "Notes", "Two documents");
        assert!(l0.starts_with("Notes: 2 documents, 5 passages"), "{l0}");
        assert!(
            l1.contains("Models") && l1.contains("Storage notes"),
            "{l1}"
        );
        assert_eq!(b.passage_at("a", 1).unwrap().heading, "Store > Index");
        assert_eq!(b.passages_of("b").len(), 2);
    }

    #[test]
    fn a_passage_is_scored_by_its_best_sentence_window_when_mixed() {
        let an = Analyzer::default();
        let mut far = row(
            "p0",
            0,
            "",
            "Long passage about many things.",
            Some(vec![1.0, 0.0]),
        );
        // One of its windows says exactly what the question asks.
        far.pieces = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let near = row(
            "p1",
            1,
            "",
            "A passage somewhat close to the question.",
            Some(vec![0.6, 0.8]),
        );
        let base = Base::new(vec![Arc::new(Part::build(
            "d",
            "Doc",
            vec![far, near],
            2,
            &an,
        ))]);
        let q = [0.0, 1.0];
        let own = base.similarities_mixed(&q, 0.0).unwrap();
        assert!(own[1] > own[0]);
        let mixed = base.similarities_mixed(&q, 1.0).unwrap();
        assert!(
            (mixed[0] - 1.0).abs() < 1e-6 && mixed[0] > mixed[1],
            "{mixed:?}"
        );
        let half = base.similarities_mixed(&q, 0.5).unwrap();
        assert!((half[0] - 0.5).abs() < 1e-6, "{half:?}");
        // A passage without windows keeps its own score.
        assert!((mixed[1] - own[1]).abs() < 1e-6);
    }

    #[test]
    fn identifiers_are_terms_of_their_own_and_words_are_stemmed() {
        let an = Analyzer::default();
        let q = an.query("What is the vector_weight of Qwen3-1.7B passages?");
        assert!(q.contains(&"vector_weight".to_string()), "{q:?}");
        assert!(q.contains(&"vector".to_string()) && q.contains(&"weight".to_string()));
        assert!(q.contains(&"qwen3-1.7b".to_string()) && q.contains(&"qwen".to_string()));
        assert!(q.contains(&"1.7".to_string()) || q.contains(&"7".to_string()));
        assert!(q.contains(&"passag".to_string()), "stemmed: {q:?}");
        assert!(!q.iter().any(|t| t == "what" || t == "the" || t == "of"));
        assert_eq!(an.query("SurrealDB's"), vec!["surreal", "db", "surrealdb"]);
        assert_eq!(an.query("café"), vec!["cafe"]);
    }

    #[test]
    fn the_words_find_the_passage_and_a_rarer_word_counts_more() {
        let (b, an) = (base(), Analyzer::default());
        let hits = b.search(
            &an,
            "which port does SurrealDB listen on",
            None,
            SearchMode::Lexical,
            3,
            &Params::default(),
        );
        assert_eq!(hits[0].chunk_id, "a0");
        assert!(hits[0].lexical_score.unwrap() > hits[1].lexical_score.unwrap());
        let hits = b.search(
            &an,
            "vector_weight",
            None,
            SearchMode::Lexical,
            3,
            &Params::default(),
        );
        // The identifier as a whole is rarer than its parts ("Vectors" is in b0 too), so its passage leads.
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, "a1");
        assert!(b
            .search(
                &an,
                "zebra",
                None,
                SearchMode::Lexical,
                3,
                &Params::default()
            )
            .is_empty());
    }

    #[test]
    fn a_heading_weight_moves_a_heading_match_up() {
        let b = base();
        let terms = vec![("index".to_string(), 1.0)];
        let flat = b.lexical_scores(&terms, &Bm25::default());
        let heavy = b.lexical_scores(
            &terms,
            &Bm25 {
                weights: [1.0, 4.0, 1.0, 1.0],
                ..Bm25::default()
            },
        );
        let a1 = b.number_of("a1").unwrap() as usize;
        let b1 = b.number_of("b1").unwrap() as usize;
        assert!(heavy[a1] / heavy[b1] > flat[a1] / flat[b1]);
    }

    #[test]
    fn the_vectors_are_scanned_exactly_and_a_passage_without_one_is_left_out() {
        let (b, an) = (base(), Analyzer::default());
        let hits = b.search(
            &an,
            "x",
            Some(&[1.0, 0.1]),
            SearchMode::Vector,
            5,
            &Params::default(),
        );
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[0].chunk_id, "a0");
        assert_eq!(hits[1].chunk_id, "b1");
        assert!(hits.iter().all(|h| h.chunk_id != "a2"));
        assert!((hits[0].vector_score.unwrap() - 0.995).abs() < 0.01);
    }

    #[test]
    fn both_fusions_keep_the_words_first_when_the_vector_arm_is_weighted_down() {
        let (b, an) = (base(), Analyzer::default());
        // The words say a0; the vector says a1.
        for fusion in [
            Fusion::Rrf { vector_weight: 0.1 },
            Fusion::Convex { alpha: 0.2 },
        ] {
            let p = Params {
                fusion,
                ..Params::default()
            };
            let hits = b.search(
                &an,
                "SurrealDB port",
                Some(&[0.0, 1.0]),
                SearchMode::Hybrid,
                3,
                &p,
            );
            assert_eq!(hits[0].chunk_id, "a0", "{fusion:?}");
            assert!(hits.iter().any(|h| h.chunk_id == "a1"));
            assert!(hits[0].vector_score.is_some() && hits[0].lexical_score.is_some());
        }
        let p = Params {
            fusion: Fusion::Convex { alpha: 0.9 },
            ..Params::default()
        };
        let hits = b.search(
            &an,
            "SurrealDB port",
            Some(&[0.0, 1.0]),
            SearchMode::Hybrid,
            3,
            &p,
        );
        assert_eq!(hits[0].chunk_id, "a1");
    }

    #[test]
    fn terms_are_names_acronyms_and_identifiers_not_sentence_openers() {
        let mut c = Vec::new();
        candidates(
            "However, the Smiley Core engine and SurrealDB use HNSW. Passages name Qwen3-1.7B and vector_weight in Phase 7.",
            false,
            &mut c,
        );
        let keys: Vec<&str> = c.iter().map(|x| &*x.key).collect();
        for k in [
            "smiley core",
            "surrealdb",
            "hnsw",
            "qwen3-1.7b",
            "vector_weight",
            "phase 7",
        ] {
            assert!(keys.contains(&k), "{k} missing from {keys:?}");
        }
        assert!(!keys.contains(&"however"));
        // A lone capitalised word at the start of a sentence is only a term when it is one elsewhere.
        let passages = c
            .iter()
            .find(|x| &*x.key == "passages")
            .expect("kept as a weak candidate");
        assert!(!passages.strong);
    }

    #[test]
    fn the_graph_links_passages_that_share_a_term_and_the_walk_reaches_them() {
        let b = base();
        let g = b.graph();
        let named: Vec<&str> = g.names.iter().map(String::as_str).collect();
        assert!(
            named.contains(&"SurrealDB") && named.contains(&"HNSW"),
            "{named:?}"
        );
        // Start at a0 (SurrealDB): b0 shares that term, a2 shares nothing.
        let a0 = b.number_of("a0").unwrap();
        let r = g.ppr(&[(a0, 1.0)], &[], 0.5, 20);
        let (b0, a2) = (
            b.number_of("b0").unwrap() as usize,
            b.number_of("a2").unwrap() as usize,
        );
        assert!(r[b0] > 0.0 && r[a2] == 0.0);
        assert!(r[a0 as usize] > r[b0]);
        assert!((r.iter().sum::<f32>()) <= 1.0 + 1e-4);
        // A question that names a term seeds it.
        assert_eq!(g.query_terms("what does hnsw do?").len(), 1);
        let view = b.graph_view(None, 10);
        assert!(view["nodes"].as_array().unwrap().len() >= 2);
        assert_eq!(view["source"], "text");
    }

    #[test]
    fn the_graph_arm_adds_a_linked_passage_without_moving_the_first() {
        let (b, an) = (base(), Analyzer::default());
        let plain = Params::default();
        let with = Params {
            ppr: Some(Ppr::default()),
            ..Params::default()
        };
        let q = "which port does the store listen on";
        let qv = [1.0, 0.0];
        let h0 = b.search(&an, q, Some(&qv), SearchMode::Hybrid, 5, &plain);
        let h1 = b.search(&an, q, Some(&qv), SearchMode::Hybrid, 5, &with);
        assert_eq!(h0[0].chunk_id, "a0");
        assert_eq!(h1[0].chunk_id, "a0");
        assert!(h1.iter().any(|h| h.chunk_id == "b0"));
    }

    #[test]
    fn a_mistyped_word_finds_the_passage_and_a_known_word_is_left_alone() {
        let b = base();
        let an = Analyzer::default();
        let on = Params::default();
        let off = Params {
            fuzzy: false,
            ..Params::default()
        };
        let right = b.search(&an, "listens", None, SearchMode::Lexical, 3, &on);
        assert!(!right.is_empty(), "the test base must hold the word");
        for typo in ["lisetns", "listns", "lirtens"] {
            let hits = b.search(&an, typo, None, SearchMode::Lexical, 3, &on);
            assert_eq!(
                hits.first().map(|h| &h.chunk_id),
                right.first().map(|h| &h.chunk_id),
                "{typo}"
            );
            assert!(b
                .search(&an, typo, None, SearchMode::Lexical, 3, &off)
                .is_empty());
        }
        assert_eq!(b.nearest_term(&an.query("listens")[0]), None);
        // A short word, an identifier and a word far from every word of the base are taken as typed.
        assert_eq!(b.nearest_term("gatx"), None);
        assert_eq!(b.nearest_term("zzzzzzzzzz"), None);
        assert_eq!(
            edit_distance(&['a', 'b', 'c'], &['a', 'c', 'b'], 1),
            Some(1)
        );
        assert_eq!(edit_distance(&['a', 'b', 'c'], &['x', 'y', 'z'], 2), None);
    }

    #[test]
    fn feedback_terms_join_the_query_with_less_weight() {
        let (b, an) = (base(), Analyzer::default());
        let p = Params {
            prf: Some(Prf {
                passages: 1,
                terms: 3,
                weight: 0.3,
            }),
            ..Params::default()
        };
        let terms = b.weighted_terms(&an, "port 8300", &p);
        assert!(terms.len() > 2);
        assert!(terms[2..].iter().all(|t| t.1 <= 0.3 + 1e-6));
    }

    #[test]
    fn vectors_can_be_set_on_a_part_and_unknown_passages_are_refused() {
        let b = base();
        let mut part = (*b.parts[0]).clone();
        assert_eq!(part.vectors(), 2);
        assert!(part.set_vectors(&[("a2".into(), vec![3.0, 4.0])]));
        assert_eq!(part.vectors(), 3);
        assert!(!part.set_vectors(&[("nope".into(), vec![1.0, 0.0])]));
        assert!(!part.set_vectors(&[("a2".into(), vec![1.0])]));
    }
}
