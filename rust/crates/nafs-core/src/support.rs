//! Is an answer from documents supported by the passages it was given? (Phase 26.)
//!
//! An answer written from retrieved passages has no exact verifier, so until now it was always "not checked". This
//! check needs no model: every sentence of the answer that makes a claim must find its content words in one passage,
//! and every number it states must occur in the passages (or in the question). A sentence that fails either test is
//! reported with the reason. The check says "this sentence is not in your sources"; it does not say the sentence is
//! false, and a sentence that only rewords a passage with other words can fail it.
//!
//! vLLM Semantic Router runs a 149M to 307M token classifier for the same purpose; on four cores that is a model
//! call per answer. This is word and number matching, a few microseconds per sentence.

use std::collections::HashSet;

use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};

use crate::knowledge::split_sentences;

/// One sentence of the answer that makes a claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    /// The largest share of the sentence's content words found in one passage.
    pub overlap: f32,
    /// That passage's number (from 1), when any word matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passage: Option<usize>,
    /// Numbers the sentence states that no passage holds.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_numbers: Vec<String>,
    pub supported: bool,
}

/// The check of one answer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Support {
    pub claims: usize,
    pub supported: usize,
    pub sentences: Vec<Claim>,
}

impl Support {
    /// The share of claims that are supported; 1 for an answer that makes none.
    pub fn share(&self) -> f32 {
        if self.claims == 0 {
            1.0
        } else {
            self.supported as f32 / self.claims as f32
        }
    }

    /// The row a reply's `sources` carries: the counts and the sentences that are not in the passages.
    pub fn row(&self) -> serde_json::Value {
        let unsupported: Vec<serde_json::Value> = self
            .sentences
            .iter()
            .filter(|c| !c.supported)
            .map(|c| {
                let reason = if c.missing_numbers.is_empty() {
                    "its words are not in the passages".to_string()
                } else {
                    format!("the passages do not state {}", c.missing_numbers.join(", "))
                };
                serde_json::json!({"text": c.text, "reason": reason})
            })
            .collect();
        serde_json::json!({"support": {"claims": self.claims, "supported": self.supported,
            "share": self.share(), "unsupported": unsupported}})
    }
}

const STOP: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "many",
    "much",
    "from",
    "that",
    "this",
    "these",
    "those",
    "are",
    "was",
    "were",
    "his",
    "her",
    "its",
    "you",
    "your",
    "into",
    "then",
    "than",
    "who",
    "has",
    "have",
    "had",
    "not",
    "but",
    "can",
    "will",
    "would",
    "could",
    "should",
    "which",
    "when",
    "where",
    "why",
    "also",
    "there",
    "their",
    "they",
    "them",
    "been",
    "being",
    "does",
    "did",
    "about",
    "such",
    "each",
    "more",
    "most",
    "some",
    "any",
    "all",
    "both",
    "only",
    "over",
    "under",
    "between",
    "because",
    "while",
    "after",
    "before",
    "per",
    "via",
    "our",
    "one",
    "two",
    "according",
    "passage",
    "passages",
    "source",
    "sources",
    "document",
    "documents",
    "states",
    "state",
    "stated",
    "says",
    "said",
    "mentions",
    "mentioned",
    "based",
    "provided",
    "given",
    "answer",
    "question",
    "following",
];

/// Sentences that decline rather than claim; an honest "the passages do not say" is not an unsupported claim.
const DECLINES: &[&str] = &[
    "not sure",
    "do not know",
    "don't know",
    "does not say",
    "do not say",
    "doesn't say",
    "don't say",
    "no information",
    "not mentioned",
    "not stated",
    "not specified",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "not provided",
    "do not mention",
    "does not mention",
    "do not state",
    "does not state",
    "unable to",
];

const NUMBER_WORDS: &[(&str, &str)] = &[
    ("zero", "0"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
    ("eleven", "11"),
    ("twelve", "12"),
    ("twenty", "20"),
    ("thirty", "30"),
    ("hundred", "100"),
    ("thousand", "1000"),
    ("half", "0.5"),
    ("third", "3"),
    ("quarter", "4"),
    ("twice", "2"),
    ("once", "1"),
];

/// The words and the numbers of a text. Words are lowercased, stripped of stop words and stemmed; a number is its
/// digits with thousands separators removed ("1,200" and "1200" are one number, "36 %" is 36), and a number word
/// counts as its digit.
struct Terms {
    words: HashSet<String>,
    numbers: HashSet<String>,
}

fn terms(text: &str, stem: &Stemmer) -> Terms {
    let (mut words, mut numbers) = (HashSet::new(), HashSet::new());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            // A number: digits, with `,` between digits dropped and one `.` between digits kept.
            let mut n = String::new();
            while i < chars.len() {
                let d = chars[i];
                let next_digit = chars.get(i + 1).is_some_and(|x| x.is_ascii_digit());
                if d.is_ascii_digit() {
                    n.push(d);
                } else if d == ',' && next_digit {
                } else if d == '.' && next_digit && !n.contains('.') {
                    n.push('.');
                } else {
                    break;
                }
                i += 1;
            }
            if n.contains('.') {
                let t = n.trim_end_matches('0').trim_end_matches('.');
                n = t.to_string();
            }
            numbers.insert(n);
            continue;
        }
        if c.is_alphabetic() {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let w: String = chars[start..i].iter().collect::<String>().to_lowercase();
            if let Some((_, d)) = NUMBER_WORDS.iter().find(|(n, _)| *n == w) {
                numbers.insert((*d).to_string());
            }
            if w.chars().count() >= 3 && !STOP.contains(&w.as_str()) {
                if w.chars().all(|c| c.is_ascii_alphabetic()) {
                    words.insert(stem.stem(&w).into_owned());
                } else {
                    words.insert(w);
                }
            }
            continue;
        }
        i += 1;
    }
    Terms { words, numbers }
}

/// The answer without what is not prose: citation markers (`[1]`, `[2, 3]`), emphasis marks, list markers and
/// heading marks at the start of a line.
fn clean(answer: &str) -> String {
    let mut out = String::with_capacity(answer.len());
    for line in answer.lines() {
        let mut l = line.trim_start();
        l = l.trim_start_matches(['#', '>', '-', '*', '•']).trim_start();
        // "1. " or "2) " at the start of a line is a list number, not a number the answer states.
        let digits = l.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && digits <= 2 {
            let rest = &l[digits..];
            if let Some(r) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
                l = r;
            }
        }
        let cs: Vec<char> = l.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            if cs[i] == '[' {
                // A citation marker holds only digits, commas, dashes and spaces.
                if let Some(end) = cs[i..].iter().position(|c| *c == ']') {
                    let inner = &cs[i + 1..i + end];
                    if !inner.is_empty()
                        && inner.iter().any(|c| c.is_ascii_digit())
                        && inner
                            .iter()
                            .all(|c| c.is_ascii_digit() || ", -–".contains(*c))
                    {
                        i += end + 1;
                        continue;
                    }
                }
            }
            if !matches!(cs[i], '*' | '`') {
                out.push(cs[i]);
            }
            i += 1;
        }
        out.push('\n');
    }
    out
}

/// Check `answer` against the `passages` it was written from. `question` supplies numbers the answer may repeat.
/// `min_overlap` is the share of a sentence's content words one passage must hold.
pub fn check(answer: &str, passages: &[&str], question: &str, min_overlap: f32) -> Support {
    let stem = Stemmer::create(Algorithm::English);
    let sources: Vec<Terms> = passages.iter().map(|p| terms(p, &stem)).collect();
    let asked = terms(question, &stem);
    let mut out = Support::default();
    let cleaned = clean(answer);
    for sentence in split_sentences(&cleaned) {
        let lower = sentence.to_lowercase();
        if sentence.ends_with('?') || DECLINES.iter().any(|d| lower.contains(d)) {
            continue;
        }
        let t = terms(sentence, &stem);
        // A sentence with fewer than three content words and no number states too little to check.
        if t.words.len() < 3 && t.numbers.is_empty() {
            continue;
        }
        let (mut best, mut passage) = (0f32, None);
        if !t.words.is_empty() {
            for (i, s) in sources.iter().enumerate() {
                let hit = t.words.iter().filter(|w| s.words.contains(*w)).count();
                let share = hit as f32 / t.words.len() as f32;
                if share > best {
                    best = share;
                    passage = Some(i + 1);
                }
            }
        } else {
            best = 1.0;
        }
        let mut missing: Vec<String> = t
            .numbers
            .iter()
            .filter(|n| {
                !asked.numbers.contains(*n) && !sources.iter().any(|s| s.numbers.contains(*n))
            })
            .cloned()
            .collect();
        missing.sort();
        let supported = best >= min_overlap && missing.is_empty();
        out.claims += 1;
        out.supported += usize::from(supported);
        out.sentences.push(Claim {
            text: sentence.to_string(),
            overlap: best,
            passage,
            missing_numbers: missing,
            supported,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const P1: &str = "The gate escalates only when the expected error saved is worth the waiting time. With an exact \
                      verifier visible, the gate matched deliberate-only accuracy at 36 % of its cost.";
    const P2: &str =
        "Passages are 1,200 characters long and overlap by 150. Search runs on four CPU cores.";

    #[test]
    fn a_sentence_restating_a_passage_is_supported_and_names_its_passage() {
        let s = check(
            "The gate matched the accuracy of always deliberating at 36% of the cost [1].",
            &[P1, P2],
            "What did the gate cost?",
            0.6,
        );
        assert_eq!((s.claims, s.supported), (1, 1), "{s:?}");
        assert_eq!(s.sentences[0].passage, Some(1));
    }

    #[test]
    fn a_number_the_passages_do_not_state_is_reported() {
        let s = check(
            "The gate matched deliberate-only accuracy at 63% of its cost.",
            &[P1, P2],
            "What did the gate cost?",
            0.6,
        );
        assert_eq!((s.claims, s.supported), (1, 0));
        assert_eq!(s.sentences[0].missing_numbers, vec!["63".to_string()]);
        assert!(s.row()["support"]["unsupported"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("63"));
    }

    #[test]
    fn a_sentence_about_something_else_is_not_supported() {
        let s = check(
            "Speculative decoding with a draft model doubles the throughput on every laptop.",
            &[P1, P2],
            "Is it fast?",
            0.6,
        );
        assert_eq!((s.claims, s.supported), (1, 0));
        assert!(s.sentences[0].missing_numbers.is_empty());
    }

    #[test]
    fn numbers_match_across_separators_words_and_the_question() {
        let s = check(
            "Passages are 1200 characters long. The search runs on 4 cores. Each of the 12 students gets fewer oranges.",
            &[P2],
            "A class of 12 students shares oranges; how long are passages?",
            0.5,
        );
        assert!(
            s.sentences[0].supported && s.sentences[1].supported,
            "{s:?}"
        );
        // 12 is in the question, but the sentence's words are in no passage.
        assert!(s.sentences[2].missing_numbers.is_empty() && !s.sentences[2].supported);
    }

    #[test]
    fn declining_questions_markup_and_list_numbers_are_not_claims() {
        let s = check(
            "**Answer**\n1. The passages do not say how many users were served.\n2. Do you want the details?\n- Yes.",
            &[P1],
            "How many users?",
            0.6,
        );
        assert_eq!(s.claims, 0, "{s:?}");
        assert_eq!(s.share(), 1.0);
    }

    #[test]
    fn with_no_passages_every_claim_is_unsupported() {
        let s = check(
            "The gate matched deliberate-only accuracy at 36 % of its cost.",
            &[],
            "cost?",
            0.6,
        );
        assert_eq!((s.claims, s.supported), (1, 0));
    }
}
