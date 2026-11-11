//! Does a question lean on the turn before it? (Phase 26.)
//!
//! "And how fast is it?" cannot be searched on its own: its words name nothing. The knowledge search therefore reads
//! such a question together with the previous user turn. The test is by rule, with no model and no embedding: a
//! question leans on the previous turn when it is too short to name a subject, or when it points back with a
//! reference word and brings few words of its own. A phrase that announces a new subject turns the test off.
//!
//! After the topic-continuity signal of vLLM Semantic Router (reference words, change markers, a floor of three
//! content words), reduced to the one decision this system needs.

const STOP: &[&str] = &[
    "the", "and", "for", "with", "what", "how", "many", "much", "from", "are", "was", "were",
    "you", "your", "into", "than", "who", "has", "have", "had", "not", "but", "can", "will",
    "would", "could", "should", "which", "when", "where", "why", "also", "does", "did", "about",
    "any", "all", "more", "most", "is", "of", "in", "on", "to", "do", "be", "as", "at", "by", "or",
    "an", "so", "if", "then", "there", "please", "tell", "me", "explain", "again",
];

const REFERENCES: &[&str] = &[
    "it", "its", "that", "this", "those", "these", "they", "them", "their", "he", "she", "his",
    "her", "one", "ones", "same", "former", "latter", "such", "either", "both", "above",
];

const OPENERS: &[&str] = &[
    "and ",
    "what about",
    "how about",
    "but ",
    "so ",
    "then ",
    "also ",
    "why is that",
    "why not",
];

const CHANGES: &[&str] = &[
    "new topic",
    "new question",
    "another question",
    "different question",
    "different topic",
    "unrelated",
    "by the way",
    "switching",
    "change of subject",
    "something else",
    "moving on",
];

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `question` needs the previous user turn to be understood.
pub fn leans_on_previous(question: &str) -> bool {
    let lower = question.trim().to_lowercase();
    if lower.is_empty() || CHANGES.iter().any(|c| lower.contains(c)) {
        return false;
    }
    let ws = words(&lower);
    let reference = ws.iter().any(|w| REFERENCES.contains(&w.as_str()));
    let content = ws
        .iter()
        .filter(|w| {
            w.chars().count() >= 3
                && !STOP.contains(&w.as_str())
                && !REFERENCES.contains(&w.as_str())
        })
        .count();
    if content < 3 {
        return true;
    }
    let opener = OPENERS.iter().any(|o| lower.starts_with(o));
    (reference || opener) && content <= 6
}

/// What the knowledge search reads for `question`: the question itself, or the previous user turn and the question
/// when the question leans on it. The flag says which.
pub fn search_text(question: &str, previous: Option<&str>) -> (String, bool) {
    match previous.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) if leans_on_previous(question) => {
            let clipped: String = p.chars().take(400).collect();
            (format!("{clipped}\n{question}"), true)
        }
        _ => (question.to_string(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_question_that_points_back_is_read_with_the_previous_turn() {
        for q in [
            "And how fast is it?",
            "Why was it left off by default?",
            "What about on the larger model?",
            "Why?",
            "How much did that cost in latency?",
            "Is the same true for code?",
        ] {
            assert!(leans_on_previous(q), "{q}");
        }
        let (text, joined) = search_text("And how fast is it?", Some("What is the prefix store?"));
        assert!(joined && text.starts_with("What is the prefix store?") && text.ends_with("it?"));
    }

    #[test]
    fn a_question_that_names_its_subject_is_read_alone() {
        for q in [
            "Which post-training converter turned Qwen3-1.7B into LittleBit factors at 1.0 bit per weight?",
            "How many passages does the hybrid search over the knowledge base return by default?",
            "What is the lowest window confidence signal used by the gate for math drafts?",
            "New topic: how does it handle uploads?",
        ] {
            assert!(!leans_on_previous(q), "{q}");
        }
        assert_eq!(
            search_text("And how fast is it?", None),
            ("And how fast is it?".to_string(), false)
        );
        assert!(!search_text("And how fast is it?", Some("  ")).1);
    }
}
