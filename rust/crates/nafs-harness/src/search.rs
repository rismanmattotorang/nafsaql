//! Best-first repair over a tree of candidates, after Open Math Harness's discovery service
//! (`program-files/discovery/src/lib.rs` `ProblemTree`, `InitialPolicy`; Apache-2.0) and its scoring protocol
//! (`scoring.rs`, ported unchanged).
//!
//! Node 0 is the task itself (no candidate). Every attempt becomes a child of the node it repaired; a fresh,
//! draft-blind attempt is a child of the root. Scores come only from the exact verifier's verdict, never from
//! what a model says. Differences from OMH, all because Nafs-Aql explores **one** candidate at a time on a
//! four-core CPU (OMH runs `W` workers per round):
//!
//! - With one slot, OMH's reserved repair slot would always win over the score ordering, so the repair
//!   prospect (fewest failures) is used as a tie-break among equal scores instead.
//! - Equal scores go to the **latest** node (OMH's prover `pick_best` rule) rather than the shallowest, so
//!   with all-or-nothing verdicts the tree repairs the last candidate, exactly as Nafs-Aql did before.
//! - A node stays continuable until it has `fanout` children (OMH: leaves only), so when a repair scores
//!   worse than its parent the search backtracks to the parent, and the failed child is shown as a sibling
//!   that was "already tried".
use crate::scoring::{self, Credit, Verdict};

/// Exact-verifier outcome of one candidate, in the scoring protocol's terms.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub verdict: Verdict,
    pub verified: bool,
    pub credit: Option<Credit>,
    /// Failing checks (failed asserts, Lean errors, schema errors); the repair prospect is the fewest.
    pub failures: u64,
    /// The verifier's report, shown to the model when this node is repaired.
    pub evidence: Option<String>,
}

impl Outcome {
    pub fn score(&self) -> f64 {
        scoring::score(self.verdict, Some(self.verified), self.credit)
    }

    /// An all-or-nothing verdict (math reference, JSON Schema).
    pub fn binary(passed: bool, evidence: Option<String>) -> Self {
        Outcome {
            verdict: if passed {
                Verdict::Complete
            } else {
                Verdict::Errors
            },
            verified: passed,
            credit: None,
            failures: u64::from(!passed),
            evidence,
        }
    }

    /// Code tests with per-assert credit: all passed is solved; some passed is a partial proof with
    /// `closed = passed`, `open = failed` (0.5 + 0.4·passed/total); none passed is `errors`. `total == 0`
    /// (a crash before any test ran, or one HumanEval `check`) is all-or-nothing.
    pub fn tests(passed_all: bool, passed: u64, total: u64, evidence: Option<String>) -> Self {
        if passed_all {
            return Self::binary(true, None);
        }
        let failed = total.saturating_sub(passed).max(1);
        if passed > 0 && total > 0 {
            Outcome {
                verdict: Verdict::Sorries,
                verified: false,
                credit: Some(Credit {
                    closed: passed,
                    open: failed,
                    hollow: false,
                }),
                failures: failed,
                evidence,
            }
        } else {
            Outcome {
                verdict: Verdict::Errors,
                verified: false,
                credit: None,
                failures: failed,
                evidence,
            }
        }
    }

    /// A reply with nothing to check (no code block, statement not reproduced, forbidden token).
    pub fn invalid(evidence: Option<String>) -> Self {
        Outcome {
            verdict: Verdict::Invalid,
            verified: false,
            credit: None,
            failures: 0,
            evidence,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub parent: Option<usize>,
    pub depth: usize,
    pub candidate: String,
    pub outcome: Option<Outcome>,
    pub children: Vec<usize>,
}

impl Node {
    pub fn score(&self) -> f64 {
        self.outcome.as_ref().map_or(0.0, Outcome::score)
    }
}

#[derive(Debug, Clone)]
pub struct CandidateTree {
    pub nodes: Vec<Node>,
    pub fanout: usize,
}

impl CandidateTree {
    /// A tree holding only the task (node 0). `fanout` bounds the repairs of one candidate (at least 1).
    pub fn new(fanout: usize) -> Self {
        CandidateTree {
            nodes: vec![Node {
                id: 0,
                parent: None,
                depth: 0,
                candidate: String::new(),
                outcome: None,
                children: vec![],
            }],
            fanout: fanout.max(1),
        }
    }

    /// Record an attempt made from `parent` and return its id.
    pub fn add(&mut self, parent: usize, candidate: String, outcome: Outcome) -> usize {
        let id = self.nodes.len();
        let depth = self.nodes[parent].depth + 1;
        self.nodes.push(Node {
            id,
            parent: Some(parent),
            depth,
            candidate,
            outcome: Some(outcome),
            children: vec![],
        });
        self.nodes[parent].children.push(id);
        id
    }

    fn eligible(&self, v: usize) -> bool {
        let n = &self.nodes[v];
        v == 0
            || (n.children.len() < self.fanout
                && n.outcome
                    .as_ref()
                    .is_some_and(|o| o.verdict != Verdict::Invalid))
    }

    /// The node the next attempt continues: the highest score; ties to the fewest failures, then the latest.
    /// The root (score 0) is chosen only when every candidate is invalid or exhausted, which makes the next
    /// attempt a fresh one.
    pub fn choose(&self) -> usize {
        (0..self.nodes.len())
            .filter(|&v| self.eligible(v))
            .max_by(|&a, &b| {
                let (x, y) = (&self.nodes[a], &self.nodes[b]);
                let fail = |n: &Node| n.outcome.as_ref().map_or(u64::MAX, |o| o.failures);
                x.score()
                    .total_cmp(&y.score())
                    .then(fail(y).cmp(&fail(x)))
                    .then(a.cmp(&b))
            })
            .unwrap_or(0)
    }

    /// The best candidate so far (highest score, latest on ties), or None before any attempt.
    pub fn best(&self) -> Option<&Node> {
        self.nodes[1..]
            .iter()
            .max_by(|x, y| x.score().total_cmp(&y.score()).then(x.id.cmp(&y.id)))
    }

    /// Children already made from `v`, newest first (OMH shows at most 12; we show `max`).
    pub fn siblings(&self, v: usize, max: usize) -> Vec<&Node> {
        self.nodes[v]
            .children
            .iter()
            .rev()
            .take(max)
            .map(|&c| &self.nodes[c])
            .collect()
    }
}

/// OMH's "already tried, do something different" note for a repair of a node with earlier children.
pub fn siblings_note(siblings: &[&Node], excerpt_chars: usize) -> Option<String> {
    if siblings.is_empty() {
        return None;
    }
    let mut s = format!(
        "{} earlier repair attempt(s) of this answer did not pass; do something different from them:\n",
        siblings.len()
    );
    for n in siblings {
        let ev = n
            .outcome
            .as_ref()
            .and_then(|o| o.evidence.as_deref())
            .unwrap_or("rejected");
        let ev: String = ev.chars().take(excerpt_chars).collect();
        s.push_str(&format!(
            "- score {:.2}: {}\n",
            n.score(),
            ev.replace('\n', " ")
        ));
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_verdicts_repair_the_latest_candidate() {
        let mut t = CandidateTree::new(2);
        assert_eq!(t.choose(), 0);
        let a = t.add(0, "a".into(), Outcome::binary(false, None));
        assert_eq!(t.choose(), a);
        let b = t.add(a, "b".into(), Outcome::binary(false, None));
        assert_eq!(t.choose(), b, "equal scores go to the latest node");
        let c = t.add(b, "c".into(), Outcome::binary(false, None));
        assert_eq!(t.choose(), c);
    }

    #[test]
    fn partial_credit_backtracks_to_the_better_candidate() {
        let mut t = CandidateTree::new(2);
        let a = t.add(
            0,
            "a".into(),
            Outcome::tests(false, 3, 4, Some("1 of 4 failed".into())),
        );
        assert_eq!(a, 1);
        let b = t.add(
            a,
            "b".into(),
            Outcome::tests(false, 0, 4, Some("all failed".into())),
        );
        assert!(t.nodes[a].score() > t.nodes[b].score());
        assert_eq!(
            t.choose(),
            a,
            "a repair that made things worse sends the search back"
        );
        let note = siblings_note(&t.siblings(a, 12), 80).unwrap();
        assert!(note.contains("all failed"));
        // After `fanout` repairs `a` is exhausted and the latest best leaf is continued.
        let c = t.add(a, "c".into(), Outcome::tests(false, 1, 4, None));
        assert_eq!(t.choose(), c);
        assert_eq!(t.best().unwrap().id, a);
    }

    #[test]
    fn invalid_candidates_are_never_continued() {
        let mut t = CandidateTree::new(2);
        t.add(0, "x".into(), Outcome::invalid(None));
        assert_eq!(
            t.choose(),
            0,
            "only the task itself is left: a fresh attempt"
        );
        let ok = t.add(0, "y".into(), Outcome::tests(false, 0, 3, None));
        assert_eq!(t.choose(), ok);
    }

    #[test]
    fn scores_follow_the_protocol() {
        assert_eq!(Outcome::binary(true, None).score(), 1.0);
        assert_eq!(Outcome::binary(false, None).score(), 0.05);
        assert_eq!(Outcome::tests(false, 1, 2, None).score(), 0.7);
        assert_eq!(Outcome::invalid(None).score(), 0.0);
    }
}
