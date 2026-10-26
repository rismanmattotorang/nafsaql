//! The agent's transcript, kept the way Codex keeps its history so the server's prompt cache stays warm.
//!
//! Codex's context manager (`codex-rs/core/src/context_manager/history.rs`) only ever *appends*: a change of
//! state is recorded as a new item, never by rewriting an earlier one, because the provider's prompt cache
//! (and Codex's incremental `previous_response_id` requests, `core/src/client.rs`) match the request byte for
//! byte from the start. llama-server's slot cache is the same kind of prefix match. On the reference machine
//! System 2 prefills about 31 tokens/s: a cold 1,969-token prompt took 64 s, the same prompt with one more
//! exchange appended took 3 s (2026-10-25). The loop of Phase 10 rebuilt the first user message (task plus
//! the current files) at every step, which invalidated almost the whole prefix after every edit.
//!
//! When the transcript outgrows its budget it is *compacted* (Codex `core/src/compact.rs`: at 90% of the
//! window, history is rebuilt from the user's messages plus a model-written handoff summary). Nafs-Aql's
//! summary is written by the harness, not the model: at 6-8 tokens/s a model-written summary would cost
//! minutes, and the facts it needs (what was tried and what the verifier said) are already structured. The
//! rebuilt transcript keeps the task message's prefix, so the system prompt and the task stay cached.
use crate::patch::Workspace;
use crate::truncate::truncate_middle;

/// Bytes per token for the size estimate. Codex uses 4 (`APPROX_BYTES_PER_TOKEN`); Qwen3's tokenizer packs
/// Python into fewer bytes per token, so 3 errs on the side of compacting early.
pub const BYTES_PER_TOKEN: usize = 3;

/// One attempt, as the compacted summary remembers it.
#[derive(Debug, Clone)]
pub struct Attempt {
    pub step: u32,
    pub kind: String,
    pub result: String,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    system: String,
    messages: Vec<(String, String)>,
    pub compactions: u32,
}

impl Transcript {
    pub fn new(system: &str, first_user: String) -> Self {
        Transcript {
            system: system.to_string(),
            messages: vec![("user".into(), first_user)],
            compactions: 0,
        }
    }

    /// The request's messages: the system prompt, then the transcript in order.
    pub fn messages(&self) -> Vec<(String, String)> {
        let mut m = vec![("system".to_string(), self.system.clone())];
        m.extend(self.messages.iter().cloned());
        m
    }

    pub fn push(&mut self, role: &str, content: String) {
        self.messages.push((role.to_string(), content));
    }

    pub fn estimated_tokens(&self) -> usize {
        (self.system.len()
            + self
                .messages
                .iter()
                .map(|(_, c)| c.len() + 8)
                .sum::<usize>())
            / BYTES_PER_TOKEN
    }

    /// Replace everything after the system prompt by one user message (the compacted state).
    pub fn compact(&mut self, first_user: String) {
        self.messages = vec![("user".into(), first_user)];
        self.compactions += 1;
    }
}

/// The task message: the same layout at the start and after a compaction, so the shared prefix (system
/// prompt, `Task:` and the task text) is reused by the cache in both cases.
pub fn task_message(
    task: &str,
    ws: &Workspace,
    max_file_bytes: usize,
    verdict: Option<(bool, Option<&str>)>,
    attempts: &[Attempt],
    max_evidence_bytes: usize,
) -> String {
    let mut s = format!(
        "Task:\n{task}\n\nWorkspace files:\n{}",
        workspace_view(ws, max_file_bytes)
    );
    if let Some((passed, evidence)) = verdict {
        s.push_str(&format!(
            "\nHidden tests on the current files: {}.",
            if passed { "passed" } else { "failed" }
        ));
        if let Some(e) = evidence {
            s.push_str(&format!(
                "\n<verifier_evidence>\n{}\n</verifier_evidence>",
                truncate_middle(e, max_evidence_bytes)
            ));
        }
    }
    if !attempts.is_empty() {
        // Open Math Harness's "already tried" list, in place of Codex's model-written summary.
        s.push_str("\n\nEarlier attempts (already tried; do something different):\n");
        for a in attempts {
            s.push_str(&format!(
                "{}. {}: {}\n",
                a.step,
                a.kind,
                one_line(&a.result, 200)
            ));
        }
    }
    s.push_str("\n\nReply with one action.");
    s
}

pub fn workspace_view(ws: &Workspace, max: usize) -> String {
    let mut s = String::new();
    for (p, c) in ws {
        s.push_str(&format!("--- {p}\n{}\n", truncate_middle(c, max)));
    }
    if s.is_empty() {
        s.push_str("(empty: solution.py does not exist yet)\n");
    }
    s
}

/// The first line that says something, cut to `max` bytes on a character boundary.
pub fn one_line(s: &str, max: usize) -> String {
    let joined: Vec<&str> = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(3)
        .collect();
    let line = joined.join(" | ");
    if line.len() <= max {
        return line;
    }
    let mut end = max;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &line[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> Workspace {
        [("solution.py".to_string(), "x = 1\n".to_string())].into()
    }

    #[test]
    fn appends_keep_the_prefix_and_compaction_keeps_the_task_prefix() {
        let first = task_message("Fix it.", &ws(), 100, Some((false, Some("boom"))), &[], 100);
        let mut t = Transcript::new("sys", first.clone());
        let before = t.messages();
        t.push("assistant", "a".into());
        t.push("user", "b".into());
        let after = t.messages();
        assert_eq!(
            after[..2],
            before[..],
            "earlier messages are never rewritten"
        );
        let attempts = vec![Attempt {
            step: 1,
            kind: "patch".into(),
            result: "The patch was not applied:\nFailed to find expected lines".into(),
        }];
        let changed: Workspace = [("solution.py".to_string(), "x = 2\n".to_string())].into();
        t.compact(task_message(
            "Fix it.",
            &changed,
            100,
            Some((false, None)),
            &attempts,
            100,
        ));
        let c = t.messages();
        assert_eq!(c.len(), 2);
        assert_eq!(t.compactions, 1);
        let shared = "Task:\nFix it.\n\nWorkspace files:\n--- solution.py\nx = ";
        assert!(c[1].1.starts_with(shared) && first.starts_with(shared));
        assert!(
            c[1].1
                .contains("1. patch: The patch was not applied: | Failed to find expected lines\n")
        );
    }

    #[test]
    fn an_empty_workspace_says_what_to_create_and_lines_are_cut_on_boundaries() {
        assert!(
            task_message("t", &Workspace::new(), 10, None, &[], 10)
                .contains("solution.py does not exist yet")
        );
        assert_eq!(one_line("ééé", 3), "é…");
        assert_eq!(one_line("\n a \n\n b \n", 50), "a | b");
    }
}
