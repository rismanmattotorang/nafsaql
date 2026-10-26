//! The agent's action language as a GBNF grammar, so the server can only sample well-formed actions.
//!
//! Codex exposes `apply_patch` to its models only as a *freeform* tool whose input is constrained by a Lark
//! grammar (`codex-rs/core/assets/tools/apply_patch.lark`, sent as `format: {type: "grammar", syntax:
//! "lark"}` by `core/src/tools/handlers/apply_patch_spec.rs`; reproduced in `patch/parser.rs`).
//! llama.cpp takes the same kind of constraint as GBNF in the request's `grammar` field. This module
//! translates Codex's grammar to GBNF and specialises it per step, which a fixed tool schema cannot do:
//!
//! - `*** Update File:` and `*** Delete File:` may name only files that exist in the workspace (the live run
//!   of 2026-10-24 lost a step to an update of a file that was never written);
//! - an `@@` header may not start with `-`, so the unified-diff header `@@ -1,4 +1,12 @@` that Qwen3-4B
//!   wrote in the same run cannot be sampled;
//! - a step can be limited to the actions that can still change the outcome (the last step allows no
//!   `run`), and `SUBMIT` is never offered: the verifier ends the run, so `SUBMIT` only spends a step.
//!
//! For a reasoning model the grammar starts with the reasoning span, which it leaves free up to `</think>`:
//! llama-server applies a request grammar from the first sampled token, and without the span a thinking
//! model is forced to answer at once (measured on llama.cpp 797de63 with Qwen3-4B-Thinking-2507: the
//! answer was sampled in place of the reasoning). The server's reasoning budget still closes the span.
//! When the grammar is complete only end-of-generation can be sampled, so decoding stops at the end of the
//! action: no prose after it is ever paid for.

/// Which actions one step may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allowed {
    pub patch: bool,
    pub whole: bool,
    pub run: bool,
}

impl Allowed {
    pub const ALL: Allowed = Allowed {
        patch: true,
        whole: true,
        run: true,
    };
}

/// The GBNF grammar for one step.
///
/// `existing` are the workspace's files (targets of Update/Delete), `runnable` the Python files a `run`
/// block may name, `main_file` the file a ```python block replaces. `notes` short prose lines may precede the
/// action (a non-reasoning model plans there); a reasoning model plans in its reasoning span instead.
pub fn action_grammar(
    thinking: bool,
    allowed: Allowed,
    existing: &[&str],
    runnable: &[&str],
    notes: usize,
) -> String {
    let mut g = String::new();
    let mut root = vec![];
    if thinking {
        root.push("think");
    }
    if notes > 0 {
        root.push(if notes == 1 { "note?" } else { "notes" });
    }
    root.push("action");
    g.push_str(&format!("root ::= {}\n", root.join(" ")));
    if thinking {
        // Any text that does not contain `</think>`, then `</think>`. GBNF has no negative lookahead, so each
        // proper prefix of the tag is spelled out; the alternatives overlap, which the grammar engine allows.
        g.push_str(
            "think ::= ( [^<] | \"<\" [^/] | \"</\" [^t] | \"</t\" [^h] | \"</th\" [^i] | \"</thi\" [^n] | \"</thin\" [^k] | \"</think\" [^>] )* \"</think>\" [ \\n]*\n",
        );
    }
    if notes > 1 {
        g.push_str(&format!("notes ::= note{{0,{notes}}}\n"));
    }
    if notes > 0 {
        // A note line cannot start like an action (a fence or `***`).
        g.push_str("note ::= [^`*\\n] [^\\n]* \"\\n\" \"\\n\"?\n");
    }
    let has_runnable = !runnable.is_empty();
    let mut actions = vec![];
    if allowed.patch {
        actions.push("patch");
    }
    if allowed.whole {
        actions.push("whole");
    }
    if allowed.run && has_runnable {
        actions.push("run");
    }
    if actions.is_empty() {
        actions.push("whole");
    }
    g.push_str(&format!("action ::= {}\n", actions.join(" | ")));
    if actions.contains(&"whole") {
        g.push_str("whole ::= \"```python\\n\" codeline* \"```\"\n");
        // Lines that open with neither a backtick (the closing fence ends the block) nor `*`: a Codex patch
        // written inside the block would bypass the step's edit policy, and Qwen3-4B did exactly that when
        // only a whole file was offered (round 2 of the Phase 11 A/B: four such patches failed to apply).
        // A top-level Python line never starts with `*`.
        g.push_str("codeline ::= ( [^`*\\n] [^\\n]* )? \"\\n\"\n");
    }
    if actions.contains(&"run") {
        g.push_str(&format!(
            "run ::= \"```run\\npython \" ( {} ) \"\\n```\"\n",
            alternatives(runnable)
        ));
    }
    if actions.contains(&"patch") {
        let mut hunks = vec!["add"];
        if !existing.is_empty() {
            hunks.push("update");
            hunks.push("delete");
        }
        g.push_str("patch ::= \"*** Begin Patch\\n\" hunk+ \"*** End Patch\"\n");
        g.push_str(&format!("hunk ::= {}\n", hunks.join(" | ")));
        g.push_str("add ::= \"*** Add File: \" newname \"\\n\" addline+\n");
        g.push_str("addline ::= \"+\" [^\\n]* \"\\n\"\n");
        // Flat names, as the host requires (`files: '…' must be a flat file name`).
        g.push_str("newname ::= [A-Za-z0-9_] [A-Za-z0-9_.-]*\n");
        if !existing.is_empty() {
            g.push_str(&format!("existing ::= {}\n", alternatives(existing)));
            g.push_str("delete ::= \"*** Delete File: \" existing \"\\n\"\n");
            g.push_str(
                "update ::= \"*** Update File: \" existing \"\\n\" change+ ( \"*** End of File\\n\" )?\n",
            );
            g.push_str("change ::= ctx? cline+\n");
            // `@@`, or `@@ ` and a line of the file; never a unified-diff range.
            g.push_str("ctx ::= \"@@\" ( \" \" [^-\\n] [^\\n]* )? \"\\n\"\n");
            g.push_str("cline ::= ( [-+ ] [^\\n]* )? \"\\n\"\n");
        }
    }
    g
}

/// `"a" | "b"` with GBNF string escapes.
fn alternatives(names: &[&str]) -> String {
    names
        .iter()
        .map(|n| literal(n))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn literal(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule<'a>(g: &'a str, name: &str) -> Option<&'a str> {
        g.lines()
            .find_map(|l| l.strip_prefix(&format!("{name} ::= ")))
    }

    #[test]
    fn update_and_delete_name_only_existing_files() {
        let g = action_grammar(
            false,
            Allowed::ALL,
            &["solution.py", "a\"b.py"],
            &["solution.py"],
            0,
        );
        assert_eq!(
            rule(&g, "existing"),
            Some("\"solution.py\" | \"a\\\"b.py\"")
        );
        assert_eq!(rule(&g, "hunk"), Some("add | update | delete"));
        assert_eq!(rule(&g, "root"), Some("action"));
        assert!(rule(&g, "think").is_none());
    }

    #[test]
    fn an_empty_workspace_can_only_add_or_write() {
        let g = action_grammar(true, Allowed::ALL, &[], &[], 3);
        assert_eq!(rule(&g, "hunk"), Some("add"));
        assert!(rule(&g, "update").is_none() && rule(&g, "existing").is_none());
        assert_eq!(
            rule(&g, "action"),
            Some("patch | whole"),
            "nothing to run yet"
        );
        assert_eq!(rule(&g, "root"), Some("think notes action"));
        assert!(rule(&g, "think").unwrap().ends_with("\"</think>\" [ \\n]*"));
    }

    #[test]
    fn the_last_step_cannot_run_and_unified_ranges_are_excluded() {
        let last = Allowed {
            run: false,
            ..Allowed::ALL
        };
        let g = action_grammar(false, last, &["solution.py"], &["solution.py"], 1);
        assert_eq!(rule(&g, "action"), Some("patch | whole"));
        assert!(rule(&g, "run").is_none());
        assert_eq!(rule(&g, "root"), Some("note? action"));
        assert_eq!(
            rule(&g, "ctx"),
            Some("\"@@\" ( \" \" [^-\\n] [^\\n]* )? \"\\n\"")
        );
        assert_eq!(
            rule(&g, "codeline"),
            Some("( [^`*\\n] [^\\n]* )? \"\\n\""),
            "a patch cannot hide inside a code block"
        );
    }

    #[test]
    fn every_rule_that_is_used_is_defined() {
        for (thinking, existing, runnable, notes) in [
            (true, vec!["solution.py"], vec!["solution.py"], 2),
            (false, vec![], vec![], 0),
        ] {
            let g = action_grammar(thinking, Allowed::ALL, &existing, &runnable, notes);
            let defined: Vec<&str> = g.lines().filter_map(|l| l.split(" ::= ").next()).collect();
            for l in g.lines() {
                let body = l.split_once(" ::= ").unwrap().1;
                // Strip literals and character classes, then every remaining word must be a rule.
                let mut plain = String::new();
                let (mut in_lit, mut in_class, mut esc) = (false, false, false);
                for c in body.chars() {
                    if esc {
                        esc = false;
                        continue;
                    }
                    match c {
                        '\\' if in_lit || in_class => esc = true,
                        '"' if !in_class => in_lit = !in_lit,
                        '[' if !in_lit => in_class = true,
                        ']' if in_class => in_class = false,
                        c if !in_lit && !in_class => plain.push(c),
                        _ => {}
                    }
                }
                for w in plain.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
                    if !w.is_empty() && !w.chars().all(|c| c.is_ascii_digit()) {
                        assert!(
                            defined.contains(&w),
                            "rule {w} used in `{l}` is not defined"
                        );
                    }
                }
            }
        }
    }
}
