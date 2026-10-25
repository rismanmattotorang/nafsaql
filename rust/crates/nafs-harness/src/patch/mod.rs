//! Codex's `apply_patch` format applied to an in-memory workspace (path -> text).
//!
//! The parser, the four-pass fuzzy matcher (`seek_sequence`) and the replacement computation come from
//! OpenAI Codex (Apache-2.0); see each file's header. What is Nafs-Aql's own: the workspace is a map, not a
//! filesystem, so a patch can never touch anything outside it; paths must be relative and stay inside it;
//! a patch applies all-or-nothing; and [`find_patch`] locates the envelope inside a model's prose.
use std::collections::BTreeMap;
use std::path::{Component, Path};

mod parser;
mod seek_sequence;
mod streaming_parser;
mod text_file;

pub use parser::{Hunk, ParseError, UpdateFileChunk, parse_patch};
use text_file::{Replacement, SourceFile};

/// Files of a workspace, keyed by their normalised relative path.
pub type Workspace = BTreeMap<String, String>;

/// What a successful patch changed, in Codex's summary format.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatchSummary {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
}

impl PatchSummary {
    /// `Success. Updated the following files:` followed by `A path` / `M path` / `D path` lines.
    pub fn to_model_text(&self) -> String {
        let mut s = String::from("Success. Updated the following files:\n");
        for p in &self.added {
            s.push_str(&format!("A {p}\n"));
        }
        for p in &self.modified {
            s.push_str(&format!("M {p}\n"));
        }
        for p in &self.deleted {
            s.push_str(&format!("D {p}\n"));
        }
        s
    }
}

/// The patch envelope inside `text` (from the first `*** Begin Patch` line to the next `*** End Patch`
/// line), or None. Small models put prose or a Markdown fence around the patch; only the envelope counts.
pub fn find_patch(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim() == parser::BEGIN_PATCH_MARKER)?;
    let end = lines[start..]
        .iter()
        .position(|l| l.trim() == parser::END_PATCH_MARKER)?
        + start;
    Some(lines[start..=end].join("\n"))
}

/// A workspace-relative path: no root, no `..`, no empty name; `./` segments are dropped.
pub fn normalise_path(p: &Path) -> Result<String, String> {
    let mut parts: Vec<String> = vec![];
    for c in p.components() {
        match c {
            Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            Component::CurDir => {}
            _ => {
                return Err(format!(
                    "path '{}' must be relative to the workspace and must not contain '..'",
                    p.display()
                ));
            }
        }
    }
    if parts.is_empty() {
        return Err("empty path in patch".into());
    }
    Ok(parts.join("/"))
}

/// Apply `patch` to `ws`. All hunks apply or none does: on any error `ws` is unchanged and the error text
/// is what the model is shown (Codex's messages: "Failed to find expected lines in …", parse errors with
/// line numbers).
pub fn apply_patch(ws: &mut Workspace, patch: &str) -> Result<PatchSummary, String> {
    let hunks = parse_patch(patch).map_err(|e| e.to_string())?;
    if hunks.is_empty() {
        return Err("No files were modified: the patch has no hunks.".into());
    }
    let mut next = ws.clone();
    let mut summary = PatchSummary::default();
    for h in &hunks {
        match h {
            Hunk::AddFile { path, contents } => {
                let p = normalise_path(path)?;
                next.insert(p.clone(), contents.clone());
                summary.added.push(p);
            }
            Hunk::DeleteFile { path } => {
                let p = normalise_path(path)?;
                if next.remove(&p).is_none() {
                    return Err(format!("Failed to delete file {p}: no such file"));
                }
                summary.deleted.push(p);
            }
            Hunk::UpdateFile {
                path,
                move_path,
                chunks,
            } => {
                let p = normalise_path(path)?;
                let original = next
                    .get(&p)
                    .ok_or_else(|| format!("Failed to read file to update {p}: no such file"))?;
                let new_contents = apply_chunks(original, &p, chunks)?;
                match move_path {
                    Some(dest) => {
                        let d = normalise_path(dest)?;
                        next.remove(&p);
                        next.insert(d.clone(), new_contents);
                        summary.modified.push(d);
                    }
                    None => {
                        next.insert(p.clone(), new_contents);
                        summary.modified.push(p);
                    }
                }
            }
        }
    }
    *ws = next;
    Ok(summary)
}

/// New contents of a file after the update chunks.
pub fn apply_chunks(
    original: &str,
    path: &str,
    chunks: &[UpdateFileChunk],
) -> Result<String, String> {
    let mut source = SourceFile::parse(original);
    let lines = source.line_texts();
    let replacements = compute_replacements(&lines, path, chunks)?;
    source.apply_replacements(&replacements);
    Ok(source.into_contents())
}

// Derived from OpenAI Codex (codex-rs/apply-patch/src/file_update.rs `compute_replacements`, revision
// 970111f6), Apache-2.0, Copyright 2025 OpenAI. Modified for Nafs-Aql: errors are plain strings.
fn compute_replacements(
    original_lines: &[String],
    path: &str,
    chunks: &[UpdateFileChunk],
) -> Result<Vec<Replacement>, String> {
    let mut replacements: Vec<Replacement> = Vec::new();
    let mut line_index: usize = 0;

    for chunk in chunks {
        // A chunk's `change_context` line moves the cursor to just after it.
        if let Some(ctx_line) = &chunk.change_context {
            if let Some(idx) = seek_sequence::seek_sequence(
                original_lines,
                std::slice::from_ref(ctx_line),
                line_index,
                false,
            ) {
                line_index = idx + 1;
            } else {
                return Err(format!("Failed to find context '{ctx_line}' in {path}"));
            }
        }

        if chunk.old_lines.is_empty() {
            replacements.push((original_lines.len(), 0, chunk.new_lines.clone()));
            continue;
        }

        // The last element of `old_lines` is often an empty string standing for the final newline, which
        // `SourceFile` keeps on the preceding line; retry without it so edits at the end of a file apply.
        let mut pattern: &[String] = &chunk.old_lines;
        let mut found =
            seek_sequence::seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);
        let mut new_slice: &[String] = &chunk.new_lines;
        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            pattern = &pattern[..pattern.len() - 1];
            if new_slice.last().is_some_and(String::is_empty) {
                new_slice = &new_slice[..new_slice.len() - 1];
            }
            found = seek_sequence::seek_sequence(
                original_lines,
                pattern,
                line_index,
                chunk.is_end_of_file,
            );
        }

        if let Some(start_idx) = found {
            // Context lines stay in place so their exact contents and terminators survive.
            let mut old_start = 0;
            let mut new_start = 0;
            for &(old_context, new_context) in &chunk.context_line_indices {
                if old_context >= pattern.len() || new_context >= new_slice.len() {
                    break;
                }
                if old_start != old_context || new_start != new_context {
                    replacements.push((
                        start_idx + old_start,
                        old_context - old_start,
                        new_slice[new_start..new_context].to_vec(),
                    ));
                }
                old_start = old_context + 1;
                new_start = new_context + 1;
            }
            if old_start != pattern.len() || new_start != new_slice.len() {
                replacements.push((
                    start_idx + old_start,
                    pattern.len() - old_start,
                    new_slice[new_start..].to_vec(),
                ));
            }
            line_index = start_idx + pattern.len();
        } else {
            return Err(format!(
                "Failed to find expected lines in {}:\n{}",
                path,
                chunk.old_lines.join("\n"),
            ));
        }
    }

    replacements.sort_by_key(|(index, _, _)| *index);
    Ok(replacements)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(files: &[(&str, &str)]) -> Workspace {
        files
            .iter()
            .map(|(p, c)| (p.to_string(), c.to_string()))
            .collect()
    }

    #[test]
    fn update_add_delete_and_move() {
        let mut w = ws(&[("a.py", "def f(x):\n    return x\n"), ("old.txt", "bye\n")]);
        let patch = "*** Begin Patch\n*** Update File: a.py\n@@ def f(x):\n-    return x\n+    return x + 1\n*** Add File: b.py\n+print(1)\n*** Delete File: old.txt\n*** End Patch";
        let s = apply_patch(&mut w, patch).unwrap();
        assert_eq!(w["a.py"], "def f(x):\n    return x + 1\n");
        assert_eq!(w["b.py"], "print(1)\n");
        assert!(!w.contains_key("old.txt"));
        assert_eq!(
            s.to_model_text(),
            "Success. Updated the following files:\nA b.py\nM a.py\nD old.txt\n"
        );

        let mv = "*** Begin Patch\n*** Update File: b.py\n*** Move to: src/c.py\n@@\n-print(1)\n+print(2)\n*** End Patch";
        apply_patch(&mut w, mv).unwrap();
        assert_eq!(w["src/c.py"], "print(2)\n");
        assert!(!w.contains_key("b.py"));
    }

    #[test]
    fn a_failed_hunk_leaves_the_workspace_unchanged() {
        let mut w = ws(&[("a.py", "x = 1\n")]);
        let before = w.clone();
        let patch = "*** Begin Patch\n*** Add File: new.py\n+y = 2\n*** Update File: a.py\n@@\n-x = 99\n+x = 2\n*** End Patch";
        let e = apply_patch(&mut w, patch).unwrap_err();
        assert!(e.contains("Failed to find expected lines in a.py"), "{e}");
        assert_eq!(w, before);
    }

    #[test]
    fn fuzzy_matching_tolerates_whitespace_and_typographic_punctuation() {
        let mut w = ws(&[("a.py", "s = \u{201C}hi\u{201D}  \nreturn s\n")]);
        let patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-s = \"hi\"\n+s = \"hello\"\n return s\n*** End Patch";
        apply_patch(&mut w, patch).unwrap();
        assert_eq!(w["a.py"], "s = \"hello\"\nreturn s\n");
    }

    #[test]
    fn paths_must_stay_inside_the_workspace() {
        let mut w = Workspace::new();
        for bad in ["/etc/passwd", "../x.py", "a/../../x.py"] {
            let patch = format!("*** Begin Patch\n*** Add File: {bad}\n+x\n*** End Patch");
            assert!(apply_patch(&mut w, &patch).is_err(), "{bad}");
        }
        assert!(w.is_empty());
        apply_patch(
            &mut w,
            "*** Begin Patch\n*** Add File: ./d/x.py\n+x\n*** End Patch",
        )
        .unwrap();
        assert!(w.contains_key("d/x.py"));
    }

    #[test]
    fn envelope_is_found_inside_prose_and_fences() {
        let reply = "I will fix the off-by-one.\n```patch\n*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 2\n*** End Patch\n```\nDone.";
        let p = find_patch(reply).unwrap();
        let mut w = ws(&[("a.py", "x = 1\n")]);
        apply_patch(&mut w, &p).unwrap();
        assert_eq!(w["a.py"], "x = 2\n");
        // The fenced form also parses directly.
        let fenced = "```diff\n*** Begin Patch\n*** Update File: a.py\n@@\n-x = 2\n+x = 3\n*** End Patch\n```";
        apply_patch(&mut w, fenced).unwrap();
        assert_eq!(w["a.py"], "x = 3\n");
        assert!(find_patch("no patch here").is_none());
    }

    #[test]
    fn parse_errors_name_the_problem() {
        let e = apply_patch(
            &mut Workspace::new(),
            "*** Begin Patch\n*** Frobnicate: a\n*** End Patch",
        )
        .unwrap_err();
        assert!(e.contains("is not a valid hunk header"), "{e}");
        let e = apply_patch(
            &mut Workspace::new(),
            "*** Begin Patch\n*** Add File: a\n+x",
        )
        .unwrap_err();
        assert!(e.contains("End Patch"), "{e}");
    }

    #[test]
    fn unified_diff_hunk_headers_are_tolerated() {
        // As written live by Qwen3-4B-Thinking: line ranges where Codex expects a context line.
        let mut w = ws(&[("a.py", "def f(s):\n    return s == s[::-1]\n")]);
        let p = "*** Begin Patch\n*** Update File: a.py\n@@ -1,2 +1,3 @@\n def f(s):\n-    return s == s[::-1]\n+    s = s.lower()\n+    return s == s[::-1]\n*** End Patch";
        apply_patch(&mut w, p).unwrap();
        assert_eq!(
            w["a.py"],
            "def f(s):\n    s = s.lower()\n    return s == s[::-1]\n"
        );
        // Text after the closing @@ is still used as the context line.
        let p = "*** Begin Patch\n*** Update File: a.py\n@@ -2 +2 @@ def f(s):\n-    s = s.lower()\n+    s = s.casefold()\n*** End Patch";
        apply_patch(&mut w, p).unwrap();
        assert!(w["a.py"].contains("casefold"));
        // An ordinary context line that happens to start with '-' is untouched.
        assert_eq!(unified_range("-x +y"), Some("-x +y".to_string()));
    }

    fn unified_range(s: &str) -> Option<String> {
        let mut c = UpdateFileChunk {
            change_context: Some(s.into()),
            ..Default::default()
        };
        let mut h = vec![Hunk::UpdateFile {
            path: "a".into(),
            move_path: None,
            chunks: vec![c.clone()],
        }];
        let text = format!("*** Begin Patch\n*** Update File: a\n@@ {s}\n-x\n+y\n*** End Patch");
        h = parse_patch(&text).unwrap();
        if let Hunk::UpdateFile { chunks, .. } = &h[0] {
            c = chunks[0].clone();
        }
        c.change_context
    }

    #[test]
    fn end_of_file_chunks_and_crlf_files() {
        let mut w = ws(&[("a.txt", "a\r\nb\r\nc\r\n")]);
        let patch = "*** Begin Patch\n*** Update File: a.txt\n@@\n b\n-c\n+C\n+d\n*** End of File\n*** End Patch";
        apply_patch(&mut w, patch).unwrap();
        assert_eq!(w["a.txt"], "a\r\nb\r\nC\r\nd\r\n");
    }
}
