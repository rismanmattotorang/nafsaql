// Derived from OpenAI Codex (codex-rs/apply-patch/src/parser.rs, revision 970111f6), Apache-2.0,
// Copyright 2025 OpenAI. Modified for Nafs-Aql: the PathUri resolution and environment id were removed
// (the workspace is in memory), and lenient mode also strips a Markdown fence around the patch, which
// small models add.
//! Parse a patch into hunks. The grammar (Codex's Lark file):
//!
//! ```text
//! start: begin_patch hunk+ end_patch
//! begin_patch: "*** Begin Patch" LF
//! end_patch: "*** End Patch" LF?
//! hunk: add_hunk | delete_hunk | update_hunk
//! add_hunk: "*** Add File: " filename LF add_line+
//! delete_hunk: "*** Delete File: " filename LF
//! update_hunk: "*** Update File: " filename LF change_move? change?
//! add_line: "+" /(.+)/ LF -> line
//! change_move: "*** Move to: " filename LF
//! change: (change_context | change_line)+ eof_line?
//! change_context: ("@@" | "@@ " /(.+)/) LF
//! change_line: ("+" | "-" | " ") /(.+)/ LF
//! eof_line: "*** End of File" LF
//! ```
use std::path::{Path, PathBuf};

use super::streaming_parser::StreamingPatchParser;

pub(crate) const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";
pub(crate) const END_PATCH_MARKER: &str = "*** End Patch";
pub(crate) const ADD_FILE_MARKER: &str = "*** Add File: ";
pub(crate) const DELETE_FILE_MARKER: &str = "*** Delete File: ";
pub(crate) const UPDATE_FILE_MARKER: &str = "*** Update File: ";
pub(crate) const MOVE_TO_MARKER: &str = "*** Move to: ";
pub(crate) const EOF_MARKER: &str = "*** End of File";
pub(crate) const CHANGE_CONTEXT_MARKER: &str = "@@ ";
pub(crate) const EMPTY_CHANGE_CONTEXT_MARKER: &str = "@@";

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    InvalidPatchError(String),
    InvalidHunkError { message: String, line_number: usize },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidPatchError(m) => write!(f, "invalid patch: {m}"),
            ParseError::InvalidHunkError {
                message,
                line_number,
            } => {
                write!(f, "invalid hunk at line {line_number}, {message}")
            }
        }
    }
}
impl std::error::Error for ParseError {}

#[derive(Debug, PartialEq, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum Hunk {
    AddFile {
        path: PathBuf,
        contents: String,
    },
    DeleteFile {
        path: PathBuf,
    },
    UpdateFile {
        path: PathBuf,
        move_path: Option<PathBuf>,
        /// In file order: each chunk's context lies after the previous chunk.
        chunks: Vec<UpdateFileChunk>,
    },
}

impl Hunk {
    /// The path this hunk leaves behind (the move destination for a rename).
    pub fn path(&self) -> &Path {
        match self {
            Hunk::AddFile { path, .. } | Hunk::DeleteFile { path } => path,
            Hunk::UpdateFile {
                move_path: Some(path),
                ..
            } => path,
            Hunk::UpdateFile {
                path,
                move_path: None,
                ..
            } => path,
        }
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct UpdateFileChunk {
    /// One line that narrows down where the chunk applies (usually a `def` or `class` line).
    pub change_context: Option<String>,
    /// Contiguous lines replaced by `new_lines`; they occur strictly after `change_context`.
    pub old_lines: Vec<String>,
    pub new_lines: Vec<String>,
    /// Index pairs into `old_lines` / `new_lines` of lines parsed as context.
    pub context_line_indices: Vec<(usize, usize)>,
    /// `old_lines` must occur at the end of the file.
    pub is_end_of_file: bool,
}

impl UpdateFileChunk {
    pub(crate) fn push_context_line(&mut self, line: String) {
        self.context_line_indices
            .push((self.old_lines.len(), self.new_lines.len()));
        self.old_lines.push(line.clone());
        self.new_lines.push(line);
    }
}

/// Parse `patch` leniently: surrounding whitespace, a heredoc wrapper (`<<'EOF'` … `EOF`) and a Markdown
/// fence (```` ```patch ```` / ```` ```diff ```` … ```` ``` ````) are stripped before the strict grammar applies.
pub fn parse_patch(patch: &str) -> Result<Vec<Hunk>, ParseError> {
    let lines: Vec<&str> = patch.trim().lines().collect();
    let inner = strip_wrappers(&lines)?;
    let mut parser = StreamingPatchParser::default();
    parser.push_delta(&inner.join("\n"))?;
    let mut hunks = parser.finish()?;
    for h in &mut hunks {
        if let Hunk::UpdateFile { chunks, .. } = h {
            for c in chunks {
                c.change_context = c.change_context.take().and_then(unified_range_context);
            }
        }
    }
    Ok(hunks)
}

/// Nafs-Aql leniency: small models write unified-diff hunk headers (`@@ -1,4 +1,12 @@ def f():`), which the
/// grammar would read as a context line that never matches (seen live, Qwen3-4B-Thinking). The line ranges are
/// dropped and the text after the closing `@@`, if any, is the context.
fn unified_range_context(ctx: String) -> Option<String> {
    let t = ctx.trim();
    let is_range = |s: &str| {
        let s = s.trim_start_matches(['-', '+']);
        !s.is_empty()
            && s.split(',')
                .all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut words = t.split_whitespace();
    match (words.next(), words.next()) {
        (Some(a), Some(b))
            if a.starts_with('-') && b.starts_with('+') && is_range(a) && is_range(b) =>
        {
            let rest = t.splitn(3, char::is_whitespace).nth(2).unwrap_or("").trim();
            let rest = rest.strip_prefix("@@").unwrap_or(rest).trim();
            (!rest.is_empty()).then(|| rest.to_string())
        }
        _ => Some(ctx),
    }
}

fn strip_wrappers<'a>(lines: &'a [&'a str]) -> Result<&'a [&'a str], ParseError> {
    let strict = check_boundaries(lines);
    if strict.is_ok() {
        return Ok(lines);
    }
    if let [first, .., last] = lines {
        let first = first.trim();
        let heredoc = (first == "<<EOF" || first == "<<'EOF'" || first == "<<\"EOF\"")
            && last.ends_with("EOF");
        let fence = first.starts_with("```") && last.trim() == "```";
        if (heredoc || fence) && lines.len() >= 4 {
            let inner = &lines[1..lines.len() - 1];
            return check_boundaries(inner).map(|_| inner);
        }
    }
    strict.map(|_| lines)
}

fn check_boundaries(lines: &[&str]) -> Result<(), ParseError> {
    let first = lines.first().map(|l| l.trim());
    let last = lines.last().map(|l| l.trim());
    match (first, last) {
        (Some(BEGIN_PATCH_MARKER), Some(END_PATCH_MARKER)) => Ok(()),
        (Some(BEGIN_PATCH_MARKER), _) => Err(ParseError::InvalidPatchError(
            "The last line of the patch must be '*** End Patch'".into(),
        )),
        _ => Err(ParseError::InvalidPatchError(
            "The first line of the patch must be '*** Begin Patch'".into(),
        )),
    }
}
