//! Output shaping for the model, after OpenAI Codex (Apache-2.0, Copyright 2025 OpenAI):
//! `codex-rs/utils/string/src/truncate.rs` (middle truncation on UTF-8 boundaries, a marker outside the
//! budget, the budget split evenly between head and tail) and `core/src/tools/mod.rs`
//! `format_exec_output_for_model` (exit code, wall time, total lines, output). Rewritten for Nafs-Aql
//! without the token-policy types.
use std::time::Duration;

/// Keep at most `max_bytes` of `s`, half from the start and half from the end, with `…N chars truncated…`
/// between them. The first failing assertion of a traceback is usually in the head, the exception in the
/// tail; keeping only one end loses the other.
pub fn truncate_middle(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let left = max_bytes / 2;
    let right = max_bytes - left;
    let head_end = floor_boundary(s, left);
    let tail_start = ceil_boundary(s, s.len() - right);
    let removed = s[head_end..tail_start].chars().count();
    format!(
        "{}…{removed} chars truncated…{}",
        &s[..head_end],
        &s[tail_start..]
    )
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// One command's result as the model sees it (Codex's shell format). A timeout is exit code 124 with a
/// leading notice, as in Codex's `exec.rs`.
pub fn format_exec_output(
    exit_code: i32,
    wall: Duration,
    output: &str,
    timed_out: bool,
    max_bytes: usize,
) -> String {
    let total_lines = output.lines().count();
    let shown = truncate_middle(output, max_bytes);
    let mut s = format!(
        "Exit code: {exit_code}\nWall time: {:.1} seconds\n",
        wall.as_secs_f64()
    );
    if shown.len() < output.len() {
        s.push_str(&format!("Total output lines: {total_lines}\n"));
    }
    s.push_str("Output:\n");
    // The timeout notice is outside the truncation budget, so it is never cut.
    if timed_out {
        s.push_str(&format!(
            "command timed out after {} milliseconds\n",
            wall.as_millis()
        ));
    }
    s.push_str(&shown);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_head_and_tail_on_char_boundaries() {
        assert_eq!(truncate_middle("short", 10), "short");
        let s = format!("HEAD{}TAIL", "é".repeat(100));
        let t = truncate_middle(&s, 20);
        assert!(t.starts_with("HEAD") && t.ends_with("TAIL"), "{t}");
        assert!(t.contains("chars truncated"));
    }

    #[test]
    fn exec_format_matches_codex() {
        let s = format_exec_output(1, Duration::from_millis(1500), "a\nb\n", false, 100);
        assert_eq!(s, "Exit code: 1\nWall time: 1.5 seconds\nOutput:\na\nb\n");
        let t = format_exec_output(124, Duration::from_secs(10), &"x\n".repeat(500), true, 64);
        assert!(t.starts_with("Exit code: 124\n") && t.contains("Total output lines: 500"));
        assert!(t.contains("command timed out after 10000 milliseconds"));
    }
}
