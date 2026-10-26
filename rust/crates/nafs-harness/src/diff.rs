//! The run's net change as a unified diff, after Codex's turn diff tracker
//! (`codex-rs/core/src/turn_diff_tracker.rs`: "the net text diff for the current turn ... without rereading
//! the workspace"). The workspace is in memory, so the diff is of the first and last workspace.
use crate::patch::Workspace;

const CONTEXT: usize = 3;
/// Above this many lines in either version a file is reported as changed without a line diff (the LCS
/// table is quadratic).
const MAX_LINES: usize = 2000;

/// `--- a/path` / `+++ b/path` sections with `@@ -l,n +l,n @@` hunks for every file that differs.
pub fn unified_diff(old: &Workspace, new: &Workspace) -> String {
    let mut paths: Vec<&String> = old.keys().chain(new.keys()).collect();
    paths.sort();
    paths.dedup();
    let mut out = String::new();
    for p in paths {
        let (a, b) = (old.get(p), new.get(p));
        if a == b {
            continue;
        }
        out.push_str(&match a {
            Some(_) => format!("--- a/{p}\n"),
            None => "--- /dev/null\n".into(),
        });
        out.push_str(&match b {
            Some(_) => format!("+++ b/{p}\n"),
            None => "+++ /dev/null\n".into(),
        });
        let a: Vec<&str> = a.map(|s| s.lines().collect()).unwrap_or_default();
        let b: Vec<&str> = b.map(|s| s.lines().collect()).unwrap_or_default();
        if a.len() > MAX_LINES || b.len() > MAX_LINES {
            out.push_str(&format!(
                "(file of {} -> {} lines changed; too large for a line diff)\n",
                a.len(),
                b.len()
            ));
            continue;
        }
        out.push_str(&hunks(&a, &b));
    }
    out
}

/// One edit: ' ' keeps a line, '-' removes one of `a`, '+' adds one of `b`.
fn edit_script<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(char, &'a str)> {
    let (n, m) = (a.len(), b.len());
    // lcs[i][j]: longest common subsequence of a[i..] and b[j..].
    let mut lcs = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[at(i, j)] = if a[i] == b[j] {
                lcs[at(i + 1, j + 1)] + 1
            } else {
                lcs[at(i + 1, j)].max(lcs[at(i, j + 1)])
            };
        }
    }
    let (mut i, mut j, mut ops) = (0, 0, vec![]);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            ops.push((' ', a[i]));
            i += 1;
            j += 1;
        } else if i < n && (j == m || lcs[at(i + 1, j)] >= lcs[at(i, j + 1)]) {
            // Removals before additions, as diff(1) prints them.
            ops.push(('-', a[i]));
            i += 1;
        } else {
            ops.push(('+', b[j]));
            j += 1;
        }
    }
    ops
}

fn hunks(a: &[&str], b: &[&str]) -> String {
    let ops = edit_script(a, b);
    // Line numbers (0-based) of `a` and `b` before each op.
    let mut pos = Vec::with_capacity(ops.len() + 1);
    let (mut oi, mut ni) = (0usize, 0usize);
    for (tag, _) in &ops {
        pos.push((oi, ni));
        match tag {
            ' ' => {
                oi += 1;
                ni += 1;
            }
            '-' => oi += 1,
            _ => ni += 1,
        }
    }
    pos.push((oi, ni));
    let n = ops.len();
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        if ops[i].0 == ' ' {
            i += 1;
            continue;
        }
        let start = i.saturating_sub(CONTEXT);
        let mut j = i;
        let end = loop {
            while j < n && ops[j].0 != ' ' {
                j += 1;
            }
            // Merge with the next change when the context between them would overlap.
            let mut k = j;
            while k < n && ops[k].0 == ' ' && k - j < 2 * CONTEXT {
                k += 1;
            }
            if k < n && ops[k].0 != ' ' {
                j = k;
            } else {
                break j;
            }
        };
        let stop = (end + CONTEXT).min(n);
        let old_n = ops[start..stop].iter().filter(|(t, _)| *t != '+').count();
        let new_n = ops[start..stop].iter().filter(|(t, _)| *t != '-').count();
        let line = |p: usize, count: usize| if count == 0 { p } else { p + 1 };
        out.push_str(&format!(
            "@@ -{},{old_n} +{},{new_n} @@\n",
            line(pos[start].0, old_n),
            line(pos[start].1, new_n)
        ));
        for (t, l) in &ops[start..stop] {
            out.push_str(&format!("{t}{l}\n"));
        }
        i = stop;
    }
    out
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
    fn a_changed_line_gets_context_and_a_header() {
        let old = ws(&[("s.py", "a\nb\nc\nd\ne\nf\ng\nh\n")]);
        let new = ws(&[("s.py", "a\nb\nc\nd\nE\nf\ng\nh\n")]);
        assert_eq!(
            unified_diff(&old, &new),
            "--- a/s.py\n+++ b/s.py\n@@ -2,7 +2,7 @@\n b\n c\n d\n-e\n+E\n f\n g\n h\n"
        );
    }

    #[test]
    fn added_and_deleted_files_and_unchanged_ones() {
        let old = ws(&[("keep.py", "x\n"), ("gone.py", "1\n2\n")]);
        let new = ws(&[("keep.py", "x\n"), ("new.py", "y\n")]);
        assert_eq!(
            unified_diff(&old, &new),
            "--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-1\n-2\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+y\n"
        );
        assert_eq!(unified_diff(&old, &old), "");
    }

    #[test]
    fn distant_changes_make_two_hunks() {
        let a: String = (0..20).map(|i| format!("{i}\n")).collect();
        let b = a.replace("2\n", "two\n").replace("17\n", "seventeen\n");
        let d = unified_diff(&ws(&[("f", &a)]), &ws(&[("f", &b)]));
        assert_eq!(d.matches("@@ -").count(), 2, "{d}");
        assert!(d.contains("@@ -1,6 +1,6 @@\n 0\n 1\n-2\n+two\n"));
    }
}
