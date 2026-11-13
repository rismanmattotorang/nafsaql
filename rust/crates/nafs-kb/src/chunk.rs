//! Chunking (FR-K15): `text-splitter`'s Markdown splitter cuts at headings first, then paragraphs, sentences and words;
//! each chunk keeps its heading path and, for PDFs, the page it starts on.

use text_splitter::{ChunkConfig, MarkdownSplitter};

use crate::parse::Parsed;

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub ord: usize,
    pub text: String,
    /// The headings above the chunk, outermost first, joined with " > ".
    pub heading: String,
    pub page: Option<usize>,
    pub chars: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkOpts {
    /// Target size in characters (upper bound).
    pub chars: usize,
    /// Characters shared with the previous chunk.
    pub overlap: usize,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        ChunkOpts {
            chars: 1200,
            overlap: 150,
        }
    }
}

pub fn chunk(doc: &Parsed, opts: ChunkOpts) -> Vec<Chunk> {
    let max = opts.chars.max(200);
    let min = (max / 4).max(50);
    let overlap = opts.overlap.min(min.saturating_sub(1));
    let cfg = ChunkConfig::new(min..max)
        .with_trim(true)
        .with_overlap(overlap)
        .expect("overlap is below the minimum");
    let splitter = MarkdownSplitter::new(cfg);
    let mut headings = heading_index(&doc.text);
    // The document's own title as its first heading is already the passage's title: repeating it in every heading
    // path only costs tokens in the vectors' text and in the prompt's source headers.
    if let (Some(t), Some(first)) = (&doc.title, headings.first()) {
        if first.1 == 1 && first.2 == *t && !headings[1..].iter().any(|h| h.1 == 1) {
            headings.remove(0);
        }
    }
    splitter
        .chunk_indices(&doc.text)
        .filter(|(_, t)| t.chars().any(char::is_alphanumeric))
        .enumerate()
        .map(|(ord, (at, text))| Chunk {
            ord,
            heading: heading_path(&headings, at, text),
            page: doc.page_at(at),
            chars: text.chars().count(),
            text: text.to_string(),
        })
        .collect()
}

/// Phase 28 (semtools' small search unit): the sentence windows of a passage that get vectors of their own. A
/// passage's vector averages everything it says, so a question about one of its sentences, put in other words, can
/// sit far from it; a window of two or three sentences is matched instead and scores the passage it belongs to.
/// Sentences are packed in order to about `target` characters (never across a table row or a list item); a passage
/// that makes a single window gets none, its own vector already is that window.
pub fn pieces(text: &str, target: usize) -> Vec<String> {
    // (text, is a table row): rows and sentences do not share a window.
    let mut units: Vec<(String, bool)> = Vec::new();
    for block in text.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        // Table rows and list items stay whole; a separator row carries no words.
        if block.lines().all(|l| l.trim_start().starts_with('|')) {
            units.extend(
                block
                    .lines()
                    .filter(|l| l.chars().any(char::is_alphanumeric))
                    .map(|l| (l.to_string(), true)),
            );
            continue;
        }
        let flat = block.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut start = 0;
        let b = flat.as_bytes();
        for i in 1..b.len().saturating_sub(1) {
            // A sentence ends at . ! ? followed by a space and a capital, a digit or an opening mark.
            if matches!(b[i], b'.' | b'!' | b'?')
                && b[i + 1] == b' '
                && flat[i + 2..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit() || "([\"'".contains(c))
                && i + 1 - start >= 20
            {
                units.push((flat[start..=i].to_string(), false));
                start = i + 2;
            }
        }
        if start < flat.len() {
            units.push((flat[start..].to_string(), false));
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_rows = false;
    for (u, row) in units {
        if !cur.is_empty() && (cur.len() + u.len() + 1 > target * 3 / 2 || row != cur_rows) {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&u);
        cur_rows = row;
        if cur.len() >= target {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        // A short tail joins the window before it.
        match out.last_mut() {
            Some(last) if cur.len() < target / 3 => {
                last.push(' ');
                last.push_str(&cur);
            }
            _ => out.push(cur),
        }
    }
    out.retain(|p| p.chars().filter(|c| c.is_alphanumeric()).count() >= 12);
    if out.len() < 2 {
        return Vec::new();
    }
    out
}

/// Every Markdown heading in the text: (byte offset, level, title).
fn heading_index(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut fence = false;
    for line in text.split_inclusive('\n') {
        let l = line.trim_end();
        if l.trim_start().starts_with("```") {
            fence = !fence;
        } else if !fence {
            let level = l.chars().take_while(|&c| c == '#').count();
            if (1..=6).contains(&level) && l[level..].starts_with(' ') {
                let title = l[level..].trim().trim_matches('*').trim();
                if !title.is_empty() {
                    out.push((at, level, title.to_string()));
                }
            }
        }
        at += line.len();
    }
    out
}

/// The heading path in force at byte offset `at`. A chunk that starts with its own heading is filed under it.
fn heading_path(headings: &[(usize, usize, String)], at: usize, text: &str) -> String {
    let own = text.trim_start().starts_with('#');
    let mut stack: Vec<(usize, &str)> = Vec::new();
    for (off, level, title) in headings {
        if *off > at || (*off == at && !own) {
            break;
        }
        while stack.last().is_some_and(|(l, _)| *l >= *level) {
            stack.pop();
        }
        stack.push((*level, title));
    }
    stack
        .iter()
        .map(|(_, t)| *t)
        .collect::<Vec<_>>()
        .join(" > ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Parsed {
        Parsed {
            text: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn short_text_is_one_chunk_with_its_heading() {
        let c = chunk(
            &parsed("# Title\n\nOne short paragraph."),
            ChunkOpts::default(),
        );
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].heading, "Title");
        assert_eq!(c[0].ord, 0);
        assert_eq!(c[0].page, None);
    }

    #[test]
    fn chunks_respect_the_bound_keep_order_and_heading_paths() {
        let para = "Memory slots attend to every stored document embedding. ".repeat(12);
        let text = format!(
            "# Book\n\n## Part A\n\n{para}\n\n{para}\n\n### Detail\n\n{para}\n\n## Part B\n\n{para}\n\n```\n# not a heading\n```\n"
        );
        let c = chunk(
            &parsed(&text),
            ChunkOpts {
                chars: 800,
                overlap: 100,
            },
        );
        assert!(c.len() >= 4, "{}", c.len());
        assert!(
            c.iter().all(|x| x.chars <= 800),
            "{:?}",
            c.iter().map(|x| x.chars).collect::<Vec<_>>()
        );
        assert!(c.windows(2).all(|w| w[0].ord + 1 == w[1].ord));
        assert!(
            c.iter().any(|x| x.heading == "Book > Part A"),
            "{:?}",
            c.iter().map(|x| &x.heading).collect::<Vec<_>>()
        );
        assert!(c.iter().any(|x| x.heading == "Book > Part A > Detail"));
        let last = c.last().unwrap();
        assert_eq!(
            last.heading,
            "Book > Part B",
            "{:?}",
            c.iter().map(|x| &x.heading).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pages_follow_page_starts() {
        let p1 = "First page sentence number one. ".repeat(30);
        let p2 = "Second page sentence number two. ".repeat(30);
        let text = format!("{p1}\n\n{p2}");
        let doc = Parsed {
            page_starts: vec![0, p1.len() + 2],
            pages: Some(2),
            text,
            ..Default::default()
        };
        let c = chunk(
            &doc,
            ChunkOpts {
                chars: 600,
                overlap: 0,
            },
        );
        assert_eq!(c.first().unwrap().page, Some(1));
        assert_eq!(c.last().unwrap().page, Some(2));
    }

    #[test]
    fn pieces_pack_sentences_and_keep_rows_whole() {
        let para = "The gate reads the draft. It escalates when the error is likely. The verifier decides alone. \
                    A reask counts as evidence. Budgets are fixed per request. The ledger keeps every outcome.";
        let text = format!("{para}\n\n| Model | Size |\n| --- | --- |\n| Qwen3-1.7B | 1.26 GiB |");
        let p = pieces(&text, 120);
        assert!(p.len() >= 3, "{p:?}");
        assert!(p.iter().all(|x| !x.contains("outcome. |")), "{p:?}");
        assert!(p.iter().all(|x| x.len() <= 180), "{p:?}");
        assert!(
            p.iter().any(|x| x.contains("| Qwen3-1.7B | 1.26 GiB |")),
            "{p:?}"
        );
        assert!(!p.iter().any(|x| x.contains("---")), "{p:?}");
        assert!(pieces("One short sentence only.", 120).is_empty());
    }

    #[test]
    fn punctuation_only_chunks_are_dropped() {
        let c = chunk(&parsed("---\n\n***\n\n"), ChunkOpts::default());
        assert!(c.is_empty());
    }
}
