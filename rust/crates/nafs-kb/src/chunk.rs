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
    let headings = heading_index(&doc.text);
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
    fn punctuation_only_chunks_are_dropped() {
        let c = chunk(&parsed("---\n\n***\n\n"), ChunkOpts::default());
        assert!(c.is_empty());
    }
}
