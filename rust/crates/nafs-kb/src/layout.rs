//! PDF layout (Phase 28): positioned text spans become Markdown in reading order, with headings, list items and table
//! rows marked. The rules are LiteParse's (run-llama, `crates/liteparse/src/projection.rs` and `markdown_layout/`),
//! rewritten over what `pdf_oxide` gives for a span: its box, font size, weight and style. No PDFium, no OCR.
//!
//! 1. Reading order by recursive XY-cut. A column gutter is a valley in the character density of the region's narrow
//!    lines; lines that cross it (a title, an abstract, a wide caption) are cut off first as bands above, between or
//!    below the columns, so the columns are read one after the other and a full-width line stays whole.
//! 2. Lines inside a region: spans join when their heights overlap by half of the smaller one.
//! 3. Running headers and footers: lines in the top or bottom band of the page that recur, digits ignored, on half the
//!    pages; a lone number there is a page number.
//! 4. Blocks: lines join into a paragraph unless the size, weight or style changes, the gap opens, the line is
//!    indented, the previous line ended short with a full stop, or the line starts a list item. A word hyphenated at
//!    a line end is joined; a paragraph cut by a column or a page break is not split when the next line continues it.
//! 5. Headings: the body size is the largest size holding half the weight of the most used one; a short block in a
//!    larger size, a numbered short block in bold, italic or capitals ("I. INTRODUCTION", "A. Problem", "2.5 Results"),
//!    or a short all-bold block is a heading. Levels come from the numbering depth, else from the rank of the size.
//! 6. Table rows: three or more consecutive lines that each split into three or more cells at gaps wider than the
//!    font size are written as Markdown table rows, so a row stays one line for the chunker.

use std::collections::HashMap;
use std::sync::OnceLock;

/// A run of text on a page, in PDF points with y growing upwards.
#[derive(Debug, Clone)]
pub struct Span {
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    pub size: f32,
    pub bold: bool,
    pub italic: bool,
    pub text: String,
}

impl Span {
    fn chars(&self) -> usize {
        self.text.chars().filter(|c| !c.is_whitespace()).count()
    }
    fn cy(&self) -> f32 {
        (self.y0 + self.y1) / 2.0
    }
    fn h(&self) -> f32 {
        (self.y1 - self.y0).clamp(0.5, 24.0)
    }
}

/// A page: its spans and size.
#[derive(Debug, Clone, Default)]
pub struct Page {
    pub width: f32,
    pub height: f32,
    pub spans: Vec<Span>,
}

/// Depth limit of the XY-cut and the fewest spans a region must hold to be cut again (LiteParse: 12 and 4).
const MAX_DEPTH: usize = 12;
const MIN_REGION: usize = 4;
/// A span wider than this share of its region is a full-width line (a title, an abstract line, a wide caption).
const WIDE: f32 = 0.55;
/// A gutter is a run of density at or below this share of the median density of the narrow lines (LiteParse 0.10).
const VALLEY: f32 = 0.10;
/// Top and bottom bands of a page where running headers and footers sit (LiteParse 12 %).
const BAND: f32 = 0.10;

#[derive(Debug, Clone, Default)]
struct Line {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    size: f32,
    bold: bool,
    italic: bool,
    text: String,
    cells: Vec<String>,
    /// The line ended in a dot leader (a table-of-contents entry).
    leader: bool,
    /// The XY-cut region the line came from, in reading order.
    region: usize,
    /// The left and right edges of that region.
    region_x0: f32,
    region_x1: f32,
}

impl Line {
    fn h(&self) -> f32 {
        (self.y1 - self.y0).clamp(1.0, 24.0)
    }
    /// Centred in its region, away from both edges (a title block, a centred abstract's last line).
    fn centred(&self) -> bool {
        let mid = (self.x0 + self.x1) / 2.0;
        let rmid = (self.region_x0 + self.region_x1) / 2.0;
        (mid - rmid).abs() < 0.8 * self.size && self.x0 - self.region_x0 > 2.0 * self.size
    }
}

// ---------------------------------------------------------------------------------------------- reading order

fn bounds(spans: &[Span], idx: &[usize]) -> (f32, f32, f32, f32) {
    idx.iter().fold(
        (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
        |(a, b, c, d), &i| {
            let s = &spans[i];
            (a.min(s.x0), b.max(s.x1), c.min(s.y0), d.max(s.y1))
        },
    )
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// The best column gutter of a region: (low x, high x), from the character density of its narrow spans.
fn gutter(spans: &[Span], idx: &[usize], x0: f32, x1: f32) -> Option<(f32, f32)> {
    let width = x1 - x0;
    if width < 40.0 {
        return None;
    }
    let narrow: Vec<usize> = idx
        .iter()
        .copied()
        .filter(|&i| spans[i].x1 - spans[i].x0 < WIDE * width)
        .collect();
    // Columns hold most of a region's text; a few narrow fragments inside full-width lines (code words, commas) are
    // not columns.
    let all: usize = idx.iter().map(|&i| spans[i].chars()).sum();
    let narrow_chars: usize = narrow.iter().map(|&i| spans[i].chars()).sum();
    if narrow.len() < 6 || narrow_chars * 2 < all {
        return None;
    }
    let n = width.ceil() as usize + 1;
    let mut density = vec![0f32; n];
    for &i in &narrow {
        let s = &spans[i];
        let a = ((s.x0 - x0).max(0.0)) as usize;
        let b = (((s.x1 - x0).ceil()) as usize).min(n);
        let w = (b.saturating_sub(a)).max(1) as f32;
        let per = s.chars() as f32 / w;
        for d in density.iter_mut().take(b).skip(a) {
            *d += per;
        }
    }
    let med = median(density.iter().copied().filter(|&d| d > 0.0).collect());
    if med <= 0.0 {
        return None;
    }
    let size = median(narrow.iter().map(|&i| spans[i].size).collect()).max(4.0);
    let min_gap = (0.7 * size).max(6.0);
    let total: usize = narrow.iter().map(|&i| spans[i].chars()).sum();
    let mut best: Option<(f32, f32, f32)> = None;
    let mut i = 0;
    while i < n {
        if density[i] > VALLEY * med {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && density[i] <= VALLEY * med {
            i += 1;
        }
        let (lo, hi) = (x0 + start as f32, x0 + i as f32);
        // A valley at the region's edge is a margin, not a gutter.
        if start == 0 || i >= n || hi - lo < min_gap {
            continue;
        }
        // Both sides must read as text: a fifth of the characters, a quarter of the width and three spans each.
        let (mut lc, mut rc, mut ln, mut rn) = (0, 0, 0, 0);
        for &j in &narrow {
            let s = &spans[j];
            if s.x1 <= lo + 1.0 {
                lc += s.chars();
                ln += 1;
            } else if s.x0 >= hi - 1.0 {
                rc += s.chars();
                rn += 1;
            }
        }
        let ok = lc * 5 >= total
            && rc * 5 >= total
            && ln >= 3
            && rn >= 3
            && lo - x0 >= 0.2 * width
            && x1 - hi >= 0.2 * width;
        let gap = hi - lo;
        if ok && best.is_none_or(|b| gap > b.2) {
            best = Some((lo, hi, gap));
        }
    }
    best.map(|(lo, hi, _)| (lo, hi))
}

/// A region that is a table: at least three rows that split into three or more cells at gaps wider than the font
/// size, two fifths of its rows at least, with cell starts on two or more shared tracks. A table is a leaf: the column
/// cut must not read it column by column (LiteParse's "tabular" flag; column-major order is the main table failure).
fn is_table(spans: &[Span], idx: &[usize]) -> bool {
    let mut items: Vec<&Span> = idx.iter().map(|&i| &spans[i]).collect();
    items.sort_by(|a, b| b.cy().total_cmp(&a.cy()));
    let mut rows: Vec<Vec<&Span>> = Vec::new();
    for s in items {
        match rows.last_mut() {
            Some(r) if (r[0].cy() - s.cy()).abs() < 0.5 * r[0].h().min(s.h()) => r.push(s),
            _ => rows.push(vec![s]),
        }
    }
    let mut starts: Vec<Vec<f32>> = Vec::new();
    for r in &mut rows {
        r.sort_by(|a, b| a.x0.total_cmp(&b.x0));
        let mut cells = vec![r[0].x0];
        for w in r.windows(2) {
            if w[1].x0 - w[0].x1 > 1.0 * w[0].size.max(w[1].size) {
                cells.push(w[1].x0);
            }
        }
        if cells.len() >= 3 {
            starts.push(cells);
        }
    }
    if starts.len() < 3 || starts.len() * 5 < rows.len() * 2 {
        return false;
    }
    let mut tracks: HashMap<i32, usize> = HashMap::new();
    for cells in &starts {
        let mut seen: Vec<i32> = cells.iter().map(|x| (x / 6.0).round() as i32).collect();
        seen.dedup();
        for b in seen {
            *tracks.entry(b).or_default() += 1;
        }
    }
    tracks.values().filter(|&&c| c * 2 >= starts.len()).count() >= 2
}

/// Cut a region and append its leaves (span indices) in reading order.
fn xy_cut(spans: &[Span], idx: Vec<usize>, depth: usize, out: &mut Vec<Vec<usize>>) {
    if idx.len() < MIN_REGION || depth >= MAX_DEPTH {
        out.push(idx);
        return;
    }
    let (x0, x1, _, _) = bounds(spans, &idx);
    let gut = if is_table(spans, &idx) {
        None
    } else {
        gutter(spans, &idx, x0, x1)
    };
    if let Some((lo, hi)) = gut {
        let crosses = |s: &Span| s.x0 < lo - 1.0 && s.x1 > hi + 1.0;
        // Bands: the vertical extents of the lines that cross the gutter, merged.
        let mut bands: Vec<(f32, f32)> = idx
            .iter()
            .map(|&i| &spans[i])
            .filter(|s| crosses(s))
            .map(|s| (s.y0, s.y1))
            .collect();
        bands.sort_by(|a, b| b.1.total_cmp(&a.1));
        // Lines of one wrapped title or abstract are one band: the gap between them is below a line height.
        let mut merged: Vec<(f32, f32)> = Vec::new();
        for (a, b) in bands {
            match merged.last_mut() {
                Some(m) if b >= m.0 - 0.6 * (b - a).clamp(4.0, 24.0) => m.0 = m.0.min(a),
                _ => merged.push((a, b)),
            }
        }
        if merged.is_empty() {
            let (mut left, mut right): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
            for &i in &idx {
                let s = &spans[i];
                if (s.x0 + s.x1) / 2.0 < (lo + hi) / 2.0 {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            if !left.is_empty() && !right.is_empty() {
                xy_cut(spans, left, depth + 1, out);
                xy_cut(spans, right, depth + 1, out);
                return;
            }
        } else {
            // Groups top to bottom: the stretch above the first band, the band, the stretch below it, ...
            let group_of = |s: &Span| -> usize {
                let cy = s.cy();
                let mut g = 0;
                for (k, (a, b)) in merged.iter().enumerate() {
                    if cy > *b {
                        return g;
                    }
                    if cy >= *a {
                        return 2 * k + 1;
                    }
                    g = 2 * k + 2;
                }
                g
            };
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); 2 * merged.len() + 1];
            for &i in &idx {
                groups[group_of(&spans[i])].push(i);
            }
            let filled = groups.iter().filter(|g| !g.is_empty()).count();
            if filled > 1 {
                for g in groups.into_iter().filter(|g| !g.is_empty()) {
                    xy_cut(spans, g, depth + 1, out);
                }
                return;
            }
        }
    }
    // No columns here: cut at the widest horizontal gap, so a band below may have columns of its own.
    let mut ys: Vec<(f32, f32)> = idx.iter().map(|&i| (spans[i].y0, spans[i].y1)).collect();
    ys.sort_by(|a, b| b.1.total_cmp(&a.1));
    let size = median(idx.iter().map(|&i| spans[i].size).collect()).max(4.0);
    let mut best: Option<(f32, f32)> = None;
    let mut low = ys[0].0;
    for &(a, b) in &ys[1..] {
        if b < low {
            let gap = low - b;
            if gap >= 0.5 * size && best.is_none_or(|x| gap > x.1) {
                best = Some(((low + b) / 2.0, gap));
            }
        }
        low = low.min(a);
    }
    match best {
        Some((y, _)) => {
            let (above, below): (Vec<usize>, Vec<usize>) =
                idx.iter().partition(|&&i| spans[i].cy() > y);
            if above.is_empty() || below.is_empty() {
                out.push(idx);
            } else {
                xy_cut(spans, above, depth + 1, out);
                xy_cut(spans, below, depth + 1, out);
            }
        }
        None => out.push(idx),
    }
}

// ---------------------------------------------------------------------------------------------- lines

fn lines_of(spans: &[Span], leaf: &[usize], region: usize) -> Vec<Line> {
    let mut items: Vec<&Span> = leaf.iter().map(|&i| &spans[i]).collect();
    items.sort_by(|a, b| b.cy().total_cmp(&a.cy()).then(a.x0.total_cmp(&b.x0)));
    let mut groups: Vec<Vec<&Span>> = Vec::new();
    for s in items {
        // Join the line whose vertical centre is within half the smaller height (LiteParse's guard against the
        // inflated boxes of large glyphs).
        let found = groups.iter_mut().rev().take(3).find(|g| {
            let cy = g.iter().map(|x| x.cy()).sum::<f32>() / g.len() as f32;
            let h = g.iter().map(|x| x.h()).fold(f32::MAX, f32::min);
            (s.cy() - cy).abs() < 0.5 * h.min(s.h())
        });
        match found {
            Some(g) => g.push(s),
            None => groups.push(vec![s]),
        }
    }
    let region_x1 = leaf.iter().map(|&i| spans[i].x1).fold(f32::MIN, f32::max);
    let region_x0 = leaf.iter().map(|&i| spans[i].x0).fold(f32::MAX, f32::min);
    let mut out: Vec<Line> = groups
        .into_iter()
        .filter_map(|mut g| {
            g.sort_by(|a, b| a.x0.total_cmp(&b.x0));
            let mut text = String::new();
            let mut cells = vec![String::new()];
            let mut prev: Option<&Span> = None;
            let (mut weight, mut bold_w, mut italic_w) = (0usize, 0usize, 0usize);
            let mut sizes: HashMap<i32, usize> = HashMap::new();
            for s in &g {
                if let Some(p) = prev {
                    // A fake-bold overprint repeats the same text on top of itself.
                    if p.text == s.text && s.x0 < p.x1 - 0.5 * (p.x1 - p.x0) {
                        continue;
                    }
                    let gap = s.x0 - p.x1;
                    let size = p.size.max(s.size);
                    if gap > 1.0 * size && !s.text.trim().is_empty() {
                        cells.push(String::new());
                    }
                    let spaced = text.ends_with(char::is_whitespace)
                        || s.text.starts_with(char::is_whitespace);
                    if gap > 0.12 * size && !spaced {
                        text.push(' ');
                        if !cells.last().is_some_and(|c| c.is_empty()) {
                            cells.last_mut().expect("one cell").push(' ');
                        }
                    }
                }
                text.push_str(&s.text);
                cells.last_mut().expect("one cell").push_str(&s.text);
                let c = s.chars();
                weight += c;
                if s.bold {
                    bold_w += c;
                }
                if s.italic {
                    italic_w += c;
                }
                *sizes.entry((s.size * 2.0).round() as i32).or_default() += c;
                prev = Some(s);
            }
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            // A dot leader and the page number after it (a table of contents) carry no words.
            let led = leader().is_match(&text);
            let text = leader().replace_all(&text, "").trim().to_string();
            if text.is_empty() || weight == 0 {
                return None;
            }
            let size = sizes
                .into_iter()
                .max_by_key(|(k, w)| (*w, *k))
                .map_or(0.0, |(k, _)| k as f32 / 2.0);
            let cells = cells
                .into_iter()
                .map(|c| c.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|c| !c.is_empty())
                .collect();
            Some(Line {
                x0: g.iter().map(|s| s.x0).fold(f32::MAX, f32::min),
                x1: g.iter().map(|s| s.x1).fold(f32::MIN, f32::max),
                y0: g.iter().map(|s| s.y0).fold(f32::MAX, f32::min),
                y1: g.iter().map(|s| s.y1).fold(f32::MIN, f32::max),
                size,
                bold: bold_w * 10 >= weight * 8,
                italic: italic_w * 10 >= weight * 8,
                text,
                cells,
                leader: led,
                region,
                region_x0,
                region_x1,
            })
        })
        .collect();
    out.sort_by(|a, b| b.y1.total_cmp(&a.y1));
    out
}

fn page_lines(page: &Page) -> Vec<Line> {
    let spans: Vec<Span> = page
        .spans
        .iter()
        .filter(|s| {
            s.x1 > s.x0
                && s.size > 0.5
                && !s.text.trim().is_empty()
                && s.y1 > 0.0
                && s.y0 < page.height.max(1.0)
        })
        .cloned()
        .collect();
    if spans.is_empty() {
        return Vec::new();
    }
    let mut leaves = Vec::new();
    xy_cut(&spans, (0..spans.len()).collect(), 0, &mut leaves);
    leaves
        .iter()
        .enumerate()
        .flat_map(|(r, leaf)| lines_of(&spans, leaf, r))
        .collect()
}

// ---------------------------------------------------------------------------------------------- the document

/// A running line with its page number removed, wherever the number stands (left on even pages, right on odd).
fn running_key(text: &str) -> String {
    let t: String = text
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .collect();
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn lone_number(text: &str) -> bool {
    let t = text.trim();
    t.chars().count() <= 4
        && (t.chars().all(|c| c.is_ascii_digit())
            || (t.chars().all(|c| "ivxlcIVXLC".contains(c)) && !t.is_empty()))
}

/// Remove running headers, footers and page numbers by position and repetition.
fn drop_running(pages: &mut [(Page, Vec<Line>)]) {
    let in_band = |p: &Page, l: &Line| {
        let h = p.height.max(1.0);
        l.y0 >= h * (1.0 - BAND) || l.y1 <= h * BAND
    };
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (p, lines) in pages.iter() {
        let mut keys: Vec<String> = lines
            .iter()
            .filter(|l| in_band(p, l) && l.text.chars().count() <= 120)
            .map(|l| running_key(&l.text))
            .collect();
        keys.sort();
        keys.dedup();
        for k in keys {
            *seen.entry(k).or_default() += 1;
        }
    }
    let n = pages.len();
    let threshold = n.div_ceil(2).max(2);
    for (p, lines) in pages.iter_mut() {
        let p = &*p;
        lines.retain(|l| {
            if !in_band(p, l) {
                return true;
            }
            let recurs = n >= 3
                && seen
                    .get(&running_key(&l.text))
                    .is_some_and(|&c| c >= threshold);
            !(recurs || lone_number(&l.text))
        });
    }
}

/// The body size: of the sizes holding at least half the weight of the most used one, the largest (LiteParse: a page
/// of small-print references does not win).
fn body_size(pages: &[(Page, Vec<Line>)]) -> f32 {
    let mut w: HashMap<i32, usize> = HashMap::new();
    for (_, lines) in pages {
        for l in lines {
            *w.entry((l.size * 2.0).round() as i32).or_default() += l.text.chars().count();
        }
    }
    let top = w.values().copied().max().unwrap_or(0);
    w.into_iter()
        .filter(|(_, c)| *c * 2 >= top)
        .map(|(k, _)| k)
        .max()
        .map_or(10.0, |k| k as f32 / 2.0)
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Para,
    Item,
    Row,
}

#[derive(Debug, Clone)]
struct Block {
    kind: Kind,
    lines: Vec<Line>,
    text: String,
}

fn numbering() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"^(?:(?:[IVX]{1,6}|[A-Z])\.|\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-Z](?:\.\d{1,2}){1,3}|§\s?\d+)\s+\S")
            .expect("valid regex")
    })
}

fn caption() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)^(?:fig\.?|figure|table|tab\.|algorithm|listing|source|note)\s*[\dIVX:.]",
        )
        .expect("valid regex")
    })
}

const BULLETS: &[char] = &['•', '◦', '▪', '‣', '∙', '·', '\u{f0b7}', '–', '○', '■'];

fn bullet(text: &str) -> bool {
    let mut c = text.chars();
    matches!((c.next(), c.next()), (Some(b), Some(' ')) if BULLETS.contains(&b))
}

/// "1) ", "(a) ", "(iv) ": an enumerated item. "1. " is left to the heading rules (a numbered section looks the same).
fn enumerated(text: &str) -> bool {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"^(?:\d{1,2}\)|\((?:\d{1,2}|[a-z]|[ivx]{1,4})\))\s+\S")
            .expect("valid regex")
    })
    .is_match(text)
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '!', '?', ':'])
}

/// Join a line to a paragraph, rejoining a word hyphenated at the line end.
fn join(text: &mut String, next: &str) {
    if text.is_empty() {
        text.push_str(next);
        return;
    }
    let hyphen = text.ends_with('-')
        && text[..text.len() - 1]
            .chars()
            .last()
            .is_some_and(char::is_alphabetic)
        && next.chars().next().is_some_and(char::is_lowercase);
    if hyphen {
        text.pop();
    } else {
        text.push(' ');
    }
    text.push_str(next);
}

/// Lines of one page in reading order to blocks. `carry` is the open paragraph of the previous page, if it continues.
/// A run of four or more dots, spaced or not, and the page number after it.
fn leader() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"\s*(?:\.\s?){4,}\s*(?:\d{1,4}|[ivxlc]{1,6})?\b").expect("valid regex")
    })
}

/// A table-of-contents page: at least three lines ended a dot leader. Its lines are entries, not headings.
fn toc_page(page: &Page) -> bool {
    let mut lines: HashMap<i32, String> = HashMap::new();
    for s in &page.spans {
        lines
            .entry(s.y0.round() as i32)
            .or_default()
            .push_str(&s.text);
    }
    lines.values().filter(|t| leader().is_match(t)).count() >= 3
}

fn blocks_of(lines: &[Line], toc: bool) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let l = &lines[i];
        // Table rows: three or more consecutive lines of three or more cells in one region.
        if l.cells.len() >= 3 {
            let mut j = i;
            while j < lines.len()
                && lines[j].cells.len() >= 3
                && lines[j].region == l.region
                && (j == i || lines[j - 1].y0 - lines[j].y1 < 2.5 * lines[j].h())
            {
                j += 1;
            }
            if j - i >= 3 {
                // The header line above the rows, when its cells are fewer (a merged or short header), joins them.
                let header = out.last().is_some_and(|b| {
                    b.kind == Kind::Para
                        && b.lines.len() == 1
                        && b.lines[0].region == l.region
                        && b.lines[0].cells.len() >= 2
                        && b.lines[0].y0 - l.y1 < 2.5 * l.h()
                });
                if header {
                    let h = out.pop().expect("checked above").lines.remove(0);
                    out.push(Block {
                        kind: Kind::Row,
                        text: format!("| {} |", h.cells.join(" | ")),
                        lines: vec![h],
                    });
                }
                for r in &lines[i..j] {
                    out.push(Block {
                        kind: Kind::Row,
                        text: format!("| {} |", r.cells.join(" | ")),
                        lines: vec![r.clone()],
                    });
                }
                i = j;
                continue;
            }
        }
        let item = bullet(&l.text) || enumerated(&l.text);
        let new = match out.last() {
            None => true,
            Some(b) if toc && (l.leader || b.lines.last().is_some_and(|p| p.leader)) => true,
            Some(b) if b.kind == Kind::Row => true,
            Some(b) => {
                let p = b.lines.last().expect("a block has lines");
                let same_region = p.region == l.region;
                let continues = !ends_sentence(&b.text)
                    && l.text.chars().next().is_some_and(char::is_lowercase)
                    && (p.size - l.size).abs() <= 0.5;
                let gap = p.y0 - l.y1;
                item || (p.size - l.size).abs() > 0.5
                    || p.bold != l.bold
                    || p.italic != l.italic
                    || if same_region {
                        gap > 0.9 * p.h().min(l.h())
                            || gap < -0.5 * p.h()
                            // An indented line opens a paragraph, unless it follows the block's first line (a hanging
                            // indent: "G1. Answer ..." with its second line under the text).
                            || l.x0 > p.x0 + 0.8 * l.size && b.kind == Kind::Para && b.lines.len() > 1 && !l.centred()
                            || (p.x1 < p.region_x1 - 2.0 * p.size && ends_sentence(&p.text))
                    } else {
                        !continues
                    }
            }
        };
        // A paragraph cut by a footnote or a figure at the column's foot continues in the next region: join it to
        // the last paragraph of its size that is still open, two blocks back at most.
        if new
            && !item
            && out
                .last()
                .is_some_and(|b| b.lines.last().is_some_and(|p| p.region != l.region))
        {
            let open = out.iter().rev().take(3).position(|b| {
                b.kind == Kind::Para
                    && !ends_sentence(&b.text)
                    && b.lines
                        .last()
                        .is_some_and(|p| (p.size - l.size).abs() <= 0.5 && p.bold == l.bold)
            });
            if let Some(k) = open.filter(|_| l.text.chars().next().is_some_and(char::is_lowercase))
            {
                let at = out.len() - 1 - k;
                join(&mut out[at].text, &l.text);
                out[at].lines.push(l.clone());
                i += 1;
                continue;
            }
        }
        if new {
            let (kind, text) = if bullet(&l.text) {
                (Kind::Item, l.text.chars().skip(2).collect::<String>())
            } else if item {
                (Kind::Item, l.text.clone())
            } else {
                (Kind::Para, l.text.clone())
            };
            out.push(Block {
                kind,
                lines: vec![l.clone()],
                text,
            });
        } else {
            let b = out.last_mut().expect("an open block");
            join(&mut b.text, &l.text);
            b.lines.push(l.clone());
        }
        i += 1;
    }
    out
}

/// The heading level of a block, if it is one. `ranks` maps a heading size (half points) to its rank, largest first.
fn heading_level(b: &Block, body: f32, ranks: &HashMap<i32, usize>) -> Option<usize> {
    if b.kind != Kind::Para || b.lines.len() > 3 {
        return None;
    }
    let t = b.text.trim();
    let n = t.chars().count();
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    let visible = t.chars().filter(|c| !c.is_whitespace()).count().max(1);
    if !(2..=160).contains(&n)
        || letters * 2 < visible
        || caption().is_match(t)
        || t.ends_with([',', ';'])
    {
        return None;
    }
    let l = &b.lines[0];
    let numbered = numbering().find(t).map(|m| m.as_str());
    let depth = numbered.map(|m| {
        let head = m.split_whitespace().next().unwrap_or("");
        let dots = head.trim_end_matches('.').matches('.').count();
        let letter = head
            .trim_end_matches('.')
            .chars()
            .all(|c| c.is_ascii_uppercase())
            && head.len() <= 2;
        let roman = head
            .trim_end_matches('.')
            .chars()
            .all(|c| "IVX".contains(c));
        // "A." under a roman numeral is a subsection (IEEE); "A.1" is two deep; "2.5" is two deep.
        if letter && !roman {
            2
        } else {
            1 + dots
        }
    });
    let words = t.split_whitespace().count();
    if l.size >= body + 0.9 {
        if words > 25 {
            return None;
        }
        let rank = ranks
            .get(&((l.size * 2.0).round() as i32))
            .copied()
            .unwrap_or(0);
        return Some(depth.map_or(rank + 1, |d| d + 1).min(6));
    }
    if (l.size - body).abs() > 0.9 || words > 14 || n > 100 {
        return None;
    }
    let caps = t
        .chars()
        .filter(|c| c.is_alphabetic())
        .all(|c| c.is_uppercase())
        && letters >= 3;
    let final_stop = t.ends_with('.') && !caps;
    if let Some(d) = depth {
        let body_text = numbering()
            .find(t)
            .and_then(|m| t[m.start()..].split_once(char::is_whitespace))
            .map_or(t, |(_, rest)| rest)
            .trim();
        let starts_upper = body_text.chars().next().is_some_and(|c| c.is_uppercase());
        if (b.lines.iter().all(|x| x.bold) || b.lines.iter().all(|x| x.italic) || caps)
            && starts_upper
            && !final_stop
        {
            return Some((d + 1).min(6));
        }
        return None;
    }
    if b.lines.iter().all(|x| x.bold)
        && !final_stop
        && t.chars()
            .next()
            .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())
        && words <= 10
    {
        return Some((ranks.len() + 2).min(4));
    }
    None
}

/// Lay out a document: one Markdown string per page.
pub fn markdown(pages: Vec<Page>) -> Vec<String> {
    let mut with_lines: Vec<(Page, Vec<Line>)> = pages
        .into_iter()
        .map(|p| {
            let l = page_lines(&p);
            (p, l)
        })
        .collect();
    drop_running(&mut with_lines);
    let body = body_size(&with_lines);
    let tocs: Vec<bool> = with_lines.iter().map(|(p, _)| toc_page(p)).collect();
    let blocks: Vec<Vec<Block>> = with_lines
        .iter()
        .zip(&tocs)
        .map(|((_, l), &toc)| blocks_of(l, toc))
        .collect();
    // Sizes larger than the body that some short block uses, largest first.
    let mut sizes: Vec<i32> = blocks
        .iter()
        .flatten()
        .filter(|b| b.kind == Kind::Para && b.lines.len() <= 3)
        .map(|b| b.lines[0].size)
        .filter(|&s| s >= body + 0.9)
        .map(|s| (s * 2.0).round() as i32)
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();
    let ranks: HashMap<i32, usize> = sizes
        .into_iter()
        .take(5)
        .enumerate()
        .map(|(r, s)| (s, r))
        .collect();
    blocks
        .into_iter()
        .zip(tocs)
        .map(|(bs, toc)| {
            // Heading levels, with a heading wrapped over several blocks (a long title cut into bands) joined: same
            // level and size, one under the other.
            let mut bs = bs;
            let mut levels: Vec<Option<usize>> = Vec::with_capacity(bs.len());
            let mut k = 0;
            while k < bs.len() {
                let b = &bs[k];
                let level = if b.kind == Kind::Para {
                    heading_level(b, body, &ranks).filter(|_| {
                        !(toc
                            && (b.lines.iter().any(|l| l.leader)
                                || b.text.ends_with(|c: char| c.is_ascii_digit())))
                    })
                } else {
                    None
                };
                if let (Some(l), Some(&Some(pl))) = (level, levels.last()) {
                    let (p, c) = (bs[k - 1].lines.last().expect("lines"), &b.lines[0]);
                    if pl == l
                        && (p.size - c.size).abs() <= 0.5
                        && p.y0 - c.y1 < 1.5 * c.h()
                        && p.y0 > c.y1 - 0.5 * c.h()
                        && !bs[k - 1].text.trim_end().ends_with(['.', '!', '?'])
                    {
                        let (head, tail) = bs.split_at_mut(k);
                        let prev = head.last_mut().expect("k > 0");
                        join(&mut prev.text, &tail[0].text);
                        let moved = std::mem::take(&mut tail[0].lines);
                        prev.lines.extend(moved);
                        bs.remove(k);
                        continue;
                    }
                }
                levels.push(level);
                k += 1;
            }
            let mut out = String::new();
            let mut prev_row = false;
            for (b, level) in bs.iter().zip(&levels) {
                let row = b.kind == Kind::Row;
                if !out.is_empty() {
                    out.push_str(if row && prev_row { "\n" } else { "\n\n" });
                }
                match b.kind {
                    Kind::Row => {
                        out.push_str(&b.text);
                        if !prev_row {
                            let cols = b.text.matches(" | ").count() + 1;
                            out.push('\n');
                            out.push_str(&format!("|{}", " --- |".repeat(cols)));
                        }
                    }
                    Kind::Item => {
                        out.push_str("- ");
                        out.push_str(&b.text);
                    }
                    Kind::Para => match level {
                        Some(level) => {
                            out.push_str(&"#".repeat(*level));
                            out.push(' ');
                            out.push_str(b.text.trim());
                        }
                        None => out.push_str(&b.text),
                    },
                }
                prev_row = row;
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(x0: f32, y: f32, text: &str, size: f32) -> Span {
        Span {
            x0,
            x1: x0 + text.chars().count() as f32 * size * 0.5,
            y0: y,
            y1: y + size,
            size,
            bold: false,
            italic: false,
            text: text.to_string(),
        }
    }

    fn page(spans: Vec<Span>) -> Page {
        Page {
            width: 612.0,
            height: 792.0,
            spans,
        }
    }

    #[test]
    fn two_columns_are_read_one_after_the_other_under_a_full_width_title() {
        let mut s = vec![span(
            100.0,
            720.0,
            "A Title That Runs Across Both Columns Of The Page",
            18.0,
        )];
        for i in 0..12 {
            let y = 650.0 - i as f32 * 12.0;
            s.push(span(
                50.0,
                y,
                &format!("left column line {i} of prose text"),
                10.0,
            ));
            s.push(span(
                320.0,
                y,
                &format!("right column line {i} of prose text"),
                10.0,
            ));
        }
        let md = markdown(vec![page(s)]).remove(0);
        let l11 = md.find("left column line 11").unwrap();
        let r0 = md.find("right column line 0").unwrap();
        assert!(l11 < r0, "{md}");
        assert!(md.starts_with("# A Title"), "{md}");
    }

    #[test]
    fn numbered_headings_hyphens_and_running_lines() {
        let mut pages = Vec::new();
        for p in 0..4 {
            let mut s = vec![
                span(50.0, 760.0, "Journal of Examples", 8.0),
                span(300.0, 30.0, &format!("{}", p + 1), 8.0),
            ];
            let mut head = span(50.0, 700.0, "I. INTRODUCTION", 10.0);
            if p == 0 {
                head.text = "I. INTRODUCTION".into();
            } else {
                head.text = format!("{}. RESULTS PART", ["I", "II", "III", "IV"][p]);
            }
            s.push(head);
            s.push(span(50.0, 680.0, "The method reads every docu-", 10.0));
            s.push(span(50.0, 668.0, "ment in the base and keeps its", 10.0));
            s.push(span(50.0, 656.0, "structure for the search.", 10.0));
            pages.push(page(s));
        }
        let md = markdown(pages);
        assert!(md[0].contains("## I. INTRODUCTION"), "{}", md[0]);
        assert!(md[0].contains("every document in the base"), "{}", md[0]);
        assert!(!md[1].contains("Journal of Examples"), "{}", md[1]);
        assert!(!md[1].lines().any(|l| l.trim() == "2"), "{}", md[1]);
    }

    #[test]
    fn aligned_cells_become_table_rows_and_bullets_list_items() {
        let mut s = Vec::new();
        for (i, row) in [
            ["Model", "Size", "Score"],
            ["Small", "0.6B", "0.81"],
            ["Large", "1.7B", "0.88"],
        ]
        .iter()
        .enumerate()
        {
            let y = 700.0 - i as f32 * 12.0;
            for (c, cell) in row.iter().enumerate() {
                s.push(span(60.0 + c as f32 * 120.0, y, cell, 10.0));
            }
        }
        s.push(span(60.0, 600.0, "• first point of the list", 10.0));
        s.push(span(60.0, 588.0, "• second point of the list", 10.0));
        let md = markdown(vec![page(s)]).remove(0);
        assert!(
            md.contains("| Model | Size | Score |\n| --- | --- | --- |\n| Small | 0.6B | 0.81 |"),
            "{md}"
        );
        assert!(
            md.contains("- first point of the list\n\n- second point"),
            "{md}"
        );
    }
}
