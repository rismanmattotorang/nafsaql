//! Document parsing (FR-K02, FR-K10..K14): every supported kind becomes Markdown-ish text whose headings drive the
//! chunker. Pure Rust: `pdf_oxide` for PDF, our own ZIP + XML readers for DOCX, PPTX and XLSX (bounded against
//! hostile archives, SR-K03), our own CSV reader, `htmd` for HTML, our own LaTeX stripper.

use std::io::{Cursor, Read};

use quick_xml::events::Event;
use quick_xml::Reader;

/// The document kinds the knowledge base accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Pdf,
    Docx,
    Pptx,
    /// Excel workbooks (Phase 35): each sheet becomes a heading and Markdown tables.
    Xlsx,
    /// Comma-, semicolon- or tab-separated values (Phase 35).
    Csv,
    Tex,
    Html,
    Md,
    Txt,
}

impl Kind {
    pub const ALL: [Kind; 9] = [
        Kind::Pdf,
        Kind::Docx,
        Kind::Pptx,
        Kind::Xlsx,
        Kind::Csv,
        Kind::Tex,
        Kind::Html,
        Kind::Md,
        Kind::Txt,
    ];

    /// The kind named by a file's extension, if supported.
    pub fn from_filename(name: &str) -> Option<Kind> {
        let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase())?;
        Some(match ext.as_str() {
            "pdf" => Kind::Pdf,
            "docx" => Kind::Docx,
            "pptx" => Kind::Pptx,
            "xlsx" | "xlsm" => Kind::Xlsx,
            "csv" | "tsv" => Kind::Csv,
            "tex" | "latex" => Kind::Tex,
            "html" | "htm" | "xhtml" => Kind::Html,
            "md" | "markdown" => Kind::Md,
            "txt" | "text" => Kind::Txt,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Pdf => "pdf",
            Kind::Docx => "docx",
            Kind::Pptx => "pptx",
            Kind::Xlsx => "xlsx",
            Kind::Csv => "csv",
            Kind::Tex => "tex",
            Kind::Html => "html",
            Kind::Md => "md",
            Kind::Txt => "txt",
        }
    }

    pub fn parse_str(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.as_str() == s)
    }

    pub fn extension(self) -> &'static str {
        self.as_str()
    }

    /// Check that the bytes look like the kind the extension claims (content sniffing, FR-K02).
    pub fn sniff(self, bytes: &[u8]) -> Result<(), String> {
        let zip = bytes.starts_with(b"PK\x03\x04");
        match self {
            Kind::Pdf if !bytes.starts_with(b"%PDF-") => Err("the file is not a PDF".into()),
            Kind::Docx | Kind::Pptx | Kind::Xlsx if !zip => Err(format!(
                "the file is not a {} document",
                self.as_str().to_uppercase()
            )),
            Kind::Tex | Kind::Html | Kind::Md | Kind::Txt | Kind::Csv
                if bytes.starts_with(b"%PDF-") || zip =>
            {
                Err("the file content does not match its extension".into())
            }
            Kind::Tex | Kind::Html | Kind::Md | Kind::Txt | Kind::Csv
                if bytes.iter().take(8192).any(|&b| b == 0) =>
            {
                Err("the file looks binary, not text".into())
            }
            _ => Ok(()),
        }
    }
}

/// A parsed document: Markdown-ish text plus where each page starts (PDF only).
#[derive(Debug, Clone, Default)]
pub struct Parsed {
    pub title: Option<String>,
    pub text: String,
    pub pages: Option<usize>,
    /// Byte offset in `text` where page `i + 1` starts (PDF only; `page_starts[0] == 0`).
    pub page_starts: Vec<usize>,
}

impl Parsed {
    /// The 1-based page holding byte offset `at`, when pages are known.
    pub fn page_at(&self, at: usize) -> Option<usize> {
        if self.page_starts.is_empty() {
            return None;
        }
        Some(self.page_starts.partition_point(|&s| s <= at).max(1))
    }
}

/// Limits for ZIP-based formats (SR-K03).
pub const ZIP_MAX_ENTRIES: usize = 2_000;
pub const ZIP_MAX_TOTAL: u64 = 200 * 1024 * 1024;
pub const ZIP_MAX_RATIO: u64 = 200;

pub fn parse(kind: Kind, bytes: &[u8]) -> Result<Parsed, String> {
    kind.sniff(bytes)?;
    let mut p = match kind {
        // A crafted file can make a reader panic; that is this document's error, not the worker's.
        Kind::Pdf => std::panic::catch_unwind(|| parse_pdf(bytes))
            .map_err(|_| "the PDF could not be read (the reader failed on it)".to_string())??,
        Kind::Docx => parse_docx(bytes)?,
        Kind::Pptx => parse_pptx(bytes)?,
        Kind::Xlsx => parse_xlsx(bytes)?,
        Kind::Csv => parse_csv(&String::from_utf8_lossy(bytes)),
        Kind::Tex => parse_tex(&String::from_utf8_lossy(bytes)),
        Kind::Html => parse_html(&String::from_utf8_lossy(bytes))?,
        Kind::Md | Kind::Txt => {
            let text = String::from_utf8_lossy(bytes).into_owned();
            let title = if kind == Kind::Md {
                first_heading(&text)
            } else {
                None
            };
            Parsed {
                title,
                text,
                ..Default::default()
            }
        }
    };
    if p.page_starts.is_empty() {
        p.text = clean(&p.text);
    }
    if p.text.trim().is_empty() {
        return Err(match kind {
            Kind::Pdf => {
                "no text layer found (a scanned PDF needs OCR, which is not supported)".into()
            }
            _ => "the document contains no text".into(),
        });
    }
    Ok(p)
}

/// The letters a typographic ligature stands for, and a no-break space as a space. A PDF typeset with ligatures
/// extracts "verifier" as `veri` + U+FB01, which no word search matches: on the technical paper's own PDFs 472 of
/// 724 passages carried at least one (found by the live sweep, and the cause of lexical misses on words like
/// "first", "efficient" and "workflow").
fn ligature(c: char) -> impl Iterator<Item = char> {
    let s: &'static str = match c {
        '\u{a0}' | '\u{202f}' => " ",
        '\u{fb00}' => "ff",
        '\u{fb01}' => "fi",
        '\u{fb02}' => "fl",
        '\u{fb03}' => "ffi",
        '\u{fb04}' => "ffl",
        '\u{fb05}' | '\u{fb06}' => "st",
        _ => "",
    };
    // `s` is empty for an ordinary character, which is passed through as itself.
    let (own, rest): (Option<char>, std::str::Chars<'static>) = if s.is_empty() {
        (Some(c), "".chars())
    } else {
        (None, s.chars())
    };
    own.into_iter().chain(rest)
}

/// Normalise extracted text (FR-K14): control characters removed, words hyphenated across line ends joined,
/// trailing spaces trimmed, runs of blank lines collapsed to one.
pub fn clean(text: &str) -> String {
    let text: String = text
        .chars()
        .filter(|&c| c == '\n' || c == '\t' || !c.is_control())
        // Zero-width and soft-hyphen characters carry no text and split a word for a word search.
        .filter(|c| {
            !matches!(
                c,
                '\u{ad}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'
            )
        })
        .flat_map(ligature)
        .collect();
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    let mut lines = text.lines().map(str::trim_end).peekable();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            blank += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        blank = 0;
        // "infor-\nmation" -> "information" when the next line starts with a lower-case letter.
        let joins = line.ends_with('-')
            && line.len() > 1
            && line[..line.len() - 1]
                .chars()
                .last()
                .is_some_and(char::is_alphabetic)
            && lines
                .peek()
                .and_then(|n| n.trim_start().chars().next())
                .is_some_and(char::is_lowercase);
        if joins {
            out.push_str(&line[..line.len() - 1]);
            let next = lines.next().unwrap_or_default().trim_start();
            let (word, rest) = next.split_once(' ').unwrap_or((next, ""));
            out.push_str(word);
            if !rest.is_empty() {
                out.push('\n');
                out.push_str(rest);
            }
        } else {
            out.push_str(line);
        }
    }
    out
}

fn first_heading(md: &str) -> Option<String> {
    md.lines()
        .find_map(|l| l.strip_prefix("# ").map(|t| t.trim().to_string()))
        .filter(|t| !t.is_empty())
}

// ---------------------------------------------------------------- PDF

/// How a PDF page becomes text (Phase 28).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PdfLayout {
    /// Our layout pass over the positioned spans (`crate::layout`): columns, headings, running lines, table rows.
    #[default]
    Spans,
    /// `pdf_oxide`'s own Markdown converter (the reader before Phase 28), kept for comparison.
    Converter,
}

impl PdfLayout {
    /// `NAFS_KB_PDF_LAYOUT=converter` selects the earlier reader (for comparisons); anything else the layout pass.
    pub fn from_env() -> PdfLayout {
        match std::env::var("NAFS_KB_PDF_LAYOUT").as_deref() {
            Ok("converter") => PdfLayout::Converter,
            _ => PdfLayout::Spans,
        }
    }
}

fn pdf_spans(doc: &pdf_oxide::PdfDocument, i: usize) -> Option<crate::layout::Page> {
    let (x0, y0, x1, y1) = doc.get_page_media_box(i).ok()?;
    let spans = doc.extract_spans(i).ok()?;
    let spans = spans
        .into_iter()
        .map(|s| {
            let name = s.font_name.to_ascii_lowercase();
            let bold = s.font_weight.is_bold()
                || ["bold", "black", "heavy", "semibold", "demi", "-medi"]
                    .iter()
                    .any(|k| name.contains(k));
            let italic = s.is_italic || name.contains("ital") || name.contains("oblique");
            crate::layout::Span {
                x0: s.bbox.x - x0,
                x1: s.bbox.x + s.bbox.width - x0,
                y0: s.bbox.y - y0,
                y1: s.bbox.y + s.bbox.height - y0,
                size: s.font_size,
                bold,
                italic,
                text: s.text,
            }
        })
        .collect();
    Some(crate::layout::Page {
        width: (x1 - x0).abs(),
        height: (y1 - y0).abs(),
        spans,
    })
}

fn parse_pdf(bytes: &[u8]) -> Result<Parsed, String> {
    parse_pdf_with(bytes, PdfLayout::from_env())
}

pub fn parse_pdf_with(bytes: &[u8], layout: PdfLayout) -> Result<Parsed, String> {
    let doc = pdf_oxide::PdfDocument::from_bytes(bytes.to_vec())
        .map_err(|e| format!("the PDF could not be read: {e}"))?;
    let n = doc
        .page_count()
        .map_err(|e| format!("the PDF could not be read: {e}"))?;
    let opts = pdf_oxide::converters::ConversionOptions {
        detect_headings: true,
        ..Default::default()
    };
    // One unreadable page must not lose the document: fall back to plain text, then to nothing.
    let converter = |i: usize| {
        doc.to_markdown(i, &opts)
            .or_else(|_| doc.extract_text(i))
            .unwrap_or_default()
    };
    let pages: Vec<String> = match layout {
        PdfLayout::Converter => (0..n).map(converter).collect(),
        PdfLayout::Spans => {
            let spans: Vec<Option<crate::layout::Page>> =
                (0..n).map(|i| pdf_spans(&doc, i)).collect();
            let read: Vec<crate::layout::Page> = spans
                .iter()
                .map(|p| p.clone().unwrap_or_default())
                .collect();
            // A layout rule that fails on an odd page must not lose the document: the converter reads it instead.
            match std::panic::catch_unwind(|| crate::layout::markdown(read)) {
                Ok(md) => md
                    .into_iter()
                    .enumerate()
                    .map(|(i, md)| if spans[i].is_none() { converter(i) } else { md })
                    .collect(),
                Err(_) => {
                    tracing::warn!(
                        "the PDF layout pass failed; using the converter for this document"
                    );
                    (0..n).map(converter).collect()
                }
            }
        }
    };
    let pages: Vec<String> = pages
        .into_iter()
        .map(|p| tidy_pdf_markdown(&clean(&p)))
        .collect();
    let pages = drop_running_lines(pages);
    let mut text = String::new();
    let mut page_starts = Vec::with_capacity(n);
    for page in &pages {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        page_starts.push(text.len());
        text.push_str(page);
    }
    let title = pdf_title(pages.first().map(String::as_str).unwrap_or(""));
    Ok(Parsed {
        title,
        text,
        pages: Some(n),
        page_starts,
    })
}

/// A PDF's title: the first heading on page 1 that reads like one (three or more words, mostly letters). Table
/// headers and single words that the converter marks as headings are not titles; the file name is used instead.
fn pdf_title(page1: &str) -> Option<String> {
    page1
        .lines()
        .filter_map(|l| l.trim().strip_prefix('#'))
        .map(|h| h.trim_start_matches('#').trim().trim_matches('*').trim())
        .find(|h| {
            let letters = h.chars().filter(|c| c.is_alphabetic() || *c == ' ').count();
            h.split_whitespace().count() >= 3
                && h.chars().count() <= 200
                && letters * 10 >= h.chars().count() * 8
        })
        .map(|h| h.replace("**", ""))
}

/// Merge per-word bold runs (`**a** **b**` -> `**a b**`) and mark numbered all-caps section lines (`2.5 MEMORILLA WITH
/// QWEN3`, `B.1 TRAINING DATASETS`) as headings, which the converter's font clustering misses in many papers.
fn tidy_pdf_markdown(page: &str) -> String {
    use std::sync::OnceLock;
    static SECTION: OnceLock<regex::Regex> = OnceLock::new();
    let section = SECTION.get_or_init(|| {
        regex::Regex::new(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}|[A-H](?:\.\d{1,2}){0,2})\s+[A-Z][A-Z0-9 ,:&/\-–()']{2,80}$").expect("valid regex")
    });
    page.replace("** **", " ")
        .lines()
        .map(|l| {
            let t = l.trim();
            let plain = t.trim_matches('*').trim();
            if !t.starts_with('#')
                && section.is_match(plain)
                && !plain.chars().any(|c| c.is_ascii_lowercase())
            {
                let depth = plain
                    .split_whitespace()
                    .next()
                    .map_or(1, |n| n.matches('.').count() + 2)
                    .min(4);
                format!("{} {plain}", "#".repeat(depth))
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove running headers and footers: short lines that recur in the first or last three lines of at least half of the
/// pages (journal names, "Preprint"), and lines that are only a page number.
fn drop_running_lines(pages: Vec<String>) -> Vec<String> {
    use std::collections::HashMap;
    let edge = |p: &str| -> Vec<String> {
        let lines: Vec<&str> = p.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let k = lines.len().min(3);
        lines[..k]
            .iter()
            .chain(lines[lines.len() - k..].iter())
            .map(|l| l.to_string())
            .collect()
    };
    let mut seen: HashMap<String, usize> = HashMap::new();
    for p in &pages {
        let mut e = edge(p);
        e.sort();
        e.dedup();
        for l in e {
            if l.chars().count() <= 60 {
                *seen.entry(l).or_default() += 1;
            }
        }
    }
    let threshold = (pages.len() / 2).max(2);
    let running: std::collections::HashSet<String> = if pages.len() >= 3 {
        seen.into_iter()
            .filter(|(_, c)| *c >= threshold)
            .map(|(l, _)| l)
            .collect()
    } else {
        Default::default()
    };
    pages
        .into_iter()
        .map(|p| {
            let kept: Vec<&str> = p
                .lines()
                .filter(|l| {
                    let t = l.trim();
                    !(running.contains(t)
                        || (!t.is_empty() && t.len() <= 4 && t.chars().all(|c| c.is_ascii_digit())))
                })
                .collect();
            clean(&kept.join("\n"))
        })
        .collect()
}

// ---------------------------------------------------------------- ZIP (DOCX, PPTX)

struct Zip<'a> {
    archive: zip::ZipArchive<Cursor<&'a [u8]>>,
    /// Bytes actually decompressed so far: the headers' sizes can lie, so the sum of what is really read is bounded
    /// too (a deck of a thousand slides that each claim to be small must not expand to a thousand times the cap).
    read_total: u64,
    budget: u64,
}

impl<'a> Zip<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, String> {
        let archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| format!("the archive could not be read: {e}"))?;
        if archive.len() > ZIP_MAX_ENTRIES {
            return Err(format!(
                "the archive has too many parts ({} > {ZIP_MAX_ENTRIES})",
                archive.len()
            ));
        }
        let mut total = 0u64;
        let mut archive = archive;
        for i in 0..archive.len() {
            let f = archive
                .by_index_raw(i)
                .map_err(|e| format!("the archive could not be read: {e}"))?;
            total = total.saturating_add(f.size());
            if f.compressed_size() > 0
                && f.size() / f.compressed_size().max(1) > ZIP_MAX_RATIO
                && f.size() > 1024 * 1024
            {
                return Err(
                    "the archive is compressed suspiciously well (possible zip bomb)".into(),
                );
            }
        }
        if total > ZIP_MAX_TOTAL {
            return Err("the archive expands to more than 200 MB".into());
        }
        Ok(Zip {
            archive,
            read_total: 0,
            budget: ZIP_MAX_TOTAL,
        })
    }

    fn names(&self) -> Vec<String> {
        self.archive
            .file_names()
            .filter_map(|n| n.ok().map(|n| n.into_owned()))
            .collect()
    }

    fn read(&mut self, name: &str) -> Option<String> {
        let left = self.budget.saturating_sub(self.read_total);
        if left == 0 {
            return None;
        }
        let f = self.archive.by_name(name).ok()?;
        let mut s = String::new();
        // `take` bounds the read even if the declared size lies, and `left` bounds all the reads together.
        let n = f.take(left).read_to_string(&mut s).ok()?;
        self.read_total += n as u64;
        Some(s)
    }

    /// Whether every read so far stayed inside the budget (a part that hit the cap was cut, not read whole).
    fn within_budget(&self) -> bool {
        self.read_total < self.budget
    }
}

/// A resolved XML entity or character reference.
fn xml_ref(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        _ => {
            let num = name.strip_prefix('#')?;
            let code = match num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => num.parse().ok()?,
            };
            char::from_u32(code)?
        }
    })
}

fn parse_docx(bytes: &[u8]) -> Result<Parsed, String> {
    let mut zip = Zip::open(bytes)?;
    let xml = zip
        .read("word/document.xml")
        .ok_or("not a Word document (word/document.xml is missing)")?;
    let mut r = Reader::from_str(&xml);
    let mut out = String::new();
    let mut para = String::new();
    let mut level: Option<usize> = None; // heading level of the current paragraph
    let mut in_text = false;
    let mut cell = String::new();
    let mut row: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut table_depth = 0usize;
    let mut title = None;
    // A paragraph with numbering properties is a list item; `w:ilvl` is how deep it sits.
    let mut list: Option<usize> = None;
    loop {
        match r
            .read_event()
            .map_err(|e| format!("the document XML is malformed: {e}"))?
        {
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == "w:pStyle" => {
                if let Ok(Some(a)) = e.try_get_attribute("w:val") {
                    level = heading_level(&a.value);
                }
            }
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == "w:ilvl" => {
                if let Ok(Some(a)) = e.try_get_attribute("w:val") {
                    list = Some(a.value.parse::<usize>().unwrap_or(0).min(8));
                }
            }
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == "w:numPr" => {
                list.get_or_insert(0);
            }
            Event::Start(e) => match e.name().as_ref() {
                "w:t" => in_text = true,
                "w:p" => {
                    para.clear();
                    level = None;
                    list = None;
                }
                "w:tbl" => {
                    table_depth += 1;
                    if table_depth == 1 {
                        rows.clear();
                    }
                }
                "w:tr" => row.clear(),
                "w:tc" => cell.clear(),
                _ => {}
            },
            Event::Empty(e) => match e.name().as_ref() {
                "w:tab" => para.push('\t'),
                "w:br" | "w:cr" => para.push('\n'),
                _ => {}
            },
            Event::Text(t) if in_text => para.push_str(&t.xml10_content()),
            Event::GeneralRef(g) if in_text => para.extend(xml_ref(&g.into_inner())),
            Event::End(e) => match e.name().as_ref() {
                "w:t" => in_text = false,
                "w:p" => {
                    let text = para.trim();
                    if table_depth > 0 {
                        if !cell.is_empty() && !text.is_empty() {
                            cell.push(' ');
                        }
                        cell.push_str(text);
                    } else if !text.is_empty() {
                        match level {
                            Some(0) => {
                                title.get_or_insert_with(|| text.to_string());
                                out.push_str(&format!("# {text}\n\n"));
                            }
                            Some(n) => {
                                out.push_str(&format!("{} {text}\n\n", "#".repeat(n.min(6))))
                            }
                            None => match list {
                                // List items sit on consecutive lines; the blank line comes with the next
                                // paragraph that is not one.
                                Some(depth) => {
                                    if out.ends_with("\n\n") && last_was_list(&out) {
                                        out.pop();
                                    }
                                    out.push_str(&format!("{}- {text}\n\n", "  ".repeat(depth)));
                                }
                                None => {
                                    out.push_str(text);
                                    out.push_str("\n\n");
                                }
                            },
                        }
                    }
                }
                "w:tc" => row.push(std::mem::take(&mut cell)),
                "w:tr" => {
                    if row.iter().any(|c| !c.is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    }
                }
                "w:tbl" => {
                    table_depth = table_depth.saturating_sub(1);
                    // A table inside a cell is folded into the outer one's rows; the table is written when the
                    // outermost one ends.
                    if table_depth == 0 {
                        md_table(&mut out, None, &rows);
                        rows.clear();
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }
    if title.is_none() {
        title = docx_core_title(&mut zip);
    }
    if !zip.within_budget() {
        return Err("the archive expands to more than 200 MB".into());
    }
    Ok(Parsed {
        title,
        text: out,
        ..Default::default()
    })
}

/// Whether the text written so far ends with a list item (so the next one follows it without a blank line).
fn last_was_list(out: &str) -> bool {
    out.trim_end_matches('\n')
        .rsplit('\n')
        .next()
        .is_some_and(|l| l.trim_start().starts_with("- "))
}

// ---------------------------------------------------------------- tables, spreadsheets, CSV (Phase 35)

/// The most rows and columns of one sheet or CSV file that are kept; what is cut is said in the text.
pub const TABLE_MAX_ROWS: usize = 5_000;
pub const TABLE_MAX_COLS: usize = 64;
/// Rows per Markdown table. A long sheet is written as several tables, each with the header row and a heading that
/// says which rows it holds, so that a passage cut from the middle of a sheet still says what its columns are.
const TABLE_GROUP: usize = 40;

/// The most characters of one cell that are kept, and the most cell text of one workbook. A cell that refers to a
/// shared string costs a few bytes in the file and the whole string in memory, so a small file could otherwise ask
/// for any amount of it.
pub const TABLE_MAX_CELL: usize = 2_000;
const WORKBOOK_MAX_BYTES: usize = 16 << 20;

/// A cell's text, cut to [`TABLE_MAX_CELL`] characters (with an ellipsis when something was cut).
fn clip_cell(s: &str) -> String {
    match s.char_indices().nth(TABLE_MAX_CELL) {
        Some((at, _)) => format!("{}…", &s[..at]),
        None => s.to_string(),
    }
}

fn cell_md(s: &str) -> String {
    clip_cell(&s.split_whitespace().collect::<Vec<_>>().join(" ")).replace('|', "\\|")
}

/// Write `rows` (the first is the header) as Markdown tables under `name`.
fn md_table(out: &mut String, name: Option<&str>, rows: &[Vec<String>]) {
    let rows: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r.iter().any(|c| !c.trim().is_empty()))
        .collect();
    let width = rows
        .iter()
        .map(|r| {
            r.iter()
                .rposition(|c| !c.trim().is_empty())
                .map_or(0, |i| i + 1)
        })
        .max()
        .unwrap_or(0)
        .min(TABLE_MAX_COLS);
    if width == 0 {
        return;
    }
    let line = |r: &Vec<String>| {
        let cells: Vec<String> = (0..width)
            .map(|k| r.get(k).map(|c| cell_md(c)).unwrap_or_default())
            .collect();
        format!("| {} |\n", cells.join(" | "))
    };
    let head = line(rows[0]);
    let rule = format!("|{}\n", " --- |".repeat(width));
    let body = &rows[1..];
    if body.is_empty() {
        // A single row has no header to speak of: it is written as a one-row table all the same.
        out.push_str(&head);
        out.push_str(&rule);
        out.push('\n');
        return;
    }
    for (g, group) in body.chunks(TABLE_GROUP).enumerate() {
        if g > 0 {
            let (a, b) = (g * TABLE_GROUP + 2, g * TABLE_GROUP + 1 + group.len());
            match name {
                Some(n) => out.push_str(&format!("### {n}, rows {a} to {b}\n\n")),
                None => out.push_str(&format!("Rows {a} to {b}:\n\n")),
            }
        }
        out.push_str(&head);
        out.push_str(&rule);
        for r in group {
            out.push_str(&line(r));
        }
        out.push('\n');
    }
}

/// The 0-based column of a cell reference such as `B2` or `AA10`.
fn column_of(cell_ref: &str) -> Option<usize> {
    let mut n = 0usize;
    let mut seen = false;
    for c in cell_ref.chars() {
        if c.is_ascii_alphabetic() {
            // A reference with more letters than a column can have is not a column.
            n = n
                .checked_mul(26)?
                .checked_add(c.to_ascii_uppercase() as usize - 'A' as usize + 1)?;
            seen = true;
        } else {
            break;
        }
    }
    seen.then(|| n - 1)
}

/// An Excel workbook: a heading per sheet, then its cells as Markdown tables (first row = header). Values are the
/// ones the file holds: a formula shows its last computed value (or the formula itself when none was stored), and
/// dates stay the numbers Excel stores. Charts, images, comments and formatting are not read.
fn parse_xlsx(bytes: &[u8]) -> Result<Parsed, String> {
    let mut zip = Zip::open(bytes)?;
    let workbook = zip
        .read("xl/workbook.xml")
        .ok_or("not an Excel workbook (xl/workbook.xml is missing)")?;
    // Sheet names in workbook order, each with the relationship id that names its part.
    let mut sheets: Vec<(String, String)> = Vec::new();
    let mut r = Reader::from_str(&workbook);
    loop {
        match r.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.name().as_ref() == "sheet" => {
                let name = e
                    .try_get_attribute("name")
                    .ok()
                    .flatten()
                    .map(|a| html_unescape(&a.value))
                    .unwrap_or_default();
                let rid = e
                    .try_get_attribute("r:id")
                    .ok()
                    .flatten()
                    .map(|a| a.value.to_string())
                    .unwrap_or_default();
                sheets.push((name, rid));
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    // Relationship id -> part name.
    let mut targets: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    if let Some(rels) = zip.read("xl/_rels/workbook.xml.rels") {
        let mut r = Reader::from_str(&rels);
        loop {
            match r.read_event() {
                Ok(Event::Start(e)) | Ok(Event::Empty(e))
                    if e.name().as_ref() == "Relationship" =>
                {
                    let get = |k: &str| {
                        e.try_get_attribute(k)
                            .ok()
                            .flatten()
                            .map(|a| a.value.to_string())
                    };
                    if let (Some(id), Some(t)) = (get("Id"), get("Target")) {
                        let part = match t.strip_prefix('/') {
                            Some(abs) => abs.to_string(),
                            None => format!("xl/{t}"),
                        };
                        targets.insert(id, part);
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
        }
    }
    // Shared strings: the text of each `si`, its runs joined, phonetic guides left out.
    let mut shared: Vec<String> = Vec::new();
    if let Some(xml) = zip.read("xl/sharedStrings.xml") {
        let mut r = Reader::from_str(&xml);
        let (mut cur, mut in_t, mut skip) = (String::new(), false, 0usize);
        loop {
            match r.read_event() {
                Ok(Event::Start(e)) => match e.name().as_ref() {
                    "si" => cur.clear(),
                    "rPh" => skip += 1,
                    "t" => in_t = skip == 0,
                    _ => {}
                },
                Ok(Event::Text(t)) if in_t => cur.push_str(&t.xml10_content()),
                Ok(Event::GeneralRef(g)) if in_t => cur.extend(xml_ref(&g.into_inner())),
                Ok(Event::End(e)) => match e.name().as_ref() {
                    "t" => in_t = false,
                    "rPh" => skip = skip.saturating_sub(1),
                    "si" => shared.push(std::mem::take(&mut cur)),
                    _ => {}
                },
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
        }
    }
    let mut out = String::new();
    // Cell text still allowed for the whole workbook; a sheet that uses it up is cut there and says so.
    let mut budget = WORKBOOK_MAX_BYTES;
    for (i, (name, rid)) in sheets.iter().enumerate() {
        let part = targets
            .get(rid)
            .cloned()
            .unwrap_or_else(|| format!("xl/worksheets/sheet{}.xml", i + 1));
        let Some(xml) = zip.read(&part) else { continue };
        let (rows, total) = sheet_rows(&xml, &shared, &mut budget);
        let name = if name.trim().is_empty() {
            format!("Sheet {}", i + 1)
        } else {
            name.trim().to_string()
        };
        out.push_str(&format!("## {name}\n\n"));
        if rows.is_empty() {
            out.push_str("(This sheet is empty.)\n\n");
            continue;
        }
        md_table(&mut out, Some(&name), &rows);
        if total > rows.len() {
            out.push_str(&format!(
                "(This sheet has {total} rows; the first {} are kept.)\n\n",
                rows.len()
            ));
        }
        if budget == 0 {
            out.push_str(
                "(The workbook holds more text than is read; what follows was left out.)\n\n",
            );
            break;
        }
    }
    if !zip.within_budget() {
        return Err("the archive expands to more than 200 MB".into());
    }
    if sheets.is_empty() {
        return Err("the workbook has no sheets".into());
    }
    Ok(Parsed {
        title: docx_core_title(&mut zip),
        text: out,
        ..Default::default()
    })
}

/// The rows of one worksheet part (at most [`TABLE_MAX_ROWS`] of them, each at most [`TABLE_MAX_COLS`] wide), and
/// how many rows with content the sheet has in all. `budget` is the cell text the workbook may still keep: each kept
/// cell is cut to [`TABLE_MAX_CELL`] characters and counted, and no row is kept once it is spent.
fn sheet_rows(xml: &str, shared: &[String], budget: &mut usize) -> (Vec<Vec<String>>, usize) {
    let mut r = Reader::from_str(xml);
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut total = 0usize;
    let mut row: Vec<String> = Vec::new();
    // The cell being read: its column, its type, its value and its formula.
    let (mut col, mut next_col) = (0usize, 0usize);
    let (mut kind, mut value, mut formula) = (String::new(), String::new(), String::new());
    let mut field = 0u8; // 1 = inside <v>, 2 = inside <t> of an inline string, 3 = inside <f>
    loop {
        match r.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.name().as_ref() == "c" => {
                col = e
                    .try_get_attribute("r")
                    .ok()
                    .flatten()
                    .and_then(|a| column_of(&a.value))
                    .unwrap_or(next_col);
                next_col = col + 1;
                kind = e
                    .try_get_attribute("t")
                    .ok()
                    .flatten()
                    .map(|a| a.value.to_string())
                    .unwrap_or_default();
                value.clear();
                formula.clear();
            }
            Ok(Event::Start(e)) => match e.name().as_ref() {
                "row" => {
                    row.clear();
                    next_col = 0;
                }
                "v" => field = 1,
                "t" => field = 2,
                "f" => field = 3,
                _ => {}
            },
            Ok(Event::Text(t)) if field != 0 => {
                let text = t.xml10_content();
                if field == 3 {
                    formula.push_str(&text)
                } else {
                    value.push_str(&text)
                }
            }
            Ok(Event::GeneralRef(g)) if field != 0 => {
                let c = xml_ref(&g.into_inner());
                if field == 3 {
                    formula.extend(c)
                } else {
                    value.extend(c)
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                "v" | "t" | "f" => field = 0,
                "c" => {
                    let text = match kind.as_str() {
                        "s" => value
                            .trim()
                            .parse::<usize>()
                            .ok()
                            .and_then(|i| shared.get(i))
                            .map(|s| clip_cell(s))
                            .unwrap_or_default(),
                        "b" => (if value.trim() == "1" { "TRUE" } else { "FALSE" }).to_string(),
                        _ if value.is_empty() && !formula.is_empty() => {
                            clip_cell(&format!("={formula}"))
                        }
                        _ => clip_cell(&value),
                    };
                    if col < TABLE_MAX_COLS && !text.trim().is_empty() && *budget > 0 {
                        *budget = budget.saturating_sub(text.len());
                        if row.len() <= col {
                            row.resize(col + 1, String::new());
                        }
                        row[col] = text;
                    }
                }
                "row" if row.iter().any(|c| !c.trim().is_empty()) => {
                    total += 1;
                    if rows.len() < TABLE_MAX_ROWS && *budget > 0 {
                        rows.push(std::mem::take(&mut row));
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    (rows, total)
}

/// Comma-, semicolon- or tab-separated values: the delimiter is the one the first lines use most; quoted fields may
/// hold the delimiter, line breaks and doubled quotes. The first row is the header.
pub fn parse_csv(src: &str) -> Parsed {
    let src = src.strip_prefix('\u{feff}').unwrap_or(src);
    let head: String = src.lines().take(10).collect::<Vec<_>>().join("\n");
    let delim = [',', ';', '\t', '|']
        .into_iter()
        .max_by_key(|d| (head.matches(*d).count(), *d == ','))
        .unwrap_or(',');
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut total = 0usize;
    let (mut row, mut field, mut quoted, mut any) = (Vec::new(), String::new(), false, false);
    let mut chars = src.chars().peekable();
    let mut end_row = |row: &mut Vec<String>, field: &mut String, any: &mut bool| {
        row.push(std::mem::take(field));
        if row.iter().any(|c: &String| !c.trim().is_empty()) {
            total += 1;
            if rows.len() < TABLE_MAX_ROWS {
                rows.push(std::mem::take(row));
            }
        }
        row.clear();
        *any = false;
    };
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' && field.is_empty() {
            quoted = true;
            any = true;
        } else if c == delim {
            row.push(std::mem::take(&mut field));
            any = true;
        } else if c == '\n' {
            end_row(&mut row, &mut field, &mut any);
        } else if c != '\r' {
            field.push(c);
            any = true;
        }
    }
    if any || !field.is_empty() || !row.is_empty() {
        end_row(&mut row, &mut field, &mut any);
    }
    let mut out = String::new();
    md_table(&mut out, None, &rows);
    if total > rows.len() {
        out.push_str(&format!(
            "(The file has {total} rows; the first {} are kept.)\n",
            rows.len()
        ));
    }
    Parsed {
        text: out,
        ..Default::default()
    }
}

/// `Title` -> 0, `Heading1`..`Heading9` (and localised `heading 1`) -> 1..9.
fn heading_level(style: &str) -> Option<usize> {
    let s = style.to_ascii_lowercase().replace(' ', "");
    if s == "title" {
        return Some(0);
    }
    s.strip_prefix("heading")
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|n| (1..=9).contains(n))
}

fn docx_core_title(zip: &mut Zip) -> Option<String> {
    let xml = zip.read("docProps/core.xml")?;
    let start = xml.find("<dc:title>")? + "<dc:title>".len();
    let end = xml[start..].find("</dc:title>")? + start;
    let t = xml[start..end].trim();
    (!t.is_empty()).then(|| t.to_string())
}

fn parse_pptx(bytes: &[u8]) -> Result<Parsed, String> {
    let mut zip = Zip::open(bytes)?;
    if !zip.names().iter().any(|n| n == "ppt/presentation.xml") {
        return Err("not a PowerPoint presentation (ppt/presentation.xml is missing)".into());
    }
    let mut slides: Vec<(usize, String)> = zip
        .names()
        .into_iter()
        .filter_map(|n| {
            let num = n
                .strip_prefix("ppt/slides/slide")?
                .strip_suffix(".xml")?
                .parse()
                .ok()?;
            Some((num, n))
        })
        .collect();
    slides.sort();
    let mut out = String::new();
    let mut title = None;
    for (num, name) in slides {
        let Some(xml) = zip.read(&name) else { continue };
        let paras = drawing_paragraphs(&xml);
        let notes = zip
            .read(&format!("ppt/notesSlides/notesSlide{num}.xml"))
            .map(|x| drawing_paragraphs(&x))
            .unwrap_or_default();
        let heading = paras.first().cloned().unwrap_or_default();
        if num == 1 && !heading.is_empty() {
            title = Some(heading.clone());
        }
        out.push_str(&format!("# Slide {num}: {heading}\n\n"));
        for p in paras.iter().skip(1) {
            out.push_str(p);
            out.push('\n');
        }
        // Notes repeat the slide number and placeholder text; keep only the speaker's sentences.
        let notes: Vec<&String> = notes
            .iter()
            .filter(|p| p.parse::<usize>().is_err() && **p != heading)
            .collect();
        if !notes.is_empty() {
            out.push_str("\nNotes: ");
            out.push_str(
                &notes
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            out.push('\n');
        }
        out.push('\n');
    }
    if !zip.within_budget() {
        return Err("the archive expands to more than 200 MB".into());
    }
    Ok(Parsed {
        title,
        text: out,
        ..Default::default()
    })
}

/// The text paragraphs (`a:p`) of a DrawingML part, in order, empty ones dropped.
fn drawing_paragraphs(xml: &str) -> Vec<String> {
    let mut r = Reader::from_str(xml);
    let mut out = Vec::new();
    let mut para = String::new();
    let mut in_text = false;
    loop {
        match r.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == "a:t" => in_text = true,
            Ok(Event::Start(e)) if e.name().as_ref() == "a:p" => para.clear(),
            Ok(Event::Empty(e)) if e.name().as_ref() == "a:br" => para.push(' '),
            Ok(Event::Text(t)) if in_text => para.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(g)) if in_text => para.extend(xml_ref(&g.into_inner())),
            Ok(Event::End(e)) if e.name().as_ref() == "a:t" => in_text = false,
            Ok(Event::End(e)) if e.name().as_ref() == "a:p" => {
                let t = para.split_whitespace().collect::<Vec<_>>().join(" ");
                if !t.is_empty() {
                    out.push(t);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------- HTML

fn parse_html(html: &str) -> Result<Parsed, String> {
    let title = regex::Regex::new(r"(?is)<title[^>]*>(.*?)</title>")
        .ok()
        .and_then(|re| re.captures(html))
        .map(|c| html_unescape(c[1].trim()))
        .filter(|t| !t.is_empty());
    let conv = htmd::HtmlToMarkdown::builder()
        .skip_tags(vec![
            "script", "style", "noscript", "nav", "footer", "header", "aside", "head", "img",
            "svg", "form", "iframe",
        ])
        .build();
    let md = conv
        .convert(html)
        .map_err(|e| format!("the HTML could not be read: {e}"))?;
    let md = drop_unsafe_links(&md);
    let title = title.or_else(|| first_heading(&md));
    Ok(Parsed {
        title,
        text: md,
        ..Default::default()
    })
}

/// A Markdown link whose target runs script (`javascript:`, `vbscript:`, `data:`) keeps its words and loses the
/// target: stored text may be shown by a viewer that renders links, and a passage is only ever data.
fn drop_unsafe_links(md: &str) -> String {
    static LINK: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = LINK.get_or_init(|| {
        regex::Regex::new(
            r"(?is)\[([^\]]*)\]\(\s*<?\s*(?:javascript|vbscript|data)\s*:(?:[^()]|\([^()]*\))*\)",
        )
        .expect("a fixed pattern")
    });
    re.replace_all(md, "$1").into_owned()
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

// ---------------------------------------------------------------- LaTeX

/// A small LaTeX-to-Markdown stripper (FR-K13). It keeps the prose and the maths, not the layout.
pub fn parse_tex(src: &str) -> Parsed {
    // Comments: an unescaped % to the end of the line.
    let mut s = String::with_capacity(src.len());
    for line in src.lines() {
        let mut prev = '\0';
        let mut cut = line.len();
        for (i, c) in line.char_indices() {
            if c == '%' && prev != '\\' {
                cut = i;
                break;
            }
            prev = c;
        }
        s.push_str(&line[..cut]);
        s.push('\n');
    }
    let s = expand_macros(&s);
    let title = braced_arg(&s, "\\title").map(|t| strip_inline(&t));
    let body = match (s.find("\\begin{document}"), s.find("\\end{document}")) {
        (Some(b), Some(e)) if e > b => s[b + "\\begin{document}".len()..e].to_string(),
        (Some(b), None) => s[b + "\\begin{document}".len()..].to_string(),
        _ => s,
    };
    let mut out = String::with_capacity(body.len());
    let sections = [
        ("\\part", 1),
        ("\\chapter", 1),
        ("\\section", 2),
        ("\\subsection", 3),
        ("\\subsubsection", 4),
        ("\\paragraph", 5),
        ("\\subparagraph", 6),
    ];
    let drop_envs = [
        "tikzpicture",
        "thebibliography",
        "lstlisting",
        "verbatim",
        "minted",
    ];
    let mut skip_until: Option<String> = None;
    if let Some(t) = &title {
        out.push_str(&format!("# {t}\n\n"));
    }
    for line in body.lines() {
        let l = line.trim();
        if let Some(end) = &skip_until {
            if l.contains(end.as_str()) {
                skip_until = None;
            }
            continue;
        }
        if let Some(env) = drop_envs
            .iter()
            .find(|e| l.starts_with(&format!("\\begin{{{e}}}")))
        {
            skip_until = Some(format!("\\end{{{env}}}"));
            continue;
        }
        if l == "\\maketitle"
            || l.starts_with("\\label{")
            || l.starts_with("\\bibliography")
            || l.starts_with("\\bibliographystyle")
        {
            continue;
        }
        if let Some((cmd, level)) = sections.iter().find(|(c, _)| {
            l.starts_with(&format!("{c}{{"))
                || l.starts_with(&format!("{c}*{{"))
                || l.starts_with(&format!("{c}["))
        }) {
            if let Some(arg) = braced_arg(l, cmd) {
                out.push_str(&format!(
                    "\n{} {}\n\n",
                    "#".repeat(*level),
                    strip_inline(&arg)
                ));
                continue;
            }
        }
        if l.starts_with("\\begin{abstract}") {
            out.push_str("\n## Abstract\n\n");
            continue;
        }
        if l.starts_with("\\item") {
            out.push_str("- ");
            out.push_str(
                strip_inline(
                    l.trim_start_matches("\\item")
                        .trim_start_matches(['[', ']']),
                )
                .trim(),
            );
            out.push('\n');
            continue;
        }
        if l.contains("\\caption") {
            if let Some(arg) = braced_arg(l, "\\caption") {
                out.push_str(&format!("Caption: {}\n", strip_inline(&arg)));
            }
            continue;
        }
        if l.starts_with("\\begin{") || l.starts_with("\\end{") {
            // Keep display maths environments as $$ blocks; drop other environment markers.
            if ["equation", "align", "gather", "multline"]
                .iter()
                .any(|m| l.contains(&format!("{{{m}")))
            {
                out.push_str("$$\n");
            }
            continue;
        }
        out.push_str(&strip_inline(line));
        out.push('\n');
    }
    Parsed {
        title,
        text: out,
        ..Default::default()
    }
}

/// Expand argument-free macros the file defines itself (`\\newcommand{\\X}{text}`, `\\newcommand\\X{text}`,
/// `\\def\\X{text}`), so `\\Sone` reads as what it stands for. Macros defined in other files stay as their names.
fn expand_macros(src: &str) -> String {
    use std::sync::OnceLock;
    static DEF: OnceLock<regex::Regex> = OnceLock::new();
    let def = DEF.get_or_init(|| {
        regex::Regex::new(
            r"\\(?:(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?|def\\([A-Za-z]+))\s*\{([^{}]*)\}",
        )
        .expect("valid regex")
    });
    let defs: Vec<(String, String)> = def
        .captures_iter(src)
        .filter_map(|c| {
            Some((
                c.get(1).or(c.get(2))?.as_str().to_string(),
                c[3].to_string(),
            ))
        })
        .filter(|(_, v)| !v.contains('#'))
        .collect();
    if defs.is_empty() {
        return src.to_string();
    }
    let body = def.replace_all(src, "");
    let mut out = body.into_owned();
    for (name, value) in defs.iter().rev() {
        let value = value.replace('~', " ");
        // `\X{}` and `\X\ ` keep the space that follows; a bare `\X` must end at a word boundary.
        let re = regex::Regex::new(&format!(r"\\{name}(\{{\}}|\\ |\b|$)")).expect("valid regex");
        out = re
            .replace_all(&out, |c: &regex::Captures| {
                if &c[1] == "\\ " {
                    format!("{value} ")
                } else {
                    value.clone()
                }
            })
            .into_owned();
    }
    out
}

/// The `{...}` argument after `cmd` (optional `*` and `[...]` skipped), braces balanced.
fn braced_arg(s: &str, cmd: &str) -> Option<String> {
    let start = s.find(cmd)? + cmd.len();
    let rest = s[start..].trim_start_matches('*');
    let rest = if rest.starts_with('[') {
        &rest[rest.find(']')? + 1..]
    } else {
        rest
    };
    let rest = rest.strip_prefix('{')?;
    let mut depth = 1;
    for (i, c) in rest.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Unwrap formatting commands, drop references, keep maths and text.
fn strip_inline(s: &str) -> String {
    use std::sync::OnceLock;
    static RES: OnceLock<Vec<(regex::Regex, &'static str)>> = OnceLock::new();
    let res = RES.get_or_init(|| {
        [
            (r"[ ~]*\\(?:cite|citep|citet|ref|eqref|autoref|cref|Cref|label|footnotemark|hspace|vspace|includegraphics)\*?(?:\[[^\]]*\])*\{[^}]*\}", ""),
            (r"\\(?:emph|textbf|textit|texttt|textsc|underline|mbox|text|textrm|textsf|footnote|url|href\{[^}]*\})\{([^{}]*)\}", "$1"),
            (r"\\(?:noindent|centering|newline|medskip|bigskip|smallskip|par|clearpage|newpage|small|large|Large|footnotesize|normalsize)\b", ""),
            (r"~", " "),
            (r"\\\\", "\n"),
            (r"\\([%&#_$])", "$1"),
            (r"``|''", "\""),
        ]
        .into_iter()
        .map(|(p, r)| (regex::Regex::new(p).expect("valid regex"), r))
        .collect()
    });
    let mut out = s.to_string();
    // Nested formatting needs more than one pass.
    for _ in 0..3 {
        let before = out.clone();
        for (re, rep) in res {
            out = re.replace_all(&out, *rep).into_owned();
        }
        if out == before {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PDF's typographic ligatures, soft hyphens and zero-width characters become the plain letters, so the words
    /// of a PDF are found by a word search.
    #[test]
    fn ligatures_and_invisible_characters_become_plain_text() {
        assert_eq!(
            clean("Veri\u{fb01}er-Gated, e\u{fb03}cient work\u{fb02}ow, o\u{fb00}er"),
            "Verifier-Gated, efficient workflow, offer"
        );
        assert_eq!(
            clean("sof\u{ad}tware\u{200b} for\u{a0}all"),
            "software for all"
        );
        assert_eq!(clean("\u{fb06}ill and \u{fb05}ill"), "still and still");
        assert_eq!(
            clean("plain text, with accents: caf\u{e9}"),
            "plain text, with accents: caf\u{e9}"
        );
    }
    use std::io::Write;

    pub(crate) fn zip_of(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, body) in parts {
                w.start_file(*name, opts).unwrap();
                w.write_all(body.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn kinds_from_names_and_sniffing() {
        assert_eq!(Kind::from_filename("A.PDF"), Some(Kind::Pdf));
        assert_eq!(Kind::from_filename("notes.markdown"), Some(Kind::Md));
        assert_eq!(Kind::from_filename("x.exe"), None);
        assert_eq!(Kind::from_filename("noext"), None);
        assert!(Kind::Pdf.sniff(b"%PDF-1.7 ...").is_ok());
        assert!(Kind::Pdf.sniff(b"hello").is_err());
        assert!(Kind::Docx.sniff(b"%PDF-").is_err());
        assert!(Kind::Txt.sniff(b"PK\x03\x04rest").is_err());
        assert!(Kind::Txt.sniff(b"ab\0cd").is_err());
        assert!(Kind::Md.sniff("# Héllo".as_bytes()).is_ok());
    }

    #[test]
    fn clean_joins_hyphens_and_collapses_blank_lines() {
        let t = clean("The infor-\nmation is here\n\n\n\nNext\u{0007} para  \nend");
        assert_eq!(t, "The information\nis here\n\nNext para\nend");
        // A dash before an upper-case word is kept.
        assert_eq!(clean("Smith-\nJones"), "Smith-\nJones");
    }

    #[test]
    fn docx_headings_paragraphs_tables_and_entities() {
        let doc = r#"<?xml version="1.0"?><w:document xmlns:w="x"><w:body>
<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Memory &amp; Retrieval</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Sixteen tokens </w:t></w:r><w:r><w:t>carry the store.</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>K</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>16</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"#;
        let bytes = zip_of(&[
            ("word/document.xml", doc),
            ("[Content_Types].xml", "<Types/>"),
        ]);
        let p = parse(Kind::Docx, &bytes).unwrap();
        assert_eq!(p.title.as_deref(), Some("Memory & Retrieval"));
        assert!(p.text.contains("# Memory & Retrieval"), "{}", p.text);
        assert!(p.text.contains("# Intro"), "{}", p.text);
        assert!(
            p.text.contains("Sixteen tokens carry the store."),
            "{}",
            p.text
        );
        assert!(p.text.contains("| K | 16 |\n| --- | --- |"), "{}", p.text);
    }

    #[test]
    fn docx_lists_and_a_table_with_a_header_row() {
        let item = |text: &str, lvl: usize| {
            format!(
                r#"<w:p><w:pPr><w:numPr><w:ilvl w:val="{lvl}"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#
            )
        };
        let cell = |t: &str| format!("<w:tc><w:p><w:r><w:t>{t}</w:t></w:r></w:p></w:tc>");
        let doc = format!(
            r#"<?xml version="1.0"?><w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Steps:</w:t></w:r></w:p>{}{}{}<w:p><w:r><w:t>After the list.</w:t></w:r></w:p>
<w:tbl><w:tr>{}{}</w:tr><w:tr>{}{}</w:tr><w:tr>{}{}</w:tr></w:tbl></w:body></w:document>"#,
            item("Unpack", 0),
            item("Check the seal", 1),
            item("Plug in", 0),
            cell("Part"),
            cell("Count"),
            cell("Bolt | M6"),
            cell("12"),
            cell("Nut"),
            cell("12"),
        );
        let p = parse(Kind::Docx, &zip_of(&[("word/document.xml", &doc)])).unwrap();
        assert!(
            p.text
                .contains("Steps:\n\n- Unpack\n  - Check the seal\n- Plug in\n\nAfter the list."),
            "{}",
            p.text
        );
        assert!(
            p.text
                .contains("| Part | Count |\n| --- | --- |\n| Bolt \\| M6 | 12 |\n| Nut | 12 |"),
            "{}",
            p.text
        );
    }

    fn xlsx_of(sheets: &[(&str, &str)], shared: &[&str]) -> Vec<u8> {
        let wb = format!(
            r#"<workbook xmlns:r="r"><sheets>{}</sheets></workbook>"#,
            sheets
                .iter()
                .enumerate()
                .map(|(i, (n, _))| format!(
                    r#"<sheet name="{n}" sheetId="{0}" r:id="rId{0}"/>"#,
                    i + 1
                ))
                .collect::<String>()
        );
        let rels = format!(
            "<Relationships>{}</Relationships>",
            (1..=sheets.len())
                .map(|i| format!(r#"<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>"#))
                .collect::<String>()
        );
        let sst = format!(
            "<sst>{}</sst>",
            shared
                .iter()
                .map(|s| format!("<si><t>{s}</t></si>"))
                .collect::<String>()
        );
        let mut parts: Vec<(String, String)> = vec![
            ("xl/workbook.xml".into(), wb),
            ("xl/_rels/workbook.xml.rels".into(), rels),
            ("xl/sharedStrings.xml".into(), sst),
        ];
        for (i, (_, xml)) in sheets.iter().enumerate() {
            parts.push((format!("xl/worksheets/sheet{}.xml", i + 1), xml.to_string()));
        }
        let refs: Vec<(&str, &str)> = parts
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        zip_of(&refs)
    }

    /// A cell that refers to a shared string costs a few bytes in the file. One long string referred to by every
    /// cell of a full sheet must not be held once per cell.
    #[test]
    fn a_shared_string_referred_to_everywhere_is_bounded() {
        let long = "x".repeat(200_000);
        let mut xml = String::from("<worksheet><sheetData>");
        for _ in 0..TABLE_MAX_ROWS {
            xml.push_str("<row>");
            for _ in 0..TABLE_MAX_COLS {
                xml.push_str(r#"<c t="s"><v>0</v></c>"#);
            }
            xml.push_str("</row>");
        }
        xml.push_str("</sheetData></worksheet>");
        let p = parse(Kind::Xlsx, &xlsx_of(&[("Big", &xml)], &[&long])).unwrap();
        // 16 MB of cell text at most, written once, plus the header row repeated per group of rows.
        assert!(p.text.len() < 40 << 20, "{} bytes", p.text.len());
        assert!(!p.text.contains(&"x".repeat(TABLE_MAX_CELL + 1)));
        assert!(p.text.contains("holds more text than is read"));
        // A column reference with far too many letters is not a column, and is not an overflow.
        assert_eq!(column_of(&"Z".repeat(40)), None);
        assert_eq!(column_of("AA10"), Some(26));
    }

    #[test]
    fn xlsx_sheets_become_tables_with_their_names() {
        let sheet1 = r#"<worksheet><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>12</v></c><c r="C2"><f>B2*1.5</f><v>18</v></c></row>
<row r="3"><c r="A3" t="inlineStr"><is><t>Nut &amp; washer</t></is></c><c r="C3" t="b"><v>1</v></c></row>
<row r="4"></row>
<row r="5"><c r="A5" t="str"><f>A2&amp;"s"</f><v>Bolts</v></c><c r="B5"><f>SUM(B2:B3)</f></c></row>
</sheetData></worksheet>"#;
        let sheet2 = "<worksheet><sheetData/></worksheet>";
        let bytes = xlsx_of(
            &[("Parts", sheet1), ("Notes", sheet2)],
            &["Part", "Count", "Cost", "Bolt"],
        );
        assert_eq!(Kind::from_filename("Stock.XLSX"), Some(Kind::Xlsx));
        let p = parse(Kind::Xlsx, &bytes).unwrap();
        assert!(p.text.contains("## Parts\n\n| Part | Count | Cost |\n| --- | --- | --- |\n| Bolt | 12 | 18 |\n| Nut & washer |  | TRUE |\n| Bolts | =SUM(B2:B3) |  |"), "{}", p.text);
        assert!(
            p.text.contains("## Notes\n\n(This sheet is empty.)"),
            "{}",
            p.text
        );
        // Not a workbook, and a text file named .xlsx.
        assert!(parse(Kind::Xlsx, &zip_of(&[("word/document.xml", "<x/>")])).is_err());
        assert!(parse(Kind::Xlsx, b"a,b\n1,2\n").is_err());
    }

    #[test]
    fn a_long_sheet_repeats_its_header_and_says_what_was_cut() {
        let mut xml = String::from(
            r#"<worksheet><sheetData><row><c t="inlineStr"><is><t>n</t></is></c><c t="inlineStr"><is><t>square</t></is></c></row>"#,
        );
        let n = TABLE_MAX_ROWS + 25;
        for i in 1..=n {
            xml.push_str(&format!(
                "<row><c><v>{i}</v></c><c><v>{}</v></c></row>",
                i * i
            ));
        }
        xml.push_str("</sheetData></worksheet>");
        let p = parse(Kind::Xlsx, &xlsx_of(&[("Squares", &xml)], &[])).unwrap();
        // The header comes back with every group of rows, under a heading that names the rows.
        assert!(
            p.text.contains(
                "### Squares, rows 42 to 81\n\n| n | square |\n| --- | --- |\n| 41 | 1681 |"
            ),
            "{}",
            &p.text[..1200]
        );
        assert_eq!(
            p.text.matches("| n | square |").count(),
            (TABLE_MAX_ROWS - 1).div_ceil(TABLE_GROUP)
        );
        assert!(
            p.text.contains(&format!(
                "(This sheet has {} rows; the first {TABLE_MAX_ROWS} are kept.)",
                n + 1
            )),
            "tail"
        );
        assert!(!p.text.contains(&format!("| {} |", TABLE_MAX_ROWS + 5)));
    }

    #[test]
    fn csv_with_quotes_other_delimiters_and_a_byte_order_mark() {
        let p = parse(Kind::Csv, "\u{feff}name,note,qty\r\n\"Bolt, M6\",\"says \"\"hi\"\"\nsecond line\",12\r\n\r\nNut,,7\r\n".as_bytes()).unwrap();
        assert_eq!(
            p.text.trim(),
            "| name | note | qty |\n| --- | --- | --- |\n| Bolt, M6 | says \"hi\" second line | 12 |\n| Nut |  | 7 |"
        );
        let semi = parse_csv("a;b\n1,5;2\n");
        assert!(
            semi.text.contains("| a | b |\n| --- | --- |\n| 1,5 | 2 |"),
            "{}",
            semi.text
        );
        let tabs = parse_csv("a\tb\n1\tx | y\n");
        assert!(tabs.text.contains("| 1 | x \\| y |"), "{}", tabs.text);
        assert_eq!(Kind::from_filename("data.TSV"), Some(Kind::Csv));
        // Binary content named .csv is refused; an empty file is an empty document.
        assert!(parse(Kind::Csv, &[0u8, 1, 2, 3]).is_err());
        assert!(parse(Kind::Csv, b"").is_err());
    }

    #[test]
    fn pptx_slides_in_numeric_order_with_notes() {
        let slide = |t: &str, b: &str| {
            format!(
                r#"<p:sld xmlns:a="a" xmlns:p="p"><a:p><a:r><a:t>{t}</a:t></a:r></a:p><a:p><a:r><a:t>{b}</a:t></a:r></a:p></p:sld>"#
            )
        };
        let s1 = slide("Overview", "Encode once");
        let s2 = slide("Second", "Read per query");
        let s10 = slide("Tenth", "Inject");
        let notes = r#"<p:notes xmlns:a="a" xmlns:p="p"><a:p><a:r><a:t>Say this aloud</a:t></a:r></a:p><a:p><a:r><a:t>1</a:t></a:r></a:p></p:notes>"#;
        let bytes = zip_of(&[
            ("ppt/presentation.xml", "<p:presentation xmlns:p='p'/>"),
            ("ppt/slides/slide10.xml", &s10),
            ("ppt/slides/slide2.xml", &s2),
            ("ppt/slides/slide1.xml", &s1),
            ("ppt/notesSlides/notesSlide1.xml", notes),
        ]);
        let p = parse(Kind::Pptx, &bytes).unwrap();
        assert_eq!(p.title.as_deref(), Some("Overview"));
        let (a, b, c) = (
            p.text.find("Slide 1:").unwrap(),
            p.text.find("Slide 2:").unwrap(),
            p.text.find("Slide 10:").unwrap(),
        );
        assert!(a < b && b < c, "{}", p.text);
        assert!(p.text.contains("Notes: Say this aloud"), "{}", p.text);
        assert!(!p.text.contains("Notes: Say this aloud 1"), "{}", p.text);
    }

    /// SR-K03: the sum of what is really decompressed is bounded, whatever the headers claim.
    #[test]
    fn the_total_actually_read_from_an_archive_is_bounded() {
        let part = "x".repeat(60_000);
        let bytes = zip_of(&[("a.xml", &part), ("b.xml", &part), ("c.xml", &part)]);
        let mut z = Zip::open(&bytes).unwrap();
        z.budget = 100_000;
        assert_eq!(z.read("a.xml").map(|s| s.len()), Some(60_000));
        assert_eq!(
            z.read("b.xml").map(|s| s.len()),
            Some(40_000),
            "cut at what is left"
        );
        assert!(!z.within_budget());
        assert!(
            z.read("c.xml").is_none(),
            "nothing more is read once it is spent"
        );
    }

    /// SR-K03: an entity that expands exponentially (billion laughs) is not expanded.
    #[test]
    fn xml_entity_bombs_are_not_expanded() {
        let doc = "<?xml version=\"1.0\"?><!DOCTYPE d [<!ENTITY a \"aaaaaaaaaa\"><!ENTITY b \"&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;\"><!ENTITY c \"&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;\">]><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body><w:p><w:r><w:t>start &c; end</w:t></w:r></w:p></w:body></w:document>";
        let bytes = zip_of(&[("word/document.xml", doc)]);
        let p = parse(Kind::Docx, &bytes).unwrap();
        assert!(
            p.text.contains("start") && p.text.len() < 200,
            "{}",
            p.text.len()
        );
    }

    /// SR-K03: nothing in a LaTeX file reads another file off the machine.
    #[test]
    fn latex_includes_do_not_read_files() {
        let dir = std::env::temp_dir().join(format!("nafs-kb-tex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let secret = dir.join("secret.tex");
        std::fs::write(&secret, "SECRET-MARKER-9f3a").unwrap();
        let tex = format!(
            "\\section{{A}}\n\nvisible text\n\n\\input{{{p}}}\n\\include{{{p}}}\n\\verbatiminput{{{p}}}\n\\lstinputlisting{{{p}}}\n\\includegraphics{{{p}}}\n",
            p = secret.display()
        );
        let p = parse(Kind::Tex, tex.as_bytes()).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert!(p.text.contains("visible text"), "{}", p.text);
        assert!(!p.text.contains("SECRET-MARKER-9f3a"), "{}", p.text);
    }

    /// SR-K03: scripts, styles, frames and handlers in a web page never reach the text.
    #[test]
    fn html_active_content_is_dropped() {
        let html = "<html><head><title>T</title><style>.x{color:red}STYLE-MARK</style><script>alert('SCRIPT-MARK')</script></head><body onload=\"alert('HANDLER-MARK')\"><iframe src=\"http://evil.example/IFRAME-MARK\"></iframe><object data=\"OBJECT-MARK\"></object><p>Real text <a href=\"javascript:alert('HREF-MARK')\">link</a></p><noscript>NOSCRIPT-MARK</noscript></body></html>";
        let p = parse(Kind::Html, html.as_bytes()).unwrap();
        for m in [
            "STYLE-MARK",
            "SCRIPT-MARK",
            "HANDLER-MARK",
            "IFRAME-MARK",
            "OBJECT-MARK",
            "HREF-MARK",
        ] {
            assert!(!p.text.contains(m), "{m} survived: {}", p.text);
        }
        assert!(p.text.contains("Real text"));
    }

    /// SR-K03: damaged and hostile PDFs are an error for the document, not a panic and not a hang.
    #[test]
    fn damaged_pdfs_fail_cleanly_and_quickly() {
        let started = std::time::Instant::now();
        let mut inputs: Vec<Vec<u8>> = vec![
            b"%PDF-1.7\n".to_vec(),
            b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 99999999 0 R >>\nendobj\ntrailer << /Root 1 0 R >>\n%%EOF".to_vec(),
            [b"%PDF-1.4\n".as_slice(), &[0xffu8; 4096]].concat(),
            b"%PDF-1.4\nxref\n0 99999999999\n".to_vec(),
        ];
        let real = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/publication/nafs_aql_lab_results.pdf");
        if let Ok(bytes) = std::fs::read(real) {
            for cut in [bytes.len() / 4, bytes.len() / 2, bytes.len() - 40] {
                inputs.push(bytes[..cut].to_vec());
            }
            let mut flipped = bytes.clone();
            for i in (200..flipped.len()).step_by(997) {
                flipped[i] ^= 0x5a;
            }
            inputs.push(flipped);
        }
        for (i, b) in inputs.iter().enumerate() {
            // Either outcome is fine; reaching the end of the loop is the assertion.
            let _ = parse(Kind::Pdf, b);
            assert!(started.elapsed().as_secs() < 60, "input {i} was slow");
        }
    }

    #[test]
    fn zip_bomb_and_missing_parts_are_refused() {
        let big = "a".repeat(3 * 1024 * 1024);
        let bytes = zip_of(&[("word/document.xml", &big)]);
        assert!(parse(Kind::Docx, &bytes).unwrap_err().contains("zip bomb"));
        let bytes = zip_of(&[("other.xml", "<x/>")]);
        assert!(parse(Kind::Docx, &bytes)
            .unwrap_err()
            .contains("word/document.xml"));
        assert!(parse(Kind::Pptx, &bytes)
            .unwrap_err()
            .contains("presentation"));
    }

    #[test]
    fn html_drops_scripts_and_navigation() {
        let html = "<html><head><title>KB &amp; Graphs</title><style>p{}</style></head><body><nav>Home | About</nav>\
            <h1>Graph retrieval</h1><p>Walks <b>entity</b> relations.</p><script>alert(1)</script></body></html>";
        let p = parse(Kind::Html, html.as_bytes()).unwrap();
        assert_eq!(p.title.as_deref(), Some("KB & Graphs"));
        assert!(p.text.contains("# Graph retrieval"), "{}", p.text);
        assert!(p.text.contains("Walks **entity** relations."), "{}", p.text);
        assert!(
            !p.text.contains("alert") && !p.text.contains("Home"),
            "{}",
            p.text
        );
    }

    #[test]
    fn tex_sections_formatting_and_maths() {
        let tex = r"\documentclass{article}
\usepackage{amsmath} % packages
\title{Latent \textbf{Memory}}
\begin{document}
\maketitle
\begin{abstract}
We compress documents.
\end{abstract}
\section{Method}\label{sec:m}
The module uses $K=16$ slots~\cite{perceiver}. 50\% fewer tokens % a comment
\subsection*{Details}
\begin{itemize}
\item \emph{Cross} attention
\end{itemize}
\begin{equation}
M = \mathrm{LN}(R) W
\end{equation}
\begin{figure}\includegraphics{x.png}\caption{The \textit{architecture}.}\end{figure}
\end{document}";
        let p = parse(Kind::Tex, tex.as_bytes()).unwrap();
        assert_eq!(p.title.as_deref(), Some("Latent Memory"));
        let t = &p.text;
        assert!(
            t.contains("# Latent Memory") && t.contains("## Abstract") && t.contains("## Method"),
            "{t}"
        );
        assert!(
            t.contains("### Details") && t.contains("- Cross attention"),
            "{t}"
        );
        assert!(
            t.contains("The module uses $K=16$ slots. 50% fewer tokens"),
            "{t}"
        );
        assert!(t.contains("Caption: The architecture."), "{t}");
        assert!(t.contains("$$\nM = \\mathrm{LN}(R) W\n$$"), "{t}");
        assert!(
            !t.contains("documentclass") && !t.contains("a comment") && !t.contains("perceiver"),
            "{t}"
        );
    }

    #[test]
    fn tex_macros_defined_in_the_file_are_expanded() {
        let tex = r"\newcommand{\Sone}{System~1}\newcommand\Gw{the gateway}\def\kb{knowledge base}\newcommand{\twice}[1]{#1#1}
\begin{document}
\section{Paths}
The fast path (\Sone) drafts; \Gw\ routes to the \kb{} and \Sone\ answers. \Stwo stays.
\end{document}";
        let p = parse(Kind::Tex, tex.as_bytes()).unwrap();
        assert!(p.text.contains("The fast path (System 1) drafts; the gateway routes to the knowledge base and System 1 answers."), "{}", p.text);
        assert!(p.text.contains("Stwo stays"), "{}", p.text);
    }

    #[test]
    fn pdf_from_the_papers_folder() {
        // A real paper: text per page, page boundaries, a title-like first heading.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../papers/2302.01318v1.pdf"
        );
        let Ok(bytes) = std::fs::read(path) else {
            return;
        }; // papers/ is not in version control
        let p = parse(Kind::Pdf, &bytes).unwrap();
        assert!(p.pages.unwrap() > 5);
        assert_eq!(p.page_starts.len(), p.pages.unwrap());
        assert!(
            p.text.to_lowercase().contains("speculative"),
            "{}",
            p.text.chars().take(500).collect::<String>()
        );
        assert_eq!(p.page_at(0), Some(1));
        assert_eq!(p.page_at(p.page_starts[2]), Some(3));
    }

    #[test]
    fn two_column_paper_reads_in_column_order_with_its_sections() {
        // The project's own IEEE two-column paper (in version control): the layout pass must keep sentences whole
        // across the columns' lines and mark the numbered sections as headings.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/report/nafs_aql_report.pdf"
        );
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let p = parse_pdf_with(&bytes, PdfLayout::Spans).unwrap();
        assert!(
            p.text.contains("## I. INTRODUCTION"),
            "{}",
            p.text.chars().take(2000).collect::<String>()
        );
        assert!(p.text.contains("## II. RELATED WORK"));
        assert!(p
            .title
            .as_deref()
            .is_some_and(|t| t.starts_with("Nafs-Aql: A Verifier-Gated")));
        // A sentence that wraps over several lines of the left column, read whole.
        assert!(
            p.text
                .contains("there is a fast, automatic mode and a slow, effortful one"),
            "{}",
            p.text.chars().take(3000).collect::<String>()
        );
        let converter = parse_pdf_with(&bytes, PdfLayout::Converter).unwrap();
        let headings = |t: &str| t.lines().filter(|l| l.starts_with('#')).count();
        assert!(headings(&p.text) > 3 * headings(&converter.text).max(1));
    }

    #[test]
    fn pdf_tidy_merges_bold_and_marks_sections() {
        let t = tidy_pdf_markdown("**Main** **results** here\n2.5 MEMORILLA WITH QWEN3\nB.1 TRAINING DATASETS\n3 Results are good\n12 MONKEYS");
        assert!(t.contains("**Main results** here"), "{t}");
        assert!(
            t.contains("### 2.5 MEMORILLA WITH QWEN3") && t.contains("### B.1 TRAINING DATASETS"),
            "{t}"
        );
        assert!(t.contains("\n3 Results are good"), "{t}");
        assert!(t.contains("## 12 MONKEYS"), "{t}");
    }

    #[test]
    fn pdf_titles_need_words() {
        assert_eq!(pdf_title("# Tokens\n# Experts (Total / Activated) 128 / 8\n# Orca: A Distributed Serving System\n"), Some("Orca: A Distributed Serving System".into()));
        assert_eq!(pdf_title("# of GPUs\nbody"), None);
    }

    #[test]
    fn running_headers_and_page_numbers_are_dropped() {
        let pages: Vec<String> = (1..=4)
            .map(|i| format!("Preprint\n\nBody text of page {i} goes here.\n\n{i}"))
            .collect();
        let out = drop_running_lines(pages);
        assert!(out.iter().all(|p| !p.contains("Preprint")), "{out:?}");
        assert_eq!(out[1], "Body text of page 2 goes here.");
    }

    #[test]
    fn empty_documents_are_refused() {
        assert!(parse(Kind::Txt, b"   \n\n ")
            .unwrap_err()
            .contains("no text"));
    }
}
