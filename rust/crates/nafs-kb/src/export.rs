//! Documents out (Phase 35): an answer or a conversation, which is Markdown, saved as a file.
//!
//! The model is never asked to write a file format. It writes Markdown, which it does well, and a converter here
//! makes the file, which a program does exactly. (An 8B model ran 2 % of raw slide-library programs and 54 % with a
//! high-level form in between: Ge et al., *AutoPresent*, 2025.) HTML, plain text, CSV and LaTeX are written by this
//! module, and so is Excel (a sheet per table, numbers as numbers, the header row bold and frozen); Word and
//! PowerPoint are written by `office_oxide` and PDF by `pdf_oxide`, both already part of the build.
//!
//! What is written is data, not a program: raw HTML in the Markdown is escaped, a link that is not `http`, `https`
//! or `mailto` loses its target, and a spreadsheet cell that would be run as a formula is written as text.

use std::io::Cursor;

use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// The most Markdown one export takes.
pub const MAX_MARKDOWN: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Md,
    Txt,
    Html,
    Docx,
    Pptx,
    Xlsx,
    Csv,
    Tex,
    Pdf,
}

impl Format {
    pub const ALL: [Format; 9] = [
        Format::Md,
        Format::Txt,
        Format::Html,
        Format::Docx,
        Format::Pptx,
        Format::Xlsx,
        Format::Csv,
        Format::Tex,
        Format::Pdf,
    ];

    pub fn from_name(s: &str) -> Option<Format> {
        Format::ALL
            .into_iter()
            .find(|f| f.ext().eq_ignore_ascii_case(s.trim()))
    }

    /// The file extension, which is also the name the API uses.
    pub fn ext(self) -> &'static str {
        match self {
            Format::Md => "md",
            Format::Txt => "txt",
            Format::Html => "html",
            Format::Docx => "docx",
            Format::Pptx => "pptx",
            Format::Xlsx => "xlsx",
            Format::Csv => "csv",
            Format::Tex => "tex",
            Format::Pdf => "pdf",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            Format::Md => "text/markdown; charset=utf-8",
            Format::Txt => "text/plain; charset=utf-8",
            Format::Html => "text/html; charset=utf-8",
            Format::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            Format::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            Format::Xlsx => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Format::Csv => "text/csv; charset=utf-8",
            Format::Tex => "application/x-tex; charset=utf-8",
            Format::Pdf => "application/pdf",
        }
    }

    /// What the format is called in the app.
    pub fn label(self) -> &'static str {
        match self {
            Format::Md => "Markdown",
            Format::Txt => "Plain text",
            Format::Html => "Web page",
            Format::Docx => "Word",
            Format::Pptx => "PowerPoint",
            Format::Xlsx => "Excel",
            Format::Csv => "CSV",
            Format::Tex => "LaTeX",
            Format::Pdf => "PDF",
        }
    }
}

fn options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_MATH
}

/// `\( .. \)` and `\[ .. \]` become `$ .. $` and `$$ .. $$` outside code fences, which is the form the Markdown
/// reader knows as maths; and the document gets its title as a first heading when it has none.
fn prepare(markdown: &str, title: &str) -> String {
    let mut out = String::with_capacity(markdown.len() + title.len() + 8);
    let title = title.trim();
    let has_heading = markdown
        .lines()
        .find(|l| !l.trim().is_empty())
        .is_some_and(|l| l.trim_start().starts_with('#'));
    if !title.is_empty() && !has_heading {
        out.push_str(&format!("# {title}\n\n"));
    }
    let mut fenced = false;
    for line in markdown.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
        }
        if fenced {
            out.push_str(line);
        } else {
            out.push_str(
                &line
                    .replace("\\[", "$$")
                    .replace("\\]", "$$")
                    .replace("\\(", "$")
                    .replace("\\)", "$"),
            );
        }
        out.push('\n');
    }
    out
}

/// The first heading of the Markdown, else the given title, else "Document".
fn title_of(md: &str, given: &str) -> String {
    md.lines()
        .find_map(|l| {
            let t = l.trim_start();
            t.starts_with('#')
                .then(|| t.trim_start_matches('#').trim().to_string())
        })
        .filter(|t| !t.is_empty())
        .or_else(|| (!given.trim().is_empty()).then(|| given.trim().to_string()))
        .unwrap_or_else(|| "Document".to_string())
}

/// Save `markdown` as `format`. `title` names the document when the text has no heading of its own.
pub fn export(markdown: &str, title: &str, format: Format) -> Result<Vec<u8>, String> {
    if markdown.trim().is_empty() {
        return Err("there is nothing to save".into());
    }
    if markdown.len() > MAX_MARKDOWN {
        return Err(format!(
            "the text is {:.1} MB; at most {} MB can be saved as one file",
            markdown.len() as f64 / 1_048_576.0,
            MAX_MARKDOWN / 1_048_576
        ));
    }
    let md = prepare(markdown, title);
    // A converter that fails on some input is that export's error, not the server's.
    std::panic::catch_unwind(|| match format {
        Format::Md => Ok(md.clone().into_bytes()),
        Format::Txt => Ok(to_text(&md).into_bytes()),
        Format::Html => Ok(to_html(&md, &title_of(&md, title)).into_bytes()),
        Format::Tex => Ok(to_tex(&md, &title_of(&md, title)).into_bytes()),
        Format::Csv => to_csv(&md).map(String::into_bytes),
        Format::Docx => office(&md, office_oxide::format::DocumentFormat::Docx),
        Format::Pptx => office(&md, office_oxide::format::DocumentFormat::Pptx),
        Format::Xlsx => {
            if tables(&md).is_empty() {
                return Err("there is no table in this text to save as a spreadsheet".into());
            }
            to_xlsx(&md, &title_of(&md, title))
        }
        Format::Pdf => pdf_oxide::api::Pdf::from_markdown(&plain_maths(&md))
            .and_then(|mut p| p.save_to_bytes())
            .map_err(|e| format!("the PDF could not be written: {e}")),
    })
    .map_err(|_| format!("the {} file could not be written", format.label()))?
}

fn office(md: &str, format: office_oxide::format::DocumentFormat) -> Result<Vec<u8>, String> {
    let mut out = Cursor::new(Vec::new());
    office_oxide::create::create_from_markdown_to_writer(&plain_maths(md), format, &mut out)
        .map_err(|e| format!("the file could not be written: {e}"))?;
    Ok(out.into_inner())
}

/// Maths as the text between its delimiters, for the writers that know no maths: `$x^2$` stays readable as `x^2`
/// rather than appearing with stray dollar signs or vanishing.
fn plain_maths(md: &str) -> String {
    let mut out = String::with_capacity(md.len());
    let mut fenced = false;
    for line in md.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
        }
        if fenced {
            out.push_str(line);
        } else {
            out.push_str(&line.replace("$$", ""));
        }
        out.push('\n');
    }
    out
}

/// Every table of the Markdown: its rows of cell texts (the first is the header) and the heading above it.
pub fn tables(md: &str) -> Vec<(String, Vec<Vec<String>>)> {
    let mut out = Vec::new();
    let (mut heading, mut in_heading, mut last_heading) = (String::new(), false, String::new());
    let (mut rows, mut row, mut cell, mut in_table): (Vec<Vec<String>>, Vec<String>, String, bool) =
        (Vec::new(), Vec::new(), String::new(), false);
    for ev in Parser::new_ext(md, options()) {
        match ev {
            Event::Start(Tag::Heading { .. }) => {
                in_heading = true;
                heading.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                in_heading = false;
                last_heading = heading.trim().to_string();
            }
            Event::Start(Tag::Table(_)) => {
                in_table = true;
                rows.clear();
            }
            Event::End(TagEnd::Table) => {
                in_table = false;
                out.push((last_heading.clone(), std::mem::take(&mut rows)));
            }
            Event::Start(Tag::TableHead) | Event::Start(Tag::TableRow) => row.clear(),
            Event::End(TagEnd::TableHead) | Event::End(TagEnd::TableRow) => {
                rows.push(std::mem::take(&mut row))
            }
            Event::Start(Tag::TableCell) => cell.clear(),
            Event::End(TagEnd::TableCell) => row.push(cell.trim().to_string()),
            Event::Text(t) | Event::Code(t) | Event::InlineMath(t) | Event::DisplayMath(t) => {
                if in_table {
                    cell.push_str(&t)
                } else if in_heading {
                    heading.push_str(&t)
                }
            }
            Event::SoftBreak | Event::HardBreak if in_table => cell.push(' '),
            _ => {}
        }
    }
    out
}

fn xml_esc(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect::<String>()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `A`, `B`, … `Z`, `AA`, … for a 0-based column.
fn column_name(mut k: usize) -> String {
    let mut out = Vec::new();
    loop {
        out.push(b'A' + (k % 26) as u8);
        if k < 26 {
            break;
        }
        k = k / 26 - 1;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// The number a cell's text is, when it is nothing but a number: `12`, `-3.5`, `1,200`, `1,234.5`. Anything else
/// (a percentage, a date, a unit) stays text, exactly as it was written.
fn as_number(v: &str) -> Option<String> {
    let t = v.trim();
    let body = t.strip_prefix('-').unwrap_or(t);
    if body.is_empty() || !body.as_bytes()[0].is_ascii_digit() || body.len() > 15 {
        return None;
    }
    let (int, frac) = body.split_once('.').unwrap_or((body, ""));
    if body.contains('.') && (frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let grouped = int.contains(',');
    let ok = if grouped {
        let mut parts = int.split(',');
        parts
            .next()
            .is_some_and(|p| (1..=3).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            && parts.all(|p| p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit()))
    } else {
        int.bytes().all(|b| b.is_ascii_digit()) && (int.len() == 1 || !int.starts_with('0'))
    };
    ok.then(|| t.replace(',', ""))
}

/// A sheet name Excel accepts (31 characters, none of `[]:*?/\`), made different from the ones already used.
fn sheet_name(want: &str, used: &mut Vec<String>) -> String {
    let base: String = want
        .chars()
        .map(|c| {
            if matches!(c, '[' | ']' | ':' | '*' | '?' | '/' | '\\' | '\'') {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let base: String = if base.is_empty() {
        "Table".into()
    } else {
        base.chars().take(31).collect()
    };
    let mut name = base.clone();
    let mut n = 2;
    while used.iter().any(|u| u.eq_ignore_ascii_case(&name)) {
        let tail = format!(" {n}");
        name = format!(
            "{}{tail}",
            base.chars().take(31 - tail.len()).collect::<String>()
        );
        n += 1;
    }
    used.push(name.clone());
    name
}

/// An Excel workbook of the text's tables: a sheet for each, named after the heading above it; the first row is the
/// header (bold, and it stays in view when the sheet is scrolled); a cell that is a number is stored as one, so it
/// can be summed, and everything else as the text that was written. Nothing is stored as a formula.
fn to_xlsx(md: &str, title: &str) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let all = tables(md);
    let mut used = Vec::new();
    let names: Vec<String> = all
        .iter()
        .enumerate()
        .map(|(i, (h, _))| {
            let want = if h.is_empty() {
                format!("Table {}", i + 1)
            } else {
                h.clone()
            };
            sheet_name(&want, &mut used)
        })
        .collect();
    let mut parts: Vec<(String, String)> = Vec::new();
    let overrides: String = (1..=all.len())
        .map(|i| format!("<Override PartName=\"/xl/worksheets/sheet{i}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"))
        .collect();
    parts.push(("[Content_Types].xml".into(), format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/><Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>{overrides}</Types>")));
    parts.push(("_rels/.rels".into(), "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/></Relationships>".into()));
    parts.push(("docProps/core.xml".into(), format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{}</dc:title><dc:creator>Nafs-Aql</dc:creator></cp:coreProperties>", xml_esc(title))));
    let sheets: String = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            format!(
                "<sheet name=\"{}\" sheetId=\"{}\" r:id=\"rId{}\"/>",
                xml_esc(n),
                i + 1,
                i + 1
            )
        })
        .collect();
    parts.push(("xl/workbook.xml".into(), format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>{sheets}</sheets></workbook>")));
    let rels: String = (1..=all.len())
        .map(|i| format!("<Relationship Id=\"rId{i}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{i}.xml\"/>"))
        .collect();
    parts.push(("xl/_rels/workbook.xml.rels".into(), format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">{rels}<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/></Relationships>", all.len() + 1)));
    // Style 0 is plain; style 1 is the header: bold on a light fill.
    parts.push(("xl/styles.xml".into(), "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><fonts count=\"2\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font><font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts><fills count=\"3\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFEEF1F5\"/><bgColor indexed=\"64\"/></patternFill></fill></fills><borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs><cellXfs count=\"2\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/><xf numFmtId=\"0\" fontId=\"1\" fillId=\"2\" borderId=\"0\" xfId=\"0\" applyFont=\"1\" applyFill=\"1\"/></cellXfs></styleSheet>".into()));
    for (i, (_, rows)) in all.iter().enumerate() {
        let width = rows.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let cols: String = (0..width)
            .map(|k| {
                let w = rows
                    .iter()
                    .filter_map(|r| r.get(k))
                    .map(|c| c.chars().count())
                    .max()
                    .unwrap_or(8)
                    .clamp(8, 60)
                    + 2;
                format!(
                    "<col min=\"{0}\" max=\"{0}\" width=\"{w}\" customWidth=\"1\"/>",
                    k + 1
                )
            })
            .collect();
        let mut data = String::new();
        for (r, row) in rows.iter().enumerate() {
            data.push_str(&format!("<row r=\"{}\">", r + 1));
            for (k, v) in row.iter().enumerate() {
                if v.is_empty() {
                    continue;
                }
                let at = format!("{}{}", column_name(k), r + 1);
                let style = if r == 0 { " s=\"1\"" } else { "" };
                match (r > 0).then(|| as_number(v)).flatten() {
                    Some(n) => data.push_str(&format!("<c r=\"{at}\"{style}><v>{n}</v></c>")),
                    // Inline text: what was written, never a formula.
                    None => data.push_str(&format!("<c r=\"{at}\"{style} t=\"inlineStr\"><is><t xml:space=\"preserve\">{}</t></is></c>", xml_esc(v))),
                }
            }
            data.push_str("</row>");
        }
        parts.push((format!("xl/worksheets/sheet{}.xml", i + 1), format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetViews><sheetView workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/></sheetView></sheetViews><cols>{cols}</cols><sheetData>{data}</sheetData></worksheet>")));
    }
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, body) in &parts {
            w.start_file(name.as_str(), opts)
                .map_err(|e| format!("the Excel file could not be written: {e}"))?;
            w.write_all(body.as_bytes())
                .map_err(|e| format!("the Excel file could not be written: {e}"))?;
        }
        w.finish()
            .map_err(|e| format!("the Excel file could not be written: {e}"))?;
    }
    Ok(buf.into_inner())
}

/// A cell a spreadsheet would run as a formula (`=`, `+`, `-`, `@` first, unless it is just a number) is given a
/// leading apostrophe, so a saved table cannot run anything when it is opened.
fn formula_safe(v: &str) -> String {
    let risky = v.starts_with(['=', '+', '-', '@', '\t', '\r']);
    let number = v
        .trim_start_matches(['+', '-'])
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '%' | ' '))
        && v.chars().any(|c| c.is_ascii_digit());
    if risky && !number {
        format!("'{v}")
    } else {
        v.to_string()
    }
}

fn to_csv(md: &str) -> Result<String, String> {
    let all = tables(md);
    if all.is_empty() {
        return Err("there is no table in this text to save as CSV".into());
    }
    let field = |v: &String| {
        let s = formula_safe(v);
        if s.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s
        }
    };
    let mut out = String::new();
    for (i, (_, rows)) in all.iter().enumerate() {
        if i > 0 {
            out.push_str("\r\n");
        }
        for r in rows {
            out.push_str(&r.iter().map(field).collect::<Vec<_>>().join(","));
            out.push_str("\r\n");
        }
    }
    Ok(out)
}

fn to_text(md: &str) -> String {
    let mut out = String::new();
    let mut lists: Vec<Option<u64>> = Vec::new();
    let mut cells = 0usize;
    let gap = |out: &mut String| {
        while out.ends_with(' ') {
            out.pop();
        }
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push_str(if out.ends_with('\n') { "\n" } else { "\n\n" });
        }
    };
    for ev in Parser::new_ext(md, options()) {
        match ev {
            Event::Start(Tag::Item) => {
                if !out.ends_with('\n') && !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&"  ".repeat(lists.len().saturating_sub(1)));
                match lists.last_mut() {
                    Some(Some(n)) => {
                        out.push_str(&format!("{n}. "));
                        *n += 1;
                    }
                    _ => out.push_str("- "),
                }
            }
            Event::Start(Tag::List(start)) => {
                if lists.is_empty() {
                    gap(&mut out);
                }
                lists.push(start);
            }
            Event::End(TagEnd::List(_)) => {
                lists.pop();
                if lists.is_empty() {
                    gap(&mut out);
                }
            }
            Event::Start(Tag::Paragraph) if lists.is_empty() => gap(&mut out),
            Event::Start(Tag::Heading { .. })
            | Event::Start(Tag::CodeBlock(_))
            | Event::Start(Tag::Table(_))
            | Event::Start(Tag::BlockQuote(_)) => gap(&mut out),
            Event::End(TagEnd::Paragraph) if lists.is_empty() => gap(&mut out),
            Event::End(TagEnd::Heading(_))
            | Event::End(TagEnd::CodeBlock)
            | Event::End(TagEnd::Table) => gap(&mut out),
            Event::Start(Tag::TableHead) | Event::Start(Tag::TableRow) => cells = 0,
            Event::End(TagEnd::TableHead) | Event::End(TagEnd::TableRow) => out.push('\n'),
            Event::Start(Tag::TableCell) => {
                if cells > 0 {
                    out.push('\t');
                }
                cells += 1;
            }
            Event::Text(t) | Event::Code(t) | Event::Html(t) | Event::InlineHtml(t) => {
                out.push_str(&t)
            }
            Event::InlineMath(t) | Event::DisplayMath(t) => out.push_str(&t),
            Event::SoftBreak => out.push(' '),
            Event::HardBreak => out.push('\n'),
            Event::Rule => {
                gap(&mut out);
                out.push_str("----------\n\n");
            }
            Event::TaskListMarker(done) => out.push_str(if done { "[x] " } else { "[ ] " }),
            Event::FootnoteReference(n) => out.push_str(&format!("[{n}]")),
            _ => {}
        }
    }
    let mut out = out.trim_end().to_string();
    out.push('\n');
    out
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Whether a link target may be kept: the web, mail, or a place in the same document.
fn safe_link(url: &str) -> bool {
    let u = url.trim().to_ascii_lowercase();
    u.starts_with("http://")
        || u.starts_with("https://")
        || u.starts_with("mailto:")
        || u.starts_with('#')
}

const HTML_STYLE: &str = "body{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#18212b;background:#fff;margin:0}main{max-width:46rem;margin:0 auto;padding:2.5rem 1.25rem}h1,h2,h3,h4{line-height:1.25;margin:1.6em 0 .5em}h1{font-size:1.9rem;margin-top:0}pre{background:#0f1720;color:#e3e8ee;padding:14px 16px;border-radius:10px;overflow:auto;font-size:13.5px}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}:not(pre)>code{background:#f0f2f5;padding:.1em .35em;border-radius:5px}table{border-collapse:collapse;margin:1em 0;font-size:.95em}th,td{border:1px solid #dce1e8;padding:6px 12px;text-align:left;vertical-align:top}th{background:#f0f2f5}blockquote{margin:1em 0;padding:.2em 1em;border-left:4px solid #dce1e8;color:#5e6b78}a{color:#1d5fbf}hr{border:0;border-top:1px solid #dce1e8;margin:2em 0}.math{font-family:ui-monospace,Menlo,monospace}@media(prefers-color-scheme:dark){body{background:#0e141b;color:#e6ebf0}:not(pre)>code,th{background:#1b2531}th,td{border-color:#2a3746}a{color:#8fb8ff}}@media print{main{max-width:none;padding:0}pre{white-space:pre-wrap}}";

/// A page that stands alone: its own style, no script, nothing fetched from anywhere. Written here rather than by a
/// general Markdown-to-HTML writer, so that what may appear in the page is exactly this list: raw HTML in the text
/// is shown as text, a link the page should not follow keeps its words and loses its target, a picture is named,
/// not loaded, and maths stays as the TeX that was written.
fn to_html(md: &str, title: &str) -> String {
    let mut b = String::new();
    let (mut aligns, mut in_head, mut col): (Vec<Alignment>, bool, usize) = (Vec::new(), false, 0);
    for ev in Parser::new_ext(md, options()) {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Paragraph => b.push_str("<p>"),
                Tag::Heading { level, .. } => b.push_str(&format!("<h{}>", level as usize)),
                Tag::BlockQuote(_) => b.push_str("<blockquote>\n"),
                Tag::CodeBlock(kind) => match kind {
                    CodeBlockKind::Fenced(lang) if !lang.trim().is_empty() => b.push_str(&format!(
                        "<pre><code class=\"language-{}\">",
                        html_esc(lang.split_whitespace().next().unwrap_or(""))
                    )),
                    _ => b.push_str("<pre><code>"),
                },
                Tag::List(None) => b.push_str("<ul>\n"),
                Tag::List(Some(1)) => b.push_str("<ol>\n"),
                Tag::List(Some(n)) => b.push_str(&format!("<ol start=\"{n}\">\n")),
                Tag::Item => b.push_str("<li>"),
                Tag::Table(al) => {
                    aligns = al;
                    b.push_str("<table>\n");
                }
                Tag::TableHead => {
                    in_head = true;
                    col = 0;
                    b.push_str("<thead><tr>");
                }
                Tag::TableRow => {
                    col = 0;
                    b.push_str("<tr>");
                }
                Tag::TableCell => {
                    let style = match aligns.get(col) {
                        Some(Alignment::Right) => " style=\"text-align: right\"",
                        Some(Alignment::Center) => " style=\"text-align: center\"",
                        _ => "",
                    };
                    b.push_str(&format!("<{}{style}>", if in_head { "th" } else { "td" }));
                }
                Tag::Emphasis => b.push_str("<em>"),
                Tag::Strong => b.push_str("<strong>"),
                Tag::Strikethrough => b.push_str("<del>"),
                Tag::Link { dest_url, .. } => b.push_str(&format!(
                    "<a href=\"{}\">",
                    if safe_link(&dest_url) {
                        html_esc(&dest_url)
                    } else {
                        "#".to_string()
                    }
                )),
                Tag::Image { .. } => b.push_str("<em>[picture: "),
                Tag::FootnoteDefinition(name) => b.push_str(&format!(
                    "<div class=\"footnote\"><sup>[{}]</sup> ",
                    html_esc(&name)
                )),
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph => b.push_str("</p>\n"),
                TagEnd::Heading(level) => b.push_str(&format!("</h{}>\n", level as usize)),
                TagEnd::BlockQuote(_) => b.push_str("</blockquote>\n"),
                TagEnd::CodeBlock => b.push_str("</code></pre>\n"),
                TagEnd::List(true) => b.push_str("</ol>\n"),
                TagEnd::List(false) => b.push_str("</ul>\n"),
                TagEnd::Item => b.push_str("</li>\n"),
                TagEnd::Table => b.push_str("</tbody>\n</table>\n"),
                TagEnd::TableHead => {
                    in_head = false;
                    b.push_str("</tr></thead>\n<tbody>\n");
                }
                TagEnd::TableRow => b.push_str("</tr>\n"),
                TagEnd::TableCell => {
                    b.push_str(if in_head { "</th>" } else { "</td>" });
                    col += 1;
                }
                TagEnd::Emphasis => b.push_str("</em>"),
                TagEnd::Strong => b.push_str("</strong>"),
                TagEnd::Strikethrough => b.push_str("</del>"),
                TagEnd::Link => b.push_str("</a>"),
                TagEnd::Image => b.push_str("]</em>"),
                TagEnd::FootnoteDefinition => b.push_str("</div>\n"),
                _ => {}
            },
            Event::Text(t) | Event::Html(t) | Event::InlineHtml(t) => b.push_str(&html_esc(&t)),
            Event::Code(t) => b.push_str(&format!("<code>{}</code>", html_esc(&t))),
            Event::InlineMath(t) => {
                b.push_str(&format!("<span class=\"math\">{}</span>", html_esc(&t)))
            }
            Event::DisplayMath(t) => {
                b.push_str(&format!("<p class=\"math\">{}</p>\n", html_esc(t.trim())))
            }
            Event::SoftBreak => b.push('\n'),
            Event::HardBreak => b.push_str("<br>\n"),
            Event::Rule => b.push_str("<hr>\n"),
            Event::TaskListMarker(done) => b.push_str(if done { "&#9745; " } else { "&#9744; " }),
            Event::FootnoteReference(n) => b.push_str(&format!("<sup>[{}]</sup>", html_esc(&n))),
        }
    }
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n<style>{HTML_STYLE}</style>\n</head>\n<body>\n<main>\n{b}</main>\n</body>\n</html>\n",
        html_esc(title)
    )
}

fn tex_esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '&' | '%' | '$' | '#' | '_' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => out.push(c),
        }
    }
    out
}

/// A LaTeX article that compiles with a plain `pdflatex` or `tectonic`: sections, lists, tables, code and the
/// maths as it was written.
fn to_tex(md: &str, title: &str) -> String {
    let mut body = String::new();
    let mut first_heading = true;
    let mut in_code = false;
    // A table is gathered whole, because its column count is needed before its first row is written.
    let mut table: Option<(Vec<Alignment>, Vec<Vec<String>>)> = None;
    let mut cell: Option<String> = None;
    let mut row: Vec<String> = Vec::new();
    let mut skip_title = false;
    for ev in Parser::new_ext(md, options()) {
        // Text goes to the open table cell when there is one.
        let mut put = |body: &mut String, s: &str| match cell.as_mut() {
            Some(c) => c.push_str(s),
            None => body.push_str(s),
        };
        match ev {
            Event::Start(Tag::Heading { level, .. }) => {
                // The document's first heading is its title (`\maketitle`), not a section.
                if first_heading && level == HeadingLevel::H1 {
                    skip_title = true;
                } else {
                    put(
                        &mut body,
                        match level {
                            HeadingLevel::H1 => "\\section{",
                            HeadingLevel::H2 => "\\subsection{",
                            HeadingLevel::H3 => "\\subsubsection{",
                            _ => "\\paragraph{",
                        },
                    );
                }
                first_heading = false;
            }
            Event::End(TagEnd::Heading(_)) => {
                if skip_title {
                    skip_title = false;
                } else {
                    put(&mut body, "}\n\n");
                }
            }
            Event::Start(Tag::Paragraph) => first_heading = false,
            Event::End(TagEnd::Paragraph) => put(&mut body, "\n\n"),
            Event::Start(Tag::Emphasis) => put(&mut body, "\\emph{"),
            Event::Start(Tag::Strong) => put(&mut body, "\\textbf{"),
            Event::Start(Tag::Strikethrough) => put(&mut body, "\\sout{"),
            Event::End(TagEnd::Emphasis)
            | Event::End(TagEnd::Strong)
            | Event::End(TagEnd::Strikethrough) => put(&mut body, "}"),
            Event::Start(Tag::BlockQuote(_)) => put(&mut body, "\\begin{quote}\n"),
            Event::End(TagEnd::BlockQuote(_)) => put(&mut body, "\\end{quote}\n\n"),
            Event::Start(Tag::List(None)) => put(&mut body, "\\begin{itemize}\n"),
            Event::Start(Tag::List(Some(_))) => put(&mut body, "\\begin{enumerate}\n"),
            Event::End(TagEnd::List(false)) => put(&mut body, "\\end{itemize}\n\n"),
            Event::End(TagEnd::List(true)) => put(&mut body, "\\end{enumerate}\n\n"),
            Event::Start(Tag::Item) => put(&mut body, "\\item "),
            Event::End(TagEnd::Item) => put(&mut body, "\n"),
            Event::Start(Tag::CodeBlock(kind)) => {
                in_code = true;
                if let CodeBlockKind::Fenced(lang) = &kind {
                    if !lang.trim().is_empty() {
                        put(&mut body, &format!("% {}\n", tex_esc(lang.trim())));
                    }
                }
                put(&mut body, "\\begin{verbatim}\n");
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code = false;
                if !body.ends_with('\n') {
                    body.push('\n');
                }
                put(&mut body, "\\end{verbatim}\n\n");
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                if safe_link(&dest_url) && !dest_url.starts_with('#') {
                    let url = dest_url
                        .replace('\\', "")
                        .replace('%', "\\%")
                        .replace('#', "\\#")
                        .replace(['{', '}'], "");
                    put(&mut body, &format!("\\href{{{url}}}{{"));
                } else {
                    put(&mut body, "{");
                }
            }
            Event::End(TagEnd::Link) => put(&mut body, "}"),
            Event::Start(Tag::Table(al)) => table = Some((al, Vec::new())),
            Event::Start(Tag::TableHead) | Event::Start(Tag::TableRow) => row.clear(),
            Event::Start(Tag::TableCell) => cell = Some(String::new()),
            Event::End(TagEnd::TableCell) => {
                row.push(cell.take().unwrap_or_default().trim().to_string())
            }
            Event::End(TagEnd::TableHead) | Event::End(TagEnd::TableRow) => {
                if let Some((_, rows)) = table.as_mut() {
                    rows.push(std::mem::take(&mut row));
                }
            }
            Event::End(TagEnd::Table) => {
                if let Some((al, rows)) = table.take() {
                    let width = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
                    let spec: String = (0..width)
                        .map(|k| match al.get(k) {
                            Some(Alignment::Right) => 'r',
                            Some(Alignment::Center) => 'c',
                            _ => 'l',
                        })
                        .collect();
                    body.push_str(&format!(
                        "\\begin{{center}}\n\\begin{{tabular}}{{{spec}}}\n\\hline\n"
                    ));
                    for (i, r) in rows.iter().enumerate() {
                        let cells: Vec<String> = (0..width)
                            .map(|k| r.get(k).cloned().unwrap_or_default())
                            .collect();
                        body.push_str(&format!("{} \\\\\n", cells.join(" & ")));
                        if i == 0 {
                            body.push_str("\\hline\n");
                        }
                    }
                    body.push_str("\\hline\n\\end{tabular}\n\\end{center}\n\n");
                }
            }
            Event::Text(t) => {
                if skip_title {
                    // already in \title
                } else if in_code {
                    // Verbatim text ends only at its own end marker: one written inside the code is broken up.
                    put(&mut body, &t.replace("\\end{verbatim}", "\\end {verbatim}"));
                } else {
                    put(&mut body, &tex_esc(&t));
                }
            }
            Event::Code(t) => put(&mut body, &format!("\\texttt{{{}}}", tex_esc(&t))),
            Event::Html(t) | Event::InlineHtml(t) => put(&mut body, &tex_esc(&t)),
            Event::InlineMath(t) => put(&mut body, &format!("${}$", t.trim())),
            Event::DisplayMath(t) => put(&mut body, &format!("\\[\n{}\n\\]\n\n", t.trim())),
            Event::SoftBreak => put(&mut body, "\n"),
            Event::HardBreak => put(&mut body, "\\\\\n"),
            Event::Rule => put(&mut body, "\\bigskip\\hrule\\bigskip\n\n"),
            Event::TaskListMarker(done) => put(
                &mut body,
                if done { "$\\boxtimes$ " } else { "$\\square$ " },
            ),
            Event::FootnoteReference(n) => put(&mut body, &format!("[{}]", tex_esc(&n))),
            _ => {}
        }
    }
    format!(
        "\\documentclass[11pt]{{article}}\n\\usepackage[utf8]{{inputenc}}\n\\usepackage[T1]{{fontenc}}\n\\usepackage{{amsmath,amssymb}}\n\\usepackage[normalem]{{ulem}}\n\\usepackage{{hyperref}}\n\\usepackage[margin=2.5cm]{{geometry}}\n\\title{{{}}}\n\\date{{}}\n\\begin{{document}}\n\\maketitle\n\n{}\n\\end{{document}}\n",
        tex_esc(title),
        body.trim_end()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{parse, Kind};

    const DOC: &str = "# Quarterly report\n\nRevenue grew **12 %** in the north & fell in the south; see [the plan](https://example.org/plan).\n\n## Figures\n\n| Region | Revenue | Change |\n|:--|--:|--:|\n| North | 1,200 | +12% |\n| South | 800 | =SUM(B2:B3) |\n\n## Steps\n\n1. Check the *ledger*\n2. Send the summary\n\n- loose end\n\nThe area is \\(\\pi r^2\\) and\n\n$$E = mc^2$$\n\n```python\nprint(\"50% of a_b\")\n```\n\n<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n";

    #[test]
    fn formats_by_name() {
        for f in Format::ALL {
            assert_eq!(Format::from_name(f.ext()), Some(f));
            assert!(!f.mime().is_empty() && !f.label().is_empty());
        }
        assert_eq!(Format::from_name(" DOCX "), Some(Format::Docx));
        assert_eq!(Format::from_name("exe"), None);
    }

    #[test]
    fn nothing_and_too_much_are_refused() {
        assert!(export("  \n", "t", Format::Md).is_err());
        let big = "x".repeat(MAX_MARKDOWN + 1);
        assert!(export(&big, "t", Format::Txt).unwrap_err().contains("MB"));
        assert!(export("no table here", "t", Format::Csv).is_err());
        assert!(export("no table here", "t", Format::Xlsx).is_err());
    }

    #[test]
    fn markdown_gets_a_title_only_when_it_has_none() {
        let with =
            String::from_utf8(export("Just a line.", "My chat", Format::Md).unwrap()).unwrap();
        assert!(with.starts_with("# My chat\n\nJust a line."));
        let keeps = String::from_utf8(export(DOC, "Other", Format::Md).unwrap()).unwrap();
        assert!(keeps.starts_with("# Quarterly report"));
    }

    #[test]
    fn html_is_a_page_that_runs_nothing() {
        let h = String::from_utf8(export(DOC, "", Format::Html).unwrap()).unwrap();
        assert!(h.starts_with("<!doctype html>") && h.contains("<title>Quarterly report</title>"));
        assert!(h.contains("<h1>Quarterly report</h1>") && h.contains("<strong>12 %</strong>"));
        assert!(
            h.contains("<table>") && h.contains("<td style=\"text-align: right\">1,200</td>"),
            "{h}"
        );
        assert!(h.contains("<a href=\"https://example.org/plan\">the plan</a>"));
        assert!(
            h.contains("<span class=\"math\">\\pi r^2</span>")
                && h.contains("<p class=\"math\">E = mc^2</p>"),
            "{h}"
        );
        assert!(
            h.contains("<ol>\n<li>Check the <em>ledger</em></li>")
                && h.contains("<pre><code class=\"language-python\">print(&quot;50% of a_b&quot;)"),
            "{h}"
        );
        // Raw HTML is shown, not run; a script link loses its target; nothing is fetched.
        assert!(!h.contains("<script>") && h.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(!h.contains("javascript:") && h.contains("<a href=\"#\">bad</a>"));
        assert!(!h.contains("src=") && !h.contains("<link") && !h.contains("<img"));
        let pic = to_html(
            "![a chart](https://example.org/x.png) and <b onclick=x>bold</b>",
            "t",
        );
        assert!(
            pic.contains("<em>[picture: a chart]</em>")
                && pic.contains("&lt;b onclick=x&gt;")
                && !pic.contains("example.org"),
            "{pic}"
        );
    }

    #[test]
    fn text_keeps_the_words_and_the_shape() {
        let t = String::from_utf8(export(DOC, "", Format::Txt).unwrap()).unwrap();
        assert!(
            t.starts_with("Quarterly report\n\nRevenue grew 12 % in the north & fell"),
            "{t}"
        );
        assert!(
            t.contains("Region\tRevenue\tChange\nNorth\t1,200\t+12%\n"),
            "{t}"
        );
        assert!(
            t.contains("1. Check the ledger\n2. Send the summary\n\n- loose end"),
            "{t}"
        );
        assert!(t.contains("print(\"50% of a_b\")") && !t.contains("**") && !t.contains("```"));
    }

    #[test]
    fn csv_holds_the_tables_and_no_formula() {
        let c = String::from_utf8(export(DOC, "", Format::Csv).unwrap()).unwrap();
        assert_eq!(
            c,
            "Region,Revenue,Change\r\nNorth,\"1,200\",+12%\r\nSouth,800,'=SUM(B2:B3)\r\n"
        );
        let two = "| a |\n|---|\n| 1 |\n\ntext\n\n| b | c |\n|---|---|\n| \"q\" | @x |\n";
        let c = String::from_utf8(export(two, "", Format::Csv).unwrap()).unwrap();
        assert_eq!(c, "a\r\n1\r\n\r\nb,c\r\n\"\"\"q\"\"\",'@x\r\n");
        assert_eq!(formula_safe("-12.5"), "-12.5");
        assert_eq!(formula_safe("-cmd|x"), "'-cmd|x");
    }

    #[test]
    fn latex_is_an_article_with_escaped_text_and_kept_maths() {
        let t = String::from_utf8(export(DOC, "", Format::Tex).unwrap()).unwrap();
        assert!(t.contains("\\title{Quarterly report}") && t.contains("\\maketitle"));
        assert!(
            !t.contains("\\section{Quarterly report}"),
            "the first heading is the title"
        );
        assert!(t.contains("\\subsection{Figures}") && t.contains("\\subsection{Steps}"));
        assert!(
            t.contains("Revenue grew \\textbf{12 \\%} in the north \\& fell"),
            "{t}"
        );
        assert!(t.contains("\\href{https://example.org/plan}{the plan}"));
        assert!(
            t.contains("\\begin{tabular}{lrr}") && t.contains("North & 1,200 & +12\\% \\\\"),
            "{t}"
        );
        assert!(
            t.contains("\\begin{enumerate}\n\\item Check the \\emph{ledger}"),
            "{t}"
        );
        assert!(
            t.contains("$\\pi r^2$") && t.contains("\\[\nE = mc^2\n\\]"),
            "{t}"
        );
        assert!(
            t.contains("\\begin{verbatim}\nprint(\"50% of a_b\")\n\\end{verbatim}"),
            "{t}"
        );
        assert!(
            t.contains("\\textless") || t.contains("<script>") || t.contains("script"),
            "raw HTML is text"
        );
        assert!(!t.contains("javascript:"));
        assert!(t.trim_end().ends_with("\\end{document}"));
        // Code that tries to close the verbatim block cannot.
        let sly = to_tex("```\n\\end{verbatim}\n\\input{/etc/passwd}\n```\n", "t");
        assert_eq!(sly.matches("\\end{verbatim}").count(), 1, "{sly}");
    }

    #[test]
    fn word_comes_back_through_the_reader() {
        let bytes = export(DOC, "", Format::Docx).unwrap();
        assert!(bytes.starts_with(b"PK"));
        let back = parse(Kind::Docx, &bytes).unwrap().text;
        for want in [
            "Quarterly report",
            "Revenue grew",
            "Figures",
            "North",
            "1,200",
            "Check the",
            "loose end",
        ] {
            assert!(back.contains(want), "{want:?} is missing from:\n{back}");
        }
    }

    #[test]
    fn slides_come_back_through_the_reader() {
        let bytes = export(DOC, "", Format::Pptx).unwrap();
        assert!(bytes.starts_with(b"PK"));
        let back = parse(Kind::Pptx, &bytes).unwrap().text;
        for want in ["Quarterly report", "Figures", "Steps"] {
            assert!(back.contains(want), "{want:?} is missing from:\n{back}");
        }
    }

    #[test]
    fn a_spreadsheet_holds_the_tables_and_comes_back() {
        let bytes = export(DOC, "", Format::Xlsx).unwrap();
        assert!(bytes.starts_with(b"PK"));
        let back = parse(Kind::Xlsx, &bytes).unwrap().text;
        // The sheet is named after the heading above the table; the header is the first row; a number is a number
        // (its thousands separator gone), a percentage and a would-be formula are the text that was written.
        assert!(
            back.contains("## Figures\n\n| Region | Revenue | Change |\n| --- | --- | --- |\n| North | 1200 | +12% |\n| South | 800 | =SUM(B2:B3) |"),
            "{back}"
        );
        let xml = {
            let mut z = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
            let mut s = String::new();
            std::io::Read::read_to_string(
                &mut z.by_name("xl/worksheets/sheet1.xml").unwrap(),
                &mut s,
            )
            .unwrap();
            s
        };
        assert!(xml.contains("<c r=\"B2\"><v>1200</v></c>"), "{xml}");
        assert!(
            xml.contains("t=\"inlineStr\"><is><t xml:space=\"preserve\">=SUM(B2:B3)</t>")
                && !xml.contains("<f>"),
            "{xml}"
        );
        // Two tables under one heading, and a heading Excel would refuse as a sheet name.
        let two = "## Q1/Q2: [draft]\n\n| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 007 |\n";
        let back = parse(Kind::Xlsx, &export(two, "", Format::Xlsx).unwrap())
            .unwrap()
            .text;
        assert!(
            back.contains("## Q1 Q2 draft\n") && back.contains("## Q1 Q2 draft 2\n"),
            "{back}"
        );
        assert!(
            back.contains("| 007 |"),
            "a number with a leading zero is a code, kept as text:\n{back}"
        );
        assert_eq!(as_number("1,234.50"), Some("1234.50".into()));
        for not in ["12%", "1,23", "1.2.3", "2026-11-22", "", "-", "1e5", "٣"] {
            assert_eq!(as_number(not), None, "{not}");
        }
        assert_eq!(
            (
                column_name(0),
                column_name(25),
                column_name(26),
                column_name(701),
                column_name(702)
            ),
            (
                "A".into(),
                "Z".into(),
                "AA".into(),
                "ZZ".into(),
                "AAA".into()
            )
        );
    }

    #[test]
    fn a_pdf_is_written_and_holds_the_text() {
        let bytes = export(DOC, "", Format::Pdf).unwrap();
        assert!(bytes.starts_with(b"%PDF-"));
        let back = parse(Kind::Pdf, &bytes).unwrap().text;
        assert!(
            back.contains("Quarterly report") && back.contains("Revenue grew"),
            "{back}"
        );
    }
}
