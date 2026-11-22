//! Documents in and out for a single question (Phase 35).
//!
//! `POST /v1/parse` reads one file with the knowledge base's parsers and returns its text, without adding it to a
//! base: the chat page puts that text beside a question. `POST /v1/export` turns Markdown (an answer, a conversation)
//! into a file. Neither needs the knowledge store, so both work when Knowledge is turned off.

use axum::{
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

use crate::api::AppState;
use nafs_kb::export::Format;
use nafs_kb::parse::Kind;

/// The most text one parsed file returns (characters). The page cuts further to what one question can carry.
const MAX_PARSED_CHARS: usize = 400_000;

fn fail(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "code": code, "message": message.into() })),
    )
        .into_response()
}

/// Read one uploaded file (multipart part `file`) and return `{filename, kind, title, text, chars, pages, cut}`.
pub async fn parse_upload(State(s): State<AppState>, mut form: Multipart) -> Response {
    let k = &s.cfg.knowledge;
    let max_bytes = k.max_upload_mb.max(1) * 1024 * 1024;
    let field = loop {
        match form.next_field().await {
            Ok(Some(f)) if f.file_name().is_some() => break f,
            Ok(Some(_)) => continue,
            Ok(None) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    "no file in the upload (send it as a multipart part named \"file\")",
                )
            }
            Err(e) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    format!("the upload could not be read: {e}"),
                )
            }
        }
    };
    let filename = nafs_kb::display_name(field.file_name().unwrap_or("file"));
    let Some(kind) = Kind::from_filename(&filename) else {
        return fail(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported",
            format!("{filename}: only PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV, LaTeX, HTML, Markdown and text files can be read"),
        );
    };
    // Read in pieces and stop at the limit: a file far over it is refused without being held whole.
    let mut buf: Vec<u8> = Vec::new();
    let mut field = field;
    loop {
        match field.chunk().await {
            Ok(Some(c)) => {
                if buf.len() + c.len() > max_bytes {
                    return fail(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "too_large",
                        format!(
                            "{filename} is larger than the limit of {} MB",
                            max_bytes / 1_048_576
                        ),
                    );
                }
                buf.extend_from_slice(&c);
            }
            Ok(None) => break,
            Err(e) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    format!("the file could not be received: {e}"),
                )
            }
        }
    }
    if buf.is_empty() {
        return fail(
            StatusCode::BAD_REQUEST,
            "bad_request",
            format!("{filename} is empty"),
        );
    }
    let limit = Duration::from_secs(k.parse_timeout_s.max(1));
    let parsed = tokio::time::timeout(
        limit,
        tokio::task::spawn_blocking(move || nafs_kb::parse::parse(kind, &buf)),
    )
    .await;
    let parsed = match parsed {
        Err(_) => {
            return fail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unreadable",
                format!("{filename} took longer than {} s to read", limit.as_secs()),
            )
        }
        Ok(Err(_)) => {
            return fail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unreadable",
                format!("{filename} could not be read"),
            )
        }
        Ok(Ok(Err(e))) => {
            return fail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unreadable",
                format!("{filename}: {e}"),
            )
        }
        Ok(Ok(Ok(p))) => p,
    };
    let chars = parsed.text.chars().count();
    let text: String = parsed.text.chars().take(MAX_PARSED_CHARS).collect();
    Json(json!({
        "filename": filename, "kind": kind.as_str(), "title": parsed.title, "text": text,
        "chars": chars, "pages": parsed.pages, "cut": chars > MAX_PARSED_CHARS,
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct ExportReq {
    markdown: String,
    format: String,
    #[serde(default)]
    title: String,
}

/// A file name from a title: its letters, digits, dashes and spaces, at most 60 characters.
fn file_stem(title: &str) -> String {
    let s: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join("-");
    let s: String = s.chars().filter(char::is_ascii).take(60).collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "nafs-aql".into()
    } else {
        s
    }
}

/// `POST /v1/export {markdown, format, title?}` → the file, as an attachment.
pub async fn export_doc(Json(b): Json<ExportReq>) -> Response {
    let Some(format) = Format::from_name(&b.format) else {
        return fail(
            StatusCode::BAD_REQUEST,
            "bad_request",
            format!(
                "format must be one of: {}",
                Format::ALL.map(|f| f.ext()).join(", ")
            ),
        );
    };
    let (md, title) = (b.markdown, b.title);
    let name = format!("{}.{}", file_stem(&title), format.ext());
    let made = tokio::time::timeout(
        Duration::from_secs(60),
        tokio::task::spawn_blocking(move || nafs_kb::export::export(&md, &title, format)),
    )
    .await;
    match made {
        Ok(Ok(Ok(bytes))) => (
            [
                (header::CONTENT_TYPE, format.mime().to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{name}\""),
                ),
                (header::CACHE_CONTROL, "no-store".to_string()),
            ],
            bytes,
        )
            .into_response(),
        Ok(Ok(Err(e))) => fail(StatusCode::UNPROCESSABLE_ENTITY, "not_written", e),
        _ => fail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "not_written",
            format!("the {} file could not be written", format.label()),
        ),
    }
}

/// `GET /v1/export/formats`: what can be read and what can be written, for the page's menus.
pub async fn formats() -> Json<Value> {
    Json(json!({
        "read": Kind::ALL.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "write": Format::ALL.iter().map(|f| json!({ "format": f.ext(), "label": f.label() })).collect::<Vec<_>>(),
        "max_markdown_bytes": nafs_kb::export::MAX_MARKDOWN,
    }))
}

#[cfg(test)]
mod tests {
    use super::file_stem;

    #[test]
    fn file_names_are_plain() {
        assert_eq!(
            file_stem("Quarterly report: Q3/2026"),
            "Quarterly-report-Q3-2026"
        );
        assert_eq!(file_stem("../../etc/passwd"), "etc-passwd");
        assert_eq!(file_stem("  \"; rm -rf  "), "rm--rf");
        assert_eq!(file_stem("日本語"), "nafs-aql");
        assert!(file_stem(&"a".repeat(200)).len() <= 60);
    }
}
