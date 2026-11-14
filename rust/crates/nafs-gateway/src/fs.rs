//! Phase 29: the context filesystem's routes (`docs/spec/context_fs_spec.md` §2). One `nafs://` namespace over the
//! knowledge bases (`nafs-kb`), and the memory files, skills and committed sessions (`nafs-memory`); layered reads
//! (L0, L1, L2), scoped `find`, rule-planned `search`, session commit with its memory diff, and compile.

use axum::extract::{Path, Query, State};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use nafs_context::assemble::{assemble, Budget, Item};
use nafs_context::compile::{self, DocInfo, Entity, Relation};
use nafs_context::memory::{event_decay, MemoryFile};
use nafs_context::skill::Skill;
use nafs_context::uri::{glob_match, Space, MEMORY_KINDS};
use nafs_context::Uri;
use nafs_types::NafsError;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::api::{ApiError, ApiResult, AppState};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/fs/ls", get(ls))
        .route("/v1/fs/tree", get(tree))
        .route("/v1/fs/read", get(read))
        .route("/v1/fs/stat", get(stat))
        .route("/v1/fs/glob", get(glob))
        .route("/v1/fs/grep", post(grep))
        .route("/v1/fs/find", post(find))
        .route("/v1/fs/search", post(search))
        .route("/v1/fs/assemble", post(assemble_route))
        .route("/v1/fs/write", put(write))
        .route("/v1/fs/rm", axum::routing::delete(rm))
        .route("/v1/sessions/{id}/commit", post(commit))
        .route("/v1/memory/diff", get(diff))
        .route("/v1/memory/diff/{id}/undo", post(undo))
        .route("/v1/compile", post(compile_route))
}

fn bad(m: impl Into<String>) -> ApiError {
    ApiError(NafsError::BadRequest(m.into()))
}

fn not_found(u: &Uri) -> ApiError {
    ApiError(NafsError::NotFound(u.to_string()))
}

/// A base the in-process index has not loaded yet: a passing state (503), not a bad request.
fn not_indexed() -> ApiError {
    ApiError(NafsError::BackendUnavailable(
        "this base is not in the in-process index yet; try again in a moment".into(),
    ))
}

fn kb_err(e: nafs_kb::KbError) -> ApiError {
    match e {
        nafs_kb::KbError::NotFound(m) => ApiError(NafsError::NotFound(m.to_string())),
        nafs_kb::KbError::BadInput(m) => ApiError(NafsError::BadRequest(m)),
        other => ApiError(NafsError::BackendUnavailable(other.to_string())),
    }
}

fn parse_uri(s: Option<&str>) -> ApiResult<Uri> {
    Uri::parse(s.unwrap_or("nafs://")).map_err(|e| bad(e.0))
}

fn kb_of(s: &AppState) -> ApiResult<Arc<nafs_kb::Kb>> {
    s.kb.clone().ok_or_else(|| {
        ApiError(NafsError::BackendUnavailable(
            "the knowledge base is turned off".into(),
        ))
    })
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n.saturating_sub(3)).collect::<String>() + "..."
    }
}

/// One entry of a listing.
fn entry(uri: &Uri, dir: bool, l0: Option<String>, extra: Value) -> Value {
    let mut v = json!({"uri": uri.to_string(), "name": uri.name(), "dir": dir});
    if let Some(a) = l0 {
        v["l0"] = json!(a);
    }
    if let Value::Object(m) = extra {
        for (k, x) in m {
            v[k] = x;
        }
    }
    v
}

// ------------------------------------------------------------------------------------------------- listing

async fn children(s: &AppState, u: &Uri, layers: bool) -> ApiResult<Vec<Value>> {
    if u.passage.is_some() {
        return Ok(Vec::new());
    }
    match (u.space, u.path.as_slice()) {
        (None, _) => Ok(Space::ALL
            .iter()
            .map(|sp| {
                entry(
                    &Uri::of(*sp, &[]),
                    true,
                    Some(sp.describe().into()),
                    json!({}),
                )
            })
            .collect()),
        (Some(Space::Kb), []) => {
            let kb = kb_of(s)?;
            let mut out = Vec::new();
            for b in kb.store().list_kbs().await.map_err(kb_err)? {
                let l0 = if layers {
                    match kb.index_of(&b.id).await.ok().flatten() {
                        Some(ix) => Some(ix.base_layers(kb.analyzer(), &b.name, &b.description).0),
                        None => Some(format!("{}: {} documents", b.name, b.documents)),
                    }
                } else {
                    None
                };
                out.push(entry(
                    &Uri::kb(&b.id),
                    true,
                    l0,
                    json!({"title": b.name, "documents": b.documents}),
                ));
            }
            Ok(out)
        }
        (Some(Space::Kb), [base]) => {
            let kb = kb_of(s)?;
            // An unknown base is a 404, not an empty folder.
            kb.store().get_kb(base).await.map_err(kb_err)?;
            let docs = kb.store().list_documents(base).await.map_err(kb_err)?;
            let ix = kb.index_of(base).await.ok().flatten();
            Ok(docs
                .iter()
                .map(|d| {
                    let l0 = if layers {
                        ix.as_ref()
                            .and_then(|ix| ix.doc_layers(kb.analyzer(), &d.id))
                            .map(|l| l.l0)
                    } else {
                        None
                    };
                    entry(
                        &Uri::doc(base, &d.id),
                        true,
                        l0,
                        json!({"title": d.title, "passages": d.n_chunks, "status": d.status, "kind": d.kind}),
                    )
                })
                .collect())
        }
        (Some(Space::Kb), [base, doc]) => {
            let kb = kb_of(s)?;
            let ix = kb
                .index_of(base)
                .await
                .map_err(kb_err)?
                .ok_or_else(not_indexed)?;
            if !ix.documents_list().iter().any(|(d, _)| d == doc) {
                return Err(not_found(u));
            }
            Ok(ix
                .passages_of(doc)
                .iter()
                .map(|h| {
                    let l0 = layers.then(|| passage_l0(&h.heading, &h.text));
                    let leaf = h.heading.rsplit(" > ").next().unwrap_or("").trim();
                    let title = match (leaf.is_empty(), h.page) {
                        (true, Some(p)) => format!("Passage {} (page {p})", h.ord + 1),
                        (true, None) => format!("Passage {}", h.ord + 1),
                        (false, Some(p)) => format!("{leaf} (page {p})"),
                        (false, None) => leaf.to_string(),
                    };
                    entry(
                        &Uri::passage(base, doc, h.ord),
                        false,
                        l0,
                        json!({"heading": h.heading, "page": h.page, "title": title}),
                    )
                })
                .collect())
        }
        (Some(Space::Memory), []) => {
            let files = s.store.memory_files(None)?;
            Ok(MEMORY_KINDS
                .iter()
                .map(|k| {
                    let n = files.iter().filter(|f| f.kind == *k).count();
                    entry(
                        &Uri::of(Space::Memory, &[k]),
                        true,
                        Some(kind_l0(k, n)),
                        json!({"files": n}),
                    )
                })
                .collect())
        }
        (Some(Space::Memory), [kind]) => {
            if !MEMORY_KINDS.contains(&kind.as_str()) {
                return Err(not_found(u));
            }
            Ok(s.store
                .memory_files(Some(kind))?
                .iter()
                .map(|f| {
                    entry(
                        &f.uri(),
                        false,
                        Some(f.abstract_line()),
                        json!({"title": f.title, "updated": f.updated}),
                    )
                })
                .collect())
        }
        (Some(Space::Skills), []) => Ok(s
            .store
            .skills()?
            .iter()
            .map(|k| {
                entry(
                    &Uri::skill(&k.name),
                    false,
                    Some(k.abstract_line()),
                    json!({"tags": k.tags}),
                )
            })
            .collect()),
        (Some(Space::Sessions), []) => Ok(s
            .store
            .sessions()?
            .iter()
            .map(|x| {
                entry(
                    &Uri::session(&x.id),
                    false,
                    Some(format!(
                        "{}: {} messages, committed {}",
                        x.title, x.messages, x.committed_at
                    )),
                    json!({"title": x.title}),
                )
            })
            .collect()),
        _ => Ok(Vec::new()),
    }
}

fn kind_l0(kind: &str, n: usize) -> String {
    let what = match kind {
        "profile" => "who you are",
        "preferences" => "how you like things done",
        "entities" => "people, places and things in your life",
        "events" => "what happened and when",
        _ => "things you asked to be remembered",
    };
    format!("{kind}: {what} ({n} file{})", if n == 1 { "" } else { "s" })
}

fn passage_l0(heading: &str, text: &str) -> String {
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let first = clip(
        body.split_whitespace().collect::<Vec<_>>().join(" ").trim(),
        160,
    );
    if heading.is_empty() {
        first
    } else {
        format!("{heading}: {first}")
    }
}

#[derive(Deserialize)]
struct UriQ {
    uri: Option<String>,
    #[serde(default)]
    layers: Option<u8>,
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    level: Option<u8>,
    #[serde(default)]
    pattern: Option<String>,
}

async fn ls(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    let items = children(&s, &u, q.layers.unwrap_or(1) != 0).await?;
    Ok(Json(
        json!({"uri": u.to_string(), "parent": u.parent().map(|p| p.to_string()), "items": items}),
    ))
}

async fn tree(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    let depth = q.depth.unwrap_or(2).min(4);
    let mut nodes = 0usize;
    let root = walk(&s, &u, depth, &mut nodes).await?;
    Ok(Json(
        json!({"uri": u.to_string(), "items": root, "nodes": nodes}),
    ))
}

/// A subtree to `depth`, at most 2,000 nodes (a passage level is never expanded: a document's passages are many).
fn walk<'a>(
    s: &'a AppState,
    u: &'a Uri,
    depth: usize,
    nodes: &'a mut usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApiResult<Vec<Value>>> + Send + 'a>> {
    Box::pin(async move {
        let mut items = children(s, u, true).await?;
        *nodes += items.len();
        if depth > 1 {
            for it in items.iter_mut() {
                if *nodes > 2000 || it["dir"] != json!(true) {
                    continue;
                }
                let child = Uri::parse(it["uri"].as_str().unwrap_or("")).map_err(|e| bad(e.0))?;
                if child.space == Some(Space::Kb) && child.path.len() == 2 {
                    continue;
                }
                it["children"] = json!(walk(s, &child, depth - 1, nodes).await?);
            }
        }
        Ok(items)
    })
}

// ------------------------------------------------------------------------------------------------- reading

struct Layers {
    l0: String,
    l1: String,
    l2: String,
    kind: &'static str,
    meta: Value,
}

async fn layers_of(s: &AppState, u: &Uri) -> ApiResult<Layers> {
    let listing = |items: &[Value]| -> String {
        items
            .iter()
            .map(|i| {
                format!(
                    "- {}",
                    i["l0"]
                        .as_str()
                        .unwrap_or_else(|| i["name"].as_str().unwrap_or(""))
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    match (u.space, u.path.as_slice(), u.passage) {
        (None, _, _) => {
            let items = children(s, u, true).await?;
            let l1 = format!("# nafs://\n\n{}", listing(&items));
            Ok(Layers {
                l0: "Everything the assistant knows: documents, memories, skills and sessions"
                    .into(),
                l2: l1.clone(),
                l1,
                kind: "root",
                meta: json!({}),
            })
        }
        (Some(sp), [], _) | (Some(sp @ Space::Memory), [_], _) => {
            let items = children(s, u, true).await?;
            let title = if u.path.is_empty() {
                sp.as_str().to_string()
            } else {
                u.path[0].clone()
            };
            let title = title[..1].to_uppercase() + &title[1..];
            let l1 = format!("# {title}\n\n{}", listing(&items));
            Ok(Layers {
                l0: if u.path.is_empty() {
                    sp.describe().to_string()
                } else {
                    kind_l0(&u.path[0], items.len())
                },
                l2: l1.clone(),
                l1,
                kind: "directory",
                meta: json!({"items": items.len()}),
            })
        }
        (Some(Space::Kb), [base], _) => {
            let kb = kb_of(s)?;
            let info = kb.store().get_kb(base).await.map_err(kb_err)?;
            let ix = kb
                .index_of(base)
                .await
                .map_err(kb_err)?
                .ok_or_else(not_indexed)?;
            let (l0, l1) = ix.base_layers(kb.analyzer(), &info.name, &info.description);
            Ok(Layers {
                l0,
                l2: l1.clone(),
                l1,
                kind: "base",
                meta: json!({"title": info.name, "documents": info.documents, "passages": info.chunks}),
            })
        }
        (Some(Space::Kb), [base, doc], None) => {
            let kb = kb_of(s)?;
            let ix = kb
                .index_of(base)
                .await
                .map_err(kb_err)?
                .ok_or_else(not_indexed)?;
            let l = ix
                .doc_layers(kb.analyzer(), doc)
                .ok_or_else(|| not_found(u))?;
            let passages = ix.passages_of(doc);
            let mut l2 = format!("# {}\n", l.title);
            for p in &passages {
                l2.push_str(&format!(
                    "\n<!-- {} -->\n{}\n",
                    Uri::passage(base, doc, p.ord),
                    p.text.trim()
                ));
            }
            Ok(Layers {
                l0: l.l0,
                l1: l.l1,
                l2,
                kind: "document",
                meta: json!({"title": l.title, "passages": l.passages, "terms": l.terms,
                    "sections": l.sections.iter().map(|(h, o)| json!({"heading": h, "uri": Uri::passage(base, doc, *o).to_string()})).collect::<Vec<_>>()}),
            })
        }
        (Some(Space::Kb), [base, doc], Some(ord)) => {
            let kb = kb_of(s)?;
            let ix = kb
                .index_of(base)
                .await
                .map_err(kb_err)?
                .ok_or_else(not_indexed)?;
            let h = ix.passage_at(doc, ord).ok_or_else(|| not_found(u))?;
            Ok(Layers {
                l0: passage_l0(&h.heading, &h.text),
                l1: h.text.clone(),
                l2: h.text.clone(),
                kind: "passage",
                meta: json!({"title": h.doc_title, "heading": h.heading, "page": h.page, "document": Uri::doc(base, doc).to_string()}),
            })
        }
        (Some(Space::Memory), [kind, key], _) => {
            let f = s
                .store
                .memory_file(kind, key)?
                .ok_or_else(|| not_found(u))?;
            let md = f.to_markdown();
            Ok(Layers {
                l0: f.abstract_line(),
                l1: md.clone(),
                l2: md,
                kind: "memory",
                meta: json!({"title": f.title, "updated": f.updated, "sources": f.sources}),
            })
        }
        (Some(Space::Skills), [name], _) => {
            let k = s.store.skill(name)?.ok_or_else(|| not_found(u))?;
            Ok(Layers {
                l0: k.abstract_line(),
                l1: format!(
                    "# {}\n\n{}\n\nTags: {}. Tools: {}.",
                    k.name,
                    k.description,
                    k.tags.join(", "),
                    k.tools.join(", ")
                ),
                l2: k.to_markdown(),
                kind: "skill",
                meta: json!({"tags": k.tags, "tools": k.tools}),
            })
        }
        (Some(Space::Sessions), [id], _) => {
            let f = s.store.session_file(id)?.ok_or_else(|| not_found(u))?;
            let learned = f
                .extracted
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|d| {
                            format!(
                                "- {} {} ({})",
                                d["op"].as_str().unwrap_or(""),
                                d["uri"].as_str().unwrap_or(""),
                                d["reason"].as_str().unwrap_or("")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            Ok(Layers {
                l0: format!(
                    "{}: {} messages, committed {}",
                    f.title, f.messages, f.committed_at
                ),
                l1: format!(
                    "# {}\n\n{} messages, committed {}.\n\n## Learned\n\n{}",
                    f.title,
                    f.messages,
                    f.committed_at,
                    if learned.is_empty() {
                        "Nothing.".into()
                    } else {
                        learned
                    }
                ),
                l2: f.transcript,
                kind: "session",
                meta: json!({"title": f.title, "messages": f.messages}),
            })
        }
        _ => Err(not_found(u)),
    }
}

async fn read(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    let level = q.level.unwrap_or(2).min(2);
    let l = layers_of(&s, &u).await?;
    let content = match level {
        0 => l.l0,
        1 => l.l1,
        _ => l.l2,
    };
    Ok(Json(
        json!({"uri": u.to_string(), "kind": l.kind, "level": level, "content": content, "meta": l.meta}),
    ))
}

async fn stat(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    let l = layers_of(&s, &u).await?;
    Ok(Json(
        json!({"uri": u.to_string(), "kind": l.kind, "l0": l.l0,
        "sizes": {"l0": l.l0.len(), "l1": l.l1.len(), "l2": l.l2.len()}, "meta": l.meta,
        "parent": u.parent().map(|p| p.to_string())}),
    ))
}

async fn glob(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let pattern = q
        .pattern
        .ok_or_else(|| bad("give a pattern, for example nafs://memory/**"))?;
    if !nafs_context::uri::glob_ok(&pattern) {
        return Err(bad(
            "a pattern is at most 256 characters with at most four **",
        ));
    }
    // Walk from the fixed part of the pattern (before the first wildcard).
    let fixed: String = pattern.split('*').next().unwrap_or("nafs://").to_string();
    let start = fixed
        .rsplit_once('/')
        .map_or("nafs://".to_string(), |(a, _)| a.to_string());
    let start = Uri::parse(if start.len() < "nafs://".len() {
        "nafs://"
    } else {
        &start
    })
    .unwrap_or_else(|_| Uri::root());
    let mut out = Vec::new();
    let mut queue = vec![start];
    let mut seen = 0;
    while let Some(u) = queue.pop() {
        for it in children(&s, &u, false).await.unwrap_or_default() {
            seen += 1;
            let uri = it["uri"].as_str().unwrap_or("").to_string();
            if glob_match(&pattern, &uri) {
                out.push(uri.clone());
            }
            let is_doc = uri.matches('/').count() >= 4 && uri.starts_with("nafs://kb/");
            if it["dir"] == json!(true)
                && (!is_doc || pattern.contains("#") || pattern.ends_with("**"))
            {
                if let Ok(c) = Uri::parse(&uri) {
                    queue.push(c);
                }
            }
        }
        if seen > 20_000 || out.len() > 2_000 {
            break;
        }
    }
    out.sort();
    Ok(Json(json!({"pattern": pattern, "uris": out})))
}

#[derive(Deserialize)]
struct GrepReq {
    pattern: String,
    scope: Option<String>,
    #[serde(default)]
    regex: bool,
    #[serde(default)]
    limit: Option<usize>,
}

async fn grep(State(s): State<AppState>, Json(b): Json<GrepReq>) -> ApiResult<Json<Value>> {
    if b.pattern.is_empty() || b.pattern.len() > 500 {
        return Err(bad("a pattern is 1 to 500 characters"));
    }
    let re = if b.regex {
        regex::RegexBuilder::new(&b.pattern)
            .size_limit(1 << 20)
            .build()
            .map_err(|e| bad(format!("bad pattern: {e}")))?
    } else {
        regex::RegexBuilder::new(&regex::escape(&b.pattern))
            .case_insensitive(true)
            .build()
            .map_err(|e| bad(e.to_string()))?
    };
    let scope = parse_uri(b.scope.as_deref())?;
    let limit = b.limit.unwrap_or(100).min(1000);
    let mut out = Vec::new();
    let (texts, cut) = texts_in(&s, &scope, GREP_MAX_BYTES).await?;
    for (uri, text) in texts {
        for (n, line) in text.lines().enumerate() {
            if re.is_match(line) {
                out.push(json!({"uri": uri, "line": n + 1, "text": clip(line.trim(), 300)}));
                if out.len() >= limit {
                    return Ok(Json(json!({"matches": out, "truncated": true})));
                }
            }
        }
    }
    Ok(Json(json!({"matches": out, "truncated": cut})))
}

/// Text a grep reads at most (a scope of every base and every session is cut there and says so).
const GREP_MAX_BYTES: usize = 32 << 20;

/// Every readable text under a scope: (URI, text), passages for documents, at most `max_bytes` of text; the flag
/// says whether the limit cut the list.
async fn texts_in(
    s: &AppState,
    scope: &Uri,
    max_bytes: usize,
) -> ApiResult<(Vec<(String, String)>, bool)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut bytes = 0usize;
    let mut full = |out: &mut Vec<(String, String)>, uri: String, text: String| -> bool {
        bytes += text.len();
        out.push((uri, text));
        bytes >= max_bytes
    };
    let want = |sp: Space| scope.space.is_none() || scope.space == Some(sp);
    if want(Space::Kb) {
        if let Some(kb) = s.kb.clone() {
            let bases: Vec<String> = match scope.path.first() {
                Some(b) if scope.space == Some(Space::Kb) => vec![b.clone()],
                _ => kb
                    .store()
                    .list_kbs()
                    .await
                    .map_err(kb_err)?
                    .into_iter()
                    .map(|b| b.id)
                    .collect(),
            };
            for base in bases {
                let Some(ix) = kb.index_of(&base).await.ok().flatten() else {
                    continue;
                };
                for (doc, _) in ix.documents_list() {
                    let du = Uri::doc(&base, &doc);
                    if !scope.contains(&du) && !du.contains(scope) {
                        continue;
                    }
                    for h in ix.passages_of(&doc) {
                        let pu = Uri::passage(&base, &doc, h.ord);
                        if scope.contains(&pu) && full(&mut out, pu.to_string(), h.text) {
                            return Ok((out, true));
                        }
                    }
                }
            }
        }
    }
    if want(Space::Memory) {
        for f in s.store.memory_files(None)? {
            if scope.contains(&f.uri())
                && full(
                    &mut out,
                    f.uri().to_string(),
                    format!("{}\n{}", f.title, f.body),
                )
            {
                return Ok((out, true));
            }
        }
    }
    if want(Space::Skills) {
        for k in s.store.skills()? {
            let u = Uri::skill(&k.name);
            if scope.contains(&u) && full(&mut out, u.to_string(), k.to_markdown()) {
                return Ok((out, true));
            }
        }
    }
    if want(Space::Sessions) {
        for x in s.store.sessions()? {
            let u = Uri::session(&x.id);
            if scope.contains(&u) {
                if let Some(f) = s.store.session_file(&x.id)? {
                    if full(&mut out, u.to_string(), f.transcript) {
                        return Ok((out, true));
                    }
                }
            }
        }
    }
    Ok((out, false))
}

// ------------------------------------------------------------------------------------------------- find / search

#[derive(Deserialize)]
struct FindReq {
    query: String,
    scope: Option<String>,
    #[serde(default)]
    k: Option<usize>,
}

/// Word overlap score of a text against a question, by the knowledge analyzer's stems (memories and skills are few,
/// so a scan is enough; the score is BM25-like saturation of the matched share).
fn overlap(an: Option<&nafs_kb::index::Analyzer>, q: &str, text: &str) -> f64 {
    let terms = |t: &str| -> HashSet<String> {
        match an {
            Some(a) => a.query(t).into_iter().collect(),
            None => t
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() > 2)
                .map(str::to_string)
                .collect(),
        }
    };
    let qt = terms(q);
    if qt.is_empty() {
        return 0.0;
    }
    let tt = terms(text);
    qt.iter().filter(|t| tt.contains(*t)).count() as f64 / qt.len() as f64
}

/// A hit of `find` or `search`.
fn hit(
    uri: String,
    level: u8,
    score: f64,
    title: &str,
    l0: String,
    text: String,
    category: &str,
) -> Value {
    json!({"uri": uri, "level": level, "score": score, "title": title, "l0": l0, "text": text, "category": category})
}

async fn find_in(s: &AppState, query: &str, scope: &Uri, k: usize) -> ApiResult<Vec<Value>> {
    let mut lists: Vec<Vec<Value>> = Vec::new();
    let an = s.kb.as_ref().map(|kb| kb.analyzer());
    let want = |sp: Space| scope.space.is_none() || scope.space == Some(sp);
    if want(Space::Kb) {
        if let Some(kb) = s.kb.clone() {
            let bases: Vec<(String, String)> = match scope.path.first() {
                Some(b) if scope.space == Some(Space::Kb) => vec![(b.clone(), String::new())],
                _ => kb
                    .store()
                    .list_kbs()
                    .await
                    .map_err(kb_err)?
                    .into_iter()
                    .map(|b| (b.id, b.name))
                    .collect(),
            };
            for (base, _) in bases {
                let docs = scope.path.get(1).map(|d| HashSet::from([d.clone()]));
                // A base that cannot be searched just now is left out of a search of everything; a search of that
                // base alone says why.
                let r = match kb.find_scoped(&base, query, k, docs).await {
                    Ok(r) => r,
                    Err(_) if scope.space.is_none() || scope.path.is_empty() => continue,
                    Err(e) => return Err(kb_err(e)),
                };
                let mut list: Vec<Value> = r
                    .results
                    .iter()
                    // A scope of one passage keeps only that passage.
                    .filter(|h| scope.contains(&Uri::passage(&base, &h.doc_id, h.ord)))
                    .map(|h| {
                        let u = Uri::passage(&base, &h.doc_id, h.ord);
                        hit(
                            u.to_string(),
                            2,
                            h.score,
                            &h.doc_title,
                            passage_l0(&h.heading, &h.text),
                            h.text.clone(),
                            "passage",
                        )
                    })
                    .collect();
                // Documents whose overview answers the question: level 1 hits beside the passages.
                if scope.path.len() < 2 {
                    if let Some(ix) = kb.index_of(&base).await.ok().flatten() {
                        let mut docs: Vec<(f64, Value)> = ix
                            .documents_list()
                            .iter()
                            .filter_map(|(d, _)| ix.doc_layers(kb.analyzer(), d))
                            .map(|l| {
                                let sc = overlap(an, query, &format!("{}\n{}", l.title, l.l1));
                                (
                                    sc,
                                    hit(
                                        Uri::doc(&base, &l.doc_id).to_string(),
                                        1,
                                        sc,
                                        &l.title,
                                        l.l0.clone(),
                                        l.l1.clone(),
                                        "document",
                                    ),
                                )
                            })
                            .filter(|(sc, _)| *sc >= 0.5)
                            .collect();
                        docs.sort_by(|a, b| b.0.total_cmp(&a.0));
                        list.extend(docs.into_iter().take(2).map(|x| x.1));
                    }
                }
                lists.push(list);
            }
        }
    }
    if want(Space::Memory) {
        let now = chrono::Utc::now();
        let mut list: Vec<(f64, Value)> = s
            .store
            .memory_files(
                scope
                    .path
                    .first()
                    .map(String::as_str)
                    .filter(|_| scope.space == Some(Space::Memory)),
            )?
            .into_iter()
            .filter(|f| scope.contains(&f.uri()))
            .map(|f| {
                let mut sc = overlap(an, query, &format!("{}\n{}", f.title, f.body));
                if f.kind == "events" {
                    sc *= event_decay((now - f.updated).num_seconds() as f64 / 86_400.0);
                }
                (
                    sc,
                    hit(
                        f.uri().to_string(),
                        2,
                        sc,
                        &f.title,
                        f.abstract_line(),
                        f.body.clone(),
                        "memory",
                    ),
                )
            })
            .filter(|(sc, _)| *sc > 0.0)
            .collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        lists.push(list.into_iter().take(k).map(|x| x.1).collect());
    }
    if want(Space::Skills) {
        let mut list: Vec<(f64, Value)> = s
            .store
            .skills()?
            .into_iter()
            .filter(|x| scope.contains(&Uri::skill(&x.name)))
            .map(|x| {
                let sc = overlap(an, query, &x.abstract_line());
                (
                    sc,
                    hit(
                        Uri::skill(&x.name).to_string(),
                        0,
                        sc,
                        &x.name,
                        x.abstract_line(),
                        x.abstract_line(),
                        "skill",
                    ),
                )
            })
            .filter(|(sc, _)| *sc >= 0.34)
            .collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        lists.push(list.into_iter().take(k).map(|x| x.1).collect());
    }
    // Reciprocal-rank merge of the spaces' lists (their scores are not on one scale).
    let mut score: HashMap<String, (f64, Value)> = HashMap::new();
    for l in lists {
        for (rank, v) in l.into_iter().enumerate() {
            let key = v["uri"].as_str().unwrap_or("").to_string();
            let e = score.entry(key).or_insert((0.0, v));
            e.0 += 1.0 / (60.0 + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<(f64, Value)> = score.into_values().collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(out
        .into_iter()
        .take(k)
        .map(|(r, mut v)| {
            v["rank_score"] = json!(r);
            v
        })
        .collect())
}

async fn find(State(s): State<AppState>, Json(b): Json<FindReq>) -> ApiResult<Json<Value>> {
    if b.query.trim().is_empty() || b.query.chars().count() > 2000 {
        return Err(bad("a query is 1 to 2,000 characters"));
    }
    let scope = parse_uri(b.scope.as_deref())?;
    let t0 = std::time::Instant::now();
    let hits = find_in(&s, &b.query, &scope, b.k.unwrap_or(8).clamp(1, 50)).await?;
    Ok(Json(
        json!({"query": b.query, "scope": scope.to_string(), "hits": hits, "ms": t0.elapsed().as_millis() as u64}),
    ))
}

#[derive(Deserialize)]
struct SearchReq {
    query: String,
    conversation: Option<Uuid>,
    #[serde(default)]
    k: Option<usize>,
}

const PERSONAL: &[&str] = &[
    "i", "me", "my", "mine", "myself", "we", "our", "prefer", "remember", "remind",
];
const TASK: &[&str] = &[
    "how do i",
    "how to",
    "how should",
    "steps",
    "procedure",
    "checklist",
    "workflow",
    "write a",
    "draft a",
    "prepare a",
];

/// The rule planner of `search` (spec §4): which scopes, and the question as searched.
async fn plan(
    s: &AppState,
    query: &str,
    conversation: Option<Uuid>,
) -> ApiResult<(String, Vec<(Uri, &'static str)>)> {
    let lower = query.to_lowercase();
    let words: HashSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    // A question that leans on the turn before it is searched with that turn (Phase 26 rule).
    let previous = match conversation {
        Some(c) => s.store.history(c).ok().and_then(|h| {
            h.into_iter()
                .rev()
                .find(|m| m.role == "user")
                .map(|m| m.content)
        }),
        None => None,
    };
    let (text, joined) = nafs_core::continuity::search_text(query, previous.as_deref());
    let mut scopes: Vec<(Uri, &'static str)> = Vec::new();
    if let Some(kb) = s.kb.clone() {
        for b in kb.store().list_kbs().await.map_err(kb_err)? {
            let Some(ix) = kb.index_of(&b.id).await.ok().flatten() else {
                scopes.push((Uri::kb(&b.id), "the whole base"));
                continue;
            };
            // A document named by its title. A rule that also took a document's distinctive terms as naming it
            // fired on 11 of 198 questions and picked the right document 5 times; scoped and fused with the whole
            // base it lowered MRR (quoted 0.870 to 0.841), so only an exact title scopes (benchmarks/kb_phase29).
            let named = ix
                .documents_list()
                .into_iter()
                .find(|(_, title)| {
                    title.chars().count() >= 8 && lower.contains(&title.to_lowercase())
                })
                .map(|(d, _)| d);
            if let Some(d) = named {
                scopes.push((Uri::doc(&b.id, &d), "the question names this document"));
            }
            scopes.push((Uri::kb(&b.id), "the whole base"));
        }
    }
    if PERSONAL.iter().any(|w| words.contains(w)) {
        scopes.push((Uri::of(Space::Memory, &[]), "a personal question"));
    }
    if TASK.iter().any(|p| lower.contains(p)) {
        scopes.push((Uri::of(Space::Skills, &[]), "asks how to do a task"));
    }
    let text = if joined { text } else { query.to_string() };
    Ok((text, scopes))
}

async fn search(State(s): State<AppState>, Json(b): Json<SearchReq>) -> ApiResult<Json<Value>> {
    if b.query.trim().is_empty() || b.query.chars().count() > 2000 {
        return Err(bad("a query is 1 to 2,000 characters"));
    }
    let t0 = std::time::Instant::now();
    let k = b.k.unwrap_or(8).clamp(1, 50);
    let (text, scopes) = plan(&s, &b.query, b.conversation).await?;
    let mut lists = Vec::new();
    for (scope, _) in &scopes {
        lists.push(find_in(&s, &text, scope, k).await?);
    }
    let mut score: HashMap<String, (f64, Value)> = HashMap::new();
    for l in lists {
        for (rank, v) in l.into_iter().enumerate() {
            let key = v["uri"].as_str().unwrap_or("").to_string();
            let e = score.entry(key).or_insert((0.0, v));
            e.0 += 1.0 / (60.0 + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<(f64, Value)> = score.into_values().collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    let hits: Vec<Value> = out.into_iter().take(k).map(|x| x.1).collect();
    Ok(Json(json!({
        "query": b.query,
        "searched_as": text,
        "plan": scopes.iter().map(|(u, why)| json!({"scope": u.to_string(), "why": why})).collect::<Vec<_>>(),
        "hits": hits,
        "ms": t0.elapsed().as_millis() as u64,
    })))
}

#[derive(Deserialize)]
struct AssembleReq {
    query: String,
    scope: Option<String>,
    #[serde(default)]
    chars: Option<usize>,
}

/// The budgeted context block (spec §4) for a question: what the chat prompt would carry, at which level.
async fn assemble_route(
    State(s): State<AppState>,
    Json(b): Json<AssembleReq>,
) -> ApiResult<Json<Value>> {
    if b.query.trim().is_empty() || b.query.chars().count() > 2000 {
        return Err(bad("a query is 1 to 2,000 characters"));
    }
    let scope = parse_uri(b.scope.as_deref())?;
    let hits = find_in(&s, &b.query, &scope, 12).await?;
    let items: Vec<Item> = hits
        .iter()
        .map(|h| Item {
            uri: h["uri"].as_str().unwrap_or("").into(),
            category: h["category"].as_str().unwrap_or("passage").into(),
            score: h["rank_score"].as_f64().unwrap_or(0.0),
            levels: vec![
                h["l0"].as_str().map(str::to_string),
                h["text"].as_str().map(str::to_string),
            ],
        })
        .collect();
    let budget = Budget {
        chars: b.chars.unwrap_or(4800).clamp(500, 20_000),
        quotas: HashMap::from([
            ("passage".to_string(), 6),
            ("document".to_string(), 2),
            ("memory".to_string(), 3),
            ("skill".to_string(), 1),
        ]),
        max_items: 10,
    };
    let chosen = assemble(&items, &budget);
    let block: Vec<Value> = chosen
        .iter()
        .map(|c| json!({"uri": items[c.item].uri, "level": if c.level == 0 { "abstract" } else { "full" },
            "text": items[c.item].levels[c.level]}))
        .collect();
    let chars: usize = chosen
        .iter()
        .map(|c| {
            items[c.item].levels[c.level]
                .as_ref()
                .map_or(0, String::len)
        })
        .sum();
    Ok(Json(
        json!({"query": b.query, "items": block, "chars": chars, "budget": budget.chars}),
    ))
}

// ------------------------------------------------------------------------------------------------- writing

#[derive(Deserialize)]
struct WriteReq {
    content: String,
}

async fn write(
    State(s): State<AppState>,
    Query(q): Query<UriQ>,
    Json(w): Json<WriteReq>,
) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    let body = w.content;
    match (u.space, u.path.as_slice()) {
        (Some(Space::Memory), [kind, key]) => {
            if nafs_context::uri::slug(key) != *key {
                return Err(bad("a memory key is lower-case letters, digits and dashes"));
            }
            let f = MemoryFile::from_markdown(kind, key, &body, chrono::Utc::now()).map_err(bad)?;
            let d = s.store.write_memory_file(&f)?;
            Ok(Json(json!({"uri": u.to_string(), "diff": d})))
        }
        (Some(Space::Skills), [name]) => {
            let k = Skill::parse(name, &body).map_err(bad)?;
            s.store.put_skill(&k)?;
            Ok(Json(json!({"uri": u.to_string(), "skill": k})))
        }
        _ => Err(bad("only memory files (nafs://memory/<kind>/<key>) and skills (nafs://skills/<name>) are written here")),
    }
}

async fn rm(State(s): State<AppState>, Query(q): Query<UriQ>) -> ApiResult<Json<Value>> {
    let u = parse_uri(q.uri.as_deref())?;
    match (u.space, u.path.as_slice()) {
        (Some(Space::Memory), [kind, key]) => Ok(Json(json!({"removed": u.to_string(), "diff": s.store.delete_memory_file(kind, key)?}))),
        (Some(Space::Skills), [name]) => {
            s.store.delete_skill(name)?;
            Ok(Json(json!({"removed": u.to_string()})))
        }
        (Some(Space::Sessions), [id]) => {
            s.store.delete_session_file(id)?;
            Ok(Json(json!({"removed": u.to_string()})))
        }
        _ => Err(bad("only memory files, skills and committed sessions are removed here; documents are removed on the Knowledge page")),
    }
}

async fn commit(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    // The store refuses a room's chat (other people's words are not this person's memory).
    let r = s.store.commit_session(id, &[])?;
    Ok(Json(
        json!({"session": Uri::session(&r.session).to_string(), "messages_read": r.messages_read, "diff": r.diff}),
    ))
}

#[derive(Deserialize)]
struct DiffQ {
    limit: Option<usize>,
}

async fn diff(State(s): State<AppState>, Query(q): Query<DiffQ>) -> ApiResult<Json<Value>> {
    Ok(Json(
        json!({"diff": s.store.memory_diff(q.limit.unwrap_or(100).min(1000))?}),
    ))
}

async fn undo(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(json!({"undone": s.store.undo_memory_diff(&id)?})))
}

// ------------------------------------------------------------------------------------------------- compile

#[derive(Deserialize)]
struct CompileReq {
    from: String,
    to: String,
}

async fn compile_route(
    State(s): State<AppState>,
    Json(b): Json<CompileReq>,
) -> ApiResult<Json<Value>> {
    let u = parse_uri(Some(&b.from))?;
    let (Some(Space::Kb), [base]) = (u.space, u.path.as_slice()) else {
        return Err(bad("compile reads one knowledge base: nafs://kb/<base>"));
    };
    let kb = kb_of(&s)?;
    let info = kb.store().get_kb(base).await.map_err(kb_err)?;
    let ix = kb
        .index_of(base)
        .await
        .map_err(kb_err)?
        .ok_or_else(not_indexed)?;
    let (_, overview) = ix.base_layers(kb.analyzer(), &info.name, &info.description);
    let docs: Vec<DocInfo> = ix
        .documents_list()
        .iter()
        .filter_map(|(d, _)| ix.doc_layers(kb.analyzer(), d))
        .map(|l| DocInfo {
            uri: Uri::doc(base, &l.doc_id).to_string(),
            title: l.title.clone(),
            l0: l.l0.clone(),
            l1: l.l1.clone(),
            sections: l
                .sections
                .iter()
                .map(|(h, o)| (h.clone(), Uri::passage(base, &l.doc_id, *o).to_string()))
                .collect(),
            terms: l.terms.clone(),
        })
        .collect();
    let files = match b.to.as_str() {
        "wiki" => compile::wiki(&info.name, &overview, &docs),
        "report" => compile::report(&info.name, &overview, &docs),
        "graph" => {
            let view = ix.graph_view(None, 60);
            let empty = Vec::new();
            let nodes = view["nodes"].as_array().unwrap_or(&empty);
            let edges = view["edges"].as_array().unwrap_or(&empty);
            let name_of: HashMap<&str, &str> = nodes
                .iter()
                .filter_map(|n| Some((n["id"].as_str()?, n["name"].as_str()?)))
                .collect();
            // Evidence is a passage id; as a URI it can be read back.
            let uri_of = |chunk: &str| -> Option<String> {
                let h = ix.passage(ix.number_of(chunk)?);
                Some(Uri::passage(base, &h.doc_id, h.ord).to_string())
            };
            let relations: Vec<Relation> = edges
                .iter()
                .filter_map(|l| {
                    Some(Relation {
                        from: name_of.get(l["source"].as_str()?)?.to_string(),
                        relation: l["predicate"].as_str().unwrap_or("appears with").into(),
                        to: name_of.get(l["target"].as_str()?)?.to_string(),
                        evidence: l["evidence"]
                            .as_str()
                            .and_then(uri_of)
                            .into_iter()
                            .collect(),
                        quote: String::new(),
                    })
                })
                .collect();
            let entities: Vec<Entity> = nodes
                .iter()
                .map(|n| {
                    let name = n["name"].as_str().unwrap_or("").to_string();
                    let mut mentions: Vec<String> = relations
                        .iter()
                        .filter(|r| r.from == name || r.to == name)
                        .flat_map(|r| r.evidence.clone())
                        .collect();
                    mentions.sort();
                    mentions.dedup();
                    Entity {
                        kind: n["type"].as_str().unwrap_or("term").into(),
                        name,
                        mentions,
                    }
                })
                .collect();
            compile::graph(&entities, &relations)
        }
        other => {
            return Err(bad(format!(
                "unknown target \"{other}\" (wiki, graph or report)"
            )))
        }
    };
    Ok(Json(
        json!({"from": u.to_string(), "to": b.to, "files": files}),
    ))
}
