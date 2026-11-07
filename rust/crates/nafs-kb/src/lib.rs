//! nafs-kb: the Knowledge Base (Phase 20, decision 014). Documents are parsed (`parse`), cut into passages (`chunk`),
//! and stored in SurrealDB (`store`) by a background ingestion worker (`Kb`), one document at a time so ingestion never
//! competes with itself for the four cores. Phase 1: full-text search; vectors (Phase 2) and the graph (Phase 3) are
//! added to the same store.

pub mod chunk;
pub mod embed;
pub mod parse;
pub mod store;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, Mutex};

pub use chunk::{chunk, Chunk, ChunkOpts};
pub use embed::{EmbedConfig, Embedder, QUERY_INSTRUCTION};
pub use parse::{parse, Kind, Parsed};
pub use store::{Document, Hit, KbSummary, NewDocument, Store, StoreConfig, StoredChunk};

#[derive(Debug, thiserror::Error)]
pub enum KbError {
    #[error("{0}")]
    Unavailable(String),
    #[error("{0} not found")]
    NotFound(&'static str),
    #[error("{0}")]
    BadInput(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    TooLarge(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("knowledge store error: {0}")]
    Store(String),
    #[error("file error: {0}")]
    Io(String),
}

impl KbError {
    /// The stable error code of IF-K01.
    pub fn code(&self) -> &'static str {
        match self {
            KbError::Unavailable(_) => "kb_unavailable",
            KbError::NotFound(_) => "not_found",
            KbError::BadInput(_) => "bad_request",
            KbError::Conflict(_) => "conflict",
            KbError::TooLarge(_) => "too_large",
            KbError::Unsupported(_) => "unsupported_type",
            KbError::Store(_) | KbError::Io(_) => "internal",
        }
    }
}

/// Settings of the knowledge base (`[knowledge]`, IF-K02).
#[derive(Debug, Clone)]
pub struct KbConfig {
    pub store: StoreConfig,
    pub files_dir: PathBuf,
    pub max_upload_bytes: usize,
    pub max_files_per_request: usize,
    pub chunk: ChunkOpts,
    pub search_k: usize,
    /// Phase 2: passage vectors from `nafs-smiley-server --embedding`; `None` keeps full-text search only.
    pub embed: Option<EmbedConfig>,
    /// Knowledge bases with fewer vectors than this are searched exactly (a scan); larger ones through HNSW.
    pub exact_search_below: u64,
}

/// How a search ranks passages (FR-K07).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// BM25 over the words.
    Lexical,
    /// Cosine similarity of meaning vectors.
    Vector,
    /// Both, fused by reciprocal rank.
    Hybrid,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<SearchMode> {
        match s {
            "lexical" => Some(SearchMode::Lexical),
            "vector" => Some(SearchMode::Vector),
            "hybrid" => Some(SearchMode::Hybrid),
            _ => None,
        }
    }
}

/// A search's results and how it actually ran (a hybrid search falls back to lexical without vectors).
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub results: Vec<Hit>,
    pub mode: SearchMode,
    pub took_ms: u64,
    /// Time spent embedding the query.
    pub embed_ms: u64,
    /// Why the requested mode was not used, when it was not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

/// Ingestion queue counters for `/v1/kb/status`.
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct QueueStatus {
    pub pending: usize,
    pub running: usize,
}

struct Job {
    kb: String,
    doc: String,
    /// Only compute missing vectors (the passages are already stored).
    vectors_only: bool,
}

/// The knowledge-base service: the store plus the ingestion worker.
pub struct Kb {
    store: Store,
    cfg: KbConfig,
    tx: mpsc::UnboundedSender<Job>,
    pending: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,
    schema_ready: AtomicBool,
    bootstrap_lock: Mutex<()>,
    embedder: Option<Embedder>,
}

impl Kb {
    /// Create the service and start its worker (needs a Tokio runtime). Nothing touches SurrealDB until first use.
    pub fn start(cfg: KbConfig) -> Arc<Kb> {
        let (tx, rx) = mpsc::unbounded_channel();
        let embedder = cfg.embed.clone().map(Embedder::new);
        let kb = Arc::new(Kb {
            store: Store::new(cfg.store.clone()),
            cfg,
            tx,
            pending: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicUsize::new(0)),
            schema_ready: AtomicBool::new(false),
            bootstrap_lock: Mutex::new(()),
            embedder,
        });
        tokio::spawn(worker(Arc::downgrade(&kb), rx));
        let k = kb.clone();
        tokio::spawn(async move {
            // Documents a previous run left half-done are read again once the store is reachable.
            if k.ready().await.is_ok() {
                if let Ok(docs) = k.store.unfinished_documents().await {
                    for d in docs {
                        k.enqueue(&d.kb, &d.id, false);
                    }
                }
                // Passages stored before the embedder ran (Phase 1 documents, or the embedder was down) get vectors.
                if k.embedder.is_some() {
                    let _ = k.reindex(None).await;
                }
            }
        });
        kb
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn config(&self) -> &KbConfig {
        &self.cfg
    }

    pub fn queue(&self) -> QueueStatus {
        QueueStatus {
            pending: self.pending.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }

    /// Make sure the schema exists; cheap after the first success, retried after a failure.
    pub async fn ready(&self) -> Result<(), KbError> {
        if self.schema_ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let _g = self.bootstrap_lock.lock().await;
        if !self.schema_ready.load(Ordering::Acquire) {
            self.store.bootstrap().await?;
            if let Some(e) = &self.embedder {
                self.store.ensure_vector_index(e.cfg.dims).await?;
            }
            self.schema_ready.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn enqueue(&self, kb: &str, doc: &str, vectors_only: bool) {
        self.pending.fetch_add(1, Ordering::Relaxed);
        if self
            .tx
            .send(Job {
                kb: kb.into(),
                doc: doc.into(),
                vectors_only,
            })
            .is_err()
        {
            self.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn kb_dir(&self, kb: &str) -> PathBuf {
        self.cfg.files_dir.join(safe_key(kb))
    }

    /// Accept one uploaded file (FR-K02..K04): validate, store the original, create the record, queue ingestion.
    pub async fn upload(
        &self,
        kb: &str,
        filename: &str,
        bytes: &[u8],
    ) -> Result<Document, KbError> {
        self.ready().await?;
        self.store.get_kb(kb).await?;
        let filename = display_name(filename);
        if bytes.len() > self.cfg.max_upload_bytes {
            return Err(KbError::TooLarge(format!(
                "{filename} is {:.1} MB; the limit is {} MB",
                bytes.len() as f64 / 1_048_576.0,
                self.cfg.max_upload_bytes / 1_048_576
            )));
        }
        let kind = Kind::from_filename(&filename).ok_or_else(|| {
            KbError::Unsupported(format!("{filename}: only PDF, Word (.docx), PowerPoint (.pptx), LaTeX, HTML, Markdown and text files are accepted"))
        })?;
        kind.sniff(bytes)
            .map_err(|e| KbError::Unsupported(format!("{filename}: {e}")))?;
        let sha = hex(&Sha256::digest(bytes));
        if let Some(mut d) = self.store.find_by_sha(kb, &sha).await? {
            d.duplicate = true;
            return Ok(d);
        }
        let dir = self.kb_dir(kb);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| KbError::Io(e.to_string()))?;
        // The stored name is ours (SR-K02): the content hash plus the kind's extension.
        let path = dir.join(format!("{}.{}", &sha[..32], kind.extension()));
        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| KbError::Io(e.to_string()))?;
        let title = filename
            .rsplit_once('.')
            .map(|(s, _)| s)
            .unwrap_or(&filename)
            .to_string();
        let doc = self
            .store
            .create_document(
                kb,
                &NewDocument {
                    title,
                    filename: filename.clone(),
                    kind: kind.as_str().into(),
                    bytes: bytes.len() as u64,
                    sha256: sha,
                    path: path.to_string_lossy().into_owned(),
                },
            )
            .await?;
        self.enqueue(kb, &doc.id, false);
        Ok(doc)
    }

    /// Read a document again from its stored original.
    pub async fn reingest(&self, kb: &str, doc: &str) -> Result<Document, KbError> {
        self.ready().await?;
        let d = self.store.get_document(kb, doc).await?;
        if matches!(
            d.status.as_str(),
            "queued" | "parsing" | "chunking" | "storing" | "embedding"
        ) {
            return Err(KbError::Conflict(
                "the document is already being read".into(),
            ));
        }
        self.store.set_status(doc, "queued", None).await?;
        self.enqueue(kb, doc, false);
        self.store.get_document(kb, doc).await
    }

    pub async fn delete_document(&self, kb: &str, doc: &str) -> Result<u64, KbError> {
        self.ready().await?;
        let d = self.store.get_document(kb, doc).await?;
        let n = self.store.delete_document(kb, doc).await?;
        // Another document of the same base may share the stored file (same content); it cannot, by FR-K04.
        let _ = tokio::fs::remove_file(&d.path).await;
        Ok(n)
    }

    pub async fn delete_kb(&self, kb: &str) -> Result<(u64, u64), KbError> {
        self.ready().await?;
        let r = self.store.delete_kb(kb).await?;
        let _ = tokio::fs::remove_dir_all(self.kb_dir(kb)).await;
        Ok(r)
    }

    async fn ingest(&self, job: &Job) -> Result<(), KbError> {
        self.ready().await?;
        let d = match self.store.get_document(&job.kb, &job.doc).await {
            Ok(d) => d,
            Err(KbError::NotFound(_)) => return Ok(()), // deleted while queued
            Err(e) => return Err(e),
        };
        self.store.set_status(&d.id, "parsing", None).await?;
        let started = Instant::now();
        let bytes = tokio::fs::read(&d.path)
            .await
            .map_err(|e| KbError::Io(format!("the stored file is missing: {e}")))?;
        let kind = Kind::parse_str(&d.kind).ok_or_else(|| KbError::Unsupported(d.kind.clone()))?;
        let opts = self.cfg.chunk;
        // Parsing and chunking are CPU work: off the async threads.
        let (parsed, chunks) = tokio::task::spawn_blocking(move || {
            let p = parse(kind, &bytes)?;
            let c = chunk(&p, opts);
            Ok::<_, String>((p, c))
        })
        .await
        .map_err(|e| KbError::BadInput(format!("the reader stopped unexpectedly: {e}")))?
        .map_err(KbError::BadInput)?;
        let parse_ms = started.elapsed().as_millis() as u64;
        self.store.set_status(&d.id, "storing", None).await?;
        let title = parsed
            .title
            .clone()
            .filter(|t| t.chars().count() <= 200)
            .unwrap_or(d.title.clone());
        self.store
            .store_chunks(&job.kb, &d.id, &title, parsed.pages, parse_ms, &chunks)
            .await?;
        let vectors = self.embed_document(&job.kb, &d.id, &title).await;
        self.store
            .set_status(&d.id, "ready", vectors.err().as_deref())
            .await?;
        tracing::info!(doc = %d.id, kind = %d.kind, chunks = chunks.len(), parse_ms, "knowledge document ready");
        Ok(())
    }

    /// Give a document's passages that lack one a vector (state `embedding`). Returns a plain-language note when the
    /// embedder is off or failed: the document stays searchable by its words, and `reindex` finishes it later.
    async fn embed_document(&self, kb: &str, doc: &str, title: &str) -> Result<u64, String> {
        let Some(e) = &self.embedder else {
            return Ok(0);
        };
        let todo = self
            .store
            .chunks_without_vectors(doc)
            .await
            .map_err(|x| x.to_string())?;
        if todo.is_empty() {
            return self
                .store
                .finish_vectors(kb, doc, &e.cfg.model, e.cfg.dims)
                .await
                .map_err(|x| x.to_string());
        }
        self.store
            .set_status(doc, "embedding", None)
            .await
            .map_err(|x| x.to_string())?;
        let t0 = Instant::now();
        for batch in todo.chunks(e.cfg.batch.max(1)) {
            let texts: Vec<String> = batch
                .iter()
                .map(|(_, h, t)| embed::passage_text(title, h, t))
                .collect();
            let vecs = match e.embed(&texts, None).await {
                Ok(v) => v,
                Err(err) => {
                    let _ = self
                        .store
                        .finish_vectors(kb, doc, &e.cfg.model, e.cfg.dims)
                        .await;
                    return Err(format!("Searching by meaning is not available for this document yet ({err}); it is searchable by its words and will be indexed when the embedding server is back."));
                }
            };
            let rows: Vec<(String, Vec<f32>)> = batch
                .iter()
                .map(|(id, _, _)| id.clone())
                .zip(vecs)
                .collect();
            self.store
                .set_vectors(&rows)
                .await
                .map_err(|x| x.to_string())?;
            // Progress for the page: the count grows batch by batch.
            let _ = self
                .store
                .finish_vectors(kb, doc, &e.cfg.model, e.cfg.dims)
                .await;
        }
        let n = self
            .store
            .finish_vectors(kb, doc, &e.cfg.model, e.cfg.dims)
            .await
            .map_err(|x| x.to_string())?;
        tracing::info!(
            doc,
            passages = todo.len(),
            ms = t0.elapsed().as_millis() as u64,
            "knowledge passages embedded"
        );
        Ok(n)
    }

    /// Queue every ready document (of one knowledge base, or all) whose passages are not all embedded.
    pub async fn reindex(&self, kb: Option<&str>) -> Result<usize, KbError> {
        self.ready().await?;
        if self.embedder.is_none() {
            return Err(KbError::Unavailable(
                "searching by meaning is turned off ([knowledge] embed_url is empty)".into(),
            ));
        }
        let docs = self.store.documents_missing_vectors(kb).await?;
        for d in &docs {
            self.enqueue(&d.kb, &d.id, true);
        }
        Ok(docs.len())
    }

    /// Whether vectors can be made and searched now (the embedding server answers).
    pub async fn embedder_status(&self) -> Option<(String, bool)> {
        let e = self.embedder.as_ref()?;
        Some((e.cfg.model.clone(), e.health().await))
    }

    /// Search a knowledge base (FR-K07). Hybrid fuses BM25 and vector ranks (RRF, k = 60); a vector or hybrid search
    /// without vectors or without the embedding server falls back to words, and says so.
    pub async fn search(
        &self,
        kb: &str,
        query: &str,
        k: usize,
        mode: SearchMode,
    ) -> Result<SearchResult, KbError> {
        self.ready().await?;
        let t0 = Instant::now();
        let k = k.clamp(1, 100);
        let mut fallback = None;
        let mut embed_ms = 0;
        let qv = if mode == SearchMode::Lexical || query.trim().is_empty() {
            None
        } else if let Some(e) = &self.embedder {
            let n = self.store.vector_count(kb).await?;
            if n == 0 {
                fallback = Some("this knowledge base has no meaning index yet".to_string());
                None
            } else {
                let te = Instant::now();
                match e.embed(&[query.to_string()], Some(QUERY_INSTRUCTION)).await {
                    Ok(mut v) => {
                        embed_ms = te.elapsed().as_millis() as u64;
                        Some((v.remove(0), n < self.cfg.exact_search_below))
                    }
                    Err(err) => {
                        fallback = Some(format!("the embedding server is not available ({err})"));
                        None
                    }
                }
            }
        } else {
            fallback = Some("searching by meaning is turned off".to_string());
            None
        };
        let (results, used) = match (mode, qv) {
            (SearchMode::Vector, Some((v, exact))) => (
                self.store.search_vector(kb, &v, k, exact).await?,
                SearchMode::Vector,
            ),
            (SearchMode::Hybrid, Some((v, exact))) => {
                let depth = (k * 3).max(20);
                let lex = self.store.search_lexical(kb, query, depth).await?;
                let vec = self.store.search_vector(kb, &v, depth, exact).await?;
                (fuse(lex, vec, k), SearchMode::Hybrid)
            }
            _ => (
                self.store.search_lexical(kb, query, k).await?,
                SearchMode::Lexical,
            ),
        };
        Ok(SearchResult {
            results,
            mode: used,
            took_ms: t0.elapsed().as_millis() as u64,
            embed_ms,
            fallback,
        })
    }
}

async fn worker(kb: std::sync::Weak<Kb>, mut rx: mpsc::UnboundedReceiver<Job>) {
    while let Some(job) = rx.recv().await {
        let Some(kb) = kb.upgrade() else { return };
        kb.pending.fetch_sub(1, Ordering::Relaxed);
        kb.running.fetch_add(1, Ordering::Relaxed);
        if job.vectors_only {
            if let Ok(d) = kb.store.get_document(&job.kb, &job.doc).await {
                let note = kb.embed_document(&job.kb, &d.id, &d.title).await.err();
                let _ = kb.store.set_status(&d.id, "ready", note.as_deref()).await;
            }
        } else if let Err(e) = kb.ingest(&job).await {
            tracing::warn!(doc = %job.doc, error = %e, "knowledge ingestion failed");
            let _ = kb
                .store
                .set_status(&job.doc, "failed", Some(&e.to_string()))
                .await;
        }
        kb.running.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Merge lexical and vector hits by reciprocal rank; each hit keeps both scores, `score` becomes the fused one.
pub fn fuse(lex: Vec<Hit>, vec: Vec<Hit>, k: usize) -> Vec<Hit> {
    let lists = [
        lex.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
        vec.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
    ];
    let fused = embed::rrf(&lists, 60.0);
    let mut out: Vec<Hit> = Vec::with_capacity(k);
    for (id, score) in fused.into_iter().take(k) {
        let l = lex.iter().find(|h| h.chunk_id == id);
        let v = vec.iter().find(|h| h.chunk_id == id);
        let mut h = l
            .or(v)
            .cloned()
            .expect("a fused id comes from one of the lists");
        h.lexical_score = l.map(|x| x.score);
        h.vector_score = v.map(|x| x.score);
        h.score = score;
        h.rank = out.len() + 1;
        out.push(h);
    }
    out
}

/// The client's file name, reduced to a display name (no directories, no control characters, at most 200 chars).
pub fn display_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let clean: String = base.chars().filter(|c| !c.is_control()).take(200).collect();
    let clean = clean.trim().trim_start_matches('.').to_string();
    if clean.is_empty() {
        "untitled".into()
    } else {
        clean
    }
}

fn safe_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_drop_paths_and_controls() {
        assert_eq!(display_name("../../etc/passwd"), "passwd");
        assert_eq!(display_name("C:\\Users\\a\\report.pdf"), "report.pdf");
        assert_eq!(display_name("..hidden\u{7}.md"), "hidden.md");
        assert_eq!(display_name(""), "untitled");
        assert_eq!(safe_key("ab-c_1/../x"), "ab-c_1x");
    }
}
