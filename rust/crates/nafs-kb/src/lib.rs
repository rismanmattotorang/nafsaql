//! nafs-kb: the Knowledge Base (Phase 20, decision 014). Documents are parsed (`parse`), cut into passages (`chunk`),
//! and stored in SurrealDB (`store`) by a background ingestion worker (`Kb`), one document at a time so ingestion never
//! competes with itself for the four cores. Phase 1: full-text search; vectors (Phase 2) and the graph (Phase 3) are
//! added to the same store.

pub mod chunk;
pub mod dataset;
pub mod embed;
pub mod graph;
pub mod index;
pub mod parse;
pub mod store;
pub mod vcache;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, Mutex};

pub use chunk::{chunk, Chunk, ChunkOpts};
pub use embed::{EmbedConfig, Embedder, QUERY_INSTRUCTION};
pub use graph::{Extractor, GraphConfig};
pub use index::IndexConfig;
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
    /// Too much is already queued (SR-K11): try again shortly.
    #[error("{0}")]
    Busy(String),
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
            KbError::Busy(_) => "busy",
            KbError::Store(_) | KbError::Io(_) => "internal",
        }
    }
}

/// Bounds on what one upload, one document and one base may cost (Phase 8, SR-K11, NFR-K07).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Documents one base may hold.
    pub max_documents_per_base: u64,
    /// Passages one document may make; a larger one is refused with a request to split it.
    pub max_chunks_per_document: usize,
    /// Jobs (reads, embeddings, graph builds) waiting at once; further uploads are told to try again.
    pub max_queued_jobs: usize,
    /// Seconds a parser may take before the document is given up on.
    pub parse_timeout_s: u64,
    /// Seconds a words search may take. At 50,000 passages a query with a very common term scores most of them (several
    /// seconds, measured), so a hybrid search gives up on the words and answers by meaning, and says so.
    pub search_timeout_s: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_documents_per_base: 2_000,
            max_chunks_per_document: 6_000,
            max_queued_jobs: 500,
            parse_timeout_s: 180,
            search_timeout_s: 10,
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
    /// Phase 3: entity and relation extraction by the quick model; `None` turns the graph off.
    pub graph: Option<GraphConfig>,
    /// FR-K50: how much the graph's ranking counts against the search's when the two are fused (1.0 = an equal arm).
    pub graph_weight: f64,
    /// FR-K63: how much the vector arm counts against the words (1.0) when a hybrid search fuses the two by rank.
    pub vector_weight: f64,
    /// NFR-K07: the most memory (MB) the in-memory copies of large bases' vectors may use together; a base that would not
    /// fit is searched through the store's index instead.
    pub vector_cache_mb: u64,
    /// FR-K56: relations kept per entity at the second hop, and the per-hop decay of a path's score.
    pub graph_beam: usize,
    pub graph_decay: f64,
    /// Phase 25 (FR-K67): the in-process index a search uses instead of round trips to the store.
    pub index: IndexConfig,
    pub limits: Limits,
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
    /// What answered: `index` (in process, FR-K67) or `store`.
    pub engine: &'static str,
}

/// Which path a search takes (FR-K69). `Store` is the path every search took before Phase 25; it stays for bases
/// that do not fit in memory and for comparing the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// The in-process index when the base has one, else the store.
    Auto,
    Store,
}

impl Engine {
    pub fn parse(s: &str) -> Option<Engine> {
        match s {
            "auto" | "index" => Some(Engine::Auto),
            "store" => Some(Engine::Store),
            _ => None,
        }
    }
}

/// Ingestion queue counters for `/v1/kb/status`.
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct QueueStatus {
    pub pending: usize,
    pub running: usize,
}

/// What a queued job does to its document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKind {
    /// Parse, chunk and store (the fast lane); the vectors are queued as a `Vectors` job after it.
    Ingest,
    /// Only compute missing vectors (the passages are already stored).
    Vectors,
    /// Read the passages for the knowledge graph (Phase 3).
    Graph,
}

struct Job {
    kb: String,
    doc: String,
    kind: JobKind,
}

/// The knowledge-base service: the store plus the ingestion worker.
pub struct Kb {
    store: Store,
    cfg: KbConfig,
    /// Two lanes (NFR-K06): parsing and storing a document takes milliseconds and goes first, so a new upload is
    /// searchable by its words at once and never waits behind a long embedding or graph job of another base; the
    /// model work (vectors, graph) queues on the slow lane, one job at a time. Found by the live sweep: with one
    /// queue a 187-byte upload sat "queued" for five minutes behind a 517-passage re-index.
    tx_fast: mpsc::UnboundedSender<Job>,
    tx_slow: mpsc::UnboundedSender<Job>,
    pending: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,
    schema_ready: AtomicBool,
    bootstrap_lock: Mutex<()>,
    embedder: Option<Embedder>,
    extractor: Option<Extractor>,
    /// In-memory copies of large bases' vectors (NFR-K07).
    vcache: vcache::VectorCache,
    /// The in-process indexes (Phase 25, FR-K67).
    index: Arc<index::Index>,
    /// Vectors of recent questions (FR-K69): a repeated question, and the graph search after the plain one, embed once.
    questions: std::sync::Mutex<HashMap<String, (Vec<f32>, u64)>>,
    /// Phase 26: per base, the embedding model its vectors were made by when that is not the one running now.
    foreign: std::sync::Mutex<HashMap<String, Option<String>>>,
}

/// Questions whose vectors are remembered.
const QUESTION_CACHE: usize = 256;

impl Kb {
    /// Create the service and start its worker (needs a Tokio runtime). Nothing touches SurrealDB until first use.
    pub fn start(cfg: KbConfig) -> Arc<Kb> {
        let (tx_fast, rx_fast) = mpsc::unbounded_channel();
        let (tx_slow, rx_slow) = mpsc::unbounded_channel();
        let vector_cache_mb = cfg.vector_cache_mb;
        let index_memory_mb = cfg.index.memory_mb;
        let embedder = cfg.embed.clone().map(Embedder::new);
        let extractor = cfg.graph.clone().map(Extractor::new);
        let kb = Arc::new(Kb {
            store: Store::new(cfg.store.clone()),
            cfg,
            tx_fast,
            tx_slow,
            pending: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicUsize::new(0)),
            schema_ready: AtomicBool::new(false),
            bootstrap_lock: Mutex::new(()),
            embedder,
            extractor,
            vcache: vcache::VectorCache::new(vector_cache_mb),
            index: index::Index::new(index_memory_mb),
            questions: std::sync::Mutex::new(HashMap::new()),
            foreign: std::sync::Mutex::new(HashMap::new()),
        });
        tokio::spawn(worker(Arc::downgrade(&kb), rx_fast));
        tokio::spawn(worker(Arc::downgrade(&kb), rx_slow));
        let k = kb.clone();
        tokio::spawn(async move {
            // Documents a previous run left half-done are read again once the store is reachable.
            if k.ready().await.is_ok() {
                // Bases whose deletion a restart interrupted are finished in the background.
                if let Ok(ids) = k.store.deleting_kbs().await {
                    for id in ids {
                        let store = k.store.clone();
                        tokio::spawn(async move {
                            let _ = store.finish_delete_kb(&id).await;
                        });
                    }
                }
                if let Ok(docs) = k.store.unfinished_documents().await {
                    for d in docs {
                        k.enqueue(&d.kb, &d.id, JobKind::Ingest);
                    }
                }
                // Passages stored before the embedder ran (Phase 1 documents, or the embedder was down) get vectors.
                if k.embedder.is_some() {
                    let _ = k.reindex(None).await;
                }
                // The indexes are read now rather than by the first question (FR-K67).
                if let Ok(bases) = k.store.list_kbs().await {
                    for b in bases {
                        let _ = k.index.get(&k.store, &b.id, k.dims()).await;
                    }
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

    fn enqueue(&self, kb: &str, doc: &str, kind: JobKind) {
        self.pending.fetch_add(1, Ordering::Relaxed);
        let lane = match kind {
            JobKind::Ingest => &self.tx_fast,
            JobKind::Vectors | JobKind::Graph => &self.tx_slow,
        };
        if lane
            .send(Job {
                kb: kb.into(),
                doc: doc.into(),
                kind,
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
        let base = self.store.get_kb(kb).await?;
        let filename = display_name(filename);
        if self.pending.load(Ordering::Relaxed) >= self.cfg.limits.max_queued_jobs {
            return Err(KbError::Busy(format!(
                "{} jobs are already waiting; try again in a few minutes",
                self.cfg.limits.max_queued_jobs
            )));
        }
        if base.documents >= self.cfg.limits.max_documents_per_base {
            return Err(KbError::TooLarge(format!(
                "{filename}: this knowledge base already holds {} documents, which is the limit; start another one",
                base.documents
            )));
        }
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
        self.enqueue(kb, &doc.id, JobKind::Ingest);
        Ok(doc)
    }

    /// Read a document again from its stored original.
    pub async fn reingest(&self, kb: &str, doc: &str) -> Result<Document, KbError> {
        self.ready().await?;
        let d = self.store.get_document(kb, doc).await?;
        if matches!(
            d.status.as_str(),
            "queued" | "parsing" | "chunking" | "storing" | "embedding" | "extracting"
        ) {
            return Err(KbError::Conflict(
                "the document is already being read".into(),
            ));
        }
        self.store.set_status(doc, "queued", None).await?;
        self.enqueue(kb, doc, JobKind::Ingest);
        self.store.get_document(kb, doc).await
    }

    pub async fn delete_document(&self, kb: &str, doc: &str) -> Result<u64, KbError> {
        self.ready().await?;
        let d = self.store.get_document(kb, doc).await?;
        self.vcache.drop_kb(kb);
        let n = self.store.delete_document(kb, doc).await?;
        self.index.remove_doc(kb, doc);
        if self.store.prune_graph(kb).await.is_ok() {
            let _ = self.recompute_communities(kb).await;
        }
        // Another document of the same base may share the stored file (same content); it cannot, by FR-K04.
        let _ = tokio::fs::remove_file(&d.path).await;
        Ok(n)
    }

    pub async fn delete_kb(&self, kb: &str) -> Result<(u64, u64), KbError> {
        self.ready().await?;
        self.vcache.drop_kb(kb);
        let base = self.store.get_kb(kb).await?;
        // Out of use at once; its rows go in the background (under an hour at 50,000 passages), and a restart resumes it.
        self.store.mark_kb_deleting(kb).await?;
        self.index.remove_kb(kb);
        self.foreign.lock().expect("embedder identity").remove(kb);
        let _ = tokio::fs::remove_dir_all(self.kb_dir(kb)).await;
        let (store, id) = (self.store.clone(), kb.to_string());
        tokio::spawn(async move {
            if let Err(e) = store.finish_delete_kb(&id).await {
                tracing::warn!(kb = %id, error = %e, "deleting a knowledge base stopped; a restart resumes it");
            }
        });
        Ok((base.documents, base.chunks))
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
        // A parser that never returns (a crafted PDF) must not hold the lane for ever. The blocking thread cannot be
        // stopped, but the document is given up on and the queue moves on.
        let limit = Duration::from_secs(self.cfg.limits.parse_timeout_s.max(1));
        let work = tokio::task::spawn_blocking(move || {
            let p = parse(kind, &bytes)?;
            let c = chunk(&p, opts);
            Ok::<_, String>((p, c))
        });
        let (parsed, chunks) = tokio::time::timeout(limit, work)
            .await
            .map_err(|_| {
                KbError::BadInput(format!(
                    "the file took more than {} s to read and was given up on",
                    limit.as_secs()
                ))
            })?
            .map_err(|e| KbError::BadInput(format!("the reader stopped unexpectedly: {e}")))?
            .map_err(KbError::BadInput)?;
        if chunks.len() > self.cfg.limits.max_chunks_per_document {
            return Err(KbError::BadInput(format!(
                "the file makes {} passages, more than the limit of {}; split it into smaller files",
                chunks.len(),
                self.cfg.limits.max_chunks_per_document
            )));
        }
        let parse_ms = started.elapsed().as_millis() as u64;
        self.store.set_status(&d.id, "storing", None).await?;
        let title = parsed
            .title
            .clone()
            .filter(|t| t.chars().count() <= 200)
            .unwrap_or(d.title.clone());
        // FR-K73: a re-read keeps the vector of every passage whose embedded text did not change.
        let kept = if self.embedder.is_some() && title == d.title && d.n_embedded > 0 {
            self.store.embedded_texts(&d.id).await.unwrap_or_default()
        } else {
            Vec::new()
        };
        self.store
            .store_chunks(&job.kb, &d.id, &title, parsed.pages, parse_ms, &chunks)
            .await?;
        // The base changed: a copy of its vectors made before no longer describes it.
        self.vcache.drop_kb(&job.kb);
        let reused = self.reuse_vectors(&job.kb, &d.id, kept).await;
        self.index
            .refresh_doc(&self.store, &job.kb, &d.id, self.dims())
            .await;
        // A re-read replaces the passages, so what the graph held about the old ones goes.
        if self.store.prune_graph(&job.kb).await.is_ok() {
            let _ = self.recompute_communities(&job.kb).await;
        }
        // Ready by its words now; the vectors follow on the slow lane (the page shows "Words only" until then).
        self.store.set_status(&d.id, "ready", None).await?;
        tracing::info!(doc = %d.id, kind = %d.kind, chunks = chunks.len(), reused, parse_ms, "knowledge document ready");
        if self.embedder.is_some() {
            self.enqueue(&job.kb, &d.id, JobKind::Vectors);
        }
        Ok(())
    }

    /// FR-K73: give the new passages of a re-read document the vectors their text already had. `kept` is what the
    /// document held before: (embedded text, vector). Returns how many vectors were reused.
    async fn reuse_vectors(&self, kb: &str, doc: &str, kept: Vec<(String, Vec<f32>)>) -> usize {
        let Some(e) = &self.embedder else { return 0 };
        if kept.is_empty() {
            return 0;
        }
        let Ok(todo) = self.store.chunks_without_vectors(doc).await else {
            return 0;
        };
        // Passages with the same text share a vector, so a lookup, not a removal.
        let by_text: HashMap<String, Vec<f32>> = kept.into_iter().collect();
        let rows: Vec<(String, Vec<f32>)> = todo
            .iter()
            .filter_map(|c| {
                // The title is the same (checked by the caller) and a new passage has no context line yet.
                let v = by_text.get(&format!("{}\u{1f}{}", c.heading, c.text))?;
                (v.len() == e.cfg.dims).then(|| (c.id.clone(), v.clone()))
            })
            .collect();
        for batch in rows.chunks(64) {
            if self.store.set_vectors(batch).await.is_err() {
                return 0;
            }
        }
        if !rows.is_empty() {
            let _ = self
                .store
                .finish_vectors(kb, doc, &e.cfg.model, e.cfg.dims)
                .await;
        }
        rows.len()
    }

    fn dims(&self) -> usize {
        self.embedder.as_ref().map_or(0, |e| e.cfg.dims)
    }

    /// Phase 26: the model a base's vectors were made by, when it is not the one running now. Vectors of two models
    /// cannot be compared, even at the same width, so such a base is searched by its words until its meaning index is
    /// rebuilt. The store is asked once per base; a rebuild or a deletion forgets the answer.
    async fn foreign_embedder(&self, kb: &str) -> Option<String> {
        let ours = &self.embedder.as_ref()?.cfg.model;
        if let Some(known) = self.foreign.lock().expect("embedder identity").get(kb) {
            return known.clone();
        }
        // A store that cannot be asked is not remembered: the next search asks again.
        let other = foreign_model(self.store.embedder_of(kb).await.ok()?, ours);
        self.foreign
            .lock()
            .expect("embedder identity")
            .insert(kb.to_string(), other.clone());
        other
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
        // Vectors are about to be written for this base: a copy made before is stale from here on.
        self.vcache.drop_kb(kb);
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
                .map(|c| embed::passage_text(title, &c.heading, &c.context, &c.text))
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
            let rows: Vec<(String, Vec<f32>)> =
                batch.iter().map(|c| c.id.clone()).zip(vecs).collect();
            self.store
                .set_vectors(&rows)
                .await
                .map_err(|x| x.to_string())?;
            if !self.index.apply_vectors(kb, doc, &rows) {
                self.index
                    .refresh_doc(&self.store, kb, doc, e.cfg.dims)
                    .await;
            }
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
        self.reindex_fresh(kb, false).await
    }

    /// The same, and with `fresh` every passage of the base is embedded again (FR-K53: the vectors then carry the
    /// context lines the graph pass wrote) and the words index is filled for passages stored before Phase 21
    /// (FR-K51). `fresh` needs one base.
    pub async fn reindex_fresh(&self, kb: Option<&str>, fresh: bool) -> Result<usize, KbError> {
        self.ready().await?;
        if self.embedder.is_none() {
            return Err(KbError::Unavailable(
                "searching by meaning is turned off ([knowledge] embed_url is empty)".into(),
            ));
        }
        if let Some(base) = kb {
            self.store.backfill_search_text(base).await?;
            if fresh {
                self.vcache.drop_kb(base);
                self.store.clear_vectors(base).await?;
                self.index.drop_kb(base);
                self.foreign.lock().expect("embedder identity").remove(base);
            }
        }
        let docs = self.store.documents_missing_vectors(kb).await?;
        for d in &docs {
            self.enqueue(&d.kb, &d.id, JobKind::Vectors);
        }
        Ok(docs.len())
    }

    /// Queue every ready document of a knowledge base whose passages are not all read for the graph (FR-K30). This
    /// is a long job on the quick model (about 12-25 s per passage here), so it runs only when asked.
    pub async fn build_graph(&self, kb: &str, fresh: bool) -> Result<usize, KbError> {
        self.ready().await?;
        if self.extractor.is_none() {
            return Err(KbError::Unavailable(
                "the knowledge graph is turned off ([knowledge] graph_llm_url is empty)".into(),
            ));
        }
        if fresh {
            // Every passage is read again (after a change to the extraction, or to start over).
            self.store.reset_graph(kb).await?;
        }
        let docs = self.store.documents_missing_graph(kb).await?;
        for d in &docs {
            self.store.set_status(&d.id, "queued", None).await?;
            self.enqueue(&d.kb, &d.id, JobKind::Graph);
        }
        Ok(docs.len())
    }

    /// Read a document's unread passages for the graph (state `extracting`): extract, resolve each entity to an
    /// existing one by key or, failing that, by name vector (same type, similarity >= `merge_similarity`), store.
    /// A reply that fails the schema skips its passage. Communities are recomputed at the end.
    async fn graph_document(&self, kb: &str, doc: &str) -> Result<(), String> {
        let Some(x) = &self.extractor else {
            return Ok(());
        };
        let todo = self
            .store
            .chunks_for_graph(doc)
            .await
            .map_err(|e| e.to_string())?;
        if todo.is_empty() {
            return Ok(());
        }
        self.store
            .set_status(doc, "extracting", None)
            .await
            .map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let (mut ok, mut skipped, mut tokens, mut merged) = (0usize, 0usize, 0u64, 0usize);
        // FR-K55: relations whose quote is not in the passage, counted rather than hidden.
        let mut quote_drops = 0usize;
        // FR-K54: the base's own ontology, read once per document; the prompt is built from it once.
        let ontology = self
            .store
            .get_ontology(kb)
            .await
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        let prompt = ontology.extraction_prompt();
        let doc_title = self
            .store
            .get_document(kb, doc)
            .await
            .map(|d| d.title)
            .unwrap_or_default();
        // Keys resolved in this job, so a name is embedded and matched once.
        let mut resolved: HashMap<String, String> = HashMap::new();
        for (chunk_id, _heading, text) in &todo {
            let call = x.extract_with(text, &ontology, &prompt).await.map_err(|e| {
                format!("The knowledge graph could not be built: {e}. Passages read so far are kept; build again to continue.")
            })?;
            tokens += call.completion_tokens;
            let ex = match call.result {
                Ok(ex) => ex,
                Err(why) => {
                    skipped += 1;
                    tracing::debug!(chunk = %chunk_id, %why, "graph extraction skipped");
                    self.store
                        .store_extraction(kb, doc, chunk_id, &[], &[])
                        .await
                        .map_err(|e| e.to_string())?;
                    continue;
                }
            };
            // key -> record key for every entity of this passage.
            let mut rec: HashMap<String, String> = HashMap::new();
            let mut ents: Vec<Value> = Vec::new();
            let fresh: Vec<&graph::ExtractedEntity> = ex
                .entities
                .iter()
                .filter(|e| !resolved.contains_key(&e.key))
                .collect();
            let ids: Vec<String> = fresh.iter().map(|e| graph::entity_id(kb, &e.key)).collect();
            let existing = self
                .store
                .existing_entities(kb, &ids)
                .await
                .map_err(|e| e.to_string())?;
            let need: Vec<&graph::ExtractedEntity> = fresh
                .iter()
                .copied()
                .filter(|e| !existing.contains_key(&graph::entity_id(kb, &e.key)))
                .collect();
            let vecs = match (&self.embedder, need.is_empty()) {
                (Some(em), false) => em
                    .embed(
                        &need.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
                        None,
                    )
                    .await
                    .ok(),
                _ => None,
            };
            for e in &ex.entities {
                let id = if let Some(id) = resolved.get(&e.key) {
                    id.clone()
                } else {
                    let own = graph::entity_id(kb, &e.key);
                    let mut id = own.clone();
                    let mut v = None;
                    if !existing.contains_key(&own) {
                        if let Some(i) = need.iter().position(|n| n.key == e.key) {
                            v = vecs.as_ref().and_then(|vs| vs.get(i).cloned());
                            if let Some(vv) = &v {
                                if let Ok(Some((other, _))) = self
                                    .store
                                    .similar_entity(kb, vv, &e.kind, x.cfg.merge_similarity)
                                    .await
                                {
                                    id = other;
                                    merged += 1;
                                }
                            }
                        }
                    }
                    resolved.insert(e.key.clone(), id.clone());
                    if id == own {
                        ents.push(json!({ "id": id, "key": e.key, "name": e.name, "type": e.kind, "v": v }));
                    } else {
                        ents.push(json!({ "id": id, "key": e.key, "name": e.name, "type": e.kind, "v": null }));
                    }
                    id
                };
                if !ents.iter().any(|x| x["id"] == json!(id)) {
                    ents.push(json!({ "id": id, "key": e.key, "name": e.name, "type": e.kind, "v": null }));
                }
                rec.insert(e.key.clone(), id);
            }
            let rels: Vec<Value> = ex
                .relations
                .iter()
                .filter_map(|r| {
                    let (s, t) = (rec.get(&r.source)?, rec.get(&r.target)?);
                    (s != t).then(|| json!({ "id": graph::relation_id(kb, s, &r.predicate, t), "s": s, "t": t, "p": r.predicate, "quote": r.quote }))
                })
                .collect();
            self.store
                .store_extraction(kb, doc, chunk_id, &ents, &rels)
                .await
                .map_err(|e| e.to_string())?;
            // FR-K53: the context line the same call wrote, stored and folded into the words index. Its vector is
            // refreshed only by a reindex, so a passage keeps the vector it was embedded with until then.
            if !ex.context.is_empty() {
                self.store
                    .set_chunk_context(chunk_id, &ex.context, &doc_title)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            quote_drops += ex.quote_drops;
            ok += 1;
        }
        let _ = self.recompute_communities(kb).await;
        tracing::info!(
            doc,
            passages = todo.len(),
            ok,
            skipped,
            merged,
            quote_drops,
            tokens,
            ms = t0.elapsed().as_millis() as u64,
            "knowledge graph read"
        );
        Ok(())
    }

    /// Louvain communities over the relations, each labelled by its three most mentioned entities.
    pub async fn recompute_communities(&self, kb: &str) -> Result<usize, KbError> {
        let ents = self.store.graph_entities(kb).await?;
        if ents.is_empty() {
            self.store.set_communities(kb, &[], &[]).await?;
            return Ok(0);
        }
        let index: HashMap<&str, usize> = ents
            .iter()
            .enumerate()
            .map(|(i, e)| (e.0.as_str(), i))
            .collect();
        let edges: Vec<(usize, usize, f64)> = self
            .store
            .graph_edges(kb)
            .await?
            .iter()
            .filter_map(|(s, t, w)| Some((*index.get(s.as_str())?, *index.get(t.as_str())?, *w)))
            .collect();
        let comm = graph::louvain(ents.len(), &edges);
        let n = comm.iter().max().map_or(0, |m| m + 1);
        let mut comms = Vec::new();
        for c in 0..n {
            // `ents` is sorted by mentions, so the first members are the most mentioned.
            let members: Vec<&str> = ents
                .iter()
                .zip(&comm)
                .filter(|(_, x)| **x == c)
                .map(|(e, _)| e.1.as_str())
                .collect();
            comms.push((
                c,
                members.len(),
                members
                    .iter()
                    .take(3)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
        let assign: Vec<(String, usize)> = ents
            .iter()
            .zip(&comm)
            .map(|(e, c)| (e.0.clone(), *c))
            .collect();
        self.store.set_communities(kb, &assign, &comms).await?;
        Ok(n)
    }

    /// Graph search for chat (FR-K22, local search): entities nearest the question (by meaning and by name), the
    /// relations around them, and passages. Returns (facts, passages).
    ///
    /// FR-K50: the graph **supplies candidates**; it does not reorder the search's. A passage a fact cites is added
    /// to the pool when the search missed it, every candidate gets a graph signal (seed entities it mentions, and the
    /// weight of the facts citing it), and the two are fused by rank, as the lexical and vector arms are. The earlier
    /// version sorted the passages by that signal alone, which dropped a passage holding the answer below passages
    /// that merely named the question's entities: measured in Phase 7, where `graph` answered 15 of 20 against
    /// `rag`'s 18 and won none of the three discordant pairs.
    pub async fn graph_search(
        &self,
        kb: &str,
        query: &str,
        k: usize,
    ) -> Result<(Vec<Value>, SearchResult), KbError> {
        self.ready().await?;
        let mut base = self
            .search(kb, query, (k * 3).max(12), SearchMode::Hybrid)
            .await?;
        let qv = self.question_vector(query).await.ok().map(|x| x.0);
        let found = self
            .store
            .find_entities(kb, query, qv.as_deref(), 8)
            .await?;
        let seeds: Vec<String> = found
            .iter()
            .filter(|e| {
                e["score"].as_f64().is_some_and(|s| s > 0.0)
                    || e["similarity"].as_f64().is_some_and(|s| s >= 0.5)
            })
            .filter_map(|e| e["id"].as_str().map(str::to_string))
            .take(5)
            .collect();
        if seeds.is_empty() {
            // FR-K72: no model-built graph names anything in the question; the passage-and-term graph still links
            // the passages, so its walk supplies the candidates.
            if let index::Lookup::Ready(ix) = self.index.get(&self.store, kb, self.dims()).await? {
                let mut params = self.cfg.index.params;
                params.ppr = Some(params.ppr.unwrap_or_default());
                let r = self
                    .search_indexed(kb, ix, query, k, SearchMode::Hybrid, params)
                    .await;
                return Ok((Vec::new(), r));
            }
            base.results.truncate(k);
            return Ok((Vec::new(), base));
        }
        let facts = self.walk(kb, &seeds).await?;
        let mentioning = self.store.chunks_mentioning(kb, &seeds).await?;
        // Every passage a fact cites joins the candidates when the search missed it, so the graph widens the pool.
        let mut cited: HashMap<String, f64> = HashMap::new();
        for f in &facts {
            let w = f["weight"].as_f64().unwrap_or(1.0).max(1.0);
            for e in f["evidence"].as_array().into_iter().flatten() {
                if let Some(id) = e.as_str() {
                    *cited.entry(id.to_string()).or_insert(0.0) += w;
                }
            }
        }
        let extra: Vec<String> = cited
            .keys()
            .filter(|id| !base.results.iter().any(|h| &&h.chunk_id == id))
            .cloned()
            .collect();
        let mut hits = base.results;
        hits.extend(self.store.hits_by_id(&extra).await?);
        base.results = fuse_graph(hits, &mentioning, &cited, k, self.cfg.graph_weight);
        Ok((facts, base))
    }

    /// FR-K54: the ontology a knowledge base's graph is built under (the default when none was set).
    pub async fn ontology(&self, kb: &str) -> Result<(graph::Ontology, bool), KbError> {
        self.store.get_kb(kb).await?;
        Ok(match self.store.get_ontology(kb).await? {
            Some(o) => (o, false),
            None => (graph::Ontology::default(), true),
        })
    }

    /// FR-K54: set it. It applies to passages read from now on; a graph built under the old one is rebuilt with
    /// `graph/build?fresh=true`.
    pub async fn set_ontology(
        &self,
        kb: &str,
        ontology: Option<graph::Ontology>,
    ) -> Result<(), KbError> {
        self.store.get_kb(kb).await?;
        if let Some(o) = &ontology {
            o.validate().map_err(KbError::BadInput)?;
        }
        self.store.set_ontology(kb, ontology.as_ref()).await
    }

    /// FR-K56: the facts around the seed entities, two hops out, each scored by the path that reached it.
    ///
    /// Hop 1 is every relation of a seed. Hop 2 keeps, per entity the first hop reached, the best `graph_beam`
    /// relations, and `graph_beam * 3` in all. A path's score is the product over its hops of the relation's weight
    /// (normalised to the heaviest first-hop relation) and `graph_decay` raised to the hop, so a second-hop fact
    /// must be well attested to outrank a first-hop one. Each fact carries `hop`, and a second-hop fact carries
    /// `via`, the entity it came through, so the prompt can show the chain. Deliberately not a cosine beam with no
    /// path score (what OntoSmith ships): this graph carries weights, so it uses them.
    async fn walk(&self, kb: &str, seeds: &[String]) -> Result<Vec<Value>, KbError> {
        let beam = self.cfg.graph_beam.max(1);
        let decay = self.cfg.graph_decay.clamp(0.0, 1.0);
        let seed_set: std::collections::HashSet<&str> = seeds.iter().map(String::as_str).collect();
        let mut first = self.store.facts_for(kb, seeds, 12).await?;
        let top = first
            .iter()
            .filter_map(|f| f["weight"].as_f64())
            .fold(1.0_f64, f64::max);
        for f in first.iter_mut() {
            let w = f["weight"].as_f64().unwrap_or(1.0) / top;
            f["hop"] = json!(1);
            f["path_score"] = json!(w * decay);
        }
        // The frontier: entities the first hop reached that are not seeds, with the best path that reached each.
        let mut frontier: HashMap<String, (String, f64)> = HashMap::new();
        for f in &first {
            let score = f["path_score"].as_f64().unwrap_or(0.0);
            for (id, name, other) in [
                (
                    f["target_id"].as_str(),
                    f["target"].as_str(),
                    f["source"].as_str(),
                ),
                (
                    f["source_id"].as_str(),
                    f["source"].as_str(),
                    f["target"].as_str(),
                ),
            ] {
                let (Some(id), Some(_name), Some(other)) = (id, name, other) else {
                    continue;
                };
                if seed_set.contains(id) {
                    continue;
                }
                let e = frontier
                    .entry(id.to_string())
                    .or_insert((other.to_string(), 0.0));
                if score > e.1 {
                    *e = (other.to_string(), score);
                }
            }
        }
        let seen: std::collections::HashSet<String> = first
            .iter()
            .filter_map(|f| f["id"].as_str().map(str::to_string))
            .collect();
        let mut second: Vec<Value> = Vec::new();
        if !frontier.is_empty() {
            let ids: Vec<String> = frontier.keys().cloned().collect();
            let hop2 = self.store.facts_for(kb, &ids, beam * 8).await?;
            let mut per_entity: HashMap<String, usize> = HashMap::new();
            for mut f in hop2 {
                let Some(id) = f["id"].as_str().map(str::to_string) else {
                    continue;
                };
                if seen.contains(&id) {
                    continue;
                }
                // The frontier entity this relation hangs off, and the path that reached that entity.
                let through = [f["source_id"].as_str(), f["target_id"].as_str()]
                    .into_iter()
                    .flatten()
                    .find(|x| frontier.contains_key(*x))
                    .map(str::to_string);
                let Some(through) = through else { continue };
                let n = per_entity.entry(through.clone()).or_insert(0);
                if *n >= beam {
                    continue;
                }
                *n += 1;
                let (_, reached) = &frontier[&through];
                let w = f["weight"].as_f64().unwrap_or(1.0) / top;
                let via = if f["source_id"].as_str() == Some(through.as_str()) {
                    f["source"].as_str().unwrap_or("")
                } else {
                    f["target"].as_str().unwrap_or("")
                }
                .to_string();
                f["hop"] = json!(2);
                f["via"] = json!(via);
                f["path_score"] = json!(reached * w.min(1.0) * decay);
                second.push(f);
            }
        }
        second.sort_by(|a, b| {
            b["path_score"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a["path_score"].as_f64().unwrap_or(0.0))
        });
        second.truncate(beam * 3);
        let mut all = first;
        all.extend(second);
        all.sort_by(|a, b| {
            b["path_score"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a["path_score"].as_f64().unwrap_or(0.0))
        });
        Ok(all)
    }

    /// A question embedded as Memorilla reads it: no instruction prefix (the memory module was trained that way).
    pub async fn embed_plain(&self, text: &str) -> Result<Vec<f32>, KbError> {
        let e = self
            .embedder
            .as_ref()
            .ok_or_else(|| KbError::Unavailable("searching by meaning is turned off".into()))?;
        Ok(e.embed(&[text.to_string()], None).await?.remove(0))
    }

    /// Texts embedded without an instruction (Memorilla's questions), in order.
    pub async fn embed_texts_plain(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KbError> {
        let e = self
            .embedder
            .as_ref()
            .ok_or_else(|| KbError::Unavailable("searching by meaning is turned off".into()))?;
        e.embed(texts, None).await
    }

    /// The width of the passage vectors, when embeddings are on.
    pub fn embed_dims(&self) -> Option<usize> {
        self.embedder.as_ref().map(|e| e.cfg.dims)
    }

    pub fn graph_enabled(&self) -> bool {
        self.extractor.is_some()
    }

    /// Whether vectors can be made and searched now (the embedding server answers).
    pub async fn embedder_status(&self) -> Option<(String, bool)> {
        let e = self.embedder.as_ref()?;
        Some((e.cfg.model.clone(), e.health().await))
    }

    /// The nearest passages to a query vector. A base under `exact_search_below` vectors is scanned in the store; a larger one
    /// is searched exactly in memory once its vectors are loaded (`vcache`), or through the store's index when it is too big
    /// for the cache. `None` while the copy of a large base is still loading.
    async fn vector_hits(
        &self,
        kb: &str,
        v: &[f32],
        k: usize,
        exact: bool,
        n: u64,
    ) -> Result<Option<Vec<Hit>>, KbError> {
        if exact {
            return Ok(Some(self.store.search_vector(kb, v, k, true).await?));
        }
        let dims = self.embedder.as_ref().map_or(v.len(), |e| e.cfg.dims);
        match self.vcache.lookup(&self.store, kb, n, dims) {
            vcache::Lookup::TooBig => Ok(Some(self.store.search_vector(kb, v, k, false).await?)),
            vcache::Lookup::Loading => Ok(None),
            vcache::Lookup::Ready(l) => {
                let top = l.top(v, k);
                let ids: Vec<String> = top.iter().map(|t| t.0.clone()).collect();
                let mut by_id: HashMap<String, Hit> = self
                    .store
                    .hits_by_id(&ids)
                    .await?
                    .into_iter()
                    .map(|h| (h.chunk_id.clone(), h))
                    .collect();
                let mut out = Vec::with_capacity(top.len());
                for (id, sim) in top {
                    if let Some(mut h) = by_id.remove(&id) {
                        h.score = sim as f64;
                        h.vector_score = Some(sim as f64);
                        h.rank = out.len() + 1;
                        out.push(h);
                    }
                }
                Ok(Some(out))
            }
        }
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
        self.search_with(kb, query, k, mode, Engine::Auto).await
    }

    /// The vector of a question, remembered for a while (FR-K69). Returns it with the time the embedding took (0 when
    /// it was remembered).
    async fn question_vector(&self, query: &str) -> Result<(Vec<f32>, u64), KbError> {
        let e = self
            .embedder
            .as_ref()
            .ok_or_else(|| KbError::Unavailable("searching by meaning is turned off".into()))?;
        static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tick = TICK.fetch_add(1, Ordering::Relaxed);
        if let Some(hit) = self
            .questions
            .lock()
            .expect("question cache")
            .get_mut(query)
        {
            hit.1 = tick;
            return Ok((hit.0.clone(), 0));
        }
        let t = Instant::now();
        let v = e
            .embed(&[query.to_string()], Some(QUERY_INSTRUCTION))
            .await?
            .remove(0);
        let mut cache = self.questions.lock().expect("question cache");
        if cache.len() >= QUESTION_CACHE {
            if let Some(oldest) = cache.iter().min_by_key(|x| x.1 .1).map(|x| x.0.clone()) {
                cache.remove(&oldest);
            }
        }
        cache.insert(query.to_string(), (v.clone(), tick));
        Ok((v, t.elapsed().as_millis() as u64))
    }

    /// The in-process index of a base when it is loaded (FR-K67).
    pub async fn index_of(&self, kb: &str) -> Result<Option<Arc<index::Base>>, KbError> {
        self.ready().await?;
        Ok(match self.index.get(&self.store, kb, self.dims()).await? {
            index::Lookup::Ready(b) => Some(b),
            _ => None,
        })
    }

    /// (bases, passages, bytes) the in-process indexes hold.
    pub fn index_stats(&self) -> (usize, usize, usize) {
        self.index.stats()
    }

    /// FR-K71: the Graph view of the passage-and-term graph, for a base whose index is loaded.
    pub async fn text_graph_view(
        &self,
        kb: &str,
        center: Option<&str>,
        limit: usize,
    ) -> Option<Value> {
        let base = self.index_of(kb).await.ok()??;
        Some(base.graph_view(center, limit))
    }

    /// A search answered from the in-process index: no round trip to the store, and the question is embedded once.
    async fn search_indexed(
        &self,
        kb: &str,
        base: Arc<index::Base>,
        query: &str,
        k: usize,
        mode: SearchMode,
        params: index::Params,
    ) -> SearchResult {
        let t0 = Instant::now();
        let (mut fallback, mut embed_ms) = (None, 0);
        let qv = if mode == SearchMode::Lexical || query.trim().is_empty() {
            None
        } else if self.embedder.is_none() {
            fallback = Some("searching by meaning is turned off".to_string());
            None
        } else if base.vectors() == 0 {
            fallback = Some("this knowledge base has no meaning index yet".to_string());
            None
        } else if let Some(other) = self.foreign_embedder(kb).await {
            fallback = Some(foreign_note(&other));
            None
        } else {
            match self.question_vector(query).await {
                Ok((v, ms)) => {
                    embed_ms = ms;
                    Some(v)
                }
                Err(err) => {
                    fallback = Some(format!("the embedding server is not available ({err})"));
                    None
                }
            }
        };
        let used = if qv.is_some() {
            mode
        } else {
            SearchMode::Lexical
        };
        let index = self.index.clone();
        let query = query.to_string();
        let large = base.passages() > 20_000;
        let run = move || base.search(&index.analyzer, &query, qv.as_deref(), used, k, &params);
        // Scoring a small base takes well under a millisecond; a large one goes off the async threads.
        let results = if large {
            tokio::task::spawn_blocking(run).await.unwrap_or_default()
        } else {
            run()
        };
        SearchResult {
            results,
            mode: used,
            took_ms: t0.elapsed().as_millis() as u64,
            embed_ms,
            fallback,
            engine: "index",
        }
    }

    /// `search`, with the path chosen by the caller (FR-K69): `Engine::Store` is the path every search took before
    /// Phase 25, kept for bases that do not fit in memory and for comparing the two.
    pub async fn search_with(
        &self,
        kb: &str,
        query: &str,
        k: usize,
        mode: SearchMode,
        engine: Engine,
    ) -> Result<SearchResult, KbError> {
        self.ready().await?;
        let t0 = Instant::now();
        let k = k.clamp(1, 100);
        if engine == Engine::Auto {
            if let index::Lookup::Ready(base) = self.index.get(&self.store, kb, self.dims()).await?
            {
                let mut r = self
                    .search_indexed(kb, base, query, k, mode, self.cfg.index.params)
                    .await;
                r.took_ms = t0.elapsed().as_millis() as u64;
                return Ok(r);
            }
        }
        let mut fallback = None;
        let mut embed_ms = 0;
        let qv = if mode == SearchMode::Lexical || query.trim().is_empty() {
            None
        } else if let Some(e) = &self.embedder {
            let n = self.store.vector_count(kb).await?;
            if n == 0 {
                fallback = Some("this knowledge base has no meaning index yet".to_string());
                None
            } else if let Some(other) = self.foreign_embedder(kb).await {
                fallback = Some(foreign_note(&other));
                None
            } else {
                let te = Instant::now();
                match e.embed(&[query.to_string()], Some(QUERY_INSTRUCTION)).await {
                    Ok(mut v) => {
                        embed_ms = te.elapsed().as_millis() as u64;
                        Some((v.remove(0), n < self.cfg.exact_search_below, n))
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
            (SearchMode::Vector, Some((v, exact, n))) => {
                match self.vector_hits(kb, &v, k, exact, n).await? {
                    Some(h) => (h, SearchMode::Vector),
                    None => {
                        fallback = Some(LOADING.to_string());
                        (
                            self.store.search_lexical(kb, query, k).await?,
                            SearchMode::Lexical,
                        )
                    }
                }
            }
            (SearchMode::Hybrid, Some((v, exact, n))) => {
                let depth = (k * 3).max(20);
                let limit = Duration::from_secs(self.cfg.limits.search_timeout_s.max(1));
                let vec = self.vector_hits(kb, &v, depth, exact, n).await?;
                // The store stops the query itself at the limit (`TIMEOUT`); the timer here is only a backstop.
                let words = tokio::time::timeout(
                    limit + Duration::from_secs(2),
                    self.store
                        .search_lexical_within(kb, query, depth, limit.as_secs()),
                )
                .await;
                match (words, vec) {
                    (Ok(Ok(lex)), Some(vec)) => (
                        fuse(lex, vec, k, self.cfg.vector_weight),
                        SearchMode::Hybrid,
                    ),
                    // The meaning index of a large base is still being loaded: the words answer alone, and say so.
                    (Ok(Ok(mut lex)), None) => {
                        fallback = Some(LOADING.to_string());
                        lex.truncate(k);
                        (lex, SearchMode::Lexical)
                    }
                    (Ok(Err(e)), _) if !is_timeout(&e) => return Err(e),
                    (_, vec) => {
                        fallback = Some(format!(
                            "the search by words took more than {} s and was skipped",
                            limit.as_secs()
                        ));
                        let mut vec = vec.unwrap_or_default();
                        vec.truncate(k);
                        (vec, SearchMode::Vector)
                    }
                }
            }
            _ => {
                let limit = Duration::from_secs(self.cfg.limits.search_timeout_s.max(1));
                let busy = || {
                    KbError::Busy(format!(
                        "the search by words took more than {} s; try fewer or rarer words",
                        limit.as_secs()
                    ))
                };
                let hits = match tokio::time::timeout(
                    limit + Duration::from_secs(2),
                    self.store
                        .search_lexical_within(kb, query, k, limit.as_secs()),
                )
                .await
                {
                    Ok(Ok(h)) => h,
                    Ok(Err(e)) if is_timeout(&e) => return Err(busy()),
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err(busy()),
                };
                (hits, SearchMode::Lexical)
            }
        };
        Ok(SearchResult {
            results,
            mode: used,
            took_ms: t0.elapsed().as_millis() as u64,
            embed_ms,
            fallback,
            engine: "store",
        })
    }
}

async fn worker(kb: std::sync::Weak<Kb>, mut rx: mpsc::UnboundedReceiver<Job>) {
    while let Some(job) = rx.recv().await {
        let Some(kb) = kb.upgrade() else { return };
        kb.pending.fetch_sub(1, Ordering::Relaxed);
        kb.running.fetch_add(1, Ordering::Relaxed);
        if job.kind == JobKind::Vectors {
            if let Ok(d) = kb.store.get_document(&job.kb, &job.doc).await {
                let note = kb.embed_document(&job.kb, &d.id, &d.title).await.err();
                let _ = kb.store.set_status(&d.id, "ready", note.as_deref()).await;
            }
        } else if job.kind == JobKind::Graph {
            if let Ok(d) = kb.store.get_document(&job.kb, &job.doc).await {
                let note = kb.graph_document(&job.kb, &d.id).await.err();
                let _ = kb.store.set_status(&d.id, "ready", note.as_deref()).await;
                // The pass wrote context lines, which the words index covers (FR-K53).
                kb.index
                    .refresh_doc(&kb.store, &job.kb, &d.id, kb.dims())
                    .await;
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
///
/// The vector arm counts `vector_weight` against the words' 1.0 (FR-K63). Measured on 70 questions over two bases of the
/// paper, an equal weight let the weaker arm (first place for 50 to 62 % of the questions against 86 to 88 % for the
/// words) push the right passage down; 0.1 raised the mean reciprocal rank by 0.03 to 0.04 and kept the vector arm's
/// recall.
pub fn fuse(lex: Vec<Hit>, vec: Vec<Hit>, k: usize, vector_weight: f64) -> Vec<Hit> {
    let lists = [
        lex.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
        vec.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
    ];
    let fused = embed::rrf_weighted(&lists, &[1.0, vector_weight], 60.0);
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

/// FR-K50: the search's order and the graph's signal fused by rank, as the lexical and vector arms are.
///
/// The graph's own ranking puts a passage first when the facts citing it are heavy and when it names more of the
/// question's entities; `graph_weight` says how much that list counts against the search's (1.0 = an equal arm).
/// Rank fusion rather than adding the scores, because a BM25 score, a cosine and a relation weight share no scale.
pub fn fuse_graph(
    hits: Vec<Hit>,
    mentioning: &HashMap<String, usize>,
    cited: &HashMap<String, f64>,
    k: usize,
    graph_weight: f64,
) -> Vec<Hit> {
    // The search's list, in the order it came back.
    let search: Vec<&str> = hits.iter().map(|h| h.chunk_id.as_str()).collect();
    // The graph's list ranks the **whole** candidate population, heaviest cited facts first, then most seed entities
    // named, then the search order. Ranking only the passages the graph has an opinion about would make "last of
    // two" score like "second overall", and a passage with one weak mention would leapfrog the search's best.
    let mut graph: Vec<(&str, f64, usize, usize)> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            (
                h.chunk_id.as_str(),
                cited.get(&h.chunk_id).copied().unwrap_or(0.0),
                mentioning.get(&h.chunk_id).copied().unwrap_or(0),
                i,
            )
        })
        .collect();
    graph.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.cmp(&a.2)).then(a.3.cmp(&b.3)));
    let graph: Vec<&str> = graph.into_iter().map(|x| x.0).collect();
    let mut score: HashMap<&str, f64> = HashMap::new();
    for (rank, id) in search.iter().enumerate() {
        *score.entry(id).or_insert(0.0) += 1.0 / (60.0 + rank as f64 + 1.0);
    }
    for (rank, id) in graph.iter().enumerate() {
        *score.entry(id).or_insert(0.0) += graph_weight / (60.0 + rank as f64 + 1.0);
    }
    let order: HashMap<&str, usize> = search.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut ids: Vec<(&str, f64)> = score.into_iter().collect();
    ids.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then(order.get(a.0).cmp(&order.get(b.0)))
    });
    let mut out: Vec<Hit> = Vec::with_capacity(k);
    for (id, s) in ids.into_iter().take(k) {
        if let Some(h) = hits.iter().find(|h| h.chunk_id == id) {
            let mut h = h.clone();
            h.score = s;
            h.rank = out.len() + 1;
            out.push(h);
        }
    }
    out
}

const LOADING: &str = "the meaning index of this large base is loading into memory (first use); answered by words for now";

/// Whether the store stopped a query because it ran past its `TIMEOUT`.
fn is_timeout(e: &KbError) -> bool {
    matches!(e, KbError::Store(m) if {
        let m = m.to_lowercase();
        m.contains("timeout") || m.contains("timed out")
    })
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

/// The recorded model of a base when it differs from the one running (`ours`); a base with none recorded has no
/// vectors yet, and whatever is written next is ours.
fn foreign_model(recorded: Option<String>, ours: &str) -> Option<String> {
    recorded.filter(|m| m != ours)
}

/// What a search says when a base's vectors were made by another model.
fn foreign_note(model: &str) -> String {
    format!("this knowledge base's meaning index was made by another embedding model ({model}); rebuild the meaning index to search by meaning again")
}

/// The relevance filter between a search and the prompt (FR-K20), here so the evaluation can run it without a server.
///
/// A passage is kept when its meaning similarity reaches `min_similarity`, **or** when its words match strongly
/// (`lexical_rescue`, a BM25 score; 0 turns this off); a search that fell back to words keeps every match. At most `k`
/// are kept. Phase 26: with `elbow_gap > 0`, the kept passages are then cut at the largest drop in meaning similarity
/// when that drop is at least `elbow_gap` (the threshold sits in the middle of the drop); the first passage and a
/// passage whose words match strongly are never cut. A prompt then carries the passages that stand out, not always
/// `k` of them, and on a CPU every passage left out is a second or more of reading time.
pub fn keep_for_chat(
    hits: Vec<Hit>,
    mode: SearchMode,
    min_similarity: f64,
    lexical_rescue: f64,
    k: usize,
    elbow_gap: f64,
) -> Vec<Hit> {
    let rescued =
        |h: &Hit| lexical_rescue > 0.0 && h.lexical_score.is_some_and(|l| l >= lexical_rescue);
    let mut kept: Vec<Hit> = hits
        .into_iter()
        .filter(|h| {
            mode == SearchMode::Lexical
                || h.vector_score.is_some_and(|v| v >= min_similarity)
                || rescued(h)
        })
        .take(k)
        .collect();
    if elbow_gap <= 0.0 || kept.len() < 3 {
        return kept;
    }
    let mut sims: Vec<f64> = kept.iter().filter_map(|h| h.vector_score).collect();
    if sims.len() < 3 {
        return kept;
    }
    sims.sort_by(|a, b| b.total_cmp(a));
    let (mut gap, mut at) = (0.0, 0);
    for i in 1..sims.len() {
        if sims[i - 1] - sims[i] > gap {
            (gap, at) = (sims[i - 1] - sims[i], i);
        }
    }
    if gap < elbow_gap {
        return kept;
    }
    let threshold = sims[at - 1] - gap / 2.0;
    let mut first = true;
    kept.retain(|h| {
        std::mem::take(&mut first) || rescued(h) || h.vector_score.is_none_or(|v| v >= threshold)
    });
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(id: &str, rank: usize) -> Hit {
        Hit {
            chunk_id: id.into(),
            doc_id: "d".into(),
            doc_title: "t".into(),
            ord: 0,
            heading: String::new(),
            page: None,
            text: String::new(),
            score: 1.0 / rank as f64,
            rank,
            lexical_score: None,
            vector_score: None,
        }
    }

    /// FR-K63: a hybrid search lets the words lead. A passage the words rank tenth and the meaning arm ranks second beats
    /// the words' first when the arms are equal; with the meaning arm at 0.1 the words' first place stays, and what only
    /// the meaning arm found still enters the list. (RRF is flat near the top, so the small weight mostly breaks
    /// near-ties and reorders what the words rank far apart.)
    #[test]
    fn the_words_lead_a_hybrid_search_when_the_meaning_arm_is_weighted_down() {
        let mut lex: Vec<Hit> = (1..=9).map(|r| h(&format!("w{r}"), r)).collect();
        lex.push(h("both", 10));
        let vec = vec![h("v1", 1), h("both", 2), h("v3", 3)];
        let ids = |v: Vec<Hit>| v.into_iter().map(|x| x.chunk_id).collect::<Vec<_>>();
        assert_eq!(ids(fuse(lex.clone(), vec.clone(), 4, 1.0))[0], "both");
        let weighted = ids(fuse(lex.clone(), vec.clone(), 12, 0.1));
        assert_eq!(weighted[0], "w1", "{weighted:?}");
        assert!(
            weighted.contains(&"v1".to_string()),
            "the meaning arm's finds are kept: {weighted:?}"
        );
        let words_only = ids(fuse(lex, vec, 10, 0.0));
        assert_eq!(
            &words_only[..3],
            ["w1", "w2", "w3"],
            "weight 0 is the words alone"
        );
    }

    /// FR-K50: the graph is an arm, not an override. A passage the search ranked first stays ahead of one the graph
    /// alone likes, and a passage only the graph found still reaches the answer.
    #[test]
    fn the_graph_arm_moves_passages_without_overriding_the_search() {
        let hits = vec![h("a", 1), h("b", 2), h("c", 3), h("d", 4)];
        let no_ents: HashMap<String, usize> = HashMap::new();
        let no_cites: HashMap<String, f64> = HashMap::new();
        // With nothing from the graph, the search order is kept exactly.
        let out = fuse_graph(hits.clone(), &no_ents, &no_cites, 4, 1.0);
        assert_eq!(
            out.iter().map(|x| x.chunk_id.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        // "c" is cited by a heavy fact and names two seed entities: it rises from third to second. The search's
        // first stays first — it leads one list and trails the other by as much, and an exact tie breaks toward the
        // search, which is the primary signal. This is the case the old code got wrong, in the other direction.
        let cited = HashMap::from([("c".to_string(), 9.0)]);
        let mentioning = HashMap::from([("c".to_string(), 2usize), ("d".to_string(), 1usize)]);
        let out = fuse_graph(hits.clone(), &mentioning, &cited, 4, 1.0);
        assert_eq!(
            out.iter().map(|x| x.chunk_id.as_str()).collect::<Vec<_>>(),
            ["a", "c", "b", "d"]
        );
        // Weight 0 means the graph contributes facts and never moves a passage.
        let out = fuse_graph(hits.clone(), &mentioning, &cited, 4, 0.0);
        assert_eq!(
            out.iter().map(|x| x.chunk_id.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        // k truncates, and the ranks are renumbered from 1.
        let out = fuse_graph(hits, &mentioning, &cited, 2, 1.0);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].rank, out[1].rank), (1, 2));
    }

    #[test]
    fn display_names_drop_paths_and_controls() {
        assert_eq!(display_name("../../etc/passwd"), "passwd");
        assert_eq!(display_name("C:\\Users\\a\\report.pdf"), "report.pdf");
        assert_eq!(display_name("..hidden\u{7}.md"), "hidden.md");
        assert_eq!(display_name(""), "untitled");
        assert_eq!(safe_key("ab-c_1/../x"), "ab-c_1x");
    }
    #[test]
    fn a_base_made_by_another_embedding_model_is_recognised() {
        let ours = "Qwen3-Embedding-0.6B";
        assert_eq!(foreign_model(None, ours), None);
        assert_eq!(foreign_model(Some(ours.to_string()), ours), None);
        assert_eq!(
            foreign_model(Some("Qwen3-Embedding-4B".to_string()), ours).as_deref(),
            Some("Qwen3-Embedding-4B")
        );
        assert!(foreign_note("Qwen3-Embedding-4B").contains("rebuild the meaning index"));
    }

    #[test]
    fn the_elbow_cuts_the_passages_after_the_largest_drop_but_not_a_strong_word_match() {
        let hit = |id: &str, sim: f64, lex: f64| Hit {
            chunk_id: id.into(),
            doc_id: "d".into(),
            doc_title: "t".into(),
            ord: 0,
            heading: String::new(),
            page: None,
            text: String::new(),
            score: 0.0,
            rank: 0,
            lexical_score: Some(lex),
            vector_score: Some(sim),
        };
        let hits = || {
            vec![
                hit("a", 0.71, 3.0),
                hit("b", 0.69, 2.0),
                hit("c", 0.52, 1.0),
                hit("d", 0.50, 12.0),
            ]
        };
        let ids = |v: Vec<Hit>| v.into_iter().map(|h| h.chunk_id).collect::<Vec<_>>();
        assert_eq!(
            ids(keep_for_chat(
                hits(),
                SearchMode::Hybrid,
                0.45,
                10.0,
                4,
                0.0
            )),
            ["a", "b", "c", "d"]
        );
        // The drop of 0.17 between b and c cuts c; d is below it too but its words match strongly.
        assert_eq!(
            ids(keep_for_chat(
                hits(),
                SearchMode::Hybrid,
                0.45,
                10.0,
                4,
                0.08
            )),
            ["a", "b", "d"]
        );
        // No drop that large: nothing is cut.
        assert_eq!(
            ids(keep_for_chat(
                hits(),
                SearchMode::Hybrid,
                0.45,
                10.0,
                4,
                0.2
            ))
            .len(),
            4
        );
        // The first passage stays even when it is the low one.
        let low_first = vec![
            hit("x", 0.50, 1.0),
            hit("y", 0.80, 1.0),
            hit("z", 0.79, 1.0),
        ];
        assert_eq!(
            ids(keep_for_chat(
                low_first,
                SearchMode::Hybrid,
                0.45,
                10.0,
                4,
                0.08
            )),
            ["x", "y", "z"]
        );
    }
}
