//! The SurrealDB client (A-104): the RPC `query` method over HTTP with bound variables, so document text never has to
//! be escaped into SurrealQL. The wire format is **CBOR**, not JSON: SurrealDB's JSON decoding turns any string that
//! looks like a record id into one (`"Caption: Model of the gate."` arrives as the record `Caption:Model`), which would
//! corrupt passages and queries. CBOR keeps strings as strings and tags record ids. Record keys travel back as plain
//! strings (`record::id(id)`), and references are rebuilt with `type::record(table, key)`.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use serde_json::{json, Value};

use crate::chunk::Chunk;
use crate::KbError;

/// Connection settings (`[knowledge]`, IF-K02).
#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub user: String,
    pub password: String,
}

#[derive(Clone)]
pub struct Store {
    http: reqwest::Client,
    cfg: StoreConfig,
}

/// Schema of Phase 1 (spec §6). Idempotent: run at every start.
pub const SCHEMA: &str = r#"
DEFINE ANALYZER IF NOT EXISTS kb_text TOKENIZERS blank, class FILTERS lowercase, ascii, snowball(english);
DEFINE TABLE IF NOT EXISTS kb SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name        ON kb TYPE string;
DEFINE FIELD IF NOT EXISTS description ON kb TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS created_at  ON kb TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at  ON kb TYPE datetime;
DEFINE FIELD IF NOT EXISTS embedder    ON kb TYPE option<string>;
DEFINE FIELD IF NOT EXISTS dims        ON kb TYPE option<int>;
DEFINE INDEX IF NOT EXISTS kb_name     ON kb FIELDS name UNIQUE;
DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb         ON document TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS title      ON document TYPE string;
DEFINE FIELD IF NOT EXISTS filename   ON document TYPE string;
DEFINE FIELD IF NOT EXISTS kind       ON document TYPE string;
DEFINE FIELD IF NOT EXISTS bytes      ON document TYPE int;
DEFINE FIELD IF NOT EXISTS sha256     ON document TYPE string;
DEFINE FIELD IF NOT EXISTS path       ON document TYPE string;
DEFINE FIELD IF NOT EXISTS status     ON document TYPE string;
DEFINE FIELD IF NOT EXISTS error      ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS pages      ON document TYPE option<int>;
DEFINE FIELD IF NOT EXISTS n_chunks   ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS n_chars    ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS n_embedded ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS parse_ms   ON document TYPE option<int>;
DEFINE FIELD IF NOT EXISTS created_at ON document TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at ON document TYPE datetime;
DEFINE INDEX IF NOT EXISTS document_kb  ON document FIELDS kb;
DEFINE INDEX IF NOT EXISTS document_sha ON document FIELDS kb, sha256;
DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb        ON chunk TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS doc       ON chunk TYPE record<document>;
DEFINE FIELD IF NOT EXISTS ord       ON chunk TYPE int;
DEFINE FIELD IF NOT EXISTS text      ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS heading   ON chunk TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS page      ON chunk TYPE option<int>;
DEFINE FIELD IF NOT EXISTS chars     ON chunk TYPE int;
DEFINE FIELD IF NOT EXISTS embedding ON chunk TYPE option<array<float>>;
DEFINE INDEX IF NOT EXISTS chunk_kb   ON chunk FIELDS kb;
DEFINE INDEX IF NOT EXISTS chunk_doc  ON chunk FIELDS doc;
DEFINE INDEX IF NOT EXISTS chunk_text ON chunk FIELDS text FULLTEXT ANALYZER kb_text BM25;
DEFINE FIELD IF NOT EXISTS graph_done ON chunk TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS n_graphed ON document TYPE int DEFAULT 0;
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb        ON entity TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS key       ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS name      ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS type      ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS mentions  ON entity TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS community ON entity TYPE option<int>;
DEFINE FIELD IF NOT EXISTS embedding ON entity TYPE option<array<float>>;
DEFINE INDEX IF NOT EXISTS entity_kb   ON entity FIELDS kb;
DEFINE INDEX IF NOT EXISTS entity_name ON entity FIELDS name FULLTEXT ANALYZER kb_text BM25;
DEFINE TABLE IF NOT EXISTS mentions TYPE RELATION FROM chunk TO entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb ON mentions TYPE record<kb>;
DEFINE INDEX IF NOT EXISTS mentions_kb ON mentions FIELDS kb;
DEFINE TABLE IF NOT EXISTS related TYPE RELATION FROM entity TO entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb        ON related TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS predicate ON related TYPE string;
DEFINE FIELD IF NOT EXISTS weight    ON related TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS evidence  ON related TYPE array<record<chunk>> DEFAULT [];
DEFINE INDEX IF NOT EXISTS related_kb ON related FIELDS kb;
DEFINE TABLE IF NOT EXISTS community SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb      ON community TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS n       ON community TYPE int;
DEFINE FIELD IF NOT EXISTS size    ON community TYPE int;
DEFINE FIELD IF NOT EXISTS label   ON community TYPE string;
DEFINE INDEX IF NOT EXISTS community_kb ON community FIELDS kb;
"#;

/// A knowledge base with its counts (IF-K01 `KbSummary`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KbSummary {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub documents: u64,
    #[serde(default)]
    pub chunks: u64,
    #[serde(default)]
    pub chars: u64,
}

/// A document record (IF-K01 `Document`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub kb: String,
    pub title: String,
    pub filename: String,
    pub kind: String,
    pub bytes: u64,
    pub sha256: String,
    #[serde(skip_serializing)]
    pub path: String,
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub pages: Option<u64>,
    #[serde(default)]
    pub n_chunks: u64,
    #[serde(default)]
    pub n_chars: u64,
    /// Passages with a vector (Phase 2); fewer than `n_chunks` while indexing or when the embedder was down.
    #[serde(default)]
    pub n_embedded: u64,
    /// Passages read for the knowledge graph (Phase 3).
    #[serde(default)]
    pub n_graphed: u64,
    #[serde(default)]
    pub parse_ms: Option<u64>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub duplicate: bool,
}

/// A stored chunk as listed for one document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredChunk {
    pub id: String,
    pub ord: u64,
    pub text: String,
    #[serde(default)]
    pub heading: String,
    #[serde(default)]
    pub page: Option<u64>,
    pub chars: u64,
}

/// A search result (IF-K01 `Hit`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hit {
    pub chunk_id: String,
    pub doc_id: String,
    pub doc_title: String,
    pub ord: u64,
    #[serde(default)]
    pub heading: String,
    #[serde(default)]
    pub page: Option<u64>,
    pub text: String,
    pub score: f64,
    #[serde(default)]
    pub rank: usize,
    /// BM25 score when the lexical search found it (hybrid mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_score: Option<f64>,
    /// Cosine similarity to the query when the vector search found it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f64>,
}

/// New-document fields supplied by the upload.
#[derive(Debug, Clone)]
pub struct NewDocument {
    pub title: String,
    pub filename: String,
    pub kind: String,
    pub bytes: u64,
    pub sha256: String,
    pub path: String,
}

const DOC_FIELDS: &str = "record::id(id) AS id, record::id(kb) AS kb, title, filename, kind, bytes, sha256, path, status, \
    error, pages, n_chunks, n_chars, n_embedded, n_graphed, parse_ms, <string> created_at AS created_at, <string> updated_at AS updated_at";

impl Store {
    pub fn new(cfg: StoreConfig) -> Store {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(60))
            .build()
            .expect("HTTP client");
        Store { http, cfg }
    }

    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }

    /// Run SurrealQL with bound variables in this store's namespace and database; returns each statement's result.
    pub async fn query(&self, sql: &str, vars: Value) -> Result<Vec<Value>, KbError> {
        let full = format!(
            "USE NS {} DB {};\n{sql}",
            ident(&self.cfg.namespace),
            ident(&self.cfg.database)
        );
        let mut res = self.rpc(&full, vars).await?;
        res.remove(0); // the USE statement
        Ok(res)
    }

    async fn rpc(&self, sql: &str, vars: Value) -> Result<Vec<Value>, KbError> {
        let body = cbor::encode(&json!({ "id": 1, "method": "query", "params": [sql, vars] }));
        let resp = self
            .http
            .post(format!("{}/rpc", self.cfg.url.trim_end_matches('/')))
            .basic_auth(&self.cfg.user, Some(&self.cfg.password))
            .header("Content-Type", "application/cbor")
            .header("Accept", "application/cbor")
            .body(body)
            .send()
            .await
            .map_err(|e| {
                KbError::Unavailable(format!("the knowledge store is not reachable: {e}"))
            })?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            KbError::Store(format!(
                "bad reply from the knowledge store ({status}): {e}"
            ))
        })?;
        let v = cbor::decode(&bytes).map_err(|e| {
            if status.as_u16() == 401 || status.as_u16() == 403 {
                KbError::Unavailable(
                    "the knowledge store refused the credentials (check [knowledge] password_file)"
                        .into(),
                )
            } else {
                KbError::Store(format!(
                    "bad reply from the knowledge store ({status}): {e}"
                ))
            }
        })?;
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(
                if status.as_u16() == 401
                    || msg.contains("authentication")
                    || msg.contains("credentials")
                {
                    KbError::Unavailable(format!(
                        "the knowledge store refused the credentials: {msg}"
                    ))
                } else {
                    KbError::Store(msg.to_string())
                },
            );
        }
        let results = v
            .get("result")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(results.len());
        for r in results {
            if r.get("status").and_then(Value::as_str) != Some("OK") {
                let msg = r.get("result").map(|m| {
                    m.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| m.to_string())
                });
                return Err(KbError::Store(msg.unwrap_or_else(|| "query failed".into())));
            }
            out.push(r.get("result").cloned().unwrap_or(Value::Null));
        }
        Ok(out)
    }

    /// The server's version string, or an error when it is down.
    pub async fn version(&self) -> Result<String, KbError> {
        let resp = self
            .http
            .get(format!("{}/version", self.cfg.url.trim_end_matches('/')))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|e| {
                KbError::Unavailable(format!("the knowledge store is not reachable: {e}"))
            })?;
        resp.text()
            .await
            .map(|s| s.trim().to_string())
            .map_err(|e| KbError::Unavailable(e.to_string()))
    }

    /// Create the namespace, database and Phase 1 schema (idempotent).
    pub async fn bootstrap(&self) -> Result<(), KbError> {
        let ns = ident(&self.cfg.namespace);
        let db = ident(&self.cfg.database);
        self.rpc(&format!("DEFINE NAMESPACE IF NOT EXISTS {ns}; USE NS {ns}; DEFINE DATABASE IF NOT EXISTS {db};"), json!({}))
            .await?;
        self.query(SCHEMA, json!({})).await.map(|_| ())
    }

    // ------------------------------------------------------------ knowledge bases

    pub async fn list_kbs(&self) -> Result<Vec<KbSummary>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, name, description, <string> created_at AS created_at, \
                 <string> updated_at AS updated_at, \
                 (SELECT VALUE count() FROM document WHERE kb = $parent.id GROUP ALL)[0] ?? 0 AS documents, \
                 (SELECT VALUE math::sum(n_chunks) FROM document WHERE kb = $parent.id GROUP ALL)[0] ?? 0 AS chunks, \
                 (SELECT VALUE math::sum(n_chars) FROM document WHERE kb = $parent.id GROUP ALL)[0] ?? 0 AS chars \
                 FROM kb ORDER BY name ASC;",
                json!({}),
            )
            .await?;
        rows(r.into_iter().next())
    }

    pub async fn get_kb(&self, id: &str) -> Result<KbSummary, KbError> {
        self.list_kbs()
            .await?
            .into_iter()
            .find(|k| k.id == id)
            .ok_or(KbError::NotFound("knowledge base"))
    }

    pub async fn create_kb(&self, name: &str, description: &str) -> Result<KbSummary, KbError> {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 80 {
            return Err(KbError::BadInput(
                "a name of 1 to 80 characters is needed".into(),
            ));
        }
        if self
            .list_kbs()
            .await?
            .iter()
            .any(|k| k.name.eq_ignore_ascii_case(name))
        {
            return Err(KbError::Conflict(format!(
                "a knowledge base called \"{name}\" already exists"
            )));
        }
        let r = self
            .query(
                "CREATE kb SET name = $name, description = $description, created_at = time::now(), updated_at = time::now() \
                 RETURN VALUE record::id(id);",
                json!({ "name": name, "description": description.trim() }),
            )
            .await?;
        let id = first_string(r.into_iter().next())
            .ok_or_else(|| KbError::Store("create returned no id".into()))?;
        self.get_kb(&id).await
    }

    pub async fn update_kb(
        &self,
        id: &str,
        name: Option<&str>,
        description: Option<&str>,
    ) -> Result<KbSummary, KbError> {
        let current = self.get_kb(id).await?;
        let name = name.map(str::trim).unwrap_or(&current.name).to_string();
        if name.is_empty() || name.chars().count() > 80 {
            return Err(KbError::BadInput(
                "a name of 1 to 80 characters is needed".into(),
            ));
        }
        if self
            .list_kbs()
            .await?
            .iter()
            .any(|k| k.id != id && k.name.eq_ignore_ascii_case(&name))
        {
            return Err(KbError::Conflict(format!(
                "a knowledge base called \"{name}\" already exists"
            )));
        }
        let description = description
            .map(str::trim)
            .unwrap_or(&current.description)
            .to_string();
        self.query(
            "UPDATE type::record('kb', $id) SET name = $name, description = $description, updated_at = time::now();",
            json!({ "id": id, "name": name, "description": description }),
        )
        .await?;
        self.get_kb(id).await
    }

    /// Delete a knowledge base with its documents and chunks; returns (documents, chunks) removed.
    pub async fn delete_kb(&self, id: &str) -> Result<(u64, u64), KbError> {
        let kb = self.get_kb(id).await?;
        self.query(
            "LET $k = type::record('kb', $id); DELETE related WHERE kb = $k; DELETE mentions WHERE kb = $k; \
             DELETE entity WHERE kb = $k; DELETE community WHERE kb = $k; DELETE chunk WHERE kb = $k; \
             DELETE document WHERE kb = $k; DELETE $k;",
            json!({ "id": id }),
        )
        .await?;
        Ok((kb.documents, kb.chunks))
    }

    // ------------------------------------------------------------ documents

    pub async fn list_documents(&self, kb: &str) -> Result<Vec<Document>, KbError> {
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM document WHERE kb = type::record('kb', $kb) ORDER BY created_at DESC;"),
                json!({ "kb": kb }),
            )
            .await?;
        rows(r.into_iter().next())
    }

    pub async fn get_document(&self, kb: &str, id: &str) -> Result<Document, KbError> {
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM type::record('document', $id) WHERE kb = type::record('kb', $kb);"),
                json!({ "kb": kb, "id": id }),
            )
            .await?;
        rows::<Document>(r.into_iter().next())?
            .into_iter()
            .next()
            .ok_or(KbError::NotFound("document"))
    }

    pub async fn find_by_sha(&self, kb: &str, sha256: &str) -> Result<Option<Document>, KbError> {
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM document WHERE kb = type::record('kb', $kb) AND sha256 = $sha LIMIT 1;"),
                json!({ "kb": kb, "sha": sha256 }),
            )
            .await?;
        Ok(rows::<Document>(r.into_iter().next())?.into_iter().next())
    }

    pub async fn create_document(&self, kb: &str, d: &NewDocument) -> Result<Document, KbError> {
        let r = self
            .query(
                "CREATE document SET kb = type::record('kb', $kb), title = $title, filename = $filename, kind = $kind, \
                 bytes = $bytes, sha256 = $sha256, path = $path, status = 'queued', created_at = time::now(), \
                 updated_at = time::now() RETURN VALUE record::id(id);",
                json!({ "kb": kb, "title": d.title, "filename": d.filename, "kind": d.kind, "bytes": d.bytes,
                        "sha256": d.sha256, "path": d.path }),
            )
            .await?;
        let id = first_string(r.into_iter().next())
            .ok_or_else(|| KbError::Store("create returned no id".into()))?;
        self.get_document(kb, &id).await
    }

    pub async fn set_status(
        &self,
        id: &str,
        status: &str,
        error: Option<&str>,
    ) -> Result<(), KbError> {
        self.query(
            "UPDATE type::record('document', $id) SET status = $status, error = $error ?? NONE, updated_at = time::now();",
            json!({ "id": id, "status": status, "error": error }),
        )
        .await
        .map(|_| ())
    }

    /// Replace a document's chunks and record the parse results, in one request.
    pub async fn store_chunks(
        &self,
        kb: &str,
        doc: &str,
        title: &str,
        pages: Option<usize>,
        parse_ms: u64,
        chunks: &[Chunk],
    ) -> Result<(), KbError> {
        let n_chars: usize = chunks.iter().map(|c| c.chars).sum();
        // Large documents go in batches so no single request grows without bound.
        self.query(
            "DELETE chunk WHERE doc = type::record('document', $doc);",
            json!({ "doc": doc }),
        )
        .await?;
        for batch in chunks.chunks(500) {
            let rows: Vec<Value> = batch
                .iter()
                .map(|c| json!({ "ord": c.ord, "text": c.text, "heading": c.heading, "page": c.page, "chars": c.chars }))
                .collect();
            self.query(
                "LET $k = type::record('kb', $kb); LET $d = type::record('document', $doc); \
                 FOR $r IN $rows { CREATE chunk CONTENT { kb: $k, doc: $d, ord: $r.ord, text: $r.text, heading: $r.heading, \
                 page: $r.page ?? NONE, chars: $r.chars }; };",
                json!({ "kb": kb, "doc": doc, "rows": rows }),
            )
            .await?;
        }
        self.query(
            "UPDATE type::record('document', $doc) SET title = $title, pages = $pages ?? NONE, parse_ms = $ms, n_chunks = $n, n_graphed = 0, \
             n_chars = $chars, updated_at = time::now(); UPDATE type::record('kb', $kb) SET updated_at = time::now();",
            json!({ "doc": doc, "kb": kb, "title": title, "pages": pages, "ms": parse_ms, "n": chunks.len(), "chars": n_chars }),
        )
        .await
        .map(|_| ())
    }

    /// Delete one document and its chunks; returns the number of chunks removed.
    pub async fn delete_document(&self, kb: &str, id: &str) -> Result<u64, KbError> {
        let d = self.get_document(kb, id).await?;
        self.query(
            "LET $d = type::record('document', $id); DELETE chunk WHERE doc = $d; DELETE $d; \
             UPDATE type::record('kb', $kb) SET updated_at = time::now();",
            json!({ "id": id, "kb": kb }),
        )
        .await?;
        Ok(d.n_chunks)
    }

    pub async fn list_chunks(
        &self,
        kb: &str,
        doc: &str,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<StoredChunk>, u64), KbError> {
        let r = self
            .query(
                "LET $d = type::record('document', $doc); \
                 SELECT record::id(id) AS id, ord, text, heading, page, chars FROM chunk \
                   WHERE doc = $d AND kb = type::record('kb', $kb) ORDER BY ord ASC LIMIT $limit START $offset; \
                 SELECT VALUE count() FROM chunk WHERE doc = $d GROUP ALL;",
                json!({ "kb": kb, "doc": doc, "offset": offset, "limit": limit.clamp(1, 500) }),
            )
            .await?;
        let mut it = r.into_iter().skip(1);
        let chunks = rows(it.next())?;
        let total = it
            .next()
            .and_then(|v| v.get(0).and_then(Value::as_u64))
            .unwrap_or(0);
        Ok((chunks, total))
    }

    /// Full-text BM25 search within one knowledge base (FR-K07, lexical mode). Terms are OR-ed and ranked by score.
    pub async fn search_lexical(
        &self,
        kb: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<Hit>, KbError> {
        let q = query.trim();
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let r = self
            .query(
                "SELECT record::id(id) AS chunk_id, record::id(doc) AS doc_id, doc.title AS doc_title, ord, heading, page, text, \
                   search::score(1) AS score FROM chunk WHERE kb = type::record('kb', $kb) AND text @1,OR@ $q \
                   ORDER BY score DESC LIMIT $k;",
                json!({ "kb": kb, "q": q, "k": k.clamp(1, 100) }),
            )
            .await?;
        let mut hits: Vec<Hit> = rows(r.into_iter().next())?;
        for (i, h) in hits.iter_mut().enumerate() {
            h.rank = i + 1;
            h.lexical_score = Some(h.score);
        }
        Ok(hits)
    }

    // ------------------------------------------------------------ vectors (Phase 2)

    /// The HNSW index over passage vectors, for one embedding width (idempotent).
    pub async fn ensure_vector_index(&self, dims: usize) -> Result<(), KbError> {
        self.query(
            &format!("DEFINE INDEX IF NOT EXISTS chunk_vec ON chunk FIELDS embedding HNSW DIMENSION {dims} DIST COSINE TYPE F32 EFC 150 M 12;"),
            json!({}),
        )
        .await
        .map(|_| ())
    }

    /// A document's passages that have no vector yet: (id, heading, text), in order.
    pub async fn chunks_without_vectors(
        &self,
        doc: &str,
    ) -> Result<Vec<(String, String, String)>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, heading, text, ord FROM chunk \
                 WHERE doc = type::record('document', $doc) AND embedding = NONE ORDER BY ord ASC;",
                json!({ "doc": doc }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows
            .into_iter()
            .map(|v| {
                let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                (s("id"), s("heading"), s("text"))
            })
            .collect())
    }

    /// Store passage vectors (one request per call; the caller batches).
    pub async fn set_vectors(&self, rows: &[(String, Vec<f32>)]) -> Result<(), KbError> {
        let rows: Vec<Value> = rows
            .iter()
            .map(|(id, v)| json!({ "id": id, "v": v }))
            .collect();
        self.query(
            "FOR $r IN $rows { UPDATE type::record('chunk', $r.id) SET embedding = $r.v; };",
            json!({ "rows": rows }),
        )
        .await
        .map(|_| ())
    }

    /// Recount a document's vectors and record the embedder on its knowledge base.
    pub async fn finish_vectors(
        &self,
        kb: &str,
        doc: &str,
        model: &str,
        dims: usize,
    ) -> Result<u64, KbError> {
        let r = self
            .query(
                "LET $d = type::record('document', $doc); \
                 LET $n = (SELECT VALUE count() FROM chunk WHERE doc = $d AND embedding != NONE GROUP ALL)[0] ?? 0; \
                 UPDATE $d SET n_embedded = $n, updated_at = time::now(); \
                 UPDATE type::record('kb', $kb) SET embedder = $model, dims = $dims; \
                 RETURN $n;",
                json!({ "doc": doc, "kb": kb, "model": model, "dims": dims }),
            )
            .await?;
        Ok(r.last().and_then(Value::as_u64).unwrap_or(0))
    }

    /// Ready documents whose passages are not all embedded (indexed again when the embedder is back).
    pub async fn documents_missing_vectors(
        &self,
        kb: Option<&str>,
    ) -> Result<Vec<Document>, KbError> {
        let filter = if kb.is_some() {
            "AND kb = type::record('kb', $kb)"
        } else {
            ""
        };
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM document WHERE status = 'ready' AND n_embedded < n_chunks {filter} ORDER BY created_at ASC;"),
                json!({ "kb": kb }),
            )
            .await?;
        rows(r.into_iter().next())
    }

    /// Every passage vector of a knowledge base, in document and passage order, as one row-major buffer: (floats, rows).
    pub async fn vectors_of(&self, kb: &str, dims: usize) -> Result<(Vec<f32>, usize), KbError> {
        let r = self
            .query(
                "SELECT VALUE embedding FROM chunk WHERE kb = type::record('kb', $kb) AND embedding != NONE ORDER BY doc, ord;",
                json!({ "kb": kb }),
            )
            .await?;
        let rows = r
            .into_iter()
            .next()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let mut out = Vec::with_capacity(rows.len() * dims);
        let mut n = 0;
        for v in rows {
            let Some(a) = v.as_array() else { continue };
            if a.len() != dims {
                continue;
            }
            out.extend(a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32));
            n += 1;
        }
        Ok((out, n))
    }

    /// Passages of a knowledge base that have a vector.
    pub async fn vector_count(&self, kb: &str) -> Result<u64, KbError> {
        let r = self
            .query(
                "RETURN (SELECT VALUE count() FROM chunk WHERE kb = type::record('kb', $kb) AND embedding != NONE GROUP ALL)[0] ?? 0;",
                json!({ "kb": kb }),
            )
            .await?;
        Ok(r.last().and_then(Value::as_u64).unwrap_or(0))
    }

    /// Nearest passages by cosine similarity. `exact` scans the knowledge base's vectors (exact, and fast for a few
    /// thousand passages); otherwise the HNSW index answers (approximate, for large bases).
    pub async fn search_vector(
        &self,
        kb: &str,
        v: &[f32],
        k: usize,
        exact: bool,
    ) -> Result<Vec<Hit>, KbError> {
        let k = k.clamp(1, 100);
        let fields = "record::id(id) AS chunk_id, record::id(doc) AS doc_id, doc.title AS doc_title, ord, heading, page, text";
        let sql = if exact {
            format!(
                "SELECT {fields}, vector::similarity::cosine(embedding, $v) AS score FROM chunk \
                 WHERE kb = type::record('kb', $kb) AND embedding != NONE ORDER BY score DESC LIMIT $k;"
            )
        } else {
            format!(
                "SELECT {fields}, 1 - vector::distance::knn() AS score FROM chunk \
                 WHERE kb = type::record('kb', $kb) AND embedding <|{k},{ef}|> $v ORDER BY score DESC;",
                ef = (k * 4).max(64)
            )
        };
        let r = self
            .query(&sql, json!({ "kb": kb, "v": v, "k": k }))
            .await?;
        let mut hits: Vec<Hit> = rows(r.into_iter().next())?;
        for (i, h) in hits.iter_mut().enumerate() {
            h.rank = i + 1;
            h.vector_score = Some(h.score);
        }
        Ok(hits)
    }

    // ------------------------------------------------------------ graph (Phase 3)

    /// A document's passages not yet read for the graph: (id, heading, text), in order.
    pub async fn chunks_for_graph(
        &self,
        doc: &str,
    ) -> Result<Vec<(String, String, String)>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, heading, text, ord FROM chunk \
                 WHERE doc = type::record('document', $doc) AND graph_done = false ORDER BY ord ASC;",
                json!({ "doc": doc }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows
            .into_iter()
            .map(|v| {
                let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                (s("id"), s("heading"), s("text"))
            })
            .collect())
    }

    /// Which of these entity keys already exist in the knowledge base: key -> (record key, type).
    pub async fn existing_entities(
        &self,
        kb: &str,
        ids: &[String],
    ) -> Result<HashMap<String, String>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, type FROM entity WHERE kb = type::record('kb', $kb) AND record::id(id) IN $ids;",
                json!({ "kb": kb, "ids": ids }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows
            .into_iter()
            .filter_map(|v| {
                Some((
                    v.get("id")?.as_str()?.to_string(),
                    v.get("type")?.as_str()?.to_string(),
                ))
            })
            .collect())
    }

    /// The most similar entity of a type by name vector, if any reaches `min` (entity resolution, FR-K31).
    pub async fn similar_entity(
        &self,
        kb: &str,
        v: &[f32],
        kind: &str,
        min: f64,
    ) -> Result<Option<(String, f64)>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, vector::similarity::cosine(embedding, $v) AS s FROM entity \
                 WHERE kb = type::record('kb', $kb) AND type = $kind AND embedding != NONE ORDER BY s DESC LIMIT 1;",
                json!({ "kb": kb, "v": v, "kind": kind }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows.into_iter().next().and_then(|x| {
            let s = x.get("s")?.as_f64()?;
            (s >= min).then(|| {
                (
                    x.get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    s,
                )
            })
        }))
    }

    /// Record one passage's extraction: entities upserted (mention counts grow), `mentions` edges from the passage,
    /// relations merged by id (weight grows, the passage joins their evidence); the passage is marked read.
    pub async fn store_extraction(
        &self,
        kb: &str,
        doc: &str,
        chunk: &str,
        ents: &[Value],
        rels: &[Value],
    ) -> Result<(), KbError> {
        self.query(
            "LET $k = type::record('kb', $kb); LET $c = type::record('chunk', $chunk); \
             FOR $e IN $ents { \
               LET $id = type::record('entity', $e.id); \
               UPSERT $id SET kb = $k, key = key ?? $e.key, name = name ?? $e.name, type = type ?? $e.type, \
                 mentions += 1, embedding = embedding ?? ($e.v ?? NONE); \
               RELATE $c->mentions->$id SET kb = $k; \
             }; \
             FOR $r IN $rels { \
               INSERT RELATION INTO related { id: type::record('related', $r.id), in: type::record('entity', $r.s), \
                 out: type::record('entity', $r.t), kb: $k, predicate: $r.p, weight: 1, evidence: [$c] } \
               ON DUPLICATE KEY UPDATE weight += 1, evidence += $input.evidence; \
             }; \
             UPDATE $c SET graph_done = true; \
             LET $d = type::record('document', $doc); \
             UPDATE $d SET n_graphed = (SELECT VALUE count() FROM chunk WHERE doc = $d AND graph_done = true GROUP ALL)[0] ?? 0;",
            json!({ "kb": kb, "doc": doc, "chunk": chunk, "ents": ents, "rels": rels }),
        )
        .await
        .map(|_| ())
    }

    /// Forget a knowledge base's graph: entities, relations, mentions, communities; every passage unread again.
    pub async fn reset_graph(&self, kb: &str) -> Result<(), KbError> {
        self.query(
            "LET $k = type::record('kb', $kb); DELETE related WHERE kb = $k; DELETE mentions WHERE kb = $k; \
             DELETE entity WHERE kb = $k; DELETE community WHERE kb = $k; \
             UPDATE chunk SET graph_done = false WHERE kb = $k; UPDATE document SET n_graphed = 0 WHERE kb = $k;",
            json!({ "kb": kb }),
        )
        .await
        .map(|_| ())
    }

    /// Ready documents whose passages are not all read for the graph.
    pub async fn documents_missing_graph(&self, kb: &str) -> Result<Vec<Document>, KbError> {
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM document WHERE kb = type::record('kb', $kb) AND status = 'ready' AND n_graphed < n_chunks ORDER BY created_at ASC;"),
                json!({ "kb": kb }),
            )
            .await?;
        rows(r.into_iter().next())
    }

    /// Remove what deleted passages leave behind: evidence pointing at them, relations without evidence, entities no
    /// passage mentions any more.
    pub async fn prune_graph(&self, kb: &str) -> Result<(), KbError> {
        self.query(
            "LET $k = type::record('kb', $kb); \
             DELETE mentions WHERE kb = $k AND !record::exists(in); \
             UPDATE related SET evidence = evidence.filter(|$c| record::exists($c)) WHERE kb = $k; \
             DELETE related WHERE kb = $k AND array::len(evidence) = 0; \
             UPDATE entity SET mentions = count(<-mentions) WHERE kb = $k; \
             DELETE entity WHERE kb = $k AND mentions = 0;",
            json!({ "kb": kb }),
        )
        .await
        .map(|_| ())
    }

    /// Entity, relation and community counts of a knowledge base.
    pub async fn graph_stats(&self, kb: &str) -> Result<Value, KbError> {
        let r = self
            .query(
                "LET $k = type::record('kb', $kb); RETURN { \
                   entities: (SELECT VALUE count() FROM entity WHERE kb = $k GROUP ALL)[0] ?? 0, \
                   relations: (SELECT VALUE count() FROM related WHERE kb = $k GROUP ALL)[0] ?? 0, \
                   communities: (SELECT VALUE count() FROM community WHERE kb = $k GROUP ALL)[0] ?? 0 };",
                json!({ "kb": kb }),
            )
            .await?;
        Ok(r.into_iter().last().unwrap_or(Value::Null))
    }

    /// Every relation as (source key, target key, weight), for community detection.
    pub async fn graph_edges(&self, kb: &str) -> Result<Vec<(String, String, f64)>, KbError> {
        let r = self
            .query(
                "SELECT record::id(in) AS s, record::id(out) AS t, weight FROM related WHERE kb = type::record('kb', $kb);",
                json!({ "kb": kb }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows
            .into_iter()
            .filter_map(|v| {
                Some((
                    v.get("s")?.as_str()?.to_string(),
                    v.get("t")?.as_str()?.to_string(),
                    v.get("weight")?.as_f64()?,
                ))
            })
            .collect())
    }

    /// Entities as (key, name, mentions).
    pub async fn graph_entities(&self, kb: &str) -> Result<Vec<(String, String, u64)>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS id, name, mentions FROM entity WHERE kb = type::record('kb', $kb) ORDER BY mentions DESC;",
                json!({ "kb": kb }),
            )
            .await?;
        let rows: Vec<Value> = rows(r.into_iter().next())?;
        Ok(rows
            .into_iter()
            .filter_map(|v| {
                Some((
                    v.get("id")?.as_str()?.to_string(),
                    v.get("name")?.as_str()?.to_string(),
                    v.get("mentions")?.as_u64()?,
                ))
            })
            .collect())
    }

    /// Replace the communities: each entity's community number, and one record per community (size, label).
    pub async fn set_communities(
        &self,
        kb: &str,
        assign: &[(String, usize)],
        comms: &[(usize, usize, String)],
    ) -> Result<(), KbError> {
        let assign: Vec<Value> = assign
            .iter()
            .map(|(id, c)| json!({ "id": id, "c": c }))
            .collect();
        let comms: Vec<Value> = comms
            .iter()
            .map(|(n, size, label)| json!({ "n": n, "size": size, "label": label }))
            .collect();
        self.query(
            "LET $k = type::record('kb', $kb); DELETE community WHERE kb = $k; \
             FOR $a IN $assign { UPDATE type::record('entity', $a.id) SET community = $a.c; }; \
             FOR $c IN $comms { CREATE community CONTENT { kb: $k, n: $c.n, size: $c.size, label: $c.label }; };",
            json!({ "kb": kb, "assign": assign, "comms": comms }),
        )
        .await
        .map(|_| ())
    }

    /// Entities whose names match the words (BM25) or, with a vector, are nearest in meaning: (key, name, type, score).
    pub async fn find_entities(
        &self,
        kb: &str,
        q: &str,
        v: Option<&[f32]>,
        k: usize,
    ) -> Result<Vec<Value>, KbError> {
        let k = k.clamp(1, 50);
        let mut out: Vec<Value> = Vec::new();
        if !q.trim().is_empty() {
            let r = self
                .query(
                    "SELECT record::id(id) AS id, name, type, mentions, community, search::score(1) AS score FROM entity \
                     WHERE kb = type::record('kb', $kb) AND name @1,OR@ $q ORDER BY score DESC LIMIT $k;",
                    json!({ "kb": kb, "q": q, "k": k }),
                )
                .await?;
            out = rows(r.into_iter().next())?;
        }
        if let Some(v) = v {
            let r = self
                .query(
                    "SELECT record::id(id) AS id, name, type, mentions, community, vector::similarity::cosine(embedding, $v) AS similarity \
                     FROM entity WHERE kb = type::record('kb', $kb) AND embedding != NONE ORDER BY similarity DESC LIMIT $k;",
                    json!({ "kb": kb, "v": v, "k": k }),
                )
                .await?;
            for e in rows::<Value>(r.into_iter().next())? {
                if !out.iter().any(|x| x["id"] == e["id"]) {
                    out.push(e);
                }
            }
        }
        Ok(out)
    }

    /// Relations touching any of the seed entities, strongest first: subject, predicate, object, weight, evidence.
    pub async fn facts_for(
        &self,
        kb: &str,
        seeds: &[String],
        limit: usize,
    ) -> Result<Vec<Value>, KbError> {
        let r = self
            .query(
                "LET $s = $seeds.map(|$x| type::record('entity', $x)); \
                 SELECT record::id(id) AS id, record::id(in) AS source_id, in.name AS source, predicate, \
                   record::id(out) AS target_id, out.name AS target, weight, evidence.map(|$c| record::id($c)) AS evidence \
                 FROM related WHERE kb = type::record('kb', $kb) AND (in IN $s OR out IN $s) ORDER BY weight DESC LIMIT $limit;",
                json!({ "kb": kb, "seeds": seeds, "limit": limit.clamp(1, 200) }),
            )
            .await?;
        rows(r.into_iter().last())
    }

    /// Passages that mention any of the entities, most mentions first: chunk key -> entity count.
    pub async fn chunks_mentioning(
        &self,
        kb: &str,
        entities: &[String],
    ) -> Result<HashMap<String, usize>, KbError> {
        let r = self
            .query(
                "LET $s = $ents.map(|$x| type::record('entity', $x)); \
                 SELECT VALUE record::id(in) FROM mentions WHERE kb = type::record('kb', $kb) AND out IN $s;",
                json!({ "kb": kb, "ents": entities }),
            )
            .await?;
        let mut m = HashMap::new();
        for v in r
            .into_iter()
            .last()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
        {
            if let Some(id) = v.as_str() {
                *m.entry(id.to_string()).or_insert(0) += 1;
            }
        }
        Ok(m)
    }

    /// Passages by key (for evidence not already among the search results).
    pub async fn hits_by_id(&self, ids: &[String]) -> Result<Vec<Hit>, KbError> {
        let r = self
            .query(
                "SELECT record::id(id) AS chunk_id, record::id(doc) AS doc_id, doc.title AS doc_title, ord, heading, page, text, \
                 0.0 AS score FROM $ids.map(|$x| type::record('chunk', $x));",
                json!({ "ids": ids }),
            )
            .await?;
        rows(r.into_iter().next())
    }

    /// The graph around one entity (or the best-connected entities when none is given), for the Graph view:
    /// nodes {id, name, type, mentions, community} and edges {source, target, predicate, weight}.
    pub async fn graph_view(
        &self,
        kb: &str,
        center: Option<&str>,
        depth: usize,
        limit: usize,
    ) -> Result<Value, KbError> {
        let limit = limit.clamp(5, 300);
        let mut ids: Vec<String> = match center {
            Some(c) => vec![c.to_string()],
            None => self
                .graph_entities(kb)
                .await?
                .into_iter()
                .take(limit.min(40))
                .map(|e| e.0)
                .collect(),
        };
        for _ in 0..depth.clamp(0, 2) {
            if ids.len() >= limit {
                break;
            }
            let facts = self.facts_for(kb, &ids, limit * 2).await?;
            for f in facts {
                for k in ["source_id", "target_id"] {
                    if let Some(x) = f[k].as_str() {
                        if ids.len() < limit && !ids.iter().any(|i| i == x) {
                            ids.push(x.to_string());
                        }
                    }
                }
            }
        }
        let r = self
            .query(
                "LET $s = $ids.map(|$x| type::record('entity', $x)); \
                 SELECT record::id(id) AS id, name, type, mentions, community FROM $s; \
                 SELECT record::id(in) AS source, record::id(out) AS target, predicate, weight FROM related \
                   WHERE kb = type::record('kb', $kb) AND in IN $s AND out IN $s;",
                json!({ "kb": kb, "ids": ids }),
            )
            .await?;
        let mut it = r.into_iter().skip(1);
        let nodes: Vec<Value> = rows::<Value>(it.next())?
            .into_iter()
            .filter(|n| !n["id"].is_null())
            .collect();
        let edges: Vec<Value> = rows(it.next())?;
        Ok(json!({ "nodes": nodes, "edges": edges }))
    }

    /// Documents left mid-ingestion by a previous run (re-queued at start).
    pub async fn unfinished_documents(&self) -> Result<Vec<Document>, KbError> {
        let r = self
            .query(
                &format!("SELECT {DOC_FIELDS} FROM document WHERE status NOT IN ['ready', 'failed'] ORDER BY created_at ASC;"),
                json!({}),
            )
            .await?;
        rows(r.into_iter().next())
    }
}

/// A SurrealQL identifier: letters, digits and underscores only (namespace and database names come from config).
fn ident(s: &str) -> String {
    let clean: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if clean.is_empty() {
        "nafs".into()
    } else {
        clean
    }
}

fn rows<T: for<'de> Deserialize<'de>>(v: Option<Value>) -> Result<Vec<T>, KbError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(v) => serde_json::from_value(v)
            .map_err(|e| KbError::Store(format!("unexpected record shape: {e}"))),
    }
}

fn first_string(v: Option<Value>) -> Option<String> {
    match v? {
        Value::Array(a) => a
            .into_iter()
            .next()
            .and_then(|x| x.as_str().map(str::to_string)),
        Value::String(s) => Some(s),
        _ => None,
    }
}

/// CBOR <-> JSON for SurrealDB's RPC (its CBOR tags: 0 and 12 datetime, 6 NONE, 7 table, 8 record id, 9 and 37 UUID,
/// 10 decimal, 13 and 14 duration).
mod cbor {
    use ciborium::value::{Integer, Value as C};
    use serde_json::{Map, Number, Value as J};

    pub fn encode(v: &J) -> Vec<u8> {
        let mut out = Vec::new();
        ciborium::into_writer(&to_cbor(v), &mut out).expect("CBOR encoding into a Vec cannot fail");
        out
    }

    fn to_cbor(v: &J) -> C {
        match v {
            J::Null => C::Null,
            J::Bool(b) => C::Bool(*b),
            J::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => C::Integer(i.into()),
                (_, Some(u)) => C::Integer(u.into()),
                _ => C::Float(n.as_f64().unwrap_or(0.0)),
            },
            J::String(s) => C::Text(s.clone()),
            J::Array(a) => C::Array(a.iter().map(to_cbor).collect()),
            J::Object(o) => C::Map(
                o.iter()
                    .map(|(k, v)| (C::Text(k.clone()), to_cbor(v)))
                    .collect(),
            ),
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<J, String> {
        let v: C = ciborium::from_reader(bytes).map_err(|e| e.to_string())?;
        Ok(to_json(v))
    }

    fn to_json(v: C) -> J {
        match v {
            C::Null => J::Null,
            C::Bool(b) => J::Bool(b),
            C::Integer(i) => int(i),
            C::Float(f) => Number::from_f64(f).map(J::Number).unwrap_or(J::Null),
            C::Text(s) => J::String(s),
            C::Bytes(b) => J::String(b.iter().map(|x| format!("{x:02x}")).collect()),
            C::Array(a) => J::Array(a.into_iter().map(to_json).collect()),
            C::Map(m) => {
                let mut o = Map::new();
                for (k, v) in m {
                    let key = match to_json(k) {
                        J::String(s) => s,
                        other => other.to_string(),
                    };
                    o.insert(key, to_json(v));
                }
                J::Object(o)
            }
            C::Tag(tag, inner) => tagged(tag, *inner),
            _ => J::Null,
        }
    }

    fn int(i: Integer) -> J {
        let i: i128 = i.into();
        if let Ok(v) = i64::try_from(i) {
            J::Number(v.into())
        } else if let Ok(v) = u64::try_from(i) {
            J::Number(v.into())
        } else {
            J::String(i.to_string())
        }
    }

    fn tagged(tag: u64, inner: C) -> J {
        match (tag, inner) {
            (6, _) => J::Null,
            // Record id: [table, key]; written as "table:key".
            (8, C::Array(mut parts)) if parts.len() == 2 => {
                let key = parts.pop().map(to_json).unwrap_or(J::Null);
                let table = parts.pop().map(to_json).unwrap_or(J::Null);
                let key = match key {
                    J::String(s) => s,
                    other => other.to_string(),
                };
                J::String(format!("{}:{key}", table.as_str().unwrap_or("")))
            }
            (8, C::Text(s))
            | (7, C::Text(s))
            | (0, C::Text(s))
            | (9, C::Text(s))
            | (13, C::Text(s)) => J::String(s),
            // Datetime as [seconds, nanoseconds].
            (12, C::Array(parts)) => {
                let n: Vec<i128> = parts
                    .into_iter()
                    .filter_map(|p| p.as_integer().map(i128::from))
                    .collect();
                let secs = n.first().copied().unwrap_or(0) as i64;
                let nanos = n.get(1).copied().unwrap_or(0) as u32;
                chrono::DateTime::from_timestamp(secs, nanos)
                    .map(|d| J::String(d.to_rfc3339_opts(chrono::SecondsFormat::Micros, true)))
                    .unwrap_or(J::Null)
            }
            (37, C::Bytes(b)) if b.len() == 16 => {
                let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
                J::String(format!(
                    "{}-{}-{}-{}-{}",
                    &h[..8],
                    &h[8..12],
                    &h[12..16],
                    &h[16..20],
                    &h[20..]
                ))
            }
            (10, C::Text(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(Number::from_f64)
                .map(J::Number)
                .unwrap_or(J::String(s)),
            (_, other) => to_json(other),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn round_trips_and_tags() {
            let v = json!({ "s": "Caption: Model of the gate.", "n": -3, "f": 1.5, "a": [true, null], "o": { "k": 1 } });
            assert_eq!(decode(&encode(&v)).unwrap(), v);
            let rec = C::Tag(
                8,
                Box::new(C::Array(vec![C::Text("kb".into()), C::Text("x1".into())])),
            );
            assert_eq!(
                tagged(
                    8,
                    match rec {
                        C::Tag(_, b) => *b,
                        _ => unreachable!(),
                    }
                ),
                json!("kb:x1")
            );
            assert_eq!(tagged(6, C::Null), J::Null);
            let dt = C::Array(vec![
                C::Integer(1_762_480_000.into()),
                C::Integer(5_000.into()),
            ]);
            assert_eq!(tagged(12, dt), json!("2025-11-07T01:46:40.000005Z"));
            assert_eq!(tagged(10, C::Text("2.5".into())), json!(2.5));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_sanitised() {
        assert_eq!(ident("nafs"), "nafs");
        assert_eq!(ident("kb; REMOVE NAMESPACE x"), "kbREMOVENAMESPACEx");
        assert_eq!(ident(""), "nafs");
    }

    #[test]
    fn rows_and_first_string() {
        let v: Vec<StoredChunk> = rows(Some(
            json!([{ "id": "a", "ord": 0, "text": "t", "chars": 1 }]),
        ))
        .unwrap();
        assert_eq!(v[0].heading, "");
        assert!(rows::<StoredChunk>(Some(Value::Null)).unwrap().is_empty());
        assert_eq!(first_string(Some(json!(["x"]))).as_deref(), Some("x"));
        assert_eq!(first_string(Some(json!("y"))).as_deref(), Some("y"));
    }
}
