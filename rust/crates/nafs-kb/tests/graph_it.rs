//! The knowledge graph against a running SurrealDB, with a mock extraction model (a canned OpenAI-style reply chosen by
//! the passage's words), so the test is fast and deterministic. Skips without SurrealDB.

use std::time::{Duration, Instant};

use nafs_kb::{ChunkOpts, GraphConfig, Kb, KbConfig, KbError, StoreConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A minimal HTTP server answering every POST with an extraction picked by keywords in the request body.
async fn mock_llm() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                // Read headers, then the body by Content-Length.
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(h) = find(&buf, b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..h]).to_lowercase();
                        let len: usize = head
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length:")
                                    .map(|v| v.trim().parse().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        while buf.len() < h + 4 + len {
                            let n = sock.read(&mut tmp).await.unwrap_or(0);
                            if n == 0 {
                                break;
                            }
                            buf.extend_from_slice(&tmp[..n]);
                        }
                        break;
                    }
                }
                let body = String::from_utf8_lossy(&buf);
                let content = if body.contains("PagedAttention") {
                    // FR-K53/55: a context line, and quotes that occur in the passage; one relation carries an
                    // invented quote and must be dropped by the gate.
                    r#"{"context":"What vLLM is built on and where that idea came from.","entities":[{"name":"vLLM","type":"system"},{"name":"PagedAttention","type":"algorithm"},{"name":"virtual memory","type":"concept"}],"relations":[{"source":"vLLM","predicate":"uses","target":"PagedAttention","quote":"vLLM builds on PagedAttention, which is inspired by virtual memory in operating systems."},{"source":"PagedAttention","predicate":"inspired_by","target":"virtual memory","quote":"vLLM builds on PagedAttention, which is inspired by virtual memory in operating systems."},{"source":"virtual memory","predicate":"replaces","target":"vLLM","quote":"Virtual memory replaces vLLM in every serving system."}]}"#
                } else if body.contains("Orca") {
                    r#"{"context":"How vLLM compares with Orca.","entities":[{"name":"the vLLM","type":"system"},{"name":"Orca","type":"system"}],"relations":[{"source":"the vLLM","predicate":"compared to","target":"Orca","quote":"Against Orca, the vLLM system serves more requests per second at equal latency."}]}"#
                } else if body.contains("Memorilla") {
                    r#"{"context":"What Memorilla does with documents.","entities":[{"name":"Memorilla","type":"method"},{"name":"Qwen3-8B","type":"model"}],"relations":[{"source":"Memorilla","predicate":"uses","target":"Qwen3-8B","quote":"Memorilla compresses documents into sixteen memory tokens for a frozen Qwen3-8B decoder."}]}"#
                } else {
                    "I cannot do that."
                };
                let reply = serde_json::json!({"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"completion_tokens": 42}}).to_string();
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len());
                let _ = sock.write_all(resp.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}/v1")
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

async fn settle(kb: &Kb, base: &str) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(60) {
        let docs = kb.store().list_documents(base).await.unwrap();
        if docs
            .iter()
            .all(|d| d.status == "ready" || d.status == "failed")
            && kb.queue().pending == 0
            && kb.queue().running == 0
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("documents did not settle");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_build_search_and_prune() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let Ok(password) = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))
    else {
        eprintln!("skipped: no SurrealDB password file");
        return;
    };
    let dir = std::env::temp_dir().join(format!("nafs-kb-graph-{}", uuid::Uuid::new_v4().simple()));
    let db = format!(
        "kb_graph_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    let kb = Kb::start(KbConfig {
        store: StoreConfig {
            url: std::env::var("NAFS_KB_TEST_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8300".into()),
            namespace: "nafs".into(),
            database: db.clone(),
            user: "root".into(),
            password: password.trim().into(),
        },
        files_dir: dir.clone(),
        max_upload_bytes: 1 << 20,
        max_files_per_request: 5,
        chunk: ChunkOpts {
            chars: 300,
            overlap: 0,
        },
        search_k: 5,
        embed: None, // name-key merging only; resolution by meaning is covered by the live run
        exact_search_below: 5000,
        graph: Some(GraphConfig {
            llm_url: mock_llm().await,
            model: "mock".into(),
            max_tokens: 300,
            merge_similarity: 0.92,
        }),
        graph_weight: 1.0,
        graph_beam: 8,
        graph_decay: 0.6,
        limits: nafs_kb::Limits::default(),
    });
    match kb.ready().await {
        Err(KbError::Unavailable(e)) => {
            eprintln!("skipped: SurrealDB not reachable ({e})");
            return;
        }
        r => r.unwrap(),
    }
    let store = kb.store();
    let base = store.create_kb("Graph test", "").await.unwrap();
    let a = "# Serving\n\nvLLM builds on PagedAttention, which is inspired by virtual memory in operating systems.\n\n\
             # Comparison\n\nAgainst Orca, the vLLM system serves more requests per second at equal latency."
        .to_string();
    let b = "# Memory\n\nMemorilla compresses documents into sixteen memory tokens for a frozen Qwen3-8B decoder.\n\n\
             # Note\n\nThis passage names nothing the extractor knows, so the mock answers with prose."
        .to_string();
    let da = kb
        .upload(&base.id, "serving.md", a.as_bytes())
        .await
        .unwrap();
    let db_ = kb
        .upload(&base.id, "memory.md", b.as_bytes())
        .await
        .unwrap();
    settle(&kb, &base.id).await;

    let queued = kb.build_graph(&base.id, false).await.unwrap();
    assert_eq!(queued, 2);
    settle(&kb, &base.id).await;
    for d in store.list_documents(&base.id).await.unwrap() {
        assert_eq!(d.n_graphed, d.n_chunks, "{} {:?}", d.title, d.error);
        assert!(d.error.is_none(), "{:?}", d.error);
    }
    // A second build finds nothing left to read.
    assert_eq!(kb.build_graph(&base.id, false).await.unwrap(), 0);

    let stats = store.graph_stats(&base.id).await.unwrap();
    // vLLM ("vLLM" and "the vLLM" share a key), PagedAttention, virtual memory, Orca, Memorilla, Qwen3-8B.
    assert_eq!(stats["entities"], 6, "{stats}");
    assert_eq!(
        stats["relations"], 4,
        "{stats}: the relation with an invented quote was dropped"
    );
    assert!(stats["communities"].as_u64().unwrap() >= 2, "{stats}");
    // FR-K55: every stored relation carries the sentence that states it; FR-K53: the passage got its context line
    // and the words index carries it.
    let facts = store
        .facts_for(&base.id, &[nafs_kb::graph::entity_id(&base.id, "vllm")], 20)
        .await
        .unwrap();
    assert!(!facts.is_empty());
    assert!(
        facts
            .iter()
            .all(|f| f["quote"].as_str().is_some_and(|q| q.len() > 20)),
        "{facts:?}"
    );
    let lex = store
        .search_lexical(&base.id, "idea came from", 5)
        .await
        .unwrap();
    assert!(
        lex.iter().any(|h| h.text.contains("PagedAttention")),
        "the context line is searchable by its words: {lex:?}"
    );

    let ents = store.graph_entities(&base.id).await.unwrap();
    let vllm = ents
        .iter()
        .find(|e| e.1 == "vLLM")
        .expect("vLLM merged under its first spelling");
    assert_eq!(vllm.2, 2, "mentioned by two passages");
    let facts = store
        .facts_for(&base.id, std::slice::from_ref(&vllm.0), 10)
        .await
        .unwrap();
    let preds: Vec<&str> = facts
        .iter()
        .filter_map(|f| f["predicate"].as_str())
        .collect();
    assert!(
        preds.contains(&"uses") && preds.contains(&"compared to"),
        "{facts:?}"
    );
    let uses = facts.iter().find(|f| f["predicate"] == "uses").unwrap();
    assert_eq!(uses["target"], "PagedAttention");
    assert_eq!(uses["evidence"].as_array().unwrap().len(), 1);

    // The graph view around vLLM reaches virtual memory in two hops, with the edges between shown nodes.
    let view = store
        .graph_view(&base.id, Some(&vllm.0), 2, 50)
        .await
        .unwrap();
    let names: Vec<&str> = view["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();
    assert!(
        names.contains(&"virtual memory") && names.contains(&"Orca"),
        "{names:?}"
    );
    assert!(!names.contains(&"Memorilla"));
    assert_eq!(view["edges"].as_array().unwrap().len(), 3);

    // Graph search: entities by name, their facts, passages mentioning them first.
    let (facts, r) = kb
        .graph_search(&base.id, "What does vLLM use?", 3)
        .await
        .unwrap();
    assert!(
        facts
            .iter()
            .any(|f| f["source"] == "vLLM" && f["predicate"] == "uses"),
        "{facts:?}"
    );
    assert_eq!(r.results[0].doc_id, da.id);

    // Deleting a document removes what only it supported.
    kb.delete_document(&base.id, &da.id).await.unwrap();
    let stats = store.graph_stats(&base.id).await.unwrap();
    assert_eq!(stats["entities"], 2, "{stats}");
    assert_eq!(stats["relations"], 1, "{stats}");
    let _ = db_;

    kb.delete_kb(&base.id).await.unwrap();
    assert_eq!(store.graph_stats(&base.id).await.unwrap()["entities"], 0);
    let _ = store
        .query(&format!("REMOVE DATABASE {db};"), serde_json::json!({}))
        .await;
    let _ = std::fs::remove_dir_all(&dir);
}
