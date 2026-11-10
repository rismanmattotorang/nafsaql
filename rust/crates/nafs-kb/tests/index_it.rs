//! The in-process index against a running SurrealDB (Phase 25, FR-K67 to FR-K73): it answers what the store answers,
//! follows uploads, re-reads and deletions, and a re-read keeps the vectors of unchanged passages. Skips when the server
//! or its password file is missing. Uses a throwaway database, removed at the end.

use std::time::{Duration, Instant};

use nafs_kb::{ChunkOpts, EmbedConfig, Engine, Kb, KbConfig, KbError, SearchMode, StoreConfig};

fn config(db: &str, dir: &std::path::Path) -> Option<KbConfig> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let password = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))
        .ok()?
        .trim()
        .to_string();
    Some(KbConfig {
        store: StoreConfig {
            url: std::env::var("NAFS_KB_TEST_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8300".into()),
            namespace: "nafs".into(),
            database: db.into(),
            user: "root".into(),
            password,
        },
        files_dir: dir.to_path_buf(),
        max_upload_bytes: 1024 * 1024,
        max_files_per_request: 5,
        chunk: ChunkOpts {
            chars: 400,
            overlap: 50,
        },
        search_k: 5,
        embed: Some(EmbedConfig {
            url: std::env::var("NAFS_EMBED_TEST_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8203".into()),
            model: "Qwen3-Embedding-0.6B".into(),
            dims: 1024,
            batch: 8,
        }),
        exact_search_below: 5000,
        graph: None,
        graph_weight: 1.0,
        vector_weight: 0.1,
        vector_cache_mb: 0,
        graph_beam: 8,
        graph_decay: 0.6,
        index: Default::default(),
        limits: nafs_kb::Limits::default(),
    })
}

async fn settled(kb: &Kb, base: &str, doc: &str) -> nafs_kb::Document {
    let embedding = matches!(kb.embedder_status().await, Some((_, true)));
    let t = Instant::now();
    loop {
        let d = kb.store().get_document(base, doc).await.unwrap();
        let words = d.status == "ready" || d.status == "failed";
        let vectors = !embedding || d.n_embedded >= d.n_chunks || d.error.is_some();
        if (words && vectors) || t.elapsed() > Duration::from_secs(120) {
            return d;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_index_follows_the_store() {
    let dir = std::env::temp_dir().join(format!("nafs-kb-ix-{}", uuid::Uuid::new_v4().simple()));
    let db = format!("kb_ix_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
    let Some(cfg) = config(&db, &dir) else {
        eprintln!("skipped: no SurrealDB password file (scripts/serve_surreal.sh start)");
        return;
    };
    let kb = Kb::start(cfg);
    match kb.ready().await {
        Err(KbError::Unavailable(e)) => {
            eprintln!("skipped: SurrealDB not reachable ({e})");
            return;
        }
        r => r.unwrap(),
    }
    let store = kb.store();
    let base = store.create_kb("Index test", "").await.unwrap();
    let notes =
        "# Storage\n\nSurrealDB listens on port 8300 and keeps every passage.\n\n## Limits\n\n"
            .to_string()
            + &"An upload may carry fifty megabytes and twenty files. ".repeat(12);
    let d1 = kb
        .upload(&base.id, "notes.md", notes.as_bytes())
        .await
        .unwrap();
    let d1 = settled(&kb, &base.id, &d1.id).await;
    assert_eq!(d1.status, "ready", "{:?}", d1.error);

    // The index answers, and it finds what the store finds.
    let by_index = kb
        .search_with(
            &base.id,
            "which port does SurrealDB listen on",
            3,
            SearchMode::Lexical,
            Engine::Auto,
        )
        .await
        .unwrap();
    assert_eq!(by_index.engine, "index");
    let by_store = kb
        .search_with(
            &base.id,
            "which port does SurrealDB listen on",
            3,
            SearchMode::Lexical,
            Engine::Store,
        )
        .await
        .unwrap();
    assert_eq!(by_store.engine, "store");
    assert_eq!(by_index.results[0].chunk_id, by_store.results[0].chunk_id);
    assert!(by_index.results[0].text.contains("8300"));
    let ix = kb.index_of(&base.id).await.unwrap().expect("loaded");
    assert_eq!(ix.passages() as u64, d1.n_chunks);
    assert_eq!(ix.vectors() as u64, d1.n_embedded);

    // A document uploaded afterwards is in the index as soon as it is ready by its words.
    let d2 = kb
        .upload(
            &base.id,
            "graph.md",
            b"# Graph\n\nPersonalized PageRank walks the passage and term graph.",
        )
        .await
        .unwrap();
    let d2 = settled(&kb, &base.id, &d2.id).await;
    let r = kb
        .search(&base.id, "PageRank", 3, SearchMode::Lexical)
        .await
        .unwrap();
    assert_eq!(r.engine, "index");
    assert_eq!(r.results[0].doc_id, d2.id);
    let embedded = d1.n_embedded > 0;
    if embedded {
        let ix = kb.index_of(&base.id).await.unwrap().expect("loaded");
        assert_eq!(ix.vectors() as u64, d1.n_embedded + d2.n_embedded);
        let r = kb
            .search(
                &base.id,
                "a walk over a graph of terms",
                3,
                SearchMode::Hybrid,
            )
            .await
            .unwrap();
        assert_eq!(r.mode, SearchMode::Hybrid);
        assert!(r.results.iter().all(|h| h.vector_score.is_some()));

        // FR-K73: reading the document again makes no call to the embedding server for unchanged passages. The
        // vectors are there the moment the document is ready by its words.
        kb.reingest(&base.id, &d1.id).await.unwrap();
        let t = Instant::now();
        let again = loop {
            let d = store.get_document(&base.id, &d1.id).await.unwrap();
            if d.status == "ready" && d.n_chunks > 0 || t.elapsed() > Duration::from_secs(60) {
                break d;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(
            again.n_embedded, again.n_chunks,
            "vectors were reused, not recomputed"
        );
        let again = settled(&kb, &base.id, &d1.id).await;
        assert_eq!(again.n_embedded, d1.n_embedded);
    }

    // A deleted document leaves the index at once.
    kb.delete_document(&base.id, &d2.id).await.unwrap();
    let r = kb
        .search(&base.id, "PageRank", 3, SearchMode::Lexical)
        .await
        .unwrap();
    assert_eq!(r.engine, "index");
    assert!(r.results.is_empty(), "{:?}", r.results);

    // The graph made from the text needs no model: two passages naming SurrealDB make it a term.
    let d3 = kb
        .upload(
            &base.id,
            "more.md",
            b"# More\n\nSurrealDB also holds the graph tables.",
        )
        .await
        .unwrap();
    settled(&kb, &base.id, &d3.id).await;
    let view = kb
        .text_graph_view(&base.id, None, 20)
        .await
        .expect("index loaded");
    assert!(
        view["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["name"] == "SurrealDB"),
        "{view}"
    );

    kb.delete_kb(&base.id).await.unwrap();
    assert!(kb.index_of(&base.id).await.is_err());
    let _ = store
        .query(
            &format!("REMOVE DATABASE IF EXISTS {db};"),
            serde_json::json!({}),
        )
        .await;
    let _ = std::fs::remove_dir_all(&dir);
}
