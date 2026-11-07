//! Integration test against a running SurrealDB (scripts/serve_surreal.sh start). Skips when the server or its password
//! file is missing. Uses a throwaway database, removed at the end.

use std::time::{Duration, Instant};

use nafs_kb::{ChunkOpts, EmbedConfig, Kb, KbConfig, KbError, SearchMode, StoreConfig};

fn config(db: &str, dir: &std::path::Path) -> Option<KbConfig> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let url = std::env::var("NAFS_KB_TEST_URL").unwrap_or_else(|_| "http://127.0.0.1:8300".into());
    let password = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))
        .ok()?
        .trim()
        .to_string();
    Some(KbConfig {
        store: StoreConfig {
            url,
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
            url: std::env::var("NAFS_KB_TEST_EMBED_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8203".into()),
            model: "Qwen3-Embedding-0.6B".into(),
            dims: 1024,
            batch: 8,
        }),
        exact_search_below: 5000,
        graph: None,
    })
}

async fn wait_ready(kb: &Kb, base: &str, doc: &str) -> nafs_kb::Document {
    let t = Instant::now();
    loop {
        let d = kb.store().get_document(base, doc).await.unwrap();
        if d.status == "ready" || d.status == "failed" || t.elapsed() > Duration::from_secs(30) {
            return d;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upload_search_and_delete_round_trip() {
    let dir = std::env::temp_dir().join(format!("nafs-kb-it-{}", uuid::Uuid::new_v4().simple()));
    let db = format!("kb_it_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
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

    let base = store.create_kb("Test papers", "integration").await.unwrap();
    assert!(matches!(
        store.create_kb("test PAPERS", "").await,
        Err(KbError::Conflict(_))
    ));
    assert!(matches!(
        store.create_kb("  ", "").await,
        Err(KbError::BadInput(_))
    ));

    let md =
        "# Memory\n\n## Slots\n\nSixteen learned slots attend to the question embedding first.\n\n\
              ## Refinement\n\nCross attention blocks read every stored document embedding. "
            .to_string()
            + &"The projection maps slots into the decoder space. ".repeat(20);
    let html = "<html><head><title>Graphs</title></head><body><h1>Graph retrieval</h1><p>SurrealDB relates entities \
                with RELATE edges and walks them with arrows.</p></body></html>";
    let d1 = kb
        .upload(&base.id, "notes.md", md.as_bytes())
        .await
        .unwrap();
    let d2 = kb
        .upload(&base.id, "../../graphs.html", html.as_bytes())
        .await
        .unwrap();
    assert_eq!(d2.filename, "graphs.html");
    // The same bytes again: the existing record, flagged.
    let again = kb.upload(&base.id, "copy.md", md.as_bytes()).await.unwrap();
    assert!(again.duplicate && again.id == d1.id);
    assert!(matches!(
        kb.upload(&base.id, "x.exe", b"MZ").await,
        Err(KbError::Unsupported(_))
    ));
    assert!(matches!(
        kb.upload(&base.id, "fake.pdf", b"not a pdf").await,
        Err(KbError::Unsupported(_))
    ));
    assert!(matches!(
        kb.upload(&base.id, "big.txt", &vec![b'a'; 2 * 1024 * 1024])
            .await,
        Err(KbError::TooLarge(_))
    ));

    let r1 = wait_ready(&kb, &base.id, &d1.id).await;
    let r2 = wait_ready(&kb, &base.id, &d2.id).await;
    assert_eq!(r1.status, "ready", "{:?}", r1.error);
    assert_eq!(r2.status, "ready", "{:?}", r2.error);
    assert_eq!(r1.title, "Memory");
    assert_eq!(r2.title, "Graphs");
    assert!(r1.n_chunks >= 2, "{}", r1.n_chunks);

    let (chunks, total) = store.list_chunks(&base.id, &d1.id, 0, 50).await.unwrap();
    assert_eq!(total as usize, chunks.len());
    assert!(chunks.windows(2).all(|w| w[0].ord < w[1].ord));
    assert!(
        chunks.iter().any(|c| c.heading == "Memory > Refinement"),
        "{chunks:?}"
    );

    let hits = store
        .search_lexical(&base.id, "relate entities edges", 5)
        .await
        .unwrap();
    assert_eq!(
        hits.first().map(|h| h.doc_id.as_str()),
        Some(d2.id.as_str()),
        "{hits:?}"
    );
    assert_eq!(hits[0].doc_title, "Graphs");
    assert_eq!(hits[0].rank, 1);
    let hits = store
        .search_lexical(&base.id, "slots decoder projection", 5)
        .await
        .unwrap();
    assert!(
        !hits.is_empty() && hits.iter().all(|h| h.doc_id == d1.id),
        "{hits:?}"
    );
    assert!(store
        .search_lexical(&base.id, "   ", 5)
        .await
        .unwrap()
        .is_empty());

    // Strings that look like record ids ("Caption: Model", "MEMORILLA: LATENT") must stay strings, in stored text,
    // headings and queries (the JSON wire format turned them into records).
    let colon = "# MEMORILLA: LATENT MEMORY\n\nCaption: Model of the gate keeps its colons.";
    let dc = kb
        .upload(&base.id, "colon.md", colon.as_bytes())
        .await
        .unwrap();
    let rc = wait_ready(&kb, &base.id, &dc.id).await;
    assert_eq!(rc.status, "ready", "{:?}", rc.error);
    assert_eq!(rc.title, "MEMORILLA: LATENT MEMORY");
    let (cc, _) = store.list_chunks(&base.id, &dc.id, 0, 10).await.unwrap();
    assert_eq!(cc[0].text, colon);
    assert_eq!(cc[0].heading, "MEMORILLA: LATENT MEMORY");
    let hits = store
        .search_lexical(&base.id, "Caption: Model gate", 3)
        .await
        .unwrap();
    assert_eq!(
        hits.first().map(|h| h.doc_id.as_str()),
        Some(dc.id.as_str()),
        "{hits:?}"
    );
    kb.delete_document(&base.id, &dc.id).await.unwrap();

    // Phase 2: vectors, when the embedding server runs (scripts/serve_smiley.sh embed).
    let embedding = matches!(kb.embedder_status().await, Some((_, true)));
    if embedding {
        assert_eq!(r1.n_embedded, r1.n_chunks, "{:?}", r1.error);
        assert_eq!(r2.n_embedded, r2.n_chunks);
        assert!(r1.error.is_none());
        // Meaning, not words: no passage contains "linking" or "database".
        let v = kb
            .search(
                &base.id,
                "linking things together in a database",
                3,
                SearchMode::Vector,
            )
            .await
            .unwrap();
        assert_eq!(v.mode, SearchMode::Vector);
        assert_eq!(v.results[0].doc_id, d2.id, "{:?}", v.results);
        assert!(v.results[0].vector_score.unwrap() > 0.3);
        let h = kb
            .search(&base.id, "relate entities edges", 3, SearchMode::Hybrid)
            .await
            .unwrap();
        assert_eq!(h.mode, SearchMode::Hybrid);
        assert_eq!(h.results[0].doc_id, d2.id);
        assert!(h.results[0].lexical_score.is_some() && h.results[0].vector_score.is_some());
        // The exact scan and the HNSW index agree on the best passage.
        let qv = kb
            .search(
                &base.id,
                "learned slots and the question",
                1,
                SearchMode::Vector,
            )
            .await
            .unwrap();
        assert_eq!(qv.results[0].doc_id, d1.id);
    } else {
        eprintln!("vector checks skipped: no embedding server on :8203");
        let h = kb
            .search(&base.id, "relate entities edges", 3, SearchMode::Hybrid)
            .await
            .unwrap();
        assert_eq!(h.mode, SearchMode::Lexical);
        assert!(h.fallback.is_some());
        assert!(
            r1.error.is_some(),
            "a document without vectors carries a note"
        );
    }

    let summary = store.get_kb(&base.id).await.unwrap();
    assert_eq!(summary.documents, 2);
    assert_eq!(summary.chunks, r1.n_chunks + r2.n_chunks);

    // Re-reading keeps the counts; deleting removes the chunks and the stored file.
    kb.reingest(&base.id, &d2.id).await.unwrap();
    assert_eq!(
        wait_ready(&kb, &base.id, &d2.id).await.n_chunks,
        r2.n_chunks
    );
    let removed = kb.delete_document(&base.id, &d2.id).await.unwrap();
    assert_eq!(removed, r2.n_chunks);
    assert!(!std::path::Path::new(&r2.path).exists());
    assert!(store
        .search_lexical(&base.id, "relate entities edges", 5)
        .await
        .unwrap()
        .iter()
        .all(|h| h.doc_id != d2.id));
    assert!(matches!(
        store.get_document(&base.id, &d2.id).await,
        Err(KbError::NotFound(_))
    ));

    let renamed = store
        .update_kb(&base.id, Some("Renamed"), None)
        .await
        .unwrap();
    assert_eq!(
        (renamed.name.as_str(), renamed.description.as_str()),
        ("Renamed", "integration")
    );
    let (docs, chunks) = kb.delete_kb(&base.id).await.unwrap();
    assert_eq!((docs, chunks), (1, r1.n_chunks));
    assert!(store.list_kbs().await.unwrap().is_empty());
    assert!(!dir.join(&base.id).exists());

    let _ = store
        .query(&format!("REMOVE DATABASE {db};"), serde_json::json!({}))
        .await;
    let _ = std::fs::remove_dir_all(&dir);
}
