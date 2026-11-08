//! NFR-K07 (Phase 8): a knowledge base of 1,000 documents and 50,000 passages stays usable.
//!
//! Heavy and slow, so it does nothing unless `NAFS_KB_SCALE=1`. It builds a base of synthetic documents straight in
//! SurrealDB (no parser, no embedder: passages from a skewed vocabulary, and random unit vectors), then times the
//! operations a person waits for, checks that a planted word and a planted vector are found, and times the deletion.
//! `NAFS_KB_SCALE_DOCS` (default 1000) and `NAFS_KB_SCALE_CHUNKS` (default 50 per document) change the size;
//! `NAFS_KB_SCALE_OUT` names a JSON file to write the numbers to. Run it with
//! `NAFS_KB_SCALE=1 cargo test --release -p nafs-kb --test scale_it -- --nocapture`.

use std::time::Instant;

use nafs_kb::{Chunk, NewDocument, Store, StoreConfig};

const DIMS: usize = 1024;

fn store() -> Option<Store> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let password = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))
        .ok()?
        .trim()
        .to_string();
    Some(Store::new(StoreConfig {
        url: std::env::var("NAFS_KB_TEST_URL").unwrap_or_else(|_| "http://127.0.0.1:8300".into()),
        namespace: "nafs".into(),
        database: "kb_scale_it".into(),
        user: "root".into(),
        password,
    }))
}

/// A tiny deterministic generator (xorshift): the test must not need a crate for random numbers.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_base_of_a_thousand_documents_stays_usable() {
    if std::env::var("NAFS_KB_SCALE").as_deref() != Ok("1") {
        eprintln!("skipped: set NAFS_KB_SCALE=1 to run the scale test");
        return;
    }
    let Some(st) = store() else {
        eprintln!("skipped: SurrealDB root secret not found");
        return;
    };
    let docs: usize = std::env::var("NAFS_KB_SCALE_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let per: usize = std::env::var("NAFS_KB_SCALE_CHUNKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    st.bootstrap().await.unwrap();
    // `NAFS_KB_SCALE_DELETE_ONLY=<base id>`: time the deletion of a base (a leftover of a failed run) and stop.
    if let Ok(id) = std::env::var("NAFS_KB_SCALE_DELETE_ONLY") {
        let before = st.get_kb(&id).await.unwrap();
        let t = Instant::now();
        st.delete_kb(&id).await.unwrap();
        eprintln!(
            "deleted a base of {} documents and {} passages in {:.1} s",
            before.documents,
            before.chunks,
            t.elapsed().as_secs_f64()
        );
        assert!(st.get_kb(&id).await.is_err());
        return;
    }
    st.ensure_vector_index(DIMS).await.unwrap();
    // `NAFS_KB_SCALE_REUSE=<base id>` measures a base an earlier run built (the build is most of the half hour).
    let reuse = std::env::var("NAFS_KB_SCALE_REUSE").ok();
    let kb = match &reuse {
        Some(id) => st.get_kb(id).await.unwrap(),
        None => {
            // A run that failed part-way leaves its base behind (4 GB at full size): clear those first.
            for stale in st
                .list_kbs()
                .await
                .unwrap()
                .into_iter()
                .filter(|k| k.name.starts_with("scale "))
            {
                st.delete_kb(&stale.id).await.unwrap();
            }
            st.create_kb(&format!("scale {}", std::process::id()), "synthetic")
                .await
                .unwrap()
        }
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // A skewed vocabulary: word i is drawn with probability about 1/i, as in text.
    let vocab: Vec<String> = (0..6000).map(|i| format!("w{i:04}x")).collect();
    let word = |rng: &mut Rng| -> &str {
        let r = rng.unit().max(1e-6);
        &vocab[((6000f32.powf(r)) as usize - 1).min(5999)]
    };

    // ---- build
    let t_build = Instant::now();
    let mut planted: Vec<(String, String, Vec<f32>)> = vec![]; // (chunk id, planted word, vector)
    let mut all_vectors: Vec<(String, Vec<f32>)> = vec![];
    for d in 0..(if reuse.is_some() { 0 } else { docs }) {
        let doc = st
            .create_document(
                &kb.id,
                &NewDocument {
                    title: format!("Synthetic document {d}"),
                    filename: format!("doc{d}.md"),
                    kind: "md".into(),
                    bytes: 1000,
                    sha256: format!("{d:064x}"),
                    path: String::new(),
                },
            )
            .await
            .unwrap();
        let marker = format!("zq{d}kx");
        let chunks: Vec<Chunk> = (0..per)
            .map(|i| {
                let mut text: Vec<String> = (0..170).map(|_| word(&mut rng).to_string()).collect();
                if i == 7 {
                    text.push(marker.clone());
                }
                let text = text.join(" ");
                Chunk {
                    ord: i,
                    chars: text.len(),
                    heading: format!("Section {}", i / 10),
                    page: None,
                    text,
                }
            })
            .collect();
        st.store_chunks(&kb.id, &doc.id, &doc.title, None, 1, &chunks)
            .await
            .unwrap();
        let (stored, _) = st.list_chunks(&kb.id, &doc.id, 0, per).await.unwrap();
        for c in &stored {
            let mut v: Vec<f32> = (0..DIMS).map(|_| rng.unit() - 0.5).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            if c.ord == 7 && planted.len() < 100 && d % (docs / 100).max(1) == 0 {
                planted.push((c.id.clone(), marker.clone(), v.clone()));
            }
            all_vectors.push((c.id.clone(), v));
        }
        if all_vectors.len() >= 500 {
            st.set_vectors(&all_vectors).await.unwrap();
            all_vectors.clear();
        }
    }
    if !all_vectors.is_empty() {
        st.set_vectors(&all_vectors).await.unwrap();
    }
    if reuse.is_some() {
        // The planted passages are the eighth of each document: the word is the document's number, the vector is stored.
        let titles: std::collections::HashMap<String, usize> = st
            .list_documents(&kb.id)
            .await
            .unwrap()
            .into_iter()
            .filter_map(|d| {
                d.title
                    .rsplit(' ')
                    .next()
                    .and_then(|n| n.parse().ok())
                    .map(|n| (d.id, n))
            })
            .collect();
        let rows = st
            .query(
                "SELECT record::id(id) AS id, record::id(doc) AS doc, embedding FROM chunk \
                 WHERE kb = type::record('kb', $kb) AND ord = 7 LIMIT 100;",
                serde_json::json!({ "kb": kb.id }),
            )
            .await
            .unwrap();
        for r in rows[0].as_array().cloned().unwrap_or_default() {
            let (Some(id), Some(doc)) = (r["id"].as_str(), r["doc"].as_str()) else {
                continue;
            };
            let Some(n) = titles.get(doc) else { continue };
            let v: Vec<f32> = r["embedding"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64())
                        .map(|x| x as f32)
                        .collect()
                })
                .unwrap_or_default();
            if v.len() == DIMS {
                planted.push((id.to_string(), format!("zq{n}kx"), v));
            }
        }
    }
    // The documents count their own vectors (what `vector_count` reads); the embedder does this in service.
    for d in st.list_documents(&kb.id).await.unwrap() {
        st.finish_vectors(&kb.id, &d.id, "synthetic", DIMS)
            .await
            .unwrap();
    }
    if reuse.is_none() {
        assert_eq!(st.vector_count(&kb.id).await.unwrap() as usize, docs * per);
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let n_chunks = docs * per;
    eprintln!("built {docs} documents, {n_chunks} passages with vectors in {build_s:.0} s");

    // ---- what a person waits for
    let time = |label: &'static str, ms: Vec<f64>| -> (String, f64, f64) {
        let mut v = ms;
        let (p50, p95) = (percentile(&mut v, 0.5), percentile(&mut v, 0.95));
        eprintln!("{label:34} p50 {p50:8.1} ms   p95 {p95:8.1} ms");
        (label.to_string(), p50, p95)
    };
    let mut rows = vec![];
    let mut ms = vec![];
    for _ in 0..5 {
        let t = Instant::now();
        let kbs = st.list_kbs().await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(kbs.iter().any(|k| k.id == kb.id
            && k.documents as usize == docs
            && k.chunks as usize == n_chunks));
    }
    rows.push(time("list bases (with counts)", ms));
    let mut ms = vec![];
    for _ in 0..5 {
        let t = Instant::now();
        let n = st.vector_count(&kb.id).await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(reuse.is_some() || n as usize == n_chunks);
    }
    rows.push(time("count the base's vectors", ms));
    let mut ms = vec![];
    for _ in 0..5 {
        let t = Instant::now();
        let list = st.list_documents(&kb.id).await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(list.len(), docs);
    }
    rows.push(time("list documents (all of them)", ms));
    let mut ms = vec![];
    for i in 0..5 {
        let t = Instant::now();
        let list = st.list_documents(&kb.id).await.unwrap();
        let d = &list[(i * 37) % list.len()];
        let (c, total) = st.list_chunks(&kb.id, &d.id, 0, 20).await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(!c.is_empty() && total as usize == per);
    }
    rows.push(time("open a document's passages", ms));
    // The mechanism, apart from its speed: in this large base a word every passage holds is left out of a query that has a
    // rarer word, and a query of common words alone is left whole.
    {
        let t = Instant::now();
        let kept = st
            .without_common_terms(&kb.id, "zq7kx w0001x document")
            .await;
        eprintln!(
            "terms kept from \"zq7kx w0001x document\": {kept:?} ({:.0} ms to probe)",
            t.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(kept, "zq7kx");
        assert_eq!(
            st.without_common_terms(&kb.id, "w0001x w0002x").await,
            "w0001x w0002x"
        );
    }
    // The words queries, in four shapes. The store stops a query at its own 10-second TIMEOUT (the service gives up and
    // answers by meaning); here a query that hit it is counted, and its time is the time it took to be stopped.
    // 1. a rare word; 2. a question (its function words are dropped before the store sees them; "here" appears nowhere);
    // 3. a rare word with a word in every passage ("w0001x" is the vocabulary's commonest, "document" is in the title line
    // of every passage, which the words index covers): in a large base the common word is left out before scoring;
    // 4. only common words, which stay and are slow by nature.
    let mut lexical_found = 0;
    let mut lexical_done = 0;
    let shapes: Vec<(&str, Vec<String>, usize)> = vec![
        (
            "search by words: a rare word",
            planted.iter().take(50).map(|p| p.1.clone()).collect(),
            50,
        ),
        (
            "search by words: a question",
            planted
                .iter()
                .take(20)
                .map(|p| format!("What is the {} here?", p.1))
                .collect(),
            20,
        ),
        (
            "search by words: + the commonest word",
            planted
                .iter()
                .take(10)
                .map(|p| format!("{} w0001x", p.1))
                .collect(),
            10,
        ),
        (
            "search by words: + a title word",
            planted
                .iter()
                .take(10)
                .map(|p| format!("{} document", p.1))
                .collect(),
            10,
        ),
        (
            "search by words: only common words",
            (0..5).map(|_| "w0001x w0002x w0003x".to_string()).collect(),
            5,
        ),
    ];
    let skip_store = std::env::var("NAFS_KB_SCALE_SKIP_STORE_QUERIES").is_ok();
    for (label, queries, _) in if skip_store { vec![] } else { shapes } {
        let mut ms = vec![];
        let mut stopped = 0;
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let r = st.search_lexical(&kb.id, q, 8).await;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            match r {
                Ok(hits) => {
                    if label.ends_with("a rare word")
                        || label.ends_with("a question")
                        || label.contains("+")
                    {
                        lexical_done += 1;
                        let marker = &planted[qi].1;
                        lexical_found += hits
                            .first()
                            .is_some_and(|h| h.text.contains(marker.as_str()))
                            as usize;
                    }
                }
                Err(_) => stopped += 1,
            }
        }
        eprintln!(
            "   ({stopped} of {} stopped by the store's 10 s limit)",
            queries.len()
        );
        rows.push(time(label, ms));
    }
    let mut ms = vec![];
    let mut vector_found = 0;
    let mut vector_done = 0;
    // The store's own index, three queries only: at this size each takes tens of seconds.
    for (id, _, v) in planted.iter().take(if skip_store { 0 } else { 3 }) {
        let t = Instant::now();
        let r = st.search_vector(&kb.id, v, 8, false).await;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if let Ok(hits) = r {
            vector_done += 1;
            vector_found += hits.first().is_some_and(|h| &h.chunk_id == id) as usize;
        }
    }
    if !ms.is_empty() {
        rows.push(time("search by meaning: the store's HNSW index", ms));
    }
    // The in-memory copy of the base's vectors (exact search): the one-off load, then a query.
    let cache = nafs_kb::vcache::VectorCache::new(1024);
    let count = st.vector_count(&kb.id).await.unwrap();
    let t_load = Instant::now();
    let loaded = loop {
        match cache.lookup(&st, &kb.id, count, DIMS) {
            nafs_kb::vcache::Lookup::Ready(l) => break l,
            nafs_kb::vcache::Lookup::Loading => {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await
            }
            nafs_kb::vcache::Lookup::TooBig => {
                panic!("a 50,000-passage base must fit the 1 GB cache")
            }
        }
        assert!(
            t_load.elapsed().as_secs() < 1800,
            "the vectors did not load"
        );
    };
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "vectors loaded into memory: {} in {load_s:.0} s",
        loaded.len()
    );
    let mut ms = vec![];
    let (mut cache_found, mut cache_done) = (0usize, 0usize);
    for (id, _, v) in planted.iter().take(100) {
        let t = Instant::now();
        let top = loaded.top(v, 8);
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        cache_done += 1;
        cache_found += top.first().is_some_and(|h| &h.0 == id) as usize;
    }
    rows.push(time("search by meaning: in memory, exact", ms));
    eprintln!("in-memory planted vector found first: {cache_found}/{cache_done}");
    assert_eq!(
        cache_found, cache_done,
        "an exact search finds a vector that is in the base"
    );
    let (n_lex, n_vec) = (lexical_done, vector_done);
    eprintln!("planted word found first: {lexical_found}/{n_lex}   planted vector found first: {vector_found}/{n_vec}");

    // ---- deletion
    let t = Instant::now();
    if std::env::var("NAFS_KB_SCALE_KEEP").is_err() {
        st.delete_kb(&kb.id).await.unwrap();
    }
    let delete_s = t.elapsed().as_secs_f64();
    eprintln!("deleted the base in {delete_s:.1} s");
    assert!(std::env::var("NAFS_KB_SCALE_KEEP").is_ok() || st.get_kb(&kb.id).await.is_err());

    if let Ok(out) = std::env::var("NAFS_KB_SCALE_OUT") {
        let j = serde_json::json!({
            "note": "SHARED MACHINE", "documents": docs, "passages": n_chunks, "dims": DIMS, "build_s": build_s,
            "operations": rows.iter().map(|(l, a, b)| serde_json::json!({"operation": l, "p50_ms": a, "p95_ms": b})).collect::<Vec<_>>(),
            "lexical_first": [lexical_found, n_lex], "vector_first": [vector_found, n_vec], "delete_s": delete_s,
            "vector_cache": {"vectors": loaded.len(), "load_s": load_s, "found_first": [cache_found, cache_done]},
            "load_average": std::fs::read_to_string("/proc/loadavg").ok(),
        });
        std::fs::write(out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    }
    assert!(
        n_lex == 0 || lexical_found * 10 >= n_lex * 9,
        "the planted word must be found first at least 9 times in 10"
    );
    assert!(
        n_vec == 0 || vector_found * 10 >= n_vec * 9,
        "the planted vector must be found first at least 9 times in 10"
    );
}
