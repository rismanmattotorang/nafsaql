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
    st.ensure_vector_index(DIMS).await.unwrap();
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
    let kb = st
        .create_kb(&format!("scale {}", std::process::id()), "synthetic")
        .await
        .unwrap();
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
    for d in 0..docs {
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
    // Three shapes of words query: a rare word alone; a question whose function words are dropped before the store sees
    // them; and the worst case, a rare word with the most common word of the vocabulary (present in every passage).
    let mut ms = vec![];
    let mut lexical_found = 0;
    for (_, marker, _) in planted.iter().take(50) {
        let t = Instant::now();
        let hits = st.search_lexical(&kb.id, marker, 8).await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if hits
            .first()
            .is_some_and(|h| h.text.contains(marker.as_str()))
        {
            lexical_found += 1;
        }
    }
    rows.push(time("search by words: a rare word", ms));
    let mut ms = vec![];
    for (_, marker, _) in planted.iter().take(20) {
        let t = Instant::now();
        let hits = st
            .search_lexical(&kb.id, &format!("What is the {marker} in the document?"), 8)
            .await
            .unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(hits
            .first()
            .is_some_and(|h| h.text.contains(marker.as_str())));
    }
    rows.push(time("search by words: a question", ms));
    let mut ms = vec![];
    for (_, marker, _) in planted.iter().take(10) {
        let t = Instant::now();
        let _ = st
            .search_lexical(&kb.id, &format!("{marker} w0001x"), 8)
            .await
            .unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    rows.push(time("search by words: + the commonest word", ms));
    let mut ms = vec![];
    let mut vector_found = 0;
    for (id, _, v) in planted.iter().take(100) {
        let t = Instant::now();
        let hits = st.search_vector(&kb.id, v, 8, false).await.unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if hits.first().is_some_and(|h| &h.chunk_id == id) {
            vector_found += 1;
        }
    }
    rows.push(time("search by meaning (HNSW)", ms));
    let n_lex = planted.len().min(50);
    let n_vec = planted.len().min(100);
    eprintln!("planted word found first: {lexical_found}/{n_lex}   planted vector found first: {vector_found}/{n_vec}");

    // ---- deletion
    let t = Instant::now();
    st.delete_kb(&kb.id).await.unwrap();
    let delete_s = t.elapsed().as_secs_f64();
    eprintln!("deleted the base in {delete_s:.1} s");
    assert!(st.get_kb(&kb.id).await.is_err());

    if let Ok(out) = std::env::var("NAFS_KB_SCALE_OUT") {
        let j = serde_json::json!({
            "note": "SHARED MACHINE", "documents": docs, "passages": n_chunks, "dims": DIMS, "build_s": build_s,
            "operations": rows.iter().map(|(l, a, b)| serde_json::json!({"operation": l, "p50_ms": a, "p95_ms": b})).collect::<Vec<_>>(),
            "lexical_first": [lexical_found, n_lex], "vector_first": [vector_found, n_vec], "delete_s": delete_s,
            "load_average": std::fs::read_to_string("/proc/loadavg").ok(),
        });
        std::fs::write(out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    }
    assert!(
        lexical_found * 10 >= n_lex * 9,
        "the planted word must be found first at least 9 times in 10"
    );
    assert!(
        vector_found * 10 >= n_vec * 9,
        "the planted vector must be found first at least 9 times in 10"
    );
}
