//! Build a knowledge base from files through the same path an upload takes (Phase 28): parse, chunk, store, embed.
//! Used to compare two readers on the same documents (`NAFS_KB_PDF_LAYOUT=converter` selects the earlier one).
//!
//!     cargo run --release -p nafs-kb --example kb_ingest -- --name "p28 spans" file...
//!
//! Prints the new base's id when every document is ready and embedded.

use std::time::{Duration, Instant};

use nafs_kb::{ChunkOpts, EmbedConfig, IndexConfig, Kb, KbConfig, Limits, StoreConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let name = args
        .iter()
        .position(|a| a == "--name")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or("--name is required")?;
    let files: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| *a != "--name" && (*i == 0 || args[i - 1] != "--name"))
        .map(|(_, a)| a)
        .collect();
    let password = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))?
        .trim()
        .to_string();
    let kb = Kb::start(KbConfig {
        store: StoreConfig {
            url: std::env::var("NAFS_KB_URL").unwrap_or_else(|_| "http://127.0.0.1:8300".into()),
            namespace: "nafs".into(),
            database: std::env::var("NAFS_KB_DB").unwrap_or_else(|_| "kb".into()),
            user: "root".into(),
            password,
        },
        files_dir: format!("{root}/data/runtime/kb").into(),
        max_upload_bytes: 64 << 20,
        max_files_per_request: 1,
        chunk: ChunkOpts::default(),
        search_k: 8,
        embed: Some(EmbedConfig {
            url: std::env::var("NAFS_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8203".into()),
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
        // `NAFS_KB_PIECES=240` gives passages sentence-window vectors, as the service does by default.
        index: IndexConfig {
            piece_chars: std::env::var("NAFS_KB_PIECES")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0),
            ..IndexConfig::default()
        },
        limits: Limits::default(),
    });
    kb.ready().await?;
    let base = kb
        .store()
        .create_kb(&name, "Phase 28 reader comparison")
        .await?;
    let t0 = Instant::now();
    for f in &files {
        let bytes = std::fs::read(f)?;
        let shown = std::path::Path::new(f.as_str())
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| f.to_string());
        kb.upload(&base.id, &shown, &bytes).await?;
    }
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        let docs = kb.store().list_documents(&base.id).await?;
        let done = docs
            .iter()
            .filter(|d| d.status == "failed" || (d.status == "ready" && d.n_embedded >= d.n_chunks))
            .count();
        let passages: u64 = docs.iter().map(|d| d.n_chunks).sum();
        let embedded: u64 = docs.iter().map(|d| d.n_embedded).sum();
        eprintln!(
            "{:>5.0} s: {done}/{} documents, {embedded}/{passages} passages embedded",
            t0.elapsed().as_secs_f64(),
            docs.len()
        );
        if done == docs.len() {
            for d in &docs {
                if d.status == "failed" {
                    eprintln!("failed: {} {:?}", d.filename, d.error);
                }
            }
            // What the index holds once loaded: passages and sentence windows with vectors.
            if let Some(ix) = kb.index_of(&base.id).await? {
                eprintln!(
                    "index: {} passages, {} vectors",
                    ix.passages(),
                    ix.vectors()
                );
            }
            println!("{}", base.id);
            return Ok(());
        }
    }
}
