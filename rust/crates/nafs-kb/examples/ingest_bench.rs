//! Parse + chunk timing (NFR-K02): every file given (or every PDF in `papers/`), three runs each, median reported.
//! cargo run --release -p nafs-kb --example ingest_bench -- [files...] > benchmarks/kb_ingest.json

use std::time::Instant;

use nafs_kb::{chunk, parse, ChunkOpts, Kind};

fn main() {
    let mut files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../papers");
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .map(|r| {
                r.filter_map(|e| e.ok())
                    .map(|e| e.path().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        v.retain(|p| p.ends_with(".pdf"));
        v.sort();
        files = v;
    }
    let mut rows = Vec::new();
    for f in &files {
        let Some(kind) = Kind::from_filename(f) else {
            continue;
        };
        let bytes = std::fs::read(f).expect("readable file");
        let mut times = Vec::new();
        let mut result = None;
        for _ in 0..3 {
            let t = Instant::now();
            let r = parse(kind, &bytes).map(|p| {
                let c = chunk(&p, ChunkOpts::default());
                (p, c)
            });
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            result = Some(r);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let name = f.rsplit('/').next().unwrap_or(f);
        let row = match result.unwrap() {
            Ok((p, c)) => serde_json::json!({
                "file": name, "kind": kind.as_str(), "bytes": bytes.len(), "pages": p.pages, "chars": p.text.chars().count(),
                "chunks": c.len(), "median_ms": (times[1] * 10.0).round() / 10.0,
                "ms_per_page": p.pages.map(|n| (times[1] / n.max(1) as f64 * 100.0).round() / 100.0),
                "title": p.title,
            }),
            Err(e) => {
                serde_json::json!({ "file": name, "kind": kind.as_str(), "error": e, "median_ms": times[1] })
            }
        };
        eprintln!("{row}");
        rows.push(row);
    }
    let ok: Vec<&serde_json::Value> = rows.iter().filter(|r| r.get("error").is_none()).collect();
    let pages: f64 = ok.iter().filter_map(|r| r["pages"].as_f64()).sum();
    let ms: f64 = ok.iter().filter_map(|r| r["median_ms"].as_f64()).sum();
    let out = serde_json::json!({
        "what": "nafs-kb parse + chunk (no embedding), median of 3 runs per file, 1 thread",
        "machine": "Intel Core i3-8100, 4 cores, AVX2, macOS 15.4.1",
        "chunk_opts": { "chars": 1200, "overlap": 150 },
        "files": rows.len(), "failed": rows.len() - ok.len(),
        "total_pages": pages, "total_ms": (ms * 10.0).round() / 10.0,
        "ms_per_page": if pages > 0.0 { (ms / pages * 100.0).round() / 100.0 } else { 0.0 },
        "rows": rows,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
