//! Retrieval evaluation of the in-process index (Phase 25, FR-K75): where does the passage that states the answer rank,
//! for each setting of the words (BM25F weights), the fusion, the feedback terms and the graph arm, and how long does a
//! search take against the store path.
//!
//! The base is read once from the live store into the index; question vectors are embedded once and cached, so a sweep of
//! dozens of settings costs milliseconds per setting.
//!
//!     cargo run --release -p nafs-kb --example kb_eval -- --kb <id> \
//!         --set quoted=scripts/kb_p25_quoted.json --set para=scripts/kb_p25_para.json \
//!         --set multihop=scripts/kb_p25_multihop.json [--store] [--latency 30] [--out benchmarks/kb_phase25/eval.json]
//!
//! A question file is a list of `[question, [accepted phrases]]`, or for two-passage questions
//! `[question, [phrases A], [phrases B]]`. `--store` also ranks through the store path (seconds a question under load).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use nafs_kb::index::{Analyzer, Base, Bm25, Fusion, Params, Ppr, Prf};
use nafs_kb::{
    ChunkOpts, EmbedConfig, Embedder, Engine, Hit, IndexConfig, Kb, KbConfig, Limits, SearchMode,
    StoreConfig, QUERY_INSTRUCTION,
};
use serde_json::{json, Value};

struct Question {
    text: String,
    /// One list of accepted phrases per passage the answer needs.
    needs: Vec<Vec<String>>,
    vector: Vec<f32>,
}

fn norm(s: &str) -> String {
    let cleaned: String = s
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c.is_whitespace() || ".%/-".contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The rank (from 1) of the first passage that holds one of the phrases.
fn rank_of(texts: &[String], phrases: &[String]) -> Option<usize> {
    texts
        .iter()
        .position(|t| phrases.iter().any(|p| t.contains(p.as_str())))
        .map(|i| i + 1)
}

#[derive(Default, Clone)]
struct Score {
    n: usize,
    r1: f64,
    r4: f64,
    r8: f64,
    r20: f64,
    mrr: f64,
    /// Two-passage questions: both passages within the first 4, 8 and 20.
    all4: f64,
    all8: f64,
    all20: f64,
}

impl Score {
    fn add(&mut self, ranks: &[Option<usize>]) {
        self.n += 1;
        let share = 1.0 / ranks.len() as f64;
        for r in ranks.iter().flatten() {
            self.r1 += share * (*r <= 1) as u8 as f64;
            self.r4 += share * (*r <= 4) as u8 as f64;
            self.r8 += share * (*r <= 8) as u8 as f64;
            self.r20 += share * (*r <= 20) as u8 as f64;
            self.mrr += share / *r as f64;
        }
        let within = |k: usize| ranks.iter().all(|r| r.is_some_and(|r| r <= k)) as u8 as f64;
        self.all4 += within(4);
        self.all8 += within(8);
        self.all20 += within(20);
    }

    fn json(&self) -> Value {
        let n = self.n.max(1) as f64;
        let r = |x: f64| (x / n * 1000.0).round() / 1000.0;
        json!({ "n": self.n, "R@1": r(self.r1), "R@4": r(self.r4), "R@8": r(self.r8), "R@20": r(self.r20),
            "MRR": r(self.mrr), "all@4": r(self.all4), "all@8": r(self.all8), "all@20": r(self.all20) })
    }

    fn line(&self) -> String {
        let n = self.n.max(1) as f64;
        format!(
            "R@1 {:.3} R@4 {:.3} R@8 {:.3} R@20 {:.3} MRR {:.3} all@8 {:.3}",
            self.r1 / n,
            self.r4 / n,
            self.r8 / n,
            self.r20 / n,
            self.mrr / n,
            self.all8 / n
        )
    }
}

fn score(hits: &[Hit], q: &Question) -> Vec<Option<usize>> {
    let texts: Vec<String> = hits.iter().map(|h| norm(&h.text)).collect();
    q.needs.iter().map(|p| rank_of(&texts, p)).collect()
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let kb_id = arg(&args, "--kb").ok_or("--kb <id> is required")?;
    let sets: Vec<(String, String)> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--set")
        .filter_map(|(i, _)| args.get(i + 1))
        .filter_map(|s| {
            s.split_once('=')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();
    let latency: usize = arg(&args, "--latency")
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let with_store = args.iter().any(|a| a == "--store");
    let password = std::fs::read_to_string(format!("{root}/data/runtime/surreal/root.secret"))?
        .trim()
        .to_string();
    let embed = EmbedConfig {
        url: std::env::var("NAFS_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8203".into()),
        model: "Qwen3-Embedding-0.6B".into(),
        dims: 1024,
        batch: 8,
    };
    let kb = Kb::start(KbConfig {
        store: StoreConfig {
            url: std::env::var("NAFS_KB_URL").unwrap_or_else(|_| "http://127.0.0.1:8300".into()),
            namespace: "nafs".into(),
            database: std::env::var("NAFS_KB_DB").unwrap_or_else(|_| "kb".into()),
            user: "root".into(),
            password,
        },
        files_dir: std::env::temp_dir().join("nafs-kb-eval"),
        max_upload_bytes: 1 << 20,
        max_files_per_request: 1,
        chunk: ChunkOpts::default(),
        search_k: 8,
        embed: Some(embed.clone()),
        exact_search_below: 5000,
        graph: None,
        graph_weight: 1.0,
        vector_weight: 0.1,
        vector_cache_mb: 0,
        graph_beam: 8,
        graph_decay: 0.6,
        index: IndexConfig::default(),
        limits: Limits::default(),
    });
    let t0 = Instant::now();
    let base: Arc<Base> = loop {
        if let Some(b) = kb.index_of(&kb_id).await? {
            break b;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    };
    let load_ms = t0.elapsed().as_millis();
    let tg = Instant::now();
    let graph = base.graph();
    println!(
        "base {kb_id}: {} documents, {} passages, {} vectors, {} MB; read from the store and indexed in {load_ms} ms; \
         graph {} terms, {} links ({} ms more)",
        base.documents(),
        base.passages(),
        base.vectors(),
        base.bytes() / 1_048_576,
        graph.terms(),
        graph.links(),
        tg.elapsed().as_millis()
    );

    // Question vectors, embedded once and kept between runs.
    let cache_path = format!("{root}/data/runtime/kb_eval_qvec.json");
    let mut cache: HashMap<String, Vec<f32>> = std::fs::read(&cache_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let embedder = Embedder::new(embed);
    let mut loaded: Vec<(String, Vec<Question>)> = Vec::new();
    for (name, path) in &sets {
        let raw: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{root}/{path}"))?)?;
        let mut qs = Vec::new();
        for item in raw {
            let text = item[0].as_str().unwrap_or("").to_string();
            let needs: Vec<Vec<String>> = item
                .as_array()
                .into_iter()
                .flatten()
                .skip(1)
                .map(|l| {
                    l.as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(norm)
                        .collect()
                })
                .collect();
            if !cache.contains_key(&text) {
                let v = embedder
                    .embed(std::slice::from_ref(&text), Some(QUERY_INSTRUCTION))
                    .await?
                    .remove(0);
                cache.insert(text.clone(), v);
                if cache.len().is_multiple_of(20) {
                    std::fs::write(&cache_path, serde_json::to_vec(&cache)?)?;
                }
            }
            let vector = cache[&text].clone();
            qs.push(Question {
                text,
                needs,
                vector,
            });
        }
        println!("set {name}: {} questions ({path})", qs.len());
        loaded.push((name.clone(), qs));
    }
    std::fs::write(&cache_path, serde_json::to_vec(&cache)?)?;

    let an = Analyzer::default();
    let rrf = |w: f64| Fusion::Rrf { vector_weight: w };
    let bm = |w: [f32; 4]| Bm25 {
        weights: w,
        ..Bm25::default()
    };
    let p0 = Params::default();
    let mut variants: Vec<(String, SearchMode, Params)> = vec![
        ("words flat [1,1,1,1]".into(), SearchMode::Lexical, p0),
        ("vector".into(), SearchMode::Vector, p0),
    ];
    for w in [
        [0.0, 1.0, 1.0, 1.0],
        [0.5, 1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 0.5, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 3.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [0.5, 2.0, 1.0, 1.0],
        [0.5, 2.0, 2.0, 1.0],
    ] {
        variants.push((
            format!("words {w:?}"),
            SearchMode::Lexical,
            Params { bm25: bm(w), ..p0 },
        ));
    }
    for (k1, b) in [
        (0.9, 0.4),
        (1.2, 0.3),
        (1.2, 0.75),
        (1.6, 0.5),
        (2.0, 0.75),
        (1.2, 0.9),
    ] {
        variants.push((
            format!("words k1 {k1} b {b}"),
            SearchMode::Lexical,
            Params {
                bm25: Bm25 {
                    k1,
                    b,
                    ..Bm25::default()
                },
                ..p0
            },
        ));
    }
    variants.push((
        "words + feedback (5 passages, 8 terms, 0.3)".into(),
        SearchMode::Lexical,
        Params {
            prf: Some(Prf {
                passages: 5,
                terms: 8,
                weight: 0.3,
            }),
            ..p0
        },
    ));
    variants.push((
        "words + feedback (3 passages, 5 terms, 0.2)".into(),
        SearchMode::Lexical,
        Params {
            prf: Some(Prf {
                passages: 3,
                terms: 5,
                weight: 0.2,
            }),
            ..p0
        },
    ));
    for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0] {
        variants.push((
            format!("hybrid rrf w={w}"),
            SearchMode::Hybrid,
            Params {
                fusion: rrf(w),
                ..p0
            },
        ));
    }
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8] {
        variants.push((
            format!("hybrid convex alpha={a}"),
            SearchMode::Hybrid,
            Params {
                fusion: Fusion::Convex { alpha: a },
                ..p0
            },
        ));
    }
    for fusion in [rrf(0.1), Fusion::Convex { alpha: 0.3 }] {
        for weight in [0.25, 0.5, 1.0] {
            for (damping, share) in [(0.5, 0.5), (0.5, 0.0), (0.7, 0.5)] {
                variants.push((
                    format!("hybrid {fusion:?} + graph w={weight} d={damping} terms={share}"),
                    SearchMode::Hybrid,
                    Params {
                        fusion,
                        ppr: Some(Ppr {
                            weight,
                            damping,
                            seeds: 10,
                            entity_share: share,
                        }),
                        ..p0
                    },
                ));
            }
        }
    }
    let mut report = json!({
        "kb": kb_id, "passages": base.passages(), "vectors": base.vectors(), "documents": base.documents(),
        "index_mb": base.bytes() / 1_048_576, "load_ms": load_ms as u64,
        "graph": { "terms": graph.terms(), "links": graph.links() },
        "machine": "SHARED MACHINE", "variants": [], "store": {}, "latency": {},
    });
    for (name, mode, params) in &variants {
        let mut row = json!({ "variant": name });
        let mut line = format!("{name:<58}");
        for (set, qs) in &loaded {
            let mut s = Score::default();
            for q in qs {
                let hits = base.search(&an, &q.text, Some(&q.vector), *mode, 20, params);
                s.add(&score(&hits, q));
            }
            row[set] = s.json();
            line += &format!(" | {set}: {}", s.line());
        }
        println!("{line}");
        report["variants"].as_array_mut().expect("array").push(row);
    }

    // What the words score for the right passage and for the best wrong one: the scale `rag_lexical_rescue` sits on.
    for (set, qs) in &loaded {
        let (mut right, mut first): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        for q in qs {
            let hits = base.search(&an, &q.text, None, SearchMode::Lexical, 20, &p0);
            let texts: Vec<String> = hits.iter().map(|h| norm(&h.text)).collect();
            if let Some(r) = q.needs.first().and_then(|p| rank_of(&texts, p)) {
                right.push(hits[r - 1].score as f32);
            }
            if let Some(h) = hits.first() {
                first.push(h.score as f32);
            }
        }
        let pct = |v: &mut Vec<f32>, p: f64| {
            v.sort_by(f32::total_cmp);
            v.get(((v.len() as f64 - 1.0) * p) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        println!(
            "{set}: words score of the answering passage p10 {:.1} p50 {:.1} p90 {:.1}; of the first passage p10 {:.1} p50 {:.1} p90 {:.1} max {:.1}",
            pct(&mut right, 0.1), pct(&mut right, 0.5), pct(&mut right, 0.9), pct(&mut first, 0.1), pct(&mut first, 0.5),
            pct(&mut first, 0.9), pct(&mut first, 1.0)
        );
        report["lexical_scores"][set] = json!({ "answering_p10": pct(&mut right, 0.1), "answering_p50": pct(&mut right, 0.5),
            "answering_p90": pct(&mut right, 0.9), "first_p50": pct(&mut first, 0.5), "first_p90": pct(&mut first, 0.9),
            "first_max": pct(&mut first, 1.0) });
    }

    if with_store {
        for (set, qs) in &loaded {
            for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
                let mut s = Score::default();
                for q in qs {
                    match kb
                        .search_with(&kb_id, &q.text, 20, mode, Engine::Store)
                        .await
                    {
                        Ok(r) => s.add(&score(&r.results, q)),
                        Err(e) => {
                            eprintln!("store {mode:?}: {e}");
                            s.add(&vec![None; q.needs.len()]);
                        }
                    }
                }
                println!("store {mode:?} | {set}: {}", s.line());
                report["store"][format!("{mode:?}").to_lowercase()][set] = s.json();
            }
        }
    }

    if latency > 0 {
        // Interleaved: the same question through the store path, then the index, so both see the same load.
        let qs: Vec<&Question> = loaded
            .iter()
            .flat_map(|(_, q)| q.iter())
            .take(latency)
            .collect();
        for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
            let (mut store_ms, mut index_ms, mut embed_ms): (Vec<f64>, Vec<f64>, Vec<f64>) =
                (vec![], vec![], vec![]);
            for (i, q) in qs.iter().enumerate() {
                // A fresh wording each time, different from its first words on, so neither path is served a remembered
                // question vector and the embedding server can reuse only the instruction (as for a new question).
                let text = format!("(Run {i} {mode:?}, by the store) {}", q.text);
                let t = Instant::now();
                let r = kb.search_with(&kb_id, &text, 8, mode, Engine::Store).await;
                if r.is_ok() {
                    store_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                let text = format!("(Run {i} {mode:?}, by the index) {}", q.text);
                let t = Instant::now();
                if let Ok(r) = kb.search_with(&kb_id, &text, 8, mode, Engine::Auto).await {
                    index_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                    embed_ms.push(r.embed_ms as f64);
                }
            }
            let pct = |v: &mut Vec<f64>, p: f64| {
                v.sort_by(f64::total_cmp);
                v.get(((v.len() as f64 - 1.0) * p) as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            let row = json!({
                "n": store_ms.len(),
                "store_ms": { "p50": pct(&mut store_ms, 0.5), "p90": pct(&mut store_ms, 0.9), "max": pct(&mut store_ms, 1.0) },
                "index_ms": { "p50": pct(&mut index_ms, 0.5), "p90": pct(&mut index_ms, 0.9), "max": pct(&mut index_ms, 1.0) },
                "index_embed_ms": { "p50": pct(&mut embed_ms, 0.5), "p90": pct(&mut embed_ms, 0.9) },
            });
            println!("latency {mode:?}: {row}");
            report["latency"][format!("{mode:?}").to_lowercase()] = row;
        }
        // The index alone, without the embedding server: what the search itself costs.
        let t = Instant::now();
        let mut n = 0;
        for _ in 0..20 {
            for q in &qs {
                base.search(&an, &q.text, Some(&q.vector), SearchMode::Hybrid, 8, &p0);
                n += 1;
            }
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n.max(1) as f64;
        println!(
            "index hybrid search without the embedding call: {us:.0} us a question ({n} searches)"
        );
        report["latency"]["index_hybrid_us_without_embedding"] = json!(us.round());
        let p = Params {
            ppr: Some(Ppr::default()),
            ..p0
        };
        let t = Instant::now();
        for q in &qs {
            base.search(&an, &q.text, Some(&q.vector), SearchMode::Hybrid, 8, &p);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / qs.len().max(1) as f64;
        println!("index hybrid search with the graph arm, without the embedding call: {us:.0} us a question");
        report["latency"]["index_hybrid_graph_us_without_embedding"] = json!(us.round());
    }
    if let Some(out) = arg(&args, "--out") {
        let path = format!("{root}/{out}");
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("wrote {out}");
    }
    Ok(())
}
