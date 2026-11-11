//! How much retrieved text a prompt needs (Phase 27): the answering passage's phrase in the packed prompt, and the
//! size of that prompt, for each number of passages searched for (`k`) and each character budget.
//!
//! On a CPU a prompt is read at 50 to 70 tokens a second, so the passages are most of the time to the first word
//! of an answer. The sentence packer (`nafs_core::knowledge::pack_units`) fills a budget with whole sentences
//! ranked by their own relevance and their passage's rank; this measures what a smaller budget loses.
//!
//! ```text
//! cargo run --release -p nafs-kb --example rag_budget -- --kb <id> --set quoted=scripts/kb_p25_quoted.json \
//!     --set para=scripts/kb_p25_para.json --set paper50=scripts/kb_paper_questions_50.json \
//!     --out benchmarks/kb_phase27/rag_budget.json
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use nafs_core::knowledge::{
    pack_units, pack_units_for_prompt, prompt_block_for, prompt_block_sized, Passage, Retrieved,
};
use nafs_kb::index::{Analyzer, Base, Params};
use nafs_kb::{
    keep_for_chat, ChunkOpts, EmbedConfig, Embedder, IndexConfig, Kb, KbConfig, Limits, SearchMode,
    StoreConfig, QUERY_INSTRUCTION,
};
use serde_json::{json, Value};

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

fn phrases(v: &Value) -> Vec<String> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(norm)
        .collect()
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Vectors {
    embedder: Embedder,
    cache: HashMap<String, Vec<f32>>,
    path: String,
}

impl Vectors {
    async fn of(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.cache.contains_key(text) {
            let v = self
                .embedder
                .embed(&[text.to_string()], Some(QUERY_INSTRUCTION))
                .await?
                .remove(0);
            self.cache.insert(text.to_string(), v);
            if self.cache.len().is_multiple_of(20) {
                std::fs::write(&self.path, serde_json::to_vec(&self.cache)?)?;
            }
        }
        Ok(self.cache[text].clone())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let kb_id = arg(&args, "--kb").ok_or("--kb <id> is required")?;
    let read = |path: &str| -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&std::fs::read(format!(
            "{root}/{path}"
        ))?)?)
    };
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
    let base: Arc<Base> = loop {
        if let Some(b) = kb.index_of(&kb_id).await? {
            break b;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    };
    println!(
        "base {kb_id}: {} passages, {} vectors",
        base.passages(),
        base.vectors()
    );
    let path = format!("{root}/data/runtime/kb_eval_qvec.json");
    let mut vectors = Vectors {
        embedder: Embedder::new(embed),
        cache: std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default(),
        path: path.clone(),
    };
    let an = Analyzer::default();
    let params = Params::default();
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
    let sized = args.iter().any(|a| a == "--sized");
    let ks = [4usize, 6, 8];
    let budgets: &[usize] = if sized {
        &[4000, 3600, 3200, 2800, 2400, 2000, 1600]
    } else {
        &[4800, 3600, 2400, 1800, 1200, 800]
    };
    let mut report = json!({"kb": kb_id, "passages": base.passages()});
    for (name, path) in &sets {
        let items = read(path)?;
        let mut rows = serde_json::Map::new();
        for k in ks {
            for &budget in budgets {
                let (mut hit, mut chars, mut n, mut answerable) = (0usize, 0usize, 0usize, 0usize);
                let (mut block, mut n_units) = (0usize, 0usize);
                for item in items.iter() {
                    let q = item[0].as_str().unwrap_or("");
                    let want = phrases(&item[1]);
                    let v = vectors.of(q).await?;
                    let hits = base.search(&an, q, Some(&v), SearchMode::Hybrid, k.max(8), &params);
                    let kept = keep_for_chat(hits, SearchMode::Hybrid, 0.45, 10.0, k, 0.0);
                    let found = Retrieved {
                        kb_name: String::new(),
                        passages: kept
                            .into_iter()
                            .map(|h| Passage {
                                chunk_id: h.chunk_id,
                                doc_id: h.doc_id,
                                doc_title: h.doc_title,
                                heading: h.heading,
                                page: h.page,
                                text: h.text,
                                score: h.score,
                                lexical_score: h.lexical_score,
                                vector_score: h.vector_score,
                            })
                            .collect(),
                        mode: "hybrid".into(),
                        ..Default::default()
                    };
                    // `--sized`: the budget is the whole block (headers counted, titles written once).
                    let units = if sized {
                        pack_units_for_prompt(&found, q, budget)
                    } else {
                        pack_units(&found, q, budget)
                    };
                    let packed: String = units
                        .iter()
                        .map(|u| u.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    n += 1;
                    chars += packed.chars().count();
                    // What the prompt really carries: the same sentences with each source's title, heading and
                    // delimiters around them.
                    block += if sized {
                        prompt_block_sized(&found, q, budget).chars().count()
                    } else {
                        prompt_block_for(&found, q, budget).chars().count()
                    };
                    n_units += units.len();
                    if !want.is_empty() {
                        answerable += 1;
                        let t = norm(&packed);
                        hit += want.iter().any(|p| t.contains(p.as_str())) as usize;
                    }
                }
                let row = json!({"answer_in_prompt": (hit as f64 / answerable.max(1) as f64 * 1000.0).round() / 1000.0,
                    "answers": hit, "questions": answerable,
                    "chars_per_prompt": chars / n.max(1), "block_chars_per_prompt": block / n.max(1),
                    "sources_per_prompt": (n_units as f64 / n.max(1) as f64 * 10.0).round() / 10.0});
                println!("{name:<9} k {k} budget {budget:<5} {row}");
                rows.insert(format!("k{k}_b{budget}"), row);
            }
        }
        report[name.as_str()] = Value::Object(rows);
    }
    std::fs::write(&path, serde_json::to_vec(&vectors.cache)?)?;
    if let Some(out) = arg(&args, "--out") {
        let p = format!("{root}/{out}");
        if let Some(dir) = std::path::Path::new(&p).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(&report)?)?;
        println!("wrote {out}");
    }
    Ok(())
}
