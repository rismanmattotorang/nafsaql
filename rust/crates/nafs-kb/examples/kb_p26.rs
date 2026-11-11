//! Phase 26 retrieval checks on a live base, each against the setting it replaces:
//!
//! * a mistyped question word (`--typo clean.json=typo.json`): the nearest-word correction on and off;
//! * a follow-up question (`--followup pairs.json`, items `[previous, follow-up, [phrases]]`): searched alone, with
//!   the previous turn always, and with the previous turn when the rule says the question leans on it;
//! * the elbow cut of the passages put in a prompt (`--set name=file` per question set): how many passages the
//!   prompt carries and how often the answering passage is still among them, for each size of drop.
//!
//! ```text
//! cargo run --release -p nafs-kb --example kb_p26 -- --kb <id> --typo scripts/kb_p25_quoted.json=scripts/kb_p26_typo.json \
//!     --followup scripts/kb_p26_followup.json --set quoted=scripts/kb_p25_quoted.json --set para=scripts/kb_p25_para.json \
//!     --set multihop=scripts/kb_p25_multihop.json --set offtopic=scripts/kb_p25_offtopic.json \
//!     --out benchmarks/kb_phase26/retrieval.json
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use nafs_core::continuity::{leans_on_previous, search_text};
use nafs_kb::index::{Analyzer, Base, Params};
use nafs_kb::{
    keep_for_chat, ChunkOpts, EmbedConfig, Embedder, Hit, IndexConfig, Kb, KbConfig, Limits,
    SearchMode, StoreConfig, QUERY_INSTRUCTION,
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

fn rank(hits: &[Hit], phrases: &[String]) -> Option<usize> {
    hits.iter()
        .position(|h| {
            let t = norm(&h.text);
            phrases.iter().any(|p| t.contains(p.as_str()))
        })
        .map(|i| i + 1)
}

#[derive(Default)]
struct Score {
    n: usize,
    r1: f64,
    r4: f64,
    r8: f64,
    mrr: f64,
}

impl Score {
    fn add(&mut self, r: Option<usize>) {
        self.n += 1;
        if let Some(r) = r {
            self.r1 += (r <= 1) as u8 as f64;
            self.r4 += (r <= 4) as u8 as f64;
            self.r8 += (r <= 8) as u8 as f64;
            self.mrr += 1.0 / r as f64;
        }
    }
    fn json(&self) -> Value {
        let n = self.n.max(1) as f64;
        let r = |x: f64| (x / n * 1000.0).round() / 1000.0;
        json!({"n": self.n, "R@1": r(self.r1), "R@4": r(self.r4), "R@8": r(self.r8), "MRR": r(self.mrr)})
    }
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
    let on = Params::default();
    let off = Params { fuzzy: false, ..on };
    let mut report = json!({"kb": kb_id, "passages": base.passages()});

    // ---- a mistyped word
    if let Some((clean, typo)) = arg(&args, "--typo").and_then(|a| {
        a.split_once('=')
            .map(|(a, b)| (a.to_string(), b.to_string()))
    }) {
        let (clean, typo) = (read(&clean)?, read(&typo)?);
        let mut corrected = 0;
        let mut rows = serde_json::Map::new();
        for (name, set) in [("as written", &clean), ("mistyped", &typo)] {
            for (mode_name, mode) in [
                ("words", SearchMode::Lexical),
                ("hybrid", SearchMode::Hybrid),
            ] {
                for (p_name, p) in [("correction off", &off), ("correction on", &on)] {
                    let mut s = Score::default();
                    for item in set.iter() {
                        let q = item[0].as_str().unwrap_or("");
                        let v = vectors.of(q).await?;
                        let hits = base.search(&an, q, Some(&v), mode, 8, p);
                        s.add(rank(&hits, &phrases(&item[1])));
                    }
                    println!("typo  {name:<10} {mode_name:<6} {p_name:<14} {}", s.json());
                    rows.insert(format!("{name} / {mode_name} / {p_name}"), s.json());
                }
            }
        }
        for item in typo.iter() {
            let q = item[0].as_str().unwrap_or("");
            corrected += an
                .query(q)
                .iter()
                .filter(|t| base.nearest_term(t).is_some())
                .count();
        }
        // Words of well-formed questions the correction would change: each is a risk, not a repair.
        let changed_clean: usize = clean
            .iter()
            .map(|item| {
                an.query(item[0].as_str().unwrap_or(""))
                    .iter()
                    .filter(|t| base.nearest_term(t).is_some())
                    .count()
            })
            .sum();
        report["typo"] = json!({"rows": rows, "words_corrected_in_mistyped": corrected,
            "words_changed_in_as_written": changed_clean});
        println!("typo  words corrected: {corrected} in the mistyped set, {changed_clean} in the set as written");
    }

    // ---- a follow-up question
    if let Some(path) = arg(&args, "--followup") {
        let pairs = read(&path)?;
        let (mut alone, mut always, mut rule) =
            (Score::default(), Score::default(), Score::default());
        let mut fired = 0;
        for item in pairs.iter() {
            let (prev, q) = (
                item[0].as_str().unwrap_or(""),
                item[1].as_str().unwrap_or(""),
            );
            let want = phrases(&item[2]);
            let joined = format!("{prev}\n{q}");
            let v = vectors.of(q).await?;
            alone.add(rank(
                &base.search(&an, q, Some(&v), SearchMode::Hybrid, 8, &on),
                &want,
            ));
            let vj = vectors.of(&joined).await?;
            let with_prev = rank(
                &base.search(&an, &joined, Some(&vj), SearchMode::Hybrid, 8, &on),
                &want,
            );
            always.add(with_prev);
            let (text, leans) = search_text(q, Some(prev));
            fired += leans as usize;
            let vr = vectors.of(&text).await?;
            rule.add(rank(
                &base.search(&an, &text, Some(&vr), SearchMode::Hybrid, 8, &on),
                &want,
            ));
        }
        println!("followup alone            {}", alone.json());
        println!("followup with previous    {}", always.json());
        println!(
            "followup by the rule      {} (rule fired on {fired} of {})",
            rule.json(),
            pairs.len()
        );
        report["followup"] = json!({"alone": alone.json(), "with_previous": always.json(), "by_rule": rule.json(),
            "rule_fired": fired, "n": pairs.len()});
    }

    // ---- self-contained questions the rule would join, and the elbow cut
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
    let gaps = [0.0, 0.03, 0.05, 0.08, 0.12];
    let mut elbow = serde_json::Map::new();
    let mut false_fire = serde_json::Map::new();
    for (name, path) in &sets {
        let items = read(path)?;
        let leans = items
            .iter()
            .filter(|i| leans_on_previous(i[0].as_str().unwrap_or("")))
            .count();
        false_fire.insert(name.clone(), json!({"n": items.len(), "leans": leans}));
        println!("rule  {name}: {leans} of {} self-contained questions would be joined with a previous turn", items.len());
        let mut rows = serde_json::Map::new();
        for gap in gaps {
            let (mut kept, mut with_answer, mut answerable) = (0usize, 0usize, 0usize);
            for item in items.iter() {
                let q = item[0].as_str().unwrap_or("");
                let v = vectors.of(q).await?;
                let hits = base.search(&an, q, Some(&v), SearchMode::Hybrid, 8, &on);
                let keep = keep_for_chat(hits, SearchMode::Hybrid, 0.45, 10.0, 4, gap);
                kept += keep.len();
                let want = phrases(&item[1]);
                if !want.is_empty() {
                    answerable += 1;
                    with_answer += rank(&keep, &want).is_some() as usize;
                }
            }
            let n = items.len().max(1) as f64;
            let row = json!({"passages_per_prompt": (kept as f64 / n * 100.0).round() / 100.0,
                "answer_in_prompt": if answerable > 0 { json!((with_answer as f64 / answerable as f64 * 1000.0).round() / 1000.0) } else { Value::Null }});
            println!("elbow {name:<9} gap {gap:<4} {row}");
            rows.insert(format!("{gap}"), row);
        }
        elbow.insert(name.clone(), Value::Object(rows));
    }
    report["rule_on_self_contained"] = Value::Object(false_fire);
    report["elbow"] = Value::Object(elbow);
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
