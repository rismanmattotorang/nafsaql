//! Phase 28 retrieval comparison: two readers, and sentence-window vectors (semtools' small search unit).
//!
//! For each base given, every question set is ranked by words, by meaning and by the hybrid of both; questions are
//! scored only when every base holds a passage with an accepted phrase (the phrases were written against one reader's
//! text, so a question the other reader words differently would count against it unfairly). With `--pieces <chars>`
//! the passages of each base are also cut into sentence windows (`nafs_kb::chunk::pieces`), the windows embedded
//! (cached in `data/runtime/kb_eval_pvec.json`), and the meaning arm scores a passage by
//! `(1 - m) * own + m * max(own, best window)` for several `m`.
//!
//!     cargo run --release -p nafs-kb --example piece_eval -- --kb <old> --kb <new> \
//!         --set para=scripts/kb_p25_para.json ... [--pieces 240] [--out benchmarks/kb_phase28/retrieval.json]

use std::collections::HashMap;
use std::sync::Arc;

use nafs_kb::embed::passage_text;
use nafs_kb::index::{fuse, top, Analyzer, Base, Fusion, Params, Ppr};
use nafs_kb::{
    ChunkOpts, EmbedConfig, Embedder, IndexConfig, Kb, KbConfig, Limits, SearchMode, StoreConfig,
    QUERY_INSTRUCTION,
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

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Question {
    text: String,
    needs: Vec<Vec<String>>,
    vector: Vec<f32>,
}

#[derive(Default, Clone)]
struct Score {
    n: usize,
    r1: f64,
    r4: f64,
    r8: f64,
    mrr: f64,
}

impl Score {
    fn add(&mut self, ranks: &[Option<usize>]) {
        self.n += 1;
        let share = 1.0 / ranks.len() as f64;
        for r in ranks.iter().flatten() {
            self.r1 += share * (*r <= 1) as u8 as f64;
            self.r4 += share * (*r <= 4) as u8 as f64;
            self.r8 += share * (*r <= 8) as u8 as f64;
            self.mrr += share / *r as f64;
        }
    }
    fn json(&self) -> Value {
        let n = self.n.max(1) as f64;
        let r = |x: f64| (x / n * 1000.0).round() / 1000.0;
        json!({"n": self.n, "R@1": r(self.r1), "R@4": r(self.r4), "R@8": r(self.r8), "MRR": r(self.mrr)})
    }
}

fn arg_all(args: &[String], name: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == name)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let kbs = arg_all(&args, "--kb");
    let sets: Vec<(String, String)> = arg_all(&args, "--set")
        .into_iter()
        .filter_map(|s| {
            s.split_once('=')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();
    let piece_chars: usize = arg_all(&args, "--pieces")
        .first()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let out = arg_all(&args, "--out").first().cloned();
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
    let mut bases: Vec<(String, Arc<Base>)> = Vec::new();
    for id in &kbs {
        let b = loop {
            if let Some(b) = kb.index_of(id).await? {
                break b;
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        };
        println!(
            "base {id}: {} documents, {} passages, {} vectors",
            b.documents(),
            b.passages(),
            b.vectors()
        );
        bases.push((id.clone(), b));
    }
    let embedder = Embedder::new(embed);

    // Questions, with vectors cached between runs (the same cache as kb_eval).
    let qcache_path = format!("{root}/data/runtime/kb_eval_qvec.json");
    let mut qcache: HashMap<String, Vec<f32>> = std::fs::read(&qcache_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
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
            if !qcache.contains_key(&text) {
                let v = embedder
                    .embed(std::slice::from_ref(&text), Some(QUERY_INSTRUCTION))
                    .await?
                    .remove(0);
                qcache.insert(text.clone(), v);
            }
            let vector = unit(qcache[&text].clone());
            qs.push(Question {
                text,
                needs,
                vector,
            });
        }
        loaded.push((name.clone(), qs));
    }
    std::fs::write(&qcache_path, serde_json::to_vec(&qcache)?)?;

    // Sentence windows and their vectors, per base.
    let pcache_path = format!("{root}/data/runtime/kb_eval_pvec.json");
    let mut pcache: HashMap<String, Vec<f32>> = if piece_chars > 0 {
        std::fs::read(&pcache_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    } else {
        HashMap::new()
    };
    // Per base: for every passage, the unit vectors of its windows.
    let mut windows: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
    for (id, base) in &bases {
        let mut per: Vec<Vec<Vec<f32>>> = vec![Vec::new(); base.passages()];
        if piece_chars > 0 {
            let mut todo: Vec<(usize, String)> = Vec::new();
            for (g, slot) in per.iter_mut().enumerate() {
                let h = base.passage(g as u32);
                for p in nafs_kb::chunk::pieces(&h.text, piece_chars) {
                    let t = passage_text(&h.doc_title, &h.heading, "", &p);
                    match pcache.get(&t) {
                        Some(v) => slot.push(unit(v.clone())),
                        None => todo.push((g, t)),
                    }
                }
            }
            let total = todo.len();
            let t0 = std::time::Instant::now();
            for (k, batch) in todo.chunks(8).enumerate() {
                let texts: Vec<String> = batch.iter().map(|x| x.1.clone()).collect();
                let vs = embedder.embed(&texts, None).await?;
                for ((g, t), v) in batch.iter().zip(vs) {
                    per[*g].push(unit(v.clone()));
                    pcache.insert(t.clone(), v);
                }
                if k % 25 == 24 {
                    std::fs::write(&pcache_path, serde_json::to_vec(&pcache)?)?;
                    eprintln!(
                        "{id}: {} / {total} windows embedded, {:.0} s",
                        (k + 1) * 8,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            std::fs::write(&pcache_path, serde_json::to_vec(&pcache)?)?;
            let n: usize = per.iter().map(Vec::len).sum();
            println!(
                "base {id}: {n} windows of about {piece_chars} characters ({total} embedded now)"
            );
        }
        windows.push(per);
    }

    // Questions every base can answer.
    let an = Analyzer::default();
    let texts: Vec<Vec<String>> = bases
        .iter()
        .map(|(_, b)| {
            (0..b.passages() as u32)
                .map(|g| norm(b.text_of(g)))
                .collect()
        })
        .collect();
    let answerable = |q: &Question, t: &[String]| {
        q.needs
            .iter()
            .all(|ph| t.iter().any(|x| ph.iter().any(|p| x.contains(p.as_str()))))
    };
    let p0 = Params::default();
    let mut report = json!({"bases": kbs, "pieces": piece_chars});
    let mixes = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let weights = [0.1f64, 0.3, 0.5, 1.0];
    for (name, qs) in &loaded {
        let keep: Vec<&Question> = qs
            .iter()
            .filter(|q| texts.iter().all(|t| answerable(q, t)))
            .collect();
        println!(
            "\n== set {name}: {} of {} questions answerable in every base",
            keep.len(),
            qs.len()
        );
        report["sets"][name]["answerable"] = json!({"kept": keep.len(), "of": qs.len()});
        for (bi, (id, base)) in bases.iter().enumerate() {
            let mut rows: Vec<(String, Score)> = Vec::new();
            let mut add = |label: String, ranks: Vec<Option<usize>>| match rows
                .iter_mut()
                .find(|r| r.0 == label)
            {
                Some(r) => r.1.add(&ranks),
                None => {
                    let mut s = Score::default();
                    s.add(&ranks);
                    rows.push((label, s));
                }
            };
            for q in &keep {
                let rank = |order: &[u32]| -> Vec<Option<usize>> {
                    q.needs
                        .iter()
                        .map(|ph| {
                            order
                                .iter()
                                .take(20)
                                .position(|&g| {
                                    ph.iter()
                                        .any(|p| texts[bi][g as usize].contains(p.as_str()))
                                })
                                .map(|i| i + 1)
                        })
                        .collect()
                };
                let lex_hits = base.search(&an, &q.text, None, SearchMode::Lexical, 60, &p0);
                let lex: Vec<(u32, f32)> = lex_hits
                    .iter()
                    .filter_map(|h| base.number_of(&h.chunk_id).map(|g| (g, h.score as f32)))
                    .collect();
                add(
                    "words".into(),
                    rank(&lex.iter().map(|x| x.0).collect::<Vec<_>>()),
                );
                // The search as the service runs it (hybrid, meaning at 0.1), and with the term graph's walk added
                // (what the graph mode of a chat uses).
                for (label, ppr) in [
                    ("service hybrid", None),
                    ("service hybrid + graph", Some(Ppr::default())),
                ] {
                    let p = Params { ppr, ..p0 };
                    let hits =
                        base.search(&an, &q.text, Some(&q.vector), SearchMode::Hybrid, 20, &p);
                    let order: Vec<u32> = hits
                        .iter()
                        .filter_map(|h| base.number_of(&h.chunk_id))
                        .collect();
                    add(label.into(), rank(&order));
                }
                let own = base.similarities(&q.vector).unwrap_or_default();
                if own.is_empty() {
                    continue;
                }
                let best: Vec<f32> = windows[bi]
                    .iter()
                    .map(|ws| {
                        ws.iter()
                            .map(|w| dot(w, &q.vector))
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .collect();
                let used: &[f32] = if piece_chars > 0 { &mixes } else { &mixes[..1] };
                for &m in used {
                    let sims: Vec<f32> = own
                        .iter()
                        .zip(&best)
                        .map(|(&o, &b)| (1.0 - m) * o + m * o.max(b))
                        .collect();
                    let vec_top = top(&sims, 60, f32::NEG_INFINITY);
                    add(
                        format!("meaning m={m}"),
                        rank(&vec_top.iter().map(|x| x.0).collect::<Vec<_>>()),
                    );
                    for &w in &weights {
                        let fused = fuse(&lex, &vec_top, Fusion::Rrf { vector_weight: w });
                        add(
                            format!("hybrid w={w} m={m}"),
                            rank(&fused.iter().map(|x| x.0).collect::<Vec<_>>()),
                        );
                    }
                }
            }
            println!("-- base {id}");
            for (label, s) in &rows {
                let j = s.json();
                println!(
                    "  {label:<22} R@1 {:.3} R@4 {:.3} R@8 {:.3} MRR {:.3}",
                    j["R@1"].as_f64().unwrap_or(0.0),
                    j["R@4"].as_f64().unwrap_or(0.0),
                    j["R@8"].as_f64().unwrap_or(0.0),
                    j["MRR"].as_f64().unwrap_or(0.0)
                );
                report["sets"][name][id][label] = j;
            }
        }
    }
    if let Some(path) = out {
        std::fs::write(
            format!("{root}/{path}"),
            serde_json::to_string_pretty(&report)?,
        )?;
    }
    Ok(())
}
