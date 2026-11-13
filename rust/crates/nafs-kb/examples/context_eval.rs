//! Phase 29 measurements of the context filesystem's retrieval, with no model call beyond question vectors (cached):
//!
//! 1. Scope: the whole base, the rule planner's document scope fused with the whole base (what `/v1/fs/search`
//!    does), and the answering document as the scope (a ceiling no planner can pass).
//! 2. Assembly: how often the answer's phrase reaches the prompt, and how long the prompt's passage block is, for
//!    four whole passages (the chat default) against the budgeted assembler (every hit first as its best sentences,
//!    the best hits then in full) at the same and at a smaller budget.
//!
//!     cargo run --release -p nafs-kb --example context_eval -- --kb <id> --set quoted=scripts/kb_p25_quoted.json ...

use std::collections::{HashMap, HashSet};

use nafs_context::assemble::{assemble, Budget, Item};
use nafs_kb::index::{Analyzer, Params};
use nafs_kb::{
    ChunkOpts, EmbedConfig, Embedder, Hit, IndexConfig, Kb, KbConfig, Limits, SearchMode,
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

fn arg_all(args: &[String], name: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == name)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// The sentences of a passage that share most words with the question, in passage order, up to `max` characters.
fn best_sentences(an: &Analyzer, question: &str, text: &str, max: usize) -> String {
    let q: HashSet<String> = an.query(question).into_iter().collect();
    let flat = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let mut sents: Vec<&str> = Vec::new();
    let mut start = 0;
    let b = flat.as_bytes();
    for i in 0..b.len() {
        if matches!(b[i], b'.' | b'!' | b'?' | b'|') && (i + 1 == b.len() || b[i + 1] == b' ') {
            sents.push(flat[start..=i].trim());
            start = i + 1;
        }
    }
    if start < flat.len() {
        sents.push(flat[start..].trim());
    }
    let mut scored: Vec<(usize, usize)> = sents
        .iter()
        .enumerate()
        .map(|(i, s)| (an.query(s).iter().filter(|t| q.contains(*t)).count(), i))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut keep: Vec<usize> = Vec::new();
    let mut len = 0;
    for (score, i) in scored {
        if score == 0 && !keep.is_empty() {
            break;
        }
        if len + sents[i].len() > max && !keep.is_empty() {
            continue;
        }
        len += sents[i].len() + 1;
        keep.push(i);
        if len >= max {
            break;
        }
    }
    keep.sort_unstable();
    keep.iter()
        .map(|&i| sents[i])
        .collect::<Vec<_>>()
        .join(" … ")
}

#[derive(Default)]
struct Score {
    n: usize,
    r1: f64,
    r4: f64,
    mrr: f64,
}

impl Score {
    fn add(&mut self, rank: Option<usize>) {
        self.n += 1;
        if let Some(r) = rank {
            self.r1 += (r <= 1) as u8 as f64;
            self.r4 += (r <= 4) as u8 as f64;
            self.mrr += 1.0 / r as f64;
        }
    }
    fn json(&self) -> Value {
        let n = self.n.max(1) as f64;
        let r = |x: f64| (x / n * 1000.0).round() / 1000.0;
        json!({"n": self.n, "R@1": r(self.r1), "R@4": r(self.r4), "MRR": r(self.mrr)})
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let kb_id = arg_all(&args, "--kb")
        .first()
        .cloned()
        .ok_or("--kb is required")?;
    let sets: Vec<(String, String)> = arg_all(&args, "--set")
        .into_iter()
        .filter_map(|s| {
            s.split_once('=')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();
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
    let base = loop {
        if let Some(b) = kb.index_of(&kb_id).await? {
            break b;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    };
    let an = Analyzer::default();
    let p = Params::default();
    let embedder = Embedder::new(embed);
    let qcache_path = format!("{root}/data/runtime/kb_eval_qvec.json");
    let mut qcache: HashMap<String, Vec<f32>> = std::fs::read(&qcache_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    // Layers of every document, for the planner.
    let layers: Vec<_> = base
        .documents_list()
        .iter()
        .filter_map(|(d, _)| base.doc_layers(&an, d))
        .collect();
    let texts: Vec<(String, String)> = (0..base.passages() as u32)
        .map(|g| {
            let h = base.passage(g);
            (h.doc_id, norm(&h.text))
        })
        .collect();
    let mut report = json!({"kb": kb_id, "passages": base.passages(), "documents": layers.len()});
    for (name, path) in &sets {
        let raw: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{root}/{path}"))?)?;
        let (mut whole, mut planned, mut oracle) =
            (Score::default(), Score::default(), Score::default());
        let mut fired = 0usize;
        let mut fired_right = 0usize;
        // Assembly arms: (answer reached the prompt, characters of the block).
        let mut arms: Vec<(String, usize, usize, usize)> = vec![
            ("4 whole passages".into(), 0, 0, 0),
            ("assembler, 4,800 chars".into(), 0, 0, 0),
            ("assembler, 3,600 chars".into(), 0, 0, 0),
            ("assembler, 2,400 chars".into(), 0, 0, 0),
        ];
        let mut n = 0usize;
        for item in raw {
            let q = item[0].as_str().unwrap_or("").to_string();
            // Single-passage questions only: the answering document is the first passage holding a phrase.
            let phrases: Vec<String> = item[1]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(norm)
                .collect();
            if item.as_array().map_or(0, Vec::len) > 2 {
                continue;
            }
            let Some(answer_doc) = texts
                .iter()
                .find(|(_, t)| phrases.iter().any(|p| t.contains(p.as_str())))
                .map(|x| x.0.clone())
            else {
                continue;
            };
            n += 1;
            if !qcache.contains_key(&q) {
                let v = embedder
                    .embed(std::slice::from_ref(&q), Some(QUERY_INSTRUCTION))
                    .await?
                    .remove(0);
                qcache.insert(q.clone(), v);
            }
            let qv = qcache[&q].clone();
            let rank_in = |hits: &[Hit]| -> Option<usize> {
                hits.iter()
                    .take(20)
                    .position(|h| {
                        let t = norm(&h.text);
                        phrases.iter().any(|p| t.contains(p.as_str()))
                    })
                    .map(|i| i + 1)
            };
            let all = base.search_in(&an, &q, Some(&qv), SearchMode::Hybrid, 20, &p, None);
            whole.add(rank_in(&all));
            let only: HashSet<String> = [answer_doc.clone()].into();
            let o = base.search_in(&an, &q, Some(&qv), SearchMode::Hybrid, 20, &p, Some(&only));
            oracle.add(rank_in(&o));
            // The planner: a document named by its title or by one of its three most distinctive terms that no
            // other document of the base has among its own.
            let lower = q.to_lowercase();
            let words: HashSet<&str> = lower
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .collect();
            let named = layers.iter().find(|l| {
                let title = l.title.to_lowercase();
                (title.len() >= 6 && lower.contains(&title))
                    || l.terms.iter().take(3).any(|t| {
                        let t = t.to_lowercase();
                        t.len() >= 5
                            && words.contains(t.as_str())
                            && !layers.iter().any(|o| {
                                o.doc_id != l.doc_id
                                    && o.terms.iter().any(|x| x.to_lowercase() == t)
                            })
                    })
            });
            let fused: Vec<Hit> = match named {
                Some(l) => {
                    fired += 1;
                    fired_right += (l.doc_id == answer_doc) as usize;
                    let sc: HashSet<String> = [l.doc_id.clone()].into();
                    let scoped =
                        base.search_in(&an, &q, Some(&qv), SearchMode::Hybrid, 20, &p, Some(&sc));
                    let mut score: HashMap<String, (f64, Hit)> = HashMap::new();
                    for list in [&scoped, &all] {
                        for (r, h) in list.iter().enumerate() {
                            score
                                .entry(h.chunk_id.clone())
                                .or_insert((0.0, h.clone()))
                                .0 += 1.0 / (61.0 + r as f64);
                        }
                    }
                    let mut v: Vec<(f64, Hit)> = score.into_values().collect();
                    v.sort_by(|a, b| b.0.total_cmp(&a.0));
                    v.into_iter().map(|x| x.1).collect()
                }
                None => all.clone(),
            };
            planned.add(rank_in(&fused));
            // Assembly over the whole base's hits.
            let reached = |block: &str| phrases.iter().any(|p| norm(block).contains(p.as_str()));
            let four: String = all
                .iter()
                .take(4)
                .map(|h| h.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            arms[0].1 += reached(&four) as usize;
            arms[0].2 += four.len();
            for (ai, chars) in [(1usize, 4800usize), (2, 3600), (3, 2400)] {
                let items: Vec<Item> = all
                    .iter()
                    .take(10)
                    .enumerate()
                    .map(|(r, h)| Item {
                        uri: h.chunk_id.clone(),
                        category: "passage".into(),
                        score: 1.0 / (1.0 + r as f64),
                        levels: vec![
                            Some(best_sentences(&an, &q, &h.text, 320)),
                            Some(h.text.clone()),
                        ],
                    })
                    .collect();
                let chosen = assemble(
                    &items,
                    &Budget {
                        chars,
                        quotas: HashMap::new(),
                        max_items: 10,
                    },
                );
                let block: String = chosen
                    .iter()
                    .filter_map(|c| items[c.item].levels[c.level].clone())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                arms[ai].1 += reached(&block) as usize;
                arms[ai].2 += block.len();
                arms[ai].3 += chosen.len();
            }
        }
        std::fs::write(&qcache_path, serde_json::to_vec(&qcache)?)?;
        println!("\n== {name}: {n} single-passage questions");
        println!("  whole base       {}", whole.json());
        println!(
            "  planner scope    {} (fired {fired}, right document {fired_right})",
            planned.json()
        );
        println!("  answer's doc     {}", oracle.json());
        report[name] = json!({"n": n, "whole": whole.json(), "planned": planned.json(), "oracle": oracle.json(),
            "planner_fired": fired, "planner_right": fired_right, "assembly": {}});
        for (label, hit, chars, items) in &arms {
            let nn = n.max(1) as f64;
            println!(
                "  {label:<24} answer in prompt {:.3}  block {:.0} chars  {:.1} sources",
                *hit as f64 / nn,
                *chars as f64 / nn,
                if *items == 0 { 4.0 } else { *items as f64 / nn }
            );
            report[name]["assembly"][label] = json!({"answer_in_prompt": *hit as f64 / nn, "chars": *chars as f64 / nn,
                "sources": if *items == 0 { 4.0 } else { *items as f64 / nn }});
        }
    }
    if let Some(o) = out {
        std::fs::write(
            format!("{root}/{o}"),
            serde_json::to_string_pretty(&report)?,
        )?;
    }
    Ok(())
}
