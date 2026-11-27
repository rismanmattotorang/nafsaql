//! Phase 39: knowledge-graph construction speed and quality per extraction setting. The same passages (the first
//! `--passages` of the given files, chunked as the knowledge base chunks them) are extracted under every arm; arms run
//! block by block (a block = `--block` passages) in rotating order, so a load change on the shared host hits every arm.
//! Each arm is `name=url,concurrency,constrain` (constrain 0|1), e.g.
//!
//!   cargo run --release -p nafs-kb --example graph_bench -- --passages 24 --block 4 \
//!     --arm smiley=http://127.0.0.1:8201/v1,1,0 --arm peregrine=http://127.0.0.1:8211/v1,1,0 \
//!     --arm peregrine-x4=http://127.0.0.1:8211/v1,4,0 --arm peregrine-x4-schema=http://127.0.0.1:8211/v1,4,1 \
//!     papers/2309.06180v1.pdf > benchmarks/engines_phase39/graph_bench.json
//!
//! Reported per arm: wall seconds, passages per minute, replies that parsed (the rest are skipped by the builder),
//! entities and relations kept after grounding, relations dropped by the quote gate, tokens, and agreement with the
//! first arm (Jaccard of entity keys per passage). Quality is what is kept after the builder's own checks.
use futures::StreamExt;
use nafs_kb::graph::{Extractor, GraphConfig};
use nafs_kb::{chunk, parse, ChunkOpts, Kind};
use serde_json::json;
use std::collections::HashSet;
use std::time::Instant;

struct Arm {
    name: String,
    ex: Extractor,
    width: usize,
}

#[derive(Default, Clone)]
struct Row {
    parsed: bool,
    entities: Vec<String>,
    relations: usize,
    quote_drops: usize,
    tokens: u64,
    error: Option<String>,
}

fn load1() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut files, mut arms, mut n, mut block) = (vec![], vec![], 24usize, 4usize);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--passages" => {
                n = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--block" => {
                block = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--arm" => {
                let (name, rest) = args[i + 1]
                    .split_once('=')
                    .expect("--arm name=url,concurrency,constrain");
                let p: Vec<&str> = rest.split(',').collect();
                let cfg = GraphConfig {
                    llm_url: p[0].into(),
                    model: "system1".into(),
                    max_tokens: 400,
                    merge_similarity: 0.9,
                    concurrency: p.get(1).and_then(|x| x.parse().ok()).unwrap_or(1),
                    constrain: p.get(2) == Some(&"1"),
                };
                arms.push(Arm {
                    name: name.into(),
                    width: cfg.concurrency.max(1),
                    ex: Extractor::new(cfg),
                });
                i += 1;
            }
            f => files.push(f.to_string()),
        }
        i += 1;
    }
    let mut passages = vec![];
    for f in &files {
        let kind = Kind::from_filename(f).expect("a supported file");
        let parsed = parse(kind, &std::fs::read(f).expect("readable")).expect("parsed");
        // Skip very short pieces (a title line): the builder reads them too, but they say nothing about speed.
        passages.extend(
            chunk(&parsed, ChunkOpts::default())
                .into_iter()
                .filter(|c| c.chars >= 400)
                .map(|c| c.text),
        );
    }
    passages.truncate(n);
    eprintln!(
        "{} passages, {} arms, blocks of {block}",
        passages.len(),
        arms.len()
    );
    let mut rows: Vec<Vec<Row>> = vec![vec![Row::default(); passages.len()]; arms.len()];
    let mut wall = vec![0f64; arms.len()];
    let mut loads = vec![];
    for (b, start) in (0..passages.len()).step_by(block.max(1)).enumerate() {
        let end = (start + block).min(passages.len());
        let k = b % arms.len();
        let order: Vec<usize> = (k..arms.len()).chain(0..k).collect();
        for a in order {
            let arm = &arms[a];
            loads.extend(load1());
            let t = Instant::now();
            let out: Vec<_> = futures::stream::iter(passages[start..end].iter())
                .map(|p| arm.ex.extract(p))
                .buffered(arm.width)
                .collect()
                .await;
            wall[a] += t.elapsed().as_secs_f64();
            for (j, r) in out.into_iter().enumerate() {
                rows[a][start + j] = match r {
                    Ok(call) => match call.result {
                        Ok(ex) => Row {
                            parsed: true,
                            entities: ex.entities.iter().map(|e| e.key.clone()).collect(),
                            relations: ex.relations.len(),
                            quote_drops: ex.quote_drops,
                            tokens: call.completion_tokens,
                            error: None,
                        },
                        Err(why) => Row {
                            tokens: call.completion_tokens,
                            error: Some(why),
                            ..Default::default()
                        },
                    },
                    Err(e) => Row {
                        error: Some(e.to_string()),
                        ..Default::default()
                    },
                };
            }
            eprintln!("block {b} {} {:.1}s", arm.name, t.elapsed().as_secs_f64());
        }
    }
    let jacc = |a: &[String], b: &[String]| {
        let (x, y): (HashSet<&String>, HashSet<&String>) = (a.iter().collect(), b.iter().collect());
        let u = x.union(&y).count();
        if u == 0 {
            1.0
        } else {
            x.intersection(&y).count() as f64 / u as f64
        }
    };
    let out: Vec<_> = arms.iter().enumerate().map(|(a, arm)| {
        let r = &rows[a];
        let parsed = r.iter().filter(|x| x.parsed).count();
        let agree: Vec<f64> = (0..r.len()).filter(|&j| r[j].parsed && rows[0][j].parsed).map(|j| jacc(&r[j].entities, &rows[0][j].entities)).collect();
        json!({"arm": arm.name, "url": arm.ex.cfg.llm_url, "concurrency": arm.width, "constrain": arm.ex.cfg.constrain,
            "passages": r.len(), "wall_s": wall[a], "passages_per_min": r.len() as f64 * 60.0 / wall[a].max(1e-9),
            "parsed": parsed, "skipped": r.len() - parsed,
            "entities": r.iter().map(|x| x.entities.len()).sum::<usize>(), "relations": r.iter().map(|x| x.relations).sum::<usize>(),
            "quote_drops": r.iter().map(|x| x.quote_drops).sum::<usize>(), "tokens": r.iter().map(|x| x.tokens).sum::<u64>(),
            "entity_jaccard_vs_first": if agree.is_empty() { None } else { Some(agree.iter().sum::<f64>() / agree.len() as f64) },
            "errors": r.iter().filter_map(|x| x.error.clone()).take(5).collect::<Vec<_>>()})
    }).collect();
    println!("{}", serde_json::to_string_pretty(&json!({"files": files, "passages": passages.len(), "block": block,
        "host_load": {"min": loads.iter().cloned().reduce(f64::min), "max": loads.iter().cloned().reduce(f64::max)}, "arms": out})).unwrap());
}
