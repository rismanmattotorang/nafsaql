//! Head-to-head micro-benchmarks against the two projects Phases 22 and 24 read (vLLM Semantic Router, DeepSeek Harness).
//!
//! The Go side is run from `third_party/semantic-router` with `go test -bench` (see `benchmarks/headtohead/run.sh`); this
//! file times Nafs-Aql's counterpart of each workload on the same machine, so the numbers are comparable.
//!
//! Routing signals from a 2 KB prompt: their `BenchmarkBM25ThirtyRulesLongEnglish` and n-gram variant, against our domain
//! classifier on the same text. Extractive compression to a budget: their `BenchmarkCompress500/2K/16KTokens`, against
//! our sentence packer (FR-K52) on the same text and budgets. Dot product at 384, 768 and 1,024 dimensions: their `vecmath`
//! `BenchmarkDot` (AVX2 assembly and portable), against the vector cache's kernels. The gate decision has no benchmarked
//! counterpart upstream (their decision engine is rule evaluation) and is timed for the record. The harness's host overhead
//! per turn: DeepSeek Harness's `agent-continuation` benchmark (100 turns, 800 reads, a 900 ms budget on a CI runner),
//! against our transcript at the same shape.
//!
//! Run with `cargo run --release -p nafs-bench --example h2h [-- --json out.json]`.

use std::hint::black_box;
use std::time::Instant;

use nafs_core::knowledge::{pack_units, Passage, Retrieved};

fn time_ns<F: FnMut()>(mut f: F, min_ms: u128) -> (f64, u64) {
    // As Go's testing harness does: find a batch size that takes a measurable time, then time whole batches, so the clock
    // read is not part of what is measured (it is tens of nanoseconds, more than a 384-element dot product).
    for _ in 0..3 {
        f();
    }
    let mut batch = 1u64;
    loop {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        if t.elapsed().as_millis() >= 20 || batch >= 1 << 28 {
            break;
        }
        batch *= 2;
    }
    let (mut ops, mut nanos) = (0u64, 0u128);
    let start = Instant::now();
    while start.elapsed().as_millis() < min_ms {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        nanos += t.elapsed().as_nanos();
        ops += batch;
    }
    (nanos as f64 / ops as f64, ops)
}

fn bench_text(sentences: usize) -> String {
    let topics = [
        "Machine learning algorithms process training data efficiently.",
        "Distributed systems require careful consensus protocol design.",
        "The database query optimizer selects the most efficient plan.",
        "Neural network layers transform input through learned weights.",
        "Cryptographic hash functions ensure data integrity verification.",
    ];
    (0..sentences)
        .map(|i| topics[i % 5])
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let mut out = serde_json::Map::new();
    let mut row = |name: &str, ns: f64, note: &str| {
        let pretty = if ns >= 1e6 {
            format!("{:.2} ms", ns / 1e6)
        } else if ns >= 1e3 {
            format!("{:.1} us", ns / 1e3)
        } else {
            format!("{ns:.0} ns")
        };
        println!("{name:46} {pretty:>12}   {note}");
        out.insert(
            name.to_string(),
            serde_json::json!({"ns_per_op": ns, "note": note}),
        );
    };

    // ---- routing signals from a long prompt
    let long = "Please help me debug this function because the deployment is urgent and the service keeps failing. ".repeat(20);
    let (ns, _) = time_ns(
        || {
            black_box(nafs_core::domain::classify(black_box(&long), None, None));
        },
        400,
    );
    row(
        "domain classifier, 2 KB prompt",
        ns,
        "their BM25 30 rules 207 us, n-gram 8.2 ms",
    );

    // ---- extractive compression to a budget
    for (label, sentences, tokens) in [
        ("500 tokens", 40usize, 256usize),
        ("2K tokens", 160, 512),
        ("16K tokens", 1200, 512),
    ] {
        let text = bench_text(sentences);
        let r = Retrieved {
            passages: vec![Passage {
                chunk_id: "c".into(),
                doc_id: "d".into(),
                doc_title: "t".into(),
                heading: String::new(),
                page: None,
                text,
                score: 1.0,
                lexical_score: None,
                vector_score: None,
            }],
            ..Default::default()
        };
        let budget = tokens * 4; // characters for the same number of tokens
        let (ns, _) = time_ns(
            || {
                black_box(pack_units(
                    black_box(&r),
                    "database query optimizer plan",
                    budget,
                ));
            },
            400,
        );
        row(
            &format!("sentence packer, {label} to {tokens}"),
            ns,
            "their Compress: 1.66 ms / 8.35 ms / 72.8 ms",
        );
    }

    // ---- dot products
    for n in [384usize, 768, 1024] {
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.02).collect();
        let (ns, _) = time_ns(
            || {
                black_box(nafs_kb::vcache::dot(black_box(&a), black_box(&b)));
            },
            300,
        );
        row(
            &format!("dot, n={n} (AVX2 + FMA)"),
            ns,
            "their AVX2 asm: 89 / 93 / 154 ns",
        );
        let (ns, _) = time_ns(
            || {
                black_box(nafs_kb::vcache::dot_portable(black_box(&a), black_box(&b)));
            },
            300,
        );
        row(
            &format!("dot, n={n} (portable)"),
            ns,
            "their portable: 510 / 1,121 / 1,509 ns",
        );
    }

    // ---- the gate decision
    {
        let cfg = nafs_core::config::Config::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../config/nafs-aql.toml"
        ))
        .expect("config");
        let cal = nafs_core::calibration::Calibration::prior();
        let f = nafs_types::GateFeatures {
            mean_logprob: Some(-0.4),
            min_logprob: Some(-2.0),
            low_window_logprob: Some(-0.6),
            k: 2,
            agreement: Some(0.5),
            domain: nafs_types::Domain::Math,
            verifiable: false,
            draft_verified: None,
            prompt_tokens: 60,
            draft_tokens: 80,
            reask: false,
        };
        let ctx = nafs_core::gate::GateCtx {
            cfg: &cfg,
            mode: "fixed",
            lambda: 0.2,
            e2: 0.25,
            delta_c: 90.0,
            cal: &cal,
            promoted: false,
            budget: 0.3,
        };
        let (ns, _) = time_ns(
            || {
                black_box(nafs_core::gate::decide(black_box(&f), black_box(&ctx)));
            },
            300,
        );
        row(
            "gate decision (fixed rule)",
            ns,
            "no counterpart benchmarked upstream",
        );
        let ctx2 = nafs_core::gate::GateCtx {
            mode: "calibrated",
            promoted: true,
            ..ctx
        };
        let (ns, _) = time_ns(
            || {
                black_box(nafs_core::gate::decide(black_box(&f), black_box(&ctx2)));
            },
            300,
        );
        row(
            "gate decision (calibrated + certified bins)",
            ns,
            "no counterpart benchmarked upstream",
        );
    }

    // ---- the harness's host overhead: 100 turns, 800 observations of 1 KB, the budget checked every turn
    {
        use nafs_harness::context::Transcript;
        let obs = "x".repeat(1_000);
        let run = || {
            let mut t = Transcript::new("system", "task".into());
            let mut tokens = 0usize;
            for turn in 0..100 {
                t.push("assistant", format!("turn {turn}: read the files"));
                for _ in 0..8 {
                    t.push("user", obs.clone());
                }
                tokens += t.estimated_tokens();
                if t.estimated_tokens() > 2_600 {
                    t.prune_older();
                    if t.estimated_tokens() > 2_600 {
                        t.compact(format!("task and attempts after turn {turn}"));
                    }
                }
                black_box(t.messages());
            }
            tokens
        };
        let (ns, _) = time_ns(
            || {
                black_box(run());
            },
            400,
        );
        row(
            "harness host overhead, 100 turns / 800 reads",
            ns,
            "their budget: 900 ms (CI runner, enforced median)",
        );
    }
    if let Some(i) = std::env::args().position(|a| a == "--json") {
        if let Some(p) = std::env::args().nth(i + 1) {
            std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        }
    }
}
