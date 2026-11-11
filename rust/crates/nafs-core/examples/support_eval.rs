//! The support check of `nafs_core::support` over stored answers, for measurement (Phase 26).
//!
//! Reads a JSON list of `{"id", "question", "answer", "passages": [text, ...]}` and prints one JSON line per item:
//! the id, the number of claims, how many the passages support, the share, and the unsupported sentences.
//!
//!     cargo run --release -p nafs-core --example support_eval -- answers.json [min_overlap]

use nafs_core::support;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: support_eval <answers.json> [min_overlap]")?;
    let min_overlap: f32 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.6);
    let items: Vec<Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    for item in &items {
        let passages: Vec<&str> = item["passages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let s = support::check(
            item["answer"].as_str().unwrap_or(""),
            &passages,
            item["question"].as_str().unwrap_or(""),
            min_overlap,
        );
        println!(
            "{}",
            json!({"id": item["id"], "claims": s.claims, "supported": s.supported, "share": s.share(),
                "unsupported": s.sentences.iter().filter(|c| !c.supported).map(|c| &c.text).collect::<Vec<_>>()})
        );
    }
    Ok(())
}
