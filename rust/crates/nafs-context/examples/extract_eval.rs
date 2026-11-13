//! Phase 29: precision and recall of the rule extraction (`memory::extract`) on a written set of messages
//! (`scripts/memory_extract_set.json`: `[message, [expected kinds]]`), with no model.
//!
//!     cargo run --release -p nafs-context --example extract_eval -- scripts/memory_extract_set.json

fn main() {
    let path = std::env::args().nth(1).expect("a question set");
    let set: Vec<(String, Vec<String>)> =
        serde_json::from_slice(&std::fs::read(path).expect("readable")).expect("valid set");
    let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
    for (msg, want) in &set {
        let got: Vec<String> = nafs_context::memory::extract(msg)
            .into_iter()
            .map(|c| c.kind)
            .collect();
        let mut want_left = want.clone();
        for g in &got {
            match want_left.iter().position(|w| w == g) {
                Some(i) => {
                    tp += 1;
                    want_left.remove(i);
                }
                None => {
                    fp += 1;
                    println!("extra   {g:<12} {msg}");
                }
            }
        }
        for w in want_left {
            fneg += 1;
            println!("missed  {w:<12} {msg}");
        }
    }
    let p = tp as f64 / (tp + fp).max(1) as f64;
    let r = tp as f64 / (tp + fneg).max(1) as f64;
    println!(
        "{} messages: {tp} right, {fp} extra, {fneg} missed; precision {p:.3}, recall {r:.3}",
        set.len()
    );
}
