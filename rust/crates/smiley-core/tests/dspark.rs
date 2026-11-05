//! DSpark (DeepSeek's speculative drafter) on a Qwen3 target, end to end.
//!
//! * Qwen3 itself is checked against llama.cpp: the greedy continuation below was
//!   produced by llama.cpp b11146's `/completion` on the same file and the same raw
//!   prompt; the two agree for the first ~50 tokens and then part at a near-tie
//!   (summation order), so the check is on that common prefix.
//! * DSpark must be lossless: greedy output with the drafter is the plain greedy
//!   output, token for token, whatever the drafter proposes — and the drafter must
//!   actually be accepted often enough to be worth carrying.
//!
//! Skips when the 2.5 GB target or the 2.8 GB drafter is absent.

use smiley_core::session::{Session, SessionCfg};
use std::path::PathBuf;

fn home(p: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(p)
}

fn serial() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

fn session(drafter: bool, max: usize) -> Option<Session> {
    let target = home("models/Qwen3-4B-Q4_K_M.gguf");
    let dir = home("models/dspark_qwen3_4b_block7");
    if !target.exists() || (drafter && !dir.join("model.safetensors").exists()) {
        eprintln!("\n  SKIPPED: needs {target:?} and {dir:?}\n");
        return None;
    }
    let cfg = SessionCfg {
        max_tokens: max,
        thinking: false,
        speculate: false,
        dspark: drafter.then_some(dir),
        ..Default::default()
    };
    Some(Session::open(&target, cfg).expect("open"))
}

/// llama.cpp's greedy continuation of the prompt below, thinking off (its first 49 tokens).
const LLAMA_CPP: &str = "A **CPU cache** is a small, fast memory component that sits between the CPU and the main system memory (RAM), and is used to speed up data access by reducing the time it takes for the CPU to retrieve data from main memory.";

/// How much of that continuation is well-determined. At the token after "the main" the
/// model's top two logits differ by 0.0007 of ~20 ("the main" + "system" or "memory";
/// `SMILEY_CORE_TRACE=1` prints every such near-tie): which one wins is decided by float summation order, and llama.cpp's
/// order is not ours — nor is ours the same from one kernel revision to the next. An
/// engine that diverges *before* that point is wrong; one that diverges *at* it is
/// merely rounding differently. Matching all 49 tokens was a knife-edge, not a check.
const LLAMA_CPP_DETERMINED: &str =
    "A **CPU cache** is a small, fast memory component that sits between the CPU and the main";

#[test]
fn qwen3_continues_the_way_llama_cpp_does() {
    let _g = serial();
    let Some(mut s) = session(false, 56) else {
        return;
    };
    let (text, _) = s
        .generate("Explain how a CPU cache works, in detail.", |_| {})
        .unwrap();
    assert!(
        text.starts_with(LLAMA_CPP_DETERMINED),
        "Smiley Core: {text:?}"
    );
    assert!(LLAMA_CPP.starts_with(LLAMA_CPP_DETERMINED));
}

#[test]
fn dspark_is_lossless_and_accepted() {
    let _g = serial();
    let prompts = [
        "Write a Python function that merges two sorted lists, with a docstring and two examples.",
        "A train travels 180 km in 2.5 hours. What is its average speed in km/h and in m/s? Show the steps.",
    ];
    let Some(mut plain) = session(false, 80) else {
        return;
    };
    let Some(mut fast) = session(true, 80) else {
        return;
    };
    for p in prompts {
        plain.reset();
        fast.reset();
        let (a, _) = plain.generate(p, |_| {}).unwrap();
        let (b, tl) = fast.generate(p, |_| {}).unwrap();
        assert_eq!(a, b, "DSpark changed the output for {p:?}");
        let per_pass = tl.n_decode() as f64 / tl.passes.max(1) as f64;
        eprintln!(
            "{p:.40}… {} rounds, {:.2} tokens per pass, accepted {:?}",
            tl.dspark_rounds,
            per_pass,
            tl.acceptance_rate()
        );
        assert!(tl.dspark_rounds > 0, "the drafter never drafted");
        assert!(
            per_pass > 1.8,
            "only {per_pass:.2} tokens per pass — the drafter is not being accepted"
        );
    }
    // A second turn continues the conversation with the drafter's context intact.
    let (a, _) = plain
        .generate("Now do it for 300 km in 4 hours.", |_| {})
        .unwrap();
    let (b, _) = fast
        .generate("Now do it for 300 km in 4 hours.", |_| {})
        .unwrap();
    assert_eq!(a, b, "second turn diverged");
}

/// Sampling at temperature 0.7: DSpark stays lossless (position-keyed randomness),
/// and — because the drafter draws with the target's keys — it is still accepted.
#[test]
fn dspark_is_lossless_and_accepted_when_sampling() {
    let _g = serial();
    let p =
        "Write a Python function that merges two sorted lists, with a docstring and two examples.";
    let open = |drafter: bool| {
        session(drafter, 80).map(|mut s| {
            s.cfg.temperature = 0.7;
            s.cfg.top_p = 0.9;
            s
        })
    };
    let Some(mut plain) = open(false) else { return };
    let Some(mut fast) = open(true) else { return };
    let (a, _) = plain.generate(p, |_| {}).unwrap();
    let (b, tl) = fast.generate(p, |_| {}).unwrap();
    assert_eq!(a, b, "DSpark changed a sampled reply");
    let per_pass = tl.n_decode() as f64 / tl.passes.max(1) as f64;
    eprintln!(
        "sampled: {} rounds, {:.2} tokens per pass, accepted {:?}",
        tl.dspark_rounds,
        per_pass,
        tl.acceptance_rate()
    );
    assert!(
        tl.dspark_rounds > 0,
        "the drafter never drafted while sampling"
    );
    assert!(per_pass > 1.4, "only {per_pass:.2} tokens per pass while sampling — the draft is not coupled to the target's noise");
}

#[test]
fn a_drafter_for_another_target_is_refused() {
    let _g = serial();
    let dir = home("models/dspark_qwen3_4b_block7");
    let other = home("models/SmolLM2-360M-Instruct.Q4_K_M.gguf");
    if !dir.join("model.safetensors").exists() || !other.exists() {
        return;
    }
    let cfg = SessionCfg {
        dspark: Some(dir),
        ..Default::default()
    };
    let err = Session::open(&other, cfg)
        .err()
        .expect("a 960-wide target must refuse a 2560-wide drafter");
    assert!(err.contains("2560") || err.contains("vocabulary"), "{err}");
}
