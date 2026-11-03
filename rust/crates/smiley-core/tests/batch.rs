//! nafs-aql: several users in one weight pass must give each of them exactly what it would get alone.
//!
//! This is the property the whole concurrent path rests on. A batched decode step reads the weights once and
//! produces one token for every user in it; if that changed any user's distribution, the speed would be bought
//! with a silent difference in what people are told. The matmuls are row-independent by construction (one read
//! of a weight row, one dot product per row), so the result should be bit-identical, and that is what is
//! asserted here — not "close enough".

use smiley_core::kv::{KvCache, KvPrec};
use smiley_core::model::{Caches, State, StepStats};
use smiley_core::pool::Pool;
use std::path::PathBuf;

/// The model these tests run against: `SMILEY_CORE_MODEL`, else the repository's own System 1 weights, else the
/// engine's default fixture. Absent all three the test says so and passes, as the other model-backed tests do.
fn model_path() -> PathBuf {
    if let Some(p) = std::env::var_os("SMILEY_CORE_MODEL") {
        return PathBuf::from(p);
    }
    let repo =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/Qwen3-1.7B-Q4_K_M.gguf");
    if repo.exists() {
        return repo;
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
}

/// Greedy continuation of `prompt` for `steps` tokens, one sequence at a time, returning the token ids and the
/// logits of the final step.
fn solo(
    model: &smiley_core::model::Model,
    pool: &Pool,
    max_ctx: usize,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<f32>) {
    let c = model.cfg;
    let mut kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let mut st = State::new(
        &c,
        64,
        2,
        max_ctx,
        pool.width(),
        model.rope_factors.as_deref(),
    );
    let mut stats = StepStats::default();
    let mut out = vec![];
    let mut pos = 0usize;
    // Prefill, then one token per step.
    let last = prompt.len() - 1;
    model
        .forward(&mut st, &mut kv, pool, &prompt[..last], 0, &[], &mut stats)
        .unwrap();
    pos += last;
    let mut tok = prompt[last];
    let mut logits = vec![];
    for _ in 0..steps {
        model
            .forward(&mut st, &mut kv, pool, &[tok], pos, &[0], &mut stats)
            .unwrap();
        logits = st.logits_row(0, c.n_vocab).to_vec();
        tok = argmax(&logits);
        out.push(tok);
        pos += 1;
    }
    (out, logits)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

#[test]
fn two_users_in_one_pass_get_exactly_what_they_would_get_alone() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipped: no model at {}", path.display());
        return;
    }
    let model = smiley_core::model::keep::open(&path).unwrap();
    let c = model.cfg;
    let pool = Pool::new(3);
    let max_ctx = 256;
    // Two different prompts, so the rows differ in content and in length.
    let a: Vec<u32> = vec![1, 450, 29871, 13, 5618];
    let b: Vec<u32> = vec![1, 910, 338, 263, 1243, 310, 278];
    let steps = 6;

    let (solo_a, last_a) = solo(&model, &pool, max_ctx, &a, steps);
    let (solo_b, last_b) = solo(&model, &pool, max_ctx, &b, steps);

    // The same two sequences, now sharing every weight read.
    let mut kv_a = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let mut kv_b = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let mut st = State::new(
        &c,
        64,
        2,
        max_ctx,
        pool.width(),
        model.rope_factors.as_deref(),
    );
    let mut stats = StepStats::default();
    // Each prefills alone (a user arrives on its own), then both decode together.
    model
        .forward(
            &mut st,
            &mut kv_a,
            &pool,
            &a[..a.len() - 1],
            0,
            &[],
            &mut stats,
        )
        .unwrap();
    model
        .forward(
            &mut st,
            &mut kv_b,
            &pool,
            &b[..b.len() - 1],
            0,
            &[],
            &mut stats,
        )
        .unwrap();
    let (mut pos_a, mut pos_b) = (a.len() - 1, b.len() - 1);
    let (mut tok_a, mut tok_b) = (a[a.len() - 1], b[b.len() - 1]);
    let (mut got_a, mut got_b) = (vec![], vec![]);
    let (mut final_a, mut final_b) = (vec![], vec![]);
    for _ in 0..steps {
        let mut both = [&mut kv_a, &mut kv_b];
        model
            .forward_rows(
                &mut st,
                &mut Caches::PerRow(&mut both),
                &pool,
                &[tok_a, tok_b],
                &[pos_a, pos_b],
                &[0, 1],
                &mut stats,
            )
            .unwrap();
        final_a = st.logits_row(0, c.n_vocab).to_vec();
        final_b = st.logits_row(1, c.n_vocab).to_vec();
        tok_a = argmax(&final_a);
        tok_b = argmax(&final_b);
        got_a.push(tok_a);
        got_b.push(tok_b);
        pos_a += 1;
        pos_b += 1;
    }

    assert_eq!(
        got_a, solo_a,
        "user A's tokens changed when another user shared the pass"
    );
    assert_eq!(
        got_b, solo_b,
        "user B's tokens changed when another user shared the pass"
    );
    assert_eq!(
        final_a, last_a,
        "user A's final logits are not bit-identical"
    );
    assert_eq!(
        final_b, last_b,
        "user B's final logits are not bit-identical"
    );
}

/// The two rows must be read from their own caches: swapping the caches must change the answer. Without this a
/// batched pass that silently read one cache for both rows would pass the test above whenever the prompts agree.
#[test]
fn a_row_reads_its_own_cache() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipped: no model at {}", path.display());
        return;
    }
    let model = smiley_core::model::keep::open(&path).unwrap();
    let c = model.cfg;
    let pool = Pool::new(2);
    let max_ctx = 128;
    let mut st = State::new(
        &c,
        16,
        2,
        max_ctx,
        pool.width(),
        model.rope_factors.as_deref(),
    );
    let mut stats = StepStats::default();
    let mut kv_a = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let mut kv_b = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let a: Vec<u32> = vec![1, 450, 29871, 13];
    let b: Vec<u32> = vec![1, 910, 338, 263];
    model
        .forward(&mut st, &mut kv_a, &pool, &a, 0, &[], &mut stats)
        .unwrap();
    model
        .forward(&mut st, &mut kv_b, &pool, &b, 0, &[], &mut stats)
        .unwrap();
    let step =
        |st: &mut State, kvs: [&mut KvCache; 2], stats: &mut StepStats| -> (Vec<f32>, Vec<f32>) {
            let mut kvs = kvs;
            model
                .forward_rows(
                    st,
                    &mut Caches::PerRow(&mut kvs),
                    &pool,
                    &[5618, 5618],
                    &[a.len(), b.len()],
                    &[0, 1],
                    stats,
                )
                .unwrap();
            (
                st.logits_row(0, c.n_vocab).to_vec(),
                st.logits_row(1, c.n_vocab).to_vec(),
            )
        };
    let (row0, row1) = step(&mut st, [&mut kv_a, &mut kv_b], &mut stats);
    assert_ne!(row0, row1, "two different histories gave the same logits");
    // The same token at the same position, with the caches the other way round: the rows must swap too.
    let mut kv_a2 = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    let mut kv_b2 = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    model
        .forward(&mut st, &mut kv_a2, &pool, &a, 0, &[], &mut stats)
        .unwrap();
    model
        .forward(&mut st, &mut kv_b2, &pool, &b, 0, &[], &mut stats)
        .unwrap();
    let mut swapped = [&mut kv_b2, &mut kv_a2];
    model
        .forward_rows(
            &mut st,
            &mut Caches::PerRow(&mut swapped),
            &pool,
            &[5618, 5618],
            &[b.len(), a.len()],
            &[0, 1],
            &mut stats,
        )
        .unwrap();
    assert_eq!(
        st.logits_row(0, c.n_vocab),
        &row1[..],
        "row 0 did not follow its cache"
    );
    assert_eq!(
        st.logits_row(1, c.n_vocab),
        &row0[..],
        "row 1 did not follow its cache"
    );
}

/// A cache restored from a snapshot must behave exactly like one that was computed.
#[test]
fn a_restored_cache_gives_the_same_logits_as_a_computed_one() {
    let path = model_path();
    if !path.exists() {
        return;
    }
    let model = smiley_core::model::keep::open(&path).unwrap();
    let c = model.cfg;
    let pool = Pool::new(3);
    let max_ctx = 256;
    let opening: Vec<u32> = (0..80).map(|i| ((i * 13) % 900 + 11) as u32).collect();
    let mut st = State::new(
        &c,
        96,
        2,
        max_ctx,
        pool.width(),
        model.rope_factors.as_deref(),
    );
    let mut stats = StepStats::default();

    let mut a = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    model
        .forward(&mut st, &mut a, &pool, &opening, 0, &[], &mut stats)
        .unwrap();
    assert_eq!(a.filled(), opening.len());
    let snap = a.snapshot(opening.len());
    assert_eq!(snap.len(), opening.len());

    let mut b = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
    assert!(b.restore(&snap, opening.len()));
    assert_eq!(b.filled(), opening.len());

    let next = 6324u32;
    model
        .forward(
            &mut st,
            &mut a,
            &pool,
            &[next],
            opening.len(),
            &[0],
            &mut stats,
        )
        .unwrap();
    let la = st.logits_row(0, c.n_vocab).to_vec();
    model
        .forward(
            &mut st,
            &mut b,
            &pool,
            &[next],
            opening.len(),
            &[0],
            &mut stats,
        )
        .unwrap();
    let lb = st.logits_row(0, c.n_vocab).to_vec();
    let diff = la.iter().zip(&lb).filter(|(x, y)| x != y).count();
    assert_eq!(
        diff,
        0,
        "{diff} of {} logits differ after a restore",
        la.len()
    );
}
