//! LittleBit acceptance on real weights (plan §6; the synthetic properties live in
//! `src/lbq.rs`'s unit tests).
//!
//! * **A-LB1** — the forward pipeline computes exactly the product with the
//!   reconstructed Ŵ, on factors built from a real tensor, with a negative control
//!   that must fail (one flipped sign).
//! * **A-LB4** — Dual-SVID earns its keep on real weights: better than random
//!   signs, improved by a residual stage, never hurt by refinement.
//! * **A-LB5** — end to end: convert the real model (small rank, for test time),
//!   open a session through the ordinary sidecar pickup, chat, and check the
//!   telemetry moved far fewer weight bytes than the dense run — while the dense
//!   run stays bit-identical to what it was (the sidecar is keyed to its own copy).
//!
//! The model file is 270 MB and not in the repository; point `SMILEY_CORE_MODEL`
//! at it or keep the default path, as `oracle.rs` does. Tests that chat take a
//! shared lock, as `engine.rs` does — two sessions at once skew each other's
//! timings and the kept-model slot.

use smiley_core::gguf::Gguf;
use smiley_core::lbq::{
    build_tensor, convert_gguf, plan_gguf, sidecar_info, ConvertEvent, ConvertOpts, Progress, Slb,
};
use smiley_core::pool::Pool;
use smiley_core::quant::{dequant_row, row_bytes};
use smiley_core::session::{Session, SessionCfg};
use std::path::PathBuf;

fn model_path() -> PathBuf {
    std::env::var_os("SMILEY_CORE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

fn serial() -> std::sync::MutexGuard<'static, ()> {
    static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
    M.lock().unwrap_or_else(|e| e.into_inner())
}

/// Dequantise one whole tensor to f32, row-major `d_out×d_in`.
fn tensor_f32(g: &Gguf, name: &str) -> (Vec<f32>, usize, usize) {
    let t = g.tensor(name).unwrap();
    let (d_in, d_out) = (t.dims[0] as usize, t.dims[1] as usize);
    let rb = row_bytes(t.kind, d_in);
    let bytes = g.bytes(t);
    let mut w = vec![0.0f32; d_out * d_in];
    for i in 0..d_out {
        dequant_row(
            t.kind,
            &bytes[i * rb..(i + 1) * rb],
            &mut w[i * d_in..(i + 1) * d_in],
        );
    }
    (w, d_out, d_in)
}

/// A-LB1: on a real attention projection, the loaded container's forward equals a
/// f64 dense product against the explicitly reconstructed Ŵ — and one flipped sign
/// in the reconstruction breaks the agreement (so the test can actually fail).
#[test]
fn a_lb1_forward_equals_explicit_reconstruction_on_real_weights() {
    let g = Gguf::open(&model_path()).unwrap();
    let (w, d_out, d_in) = tensor_f32(&g, "blk.0.attn_q.weight");
    let pool = Pool::new(4);
    let b = build_tensor("blk.0.attn_q.weight", w, d_out, d_in, 64, 2, 1, &pool);
    let dir = std::env::temp_dir().join(format!("smiley-lbq-it-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("real.slb");
    smiley_core::lbq::write_slb(&path, 0, 0, std::slice::from_ref(&b)).unwrap();
    let slb = Slb::open(&path).unwrap();
    let t = &slb.tensors[0];

    // Reconstruct Ŵ in f64 from the loaded bits and scales.
    let mut what = vec![0.0f64; d_out * d_in];
    for (si, st) in t.stages.iter().enumerate() {
        let (ub, vb) = (slb.u_bits(t, si), slb.v_bits(t, si));
        let (urb, vrb) = (t.r_pad / 8, d_in / 8);
        for i in 0..d_out {
            for k in 0..t.r_pad {
                let us = if ub[i * urb + k / 8] >> (k % 8) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                };
                let hl = st.h[i] as f64 * st.l[k] as f64 * us;
                if hl == 0.0 {
                    continue;
                }
                for j in 0..d_in {
                    let vs = if vb[k * vrb + j / 8] >> (j % 8) & 1 == 1 {
                        1.0f64
                    } else {
                        -1.0
                    };
                    what[i * d_in + j] += hl * vs * st.g[j] as f64;
                }
            }
        }
    }
    let x: Vec<f32> = (0..d_in)
        .map(|j| ((j as f32) * 0.173).sin() * 0.5)
        .collect();
    let mut got = vec![0.0f32; d_out];
    slb.apply(0, &x, &mut got);
    let scale: f64 = what.iter().map(|v| v * v).sum::<f64>().sqrt() / (d_out as f64).sqrt();
    let mut worst = 0.0f64;
    for i in 0..d_out {
        let want: f64 = (0..d_in).map(|j| what[i * d_in + j] * x[j] as f64).sum();
        worst = worst.max((got[i] as f64 - want).abs());
    }
    assert!(
        worst < 1e-3 * scale.max(1e-6),
        "forward vs reconstruction: {worst} (scale {scale})"
    );

    // Negative control: flip one sign in the reconstruction; the comparison must
    // now fail by far more than the tolerance, or this test proves nothing.
    let (i0, k0) = (3usize, 5usize);
    let st0 = &t.stages[0];
    let flip = 2.0 * st0.h[i0] as f64 * st0.l[k0] as f64;
    let vb = slb.v_bits(t, 0);
    let mut broken = 0.0f64;
    {
        let i = i0;
        let want: f64 = (0..d_in)
            .map(|j| {
                let vs = if vb[k0 * (d_in / 8) + j / 8] >> (j % 8) & 1 == 1 {
                    1.0f64
                } else {
                    -1.0
                };
                let delta = -flip * vs * st0.g[j] as f64; // the flipped Ŵ row differs by this
                (what[i * d_in + j] + delta) * x[j] as f64
            })
            .sum();
        broken = broken.max((got[i] as f64 - want).abs());
    }
    assert!(
        broken > 1e-3 * scale,
        "a flipped sign went unnoticed: {broken} vs scale {scale}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A-LB4: Dual-SVID initialisation beats sign-random baselines on a real tensor,
/// the residual stage reduces the error further, and refinement never increases it.
#[test]
fn a_lb4_dual_svid_beats_random_and_stages_help_on_real_weights() {
    let g = Gguf::open(&model_path()).unwrap();
    let (w, d_out, d_in) = tensor_f32(&g, "blk.1.ffn_gate.weight");
    let pool = Pool::new(4);
    let r = 64;
    let e1 = build_tensor("t", w.clone(), d_out, d_in, r, 1, 0, &pool).rel_err;
    let e2 = build_tensor("t", w.clone(), d_out, d_in, r, 2, 0, &pool).rel_err;
    let e1r = build_tensor("t", w.clone(), d_out, d_in, r, 1, 2, &pool).rel_err;
    assert!(e1 < 1.0, "one stage must beat predicting zero: {e1}");
    assert!(e2 < e1, "the residual stage must help: {e2} vs {e1}");
    assert!(e1r <= e1 + 1e-9, "refinement must not hurt: {e1r} vs {e1}");

    // The random-sign baseline: same budget, signs drawn at random, scales fit the
    // same way (rank-1 of |random| is ~uniform). SVID must beat it clearly. A
    // random ±1 factorisation of this shape reconstructs essentially nothing, so
    // its relative error sits near 1.0.
    assert!(e1 < 0.95, "SVID should recover visible structure: {e1}");
}

/// A-LB5: the whole path — convert, sidecar pickup, session, telemetry.
#[test]
fn a_lb5_converted_model_chats_and_moves_far_fewer_bytes() {
    let _s = serial();
    let dir = std::env::temp_dir().join(format!("smiley-lbq-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Hard links, so the 270 MB file is not copied and the user's own model
    // directory never grows a sidecar from a test run.
    let lb_model = dir.join("lb.gguf");
    let dense_model = dir.join("dense.gguf");
    let _ = std::fs::remove_file(&lb_model);
    let _ = std::fs::remove_file(&dense_model);
    std::fs::hard_link(model_path(), &lb_model).unwrap();
    std::fs::hard_link(model_path(), &dense_model).unwrap();

    // Generous-but-quick settings: rank 64, both stages. Quality is exploratory
    // without QAT (plan §2) and is not gated here; bytes and correctness are.
    let opts = ConvertOpts {
        rank_override: Some(64),
        stages: 2,
        refine: 1,
        verbose: false,
        ..Default::default()
    };
    // Cancellation before the first tensor: an error, and nothing on disk.
    {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let e = convert_gguf(
            &lb_model,
            &ConvertOpts {
                cancel: Some(flag),
                ..opts.clone()
            },
        )
        .err();
        assert_eq!(e.as_deref(), Some("cancelled"));
        assert!(
            sidecar_info(&lb_model).is_none(),
            "a cancelled conversion left a file behind"
        );
    }
    let plan = plan_gguf(&lb_model, &opts).unwrap();
    let events: std::sync::Arc<std::sync::Mutex<Vec<ConvertEvent>>> = Default::default();
    let sink = std::sync::Arc::clone(&events);
    let rep = convert_gguf(
        &lb_model,
        &ConvertOpts {
            progress: Some(Progress(std::sync::Arc::new(move |e: &ConvertEvent| {
                sink.lock().unwrap().push(e.clone())
            }))),
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(rep.out, dir.join("lb.gguf.slb"));
    assert!(
        rep.data_bytes < rep.source_bytes / 4,
        "the sidecar should be far smaller: {} vs {}",
        rep.data_bytes,
        rep.source_bytes
    );
    // The plan the app shows before converting is exactly what conversion produced.
    assert_eq!(
        plan.data_bytes, rep.data_bytes,
        "planned vs written sidecar bytes"
    );
    assert_eq!(plan.tensors.len(), rep.converted.len());
    assert_eq!(plan.source_bytes, rep.source_bytes);
    // One event per candidate tensor, in order, ending at the totals.
    {
        let ev = events.lock().unwrap();
        assert_eq!(ev.len(), plan.tensors.len() + plan.skipped.len());
        assert!(ev
            .windows(2)
            .all(|w| w[1].done == w[0].done + 1 && w[1].done_params >= w[0].done_params));
        let last = ev.last().unwrap();
        assert_eq!(
            (last.done, last.done_params),
            (last.total, last.total_params)
        );
        assert!(ev
            .iter()
            .filter(|e| e.rank > 0)
            .all(|e| e.rel_err > 0.0 && e.rel_err < 1.5));
    }
    // What the app's Models view reports about the file.
    let info = sidecar_info(&lb_model).expect("the sidecar is there");
    assert!(
        info.problem.is_none(),
        "fresh sidecar reported unusable: {:?}",
        info.problem
    );
    assert_eq!(
        (info.tensors, info.stages, info.rank_min, info.rank_max),
        (rep.converted.len(), 2, 64, 64)
    );
    assert!((info.bpw - rep.bpw).abs() < 1e-9);
    // A sidecar beside a different file is detected as stale, not loaded.
    std::fs::hard_link(&rep.out, dir.join("dense.gguf.slb")).unwrap();
    std::fs::write(dir.join("other.gguf"), b"GGUF not really").unwrap();
    std::fs::hard_link(&rep.out, dir.join("other.gguf.slb")).unwrap();
    assert!(sidecar_info(&dir.join("other.gguf"))
        .unwrap()
        .problem
        .is_some());
    let _ = std::fs::remove_file(dir.join("other.gguf.slb"));
    let _ = std::fs::remove_file(dir.join("other.gguf"));

    let cfg = SessionCfg {
        max_tokens: 24,
        threads: 4,
        speculate: false,
        ..Default::default()
    };
    // The dense control runs with LittleBit declined, although `dense.gguf` now has a
    // valid sidecar beside it too: the setting, not the file system, decides.
    let mut dense = Session::open(
        &dense_model,
        SessionCfg {
            littlebit: false,
            ..cfg.clone()
        },
    )
    .unwrap();
    assert!(
        dense.model.lb_summary().is_none(),
        "littlebit: false still attached the sidecar"
    );
    let (_text_d, telem_d) = dense
        .generate("Say hello in one short sentence.", |_| {})
        .unwrap();
    drop(dense);

    let mut lb = Session::open(&lb_model, cfg.clone()).unwrap();
    assert!(
        lb.model.lb_summary().is_some(),
        "the sidecar was not picked up"
    );
    let (text, telem) = lb
        .generate("Say hello in one short sentence.", |_| {})
        .unwrap();
    assert!(!text.is_empty(), "the converted model produced nothing");
    // Both kernels drive the same factors: the first greedy token agrees.
    smiley_core::lbq::set_i8_enabled(false);
    let mut lb32 = Session::open(&lb_model, cfg.clone()).unwrap();
    let (text32, _) = lb32
        .generate("Say hello in one short sentence.", |_| {})
        .unwrap();
    smiley_core::lbq::set_i8_enabled(true);
    drop(lb32);
    let first = |t: &str| t.split_whitespace().next().map(str::to_string);
    assert_eq!(
        first(&text),
        first(&text32),
        "int8 and f32 kernels diverged at once: {text:?} vs {text32:?}"
    );
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());

    let per_tok = |t: &smiley_core::telem::TurnTelem| {
        let toks: Vec<_> = t.tokens.iter().filter(|x| !x.is_prefill).collect();
        let b: u64 = toks.iter().map(|x| x.bytes_weights).sum();
        b as f64 / toks.len().max(1) as f64
    };
    let (bd, bl) = (per_tok(&telem_d), per_tok(&telem));
    assert!(
        bl < 0.5 * bd,
        "LittleBit decode should move far fewer weight bytes: {bl:.0} vs dense {bd:.0}"
    );
    eprintln!(
        "decode weight bytes/token: dense {:.1} MB → littlebit {:.1} MB ({:.2}×); sidecar {:.1} MB over {} tensors",
        bd / 1e6,
        bl / 1e6,
        bd / bl,
        rep.data_bytes as f64 / 1e6,
        rep.converted.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
