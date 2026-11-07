//! Parity with the PyTorch reference (FR-K40): fixtures written by `training/memorilla/make_parity_fixtures.py` from the
//! unmodified `memory.py` with seeded random weights (default, retrieval_init, document self-attention; a padded row in
//! each). Every output element must match within 1e-4.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use nafs_memorilla::memory::{MemoryConfig, MemoryModule};

fn case(name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let dev = Device::Cpu;
    let m = MemoryModule::load(&dir, &dev).unwrap();
    let io = candle_core::safetensors::load(dir.join("io.safetensors"), &dev).unwrap();
    let docs = &io["doc_embeds"];
    let mask = io["doc_padding_mask"].to_dtype(DType::F32).unwrap();
    let q = &io["question_embeds"];
    let want = &io["memory"];
    let got = m.forward(docs, Some(&mask), q).unwrap();
    assert_eq!(got.dims(), want.dims(), "{name}");
    let diff = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    eprintln!("{name}: max |candle - pytorch| = {diff:e}");
    assert!(diff < 1e-4, "{name}: max abs difference {diff}");

    // Preparing the documents once and answering from the preparation gives the same memory.
    let prep = m.prepare(docs, Some(&mask)).unwrap();
    let again = m.forward_prepared(&prep, q).unwrap();
    let d2 = (again - want)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(d2 < 1e-4, "{name}: prepared path differs by {d2}");

    // The parameter formula agrees with the stored tensors.
    let weights = candle_core::safetensors::load(dir.join("weights.safetensors"), &dev).unwrap();
    let stored: usize = weights.values().map(Tensor::elem_count).sum();
    let cfg: MemoryConfig =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg.parameters(), stored, "{name}");
}

#[test]
fn default_module_matches_pytorch() {
    case("default");
}

#[test]
fn retrieval_init_matches_pytorch() {
    case("retrieval_init");
}

#[test]
fn document_self_attention_matches_pytorch() {
    case("self_attn");
}

#[test]
fn padding_never_leaks_into_the_memory() {
    // Changing the padded documents' values must not change the output of the padded row.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/default");
    let dev = Device::Cpu;
    let m = MemoryModule::load(&dir, &dev).unwrap();
    let io = candle_core::safetensors::load(dir.join("io.safetensors"), &dev).unwrap();
    let mask = io["doc_padding_mask"].to_dtype(DType::F32).unwrap();
    let a = m
        .forward(&io["doc_embeds"], Some(&mask), &io["question_embeds"])
        .unwrap();
    let noise = (Tensor::randn(0f32, 5.0, io["doc_embeds"].dims(), &dev).unwrap()
        * mask
            .unsqueeze(2)
            .unwrap()
            .broadcast_as(io["doc_embeds"].dims())
            .unwrap())
    .unwrap();
    let docs = (&io["doc_embeds"] + noise).unwrap();
    let b = m
        .forward(&docs, Some(&mask), &io["question_embeds"])
        .unwrap();
    let diff = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-5, "padding changed the memory by {diff}");
}
