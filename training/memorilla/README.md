# Memorilla tools (Phase 20)

Python one-offs that sit beside the Rust port in `rust/crates/nafs-memorilla`. They need only `torch` and `safetensors`.

| Script | What it does |
|---|---|
| `make_parity_fixtures.py` | Runs the unmodified reference `third_party/memorilla/memorilla/memory.py` with seeded random weights and writes the fixtures `tests/parity.rs` compares against (default module, `retrieval_init`, document self-attention; a padded row each). Re-run after changing the reference checkout. |
| `convert_checkpoint.py` | Turns a Memorilla checkpoint (`memory.pt` + `config.json`, from `train.py` or Hugging Face) into `weights.safetensors` + `config.json` for `MemoryModule::load`; strips Lightning nesting and `memory.` / `module.` prefixes, loads with `weights_only=True`. |

```bash
python3 -I training/memorilla/make_parity_fixtures.py
python3 -I training/memorilla/convert_checkpoint.py checkpoints/stage3 models/memorilla/stage3
```

The released checkpoints (`memorilla/Memorilla-Qwen3-8B`) project into Qwen3-8B's 4,096-wide input space; Nafs-Aql's
System 1 is 2,048 wide, so they serve as a reference, not for serving. Stage 1–2 pre-training for our encoder/decoder
pair is a GPU job (Phase 6 writes the recipe).
