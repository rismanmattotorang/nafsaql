# Tazkiya runbook (Phase 3)

Distillation turns verified System 2 answers into training data for System 1. Training needs a CUDA GPU with at least 16 GB of memory (TR-005), for example a Colab T4 or a rented L4. The CPU reference machine only builds data, evaluates and serves.

## 1. Build the data on the CPU host

```bash
python3 training/tazkiya.py build --out data/tazkiya/ --replay data/tazkiya/replay.jsonl
```

The manifest (`data/tazkiya/manifest.json`) lists counts, domain shares, exclusions, file hashes and any problems. `ready` is true only when all of these hold:

- at least 5,000 verified examples (`--min-examples`);
- no domain above 30%;
- 5% abstention examples;
- 20% replay examples.

Replay data is general instruction data that you supply as JSONL with `instruction` and `output` fields. Check its licence before training.

To add synthetic examples (TR-003), run System 2 on GSM8K and MBPP *training* splits and keep only the answers an exact verifier passes:

```bash
python3 training/tazkiya.py synthesize --n 200
```

## 2. Train on the GPU host

```bash
pip install "unsloth==2026.10.*" "transformers>=5" trl datasets
python3 training/train_unsloth.py --data data/tazkiya/train.jsonl --base Qwen/Qwen3-1.7B --epochs 2
```

The configuration is fixed by TR-004:

- LoRA r 16, alpha 16, dropout 0, on all attention and MLP projections;
- learning rate 2e-4 and effective batch 16;
- QLoRA for Qwen3, BF16 LoRA for Qwen3.5.

The run directory contains:

- the adapter;
- `gguf-q4_k_m/` and `gguf-q8_0/`;
- `manifest.json` with every hash and library version (TR-006).

## 3. Gate the candidate on the CPU host

1. Copy the q4_k_m GGUF to `models/`.
2. Build the fixed regression suite (once): `python3 training/tazkiya.py regression-suite`.
3. Run `regression_500` with the `fast_only` system twice, once on the current System 1 and once on the candidate. Start the candidate with `S1_MODEL=models/candidate.gguf scripts/serve_llama.sh start`.
4. Promote:

```bash
python3 training/tazkiya.py promote --baseline-run <RUN> --candidate-run <RUN> \
    --gguf models/candidate.gguf --adapter-hash <from manifest> --sign-off "Your Name"
```

TR-007 promotes only when all of the following hold:

- the lower 95% bound of the paired gain is above 0, with a point estimate of at least 2 points;
- no domain's lower bound is below −1 point;
- the coupling test does not fail on the candidate;
- a person has signed off.

On success, `config/models.registry.json` changes; this is a configuration change, not a code change. Restart serving:

```bash
scripts/serve_llama.sh stop && scripts/serve_llama.sh start
```

## 4. Roll back in one command (OR-005)

```bash
python3 training/tazkiya.py rollback && scripts/serve_llama.sh stop && scripts/serve_llama.sh start
```

The previous artefact is never deleted. `python3 training/tazkiya.py status` prints the registry.
