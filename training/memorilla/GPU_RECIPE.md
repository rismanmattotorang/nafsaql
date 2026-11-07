# Pre-training a Memorilla module for Nafs-Aql's System 1 on a GPU

A module learns to read memory in two general stages before it is fine-tuned on a knowledge base (paper, section 2.6):
Stage 1 restates 1.98 M Wikipedia passages, Stage 2 answers 1.34 M reading-comprehension, summarisation and QA questions.
On this CPU that is out of reach (measured below), so it runs on a GPU with the reference PyTorch code, and the result is
fine-tuned on each knowledge base here (`Fine-tune` on the Knowledge page, or `memorilla-train --init`).

## Cost

| | Per example | Stages 1 + 2 (3.32 M examples) |
|---|---|---|
| This CPU, Qwen3-1.7B decoder in f32, Candle (measured 2026-11-07, shared machine) | 36–94 s | about 4–10 years |
| 4 x H100, Qwen3-8B (paper: 6 h + 4 h) | | 40 GPU-hours |
| 4 x H100, Qwen3-1.7B (scaled by decoder size, 8B -> 1.7B; an estimate) | | about 8–10 GPU-hours |

The scaling assumes the step cost is dominated by the frozen decoder (forward and input-gradient backward, about 4 P T
FLOPs per example), as it is: the module is 31.5 M parameters against 1.7 B.

## Steps

```bash
# On a Linux machine with an NVIDIA GPU (the reference needs CUDA; see third_party/memorilla/README.md).
git clone https://github.com/snap-stanford/memorilla.git && cd memorilla
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e . && uv pip install --no-build-isolation flash-attn==2.7.4.post1

# Stage 1 and 2 with our decoder and encoder (the released data's embeddings are for Qwen3-Embedding-4B, so the
# 0.6B encoder's embeddings are computed on first use: about 3.3 M passages plus questions).
scripts/train.sh configs/stage1_enwiki.yaml   --decoder Qwen/Qwen3-1.7B --encoder Qwen/Qwen3-Embedding-0.6B --output_dir runs/stage1-q17
scripts/train.sh configs/stage2_mixture.yaml  --decoder Qwen/Qwen3-1.7B --encoder Qwen/Qwen3-Embedding-0.6B \
    --init_memory runs/stage1-q17/epoch-00 --output_dir runs/stage2-q17

# Optional: fine-tune on a Nafs-Aql knowledge base exported from this machine.
python3 -I training/memorilla/export_dataset.py <run>/data my_kb_data --config my_kb      # on the Nafs-Aql machine
python train.py --config configs/single_task.yaml --data_repo my_kb_data --datasets my_kb \
    --decoder Qwen/Qwen3-1.7B --encoder Qwen/Qwen3-Embedding-0.6B --init_memory runs/stage2-q17/epoch-00 --output_dir runs/my_kb
```

Back on the Nafs-Aql machine:

```bash
python3 -I training/memorilla/convert_checkpoint.py runs/stage2-q17/epoch-00 models/memorilla/stage2-q17
# Then either set [knowledge] memory_checkpoint = "models/memorilla/stage2-q17" (every knowledge base),
# or fine-tune per base from it: memorilla-train --init models/memorilla/stage2-q17 --data <run>/data --out <dir>
```

## Two things to keep identical between training and serving

- **The prompt.** Nafs-Aql serves memory with Memorilla's system prompt ("You are a helpful assistant.") followed by
  `\nMemory` + 16 placeholders + `Memory`, Qwen3's chat template with thinking off. The reference trainer builds the same
  layout with its own `<|memory|>` token; the placeholder token differs (`<|fim_pad|>` here), which does not matter
  because the vectors replace the embeddings at those positions, but the surrounding text must match.
- **The question embedding.** No instruction prefix (Memorilla's `embed_texts`); Nafs-Aql embeds memory questions the
  same way, and retrieval questions with an instruction.

Serving runs the decoder quantised (Q4_K_M on Smiley Core) while training uses f32 or bf16 weights; the memory tokens
transfer, but quality should be checked on the quantised model (Phase 7).
