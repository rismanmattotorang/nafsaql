#!/usr/bin/env python3
"""Tazkiya training: Unsloth LoRA on a GPU host (TR-004, TR-005, TR-006).

    python3 training/train_unsloth.py --data data/tazkiya/train.jsonl --base Qwen/Qwen3-1.7B            # GPU host
    python3 training/train_unsloth.py --data data/tazkiya/train.jsonl --dry-run                         # any host

Configuration fixed by TR-004: LoRA rank 16, alpha 16, dropout 0, learning rate 2e-4, 1-3 epochs, effective
batch 16, all attention and MLP projections; QLoRA (4-bit) for Qwen3 checkpoints, BF16 LoRA for Qwen3.5.
Outputs (TR-006): the adapter with a SHA-256 content hash, merged GGUF in q4_k_m and q8_0, and a manifest that
links the data snapshot, the configuration, library versions and every hash. The CPU reference machine must
never train (TR-005): without a CUDA GPU this script stops before importing Unsloth.
"""
import argparse, hashlib, json, os, platform, sys, time

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def sha256_path(path):
    h = hashlib.sha256()
    files = [path] if os.path.isfile(path) else sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)
    for fp in files:
        h.update(os.path.relpath(fp, path if os.path.isdir(path) else os.path.dirname(fp)).encode())
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

def load_examples(path, with_rationale):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            for k in ("id", "domain", "prompt", "answer", "assurance"):
                if k not in ex:
                    raise SystemExit(f"{path}:{i}: missing field '{k}' (TR-001 schema)")
            if ex["assurance"] not in ("verified", "consensus_reviewed", "abstention", "replay"):
                raise SystemExit(f"{path}:{i}: assurance '{ex['assurance']}' may not be trained on (TR-002)")
            answer = ex["answer"]
            if with_rationale and ex.get("rationale"):
                answer = f"{ex['rationale'].strip()}\n\n{answer}"
            rows.append({"messages": ex["prompt"] + [{"role": "assistant", "content": answer}], "domain": ex["domain"]})
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--val")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="Qwen3 checkpoints use QLoRA; Qwen3.5 checkpoints use BF16 LoRA")
    ap.add_argument("--out", default="runs/tazkiya"); ap.add_argument("--epochs", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--max-seq", type=int, default=2048); ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--with-rationale", action="store_true", help="prepend the <=150-token rationale to the answer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    qlora = "qwen3.5" not in a.base.lower()
    cfg = {"base": a.base, "method": "QLoRA 4-bit" if qlora else "BF16 LoRA", "r": 16, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": TARGETS,
           "learning_rate": 2e-4, "epochs": a.epochs, "per_device_batch": 4, "grad_accum": 4, "effective_batch": 16, "max_seq_length": a.max_seq,
           "seed": a.seed, "with_rationale": a.with_rationale}
    rows = load_examples(a.data, a.with_rationale)
    data_manifest = os.path.join(os.path.dirname(os.path.abspath(a.data)), "manifest.json")
    snap = {"train_sha256": sha256_path(a.data), "data_manifest": json.load(open(data_manifest)) if os.path.exists(data_manifest) else None}
    if snap["data_manifest"] and not snap["data_manifest"].get("ready", False):
        print("warning: the data manifest reports problems:", snap["data_manifest"].get("problems"), file=sys.stderr)
    print(json.dumps({"config": cfg, "examples": len(rows)}, indent=2))
    if a.dry_run:
        print("dry run: data and configuration are valid; no training performed")
        return 0
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError
    except ImportError:
        raise SystemExit("TR-005: no CUDA GPU found. Train on a GPU host (16 GB VRAM or more); see training/RUNBOOK.md.")
    from unsloth import FastLanguageModel  # noqa: E402  (imported only on a GPU host)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import transformers, unsloth
    model, tok = FastLanguageModel.from_pretrained(model_name=a.base, max_seq_length=a.max_seq, load_in_4bit=qlora, dtype=None if qlora else torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, target_modules=TARGETS, bias="none",
                                             use_gradient_checkpointing="unsloth", random_state=a.seed)
    def fmt(batch):
        return {"text": [tok.apply_chat_template(m, tokenize=False, enable_thinking=False) for m in batch["messages"]]}
    ds = Dataset.from_list(rows).map(fmt, batched=True)
    run = os.path.join(a.out, time.strftime("%Y%m%d-%H%M%S")); os.makedirs(run, exist_ok=True)
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(output_dir=run, dataset_text_field="text", per_device_train_batch_size=4,
                         gradient_accumulation_steps=4, num_train_epochs=a.epochs, learning_rate=2e-4, lr_scheduler_type="linear", warmup_ratio=0.03,
                         logging_steps=10, save_steps=100, seed=a.seed, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                         max_seq_length=a.max_seq, report_to="none"))
    stats = trainer.train()
    adapter = os.path.join(run, "adapter"); model.save_pretrained(adapter); tok.save_pretrained(adapter)
    ggufs = {}
    for q in ("q4_k_m", "q8_0"):
        out = os.path.join(run, f"gguf-{q}")
        model.save_pretrained_gguf(out, tok, quantization_method=q)
        files = [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".gguf")]
        ggufs[q] = {f: sha256_path(f) for f in files}
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg, "data": snap, "examples": len(rows),
                "train_loss": getattr(stats, "training_loss", None), "adapter_hash": sha256_path(adapter), "gguf": ggufs,
                "versions": {"unsloth": getattr(unsloth, "__version__", "?"), "transformers": transformers.__version__, "torch": torch.__version__, "python": platform.python_version()},
                "next": "copy the q4_k_m GGUF to models/ on the CPU host, run regression_500 with the candidate, then: python3 training/tazkiya.py promote ..."}
    json.dump(manifest, open(os.path.join(run, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
