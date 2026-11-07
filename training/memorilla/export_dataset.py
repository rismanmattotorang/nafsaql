#!/usr/bin/env python3
"""Export a Nafs-Aql knowledge-base dataset (`nafs_kb::dataset`: dataset.json, documents.safetensors,
questions.safetensors) to Memorilla's data-repository layout, so the reference PyTorch trainer (`train.py`) can train or
fine-tune on it with a GPU:

  <out>/data/<config>/{train,validation}-00000-of-00001.parquet    collection_id, question, answer, documents
  <out>/embeddings/<encoder slug>/<config>/<split>/                index, documents-00000, questions (float16)

Every row reads the whole knowledge base (collection_id 0), as Nafs-Aql's `memory` mode does. The passage texts are not
in the Nafs-Aql dataset (only their vectors); `documents` lists the passage ids, which Memorilla only uses to embed, and
the precomputed embeddings make that unnecessary.

Usage: python3 -I training/memorilla/export_dataset.py <dataset dir> <out dir> [--config my_kb]
Then:  python train.py --config configs/single_task.yaml --data_repo <out dir> --datasets my_kb \\
         --decoder Qwen/Qwen3-1.7B --encoder Qwen/Qwen3-Embedding-0.6B --init_memory "" --retrieval_init true
"""
import argparse, json, os

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--config", default="nafs_kb")
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.dataset, "dataset.json")))
    docs = load_file(os.path.join(a.dataset, "documents.safetensors"))["embeddings"].to(torch.float16)
    qs = load_file(os.path.join(a.dataset, "questions.safetensors"))["embeddings"].to(torch.float16)
    slug = (info.get("encoder") or "qwen3-embedding-0.6b").rstrip("/").split("/")[-1].lower()
    for split in ("train", "validation"):
        idx = [i for i, r in enumerate(info["rows"]) if r["split"] == split]
        if not idx:
            continue
        rows = info["rows"]
        table = pa.table({
            "collection_id": pa.array([0] * len(idx), pa.int64()),
            "question": [rows[i]["question"] for i in idx],
            "answer": [rows[i]["answer"] for i in idx],
            "documents": [info["documents"]] * len(idx),
        })
        d = os.path.join(a.out, "data", a.config)
        os.makedirs(d, exist_ok=True)
        pq.write_table(table, os.path.join(d, f"{split}-00000-of-00001.parquet"))
        e = os.path.join(a.out, "embeddings", slug, a.config, split)
        os.makedirs(e, exist_ok=True)
        save_file({"collection_ids": torch.tensor([0], dtype=torch.int64), "shard": torch.tensor([0], dtype=torch.int64),
                   "start": torch.tensor([0], dtype=torch.int64), "length": torch.tensor([docs.shape[0]], dtype=torch.int64)},
                  os.path.join(e, "index.safetensors"))
        save_file({"embeddings": docs.contiguous()}, os.path.join(e, "documents-00000.safetensors"))
        save_file({"embeddings": qs[idx].contiguous()}, os.path.join(e, "questions.safetensors"))
        print(f"{split}: {len(idx)} rows, {docs.shape[0]} passages -> {d}")


if __name__ == "__main__":
    main()
