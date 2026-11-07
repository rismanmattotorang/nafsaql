#!/usr/bin/env python3
"""Parity fixtures for the Candle memory module (Phase 20, Phase 4).

Runs the reference `MemoryModule` (third_party/memorilla/memorilla/memory.py, unmodified) with seeded random weights in
evaluation mode on random inputs, and writes for each case `weights.safetensors` (the state dict, PyTorch names),
`config.json` (constructor arguments) and `io.safetensors` (doc_embeds, doc_padding_mask, question_embeds, memory).
`rust/crates/nafs-memorilla/tests/parity.rs` loads them and compares.

The module is imported from the reference file with its small helper module stubbed, so only torch and safetensors
are needed (not transformers or vLLM).

Usage: python3 -I training/memorilla/make_parity_fixtures.py [--out rust/crates/nafs-memorilla/tests/fixtures]
"""
import argparse, importlib.util, json, os, sys, types

import torch
from safetensors.torch import save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "third_party", "memorilla", "memorilla", "memory.py")

# Each case: constructor arguments, batch size, documents per row (the second row is padded), and a seed.
CASES = {
    "default": dict(cfg=dict(embedding_dim=64, output_dim=48, num_memories=16, num_heads=8, num_self_attn_layers=0,
                             num_cross_attn_layers=2, dropout=0.1, retrieval_init=False), docs=[23, 9], seed=1),
    "retrieval_init": dict(cfg=dict(embedding_dim=64, output_dim=40, num_memories=8, num_heads=4, num_self_attn_layers=0,
                                    num_cross_attn_layers=1, dropout=0.1, retrieval_init=True), docs=[12, 5], seed=2),
    "self_attn": dict(cfg=dict(embedding_dim=32, output_dim=32, num_memories=4, num_heads=4, num_self_attn_layers=1,
                               num_cross_attn_layers=2, dropout=0.1, retrieval_init=False), docs=[7, 7], seed=3),
}


def load_reference():
    stub = types.ModuleType("memorilla.utils")
    stub.CHECKPOINT_FILE, stub.CONFIG_FILE = "memory.pt", "config.json"
    stub.checkpoint_file = lambda p: p
    stub.load_memory_state = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    pkg = types.ModuleType("memorilla")
    pkg.__path__ = []
    sys.modules["memorilla"], sys.modules["memorilla.utils"] = pkg, stub
    spec = importlib.util.spec_from_file_location("memorilla.memory", REF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memorilla.memory"] = mod
    spec.loader.exec_module(mod)
    return mod.MemoryModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "rust", "crates", "nafs-memorilla", "tests", "fixtures"))
    a = ap.parse_args()
    MemoryModule = load_reference()
    for name, case in CASES.items():
        torch.manual_seed(case["seed"])
        cfg = case["cfg"]
        m = MemoryModule(**cfg)
        # Non-trivial LayerNorm parameters, so a wrong affine order cannot pass.
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "norm" in n:
                    p.add_(0.1 * torch.randn_like(p))
        m.eval()
        b, n_max, e = len(case["docs"]), max(case["docs"]), cfg["embedding_dim"]
        docs = torch.randn(b, n_max, e)
        mask = torch.zeros(b, n_max, dtype=torch.bool)
        for i, n in enumerate(case["docs"]):
            mask[i, n:] = True
            docs[i, n:] = 0.0
        question = torch.randn(b, e)
        with torch.no_grad():
            out = m(docs, mask, question)
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(d, "weights.safetensors"))
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=2)
        save_file({"doc_embeds": docs, "doc_padding_mask": mask.to(torch.uint8), "question_embeds": question, "memory": out},
                  os.path.join(d, "io.safetensors"))
        params = sum(p.numel() for p in m.parameters())
        print(f"{name}: {params} parameters, output {tuple(out.shape)}, |out| mean {out.abs().mean():.4f} -> {d}")


if __name__ == "__main__":
    main()
