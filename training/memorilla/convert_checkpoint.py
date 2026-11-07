#!/usr/bin/env python3
"""Convert a Memorilla checkpoint (`memory.pt` + `config.json`, as `train.py` and the Hugging Face releases write them)
into what `nafs_memorilla::MemoryModule::load` reads: `weights.safetensors` (f32, PyTorch parameter names) and
`config.json`. Pickles are loaded with `weights_only=True`; wrapper prefixes (`module.`, `_orig_mod.`, `memory.`) and
Lightning's `state_dict` nesting are removed, as `memorilla.utils.load_memory_state` does.

Usage: python3 -I training/memorilla/convert_checkpoint.py <checkpoint dir or memory.pt> <output dir>
"""
import json, os, shutil, sys

import torch
from safetensors.torch import save_file

PREFIXES = ("module.", "_orig_mod.")


def load_state(path):
    if os.path.isdir(path):
        path = os.path.join(path, "memory.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    for p in PREFIXES:
        if state and all(k.startswith(p) for k in state):
            state = {k[len(p):]: v for k, v in state.items()}
    if any(k.startswith("memory.") for k in state):
        state = {k[len("memory."):]: v for k, v in state.items() if k.startswith("memory.")}
    return path, state


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    path, state = load_state(src)
    os.makedirs(out, exist_ok=True)
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in state.items()}, os.path.join(out, "weights.safetensors"))
    cfg = os.path.join(os.path.dirname(path), "config.json")
    if os.path.exists(cfg):
        shutil.copy(cfg, os.path.join(out, "config.json"))
    else:  # infer as MemoryConfig.from_state_dict does
        import re
        count = lambda pat: len({m.group(1) for k in state if (m := re.match(pat, k))})
        retrieval = "memory_queries" not in state
        json.dump({
            "embedding_dim": state["query_cond_norm.weight"].shape[0],
            "output_dim": state["memory_projection.weight"].shape[0],
            "num_memories": 16 if retrieval else state["memory_queries"].shape[1],
            "num_self_attn_layers": count(r"encoder\.layers\.(\d+)\."),
            "num_cross_attn_layers": count(r"cross_attn_layers\.(\d+)\."),
            "retrieval_init": retrieval,
        }, open(os.path.join(out, "config.json"), "w"), indent=2)
    n = sum(v.numel() for v in state.values())
    print(f"{len(state)} tensors, {n:,} parameters -> {out}")


if __name__ == "__main__":
    main()
