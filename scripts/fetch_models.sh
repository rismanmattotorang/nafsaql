#!/usr/bin/env bash
# Download model weights into models/.
#   scripts/fetch_models.sh            GGUF for Profile A (llama-server, native): Qwen3-1.7B + Qwen3-4B-Thinking-2507, Q4_K_M (~3.8 GB)
#   scripts/fetch_models.sh vllm       Hugging Face safetensors for the vLLM profile: Qwen3-1.7B + Qwen3-4B-Thinking-2507 (~11.5 GB)
#   scripts/fetch_models.sh draft      Qwen3-0.6B Q8_0 only: speculative-decoding draft for System 2 (~0.64 GB)
#   scripts/fetch_models.sh all        both, plus Qwen3-4B (Profile B), Qwen3-0.6B Q8_0, Qwen3-Embedding-0.6B
# Requires python3 with huggingface_hub (and hf_xet for fast transfers). Record revisions/hashes afterwards (MR-001, NFR-012).
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH=${1:-profile-a}
python3 -I - "$WHICH" <<'PY'
import sys, time
from huggingface_hub import hf_hub_download, snapshot_download
which = sys.argv[1]
gguf_a = [("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
          ("unsloth/Qwen3-4B-Thinking-2507-GGUF", "Qwen3-4B-Thinking-2507-Q4_K_M.gguf")]
gguf_extra = [("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
              ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
              ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf")]
hf_vllm = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Thinking-2507"]
patterns = ["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"]
todo_gguf = gguf_a + (gguf_extra if which == "all" else []) if which in ("profile-a", "all") else []
if which == "draft":
    todo_gguf = [("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")]
todo_hf = hf_vllm if which in ("vllm", "all") else []
for repo, name in todo_gguf:
    t = time.time(); p = hf_hub_download(repo_id=repo, filename=name, local_dir="models")
    print(f"OK gguf {repo} {name} {p} {time.time()-t:.0f}s", flush=True)
for repo in todo_hf:
    t = time.time(); p = snapshot_download(repo_id=repo, local_dir=f"models/hf/{repo.split('/')[1]}", allow_patterns=patterns)
    print(f"OK hf {repo} {p} {time.time()-t:.0f}s", flush=True)
print("DONE", flush=True)
PY
