#!/usr/bin/env python3
"""Write config/models.manifest.json: repository, pinned revision (commit) and SHA-256 of every local model file (MR-001, NFR-012).
Usage: python3 scripts/model_manifest.py"""
import hashlib, json, os, glob, datetime
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
M = os.path.join(ROOT, "models")
def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(buf)): h.update(b)
    return h.hexdigest()
def meta(dirpath, fname):
    m = os.path.join(dirpath, ".cache", "huggingface", "download", fname + ".metadata")
    if os.path.exists(m):
        lines = open(m).read().split("\n"); return {"revision": lines[0].strip(), "etag": lines[1].strip() if len(lines) > 1 else None}
    return {}
REPO = {"Qwen3-1.7B-Q4_K_M.gguf": "unsloth/Qwen3-1.7B-GGUF", "Qwen3-4B-Thinking-2507-Q4_K_M.gguf": "unsloth/Qwen3-4B-Thinking-2507-GGUF",
        "Qwen3-4B-Q4_K_M.gguf": "Qwen/Qwen3-4B-GGUF", "Qwen3-0.6B-Q8_0.gguf": "Qwen/Qwen3-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf": "Qwen/Qwen3-Embedding-0.6B-GGUF"}
out = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gguf": [], "hf": []}
for p in sorted(glob.glob(os.path.join(M, "*.gguf"))):
    n = os.path.basename(p); out["gguf"].append({"file": n, "repo": REPO.get(n), "bytes": os.path.getsize(p), "sha256": sha256(p), **meta(M, n)})
for d in sorted(glob.glob(os.path.join(M, "hf", "*"))):
    files = []
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors")) + glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.jinja"))):
        n = os.path.basename(p); files.append({"file": n, "bytes": os.path.getsize(p), "sha256": sha256(p), **meta(d, n)})
    revs = {f.get("revision") for f in files if f.get("revision")}
    out["hf"].append({"repo": "Qwen/" + os.path.basename(d), "revision": revs.pop() if len(revs) == 1 else sorted(revs), "files": files})
json.dump(out, open(os.path.join(ROOT, "config", "models.manifest.json"), "w"), indent=2)
print(json.dumps({"gguf": [(g["file"], g.get("revision", "?")[:10], g["sha256"][:12]) for g in out["gguf"]], "hf": [(h["repo"], str(h["revision"])[:10], len(h["files"])) for h in out["hf"]]}, indent=1))
