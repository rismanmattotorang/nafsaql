# models

Local GGUF weights (git-ignored). Fetch them with:

```bash
scripts/fetch_models.sh            # Profile A: Qwen3-1.7B and Qwen3-4B-Thinking-2507, Q4_K_M, about 3.8 GB
scripts/fetch_models.sh all        # plus Qwen3-4B (Profile B), Qwen3-0.6B Q8_0 and Qwen3-Embedding-0.6B
```

`config/llama_server_profile_a.sh` reads from this directory. Record revisions and SHA-256 hashes in a model manifest after downloading (requirements MR-001, NFR-012).
