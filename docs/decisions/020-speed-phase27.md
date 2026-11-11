# Decision 020: Read prompts with every core; keep whole passages in the prompt; no change that does not win (Phase 27)

Date 2026-11-11. Status: **adopted.** Research [`speed_sota_2026.md`](../research/speed_sota_2026.md). Measurements:
`benchmarks/engine_phase27/`, `benchmarks/kb_phase27/`. All timings SHARED MACHINE (load 4 to 7 during the
comparisons, stated in each file).

## Decisions

1. **The engine reads prompts with every core and writes tokens with one fewer** (`--prefill-threads`,
   `PREFILL_THREADS` in `scripts/serve_smiley.sh`, default the number of cores). Prompt reading +23 % on both models
   (49.7 → 61.0 and 18.5 → 22.7 tokens/s), decode unchanged, time to the last token −16 % (7 of 8 and 6 of 6
   interleaved pairs). The embedding server, which only reads, uses the same width. A question embeds in 252 ms against 293 ms and a
   passage in 1.08 s against 1.27 s (24 of 24 pairs each).
2. **The knowledge prompt keeps four whole passages.** A packer that counts each source's header in the budget
   and writes a document's title once is built and off (`rag_compact_sources`). With six passages in a block of
   2,400 characters it makes prompts 25 % shorter and answers 20 to 26 % sooner (42 of 50 pairs), and 39 of 50
   answers were right against 42: not significant, but the losses were wrong numbers read from excerpts, which is
   the error a small model must not be pushed towards. It is documented as a faster profile. Searching eight
   passages for 1,800 characters of text was right more often (45 against 42) and no faster, because a third of the
   block was headers; not adopted.
3. **Cache-blocking the prompt products and a smaller prompt chunk are not adopted**: within ±3 % once the benchmark
   was warmed up. The engine's kernel is at what its instruction mix allows on AVX2; the next kernel gain needs VNNI,
   which the reference CPU does not have.
4. **Not built, with the reason:** per-passage cached attention state (TurboRAG, CacheBlend: not the same
   computation, needs a tuned model, 34 MB a passage); a trained pruner (a further model); int8 or truncated vectors
   (the scan is 0.1 % of a search at this scale); PageRank-ordered graph extraction (no graph-quality measure to
   validate it against); a 4-bit embedding model (every base would be embedded again).
5. **No "state of the art" claim.** No public benchmark was run. What is claimed is each measured change against the
   setting it replaced.
6. **Release 0.1.0** is cut from `main` after this merge, as a research preview: the roadmap's gates G0, G2, G3 and
   G6 remain pending for the reasons in `docs/release/RELEASE-CANDIDATE.md`.

## Validation

507 Rust tests pass; `cargo fmt --check` and `cargo clippy -D warnings` (all targets) are clean. Live sweep on the
final build, with the engines reading prompts on four threads, 2026-11-11 21:41 to 22:16: contract passed, security
26/26, Rooms 10/10, end to end 69/69, knowledge 55 passed (1 skipped), knowledge security 33/33, harness 53/53,
acceptance 9/9 (first token after 152 ms; a verified reasoning answer in 198 s against a 240 s target). Gate G1
reads pass.
