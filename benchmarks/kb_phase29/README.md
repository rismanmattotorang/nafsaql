# Phase 29: one context filesystem (OpenViking ideas)

Reference machine (Intel i3-8100, 4 cores), shared with other workloads. Prompt: `docs/prompts/CONTEXT_FS_PROMPT.md`.
Research: `docs/research/openviking_gaps.md`. Specification: `docs/spec/context_fs_spec.md`. Decision 022.

## Files

| File | What | Tool |
|---|---|---|
| `context_eval.json` | scoped search (whole base, the rule planner's document scope, the answering document as a ceiling) and the summaries-first assembler against four whole passages, on the Phase 28 base `pir9ftsm6eja267j4fg0` (723 passages), single-passage questions | `rust/crates/nafs-kb/examples/context_eval.rs` |
| `extract_written.txt` | rule extraction of memories on 40 messages written together with the rules | `rust/crates/nafs-context/examples/extract_eval.rs` |
| `extract_heldout_after.txt` | the same on 20 messages written to differ from the first rules; before the rules were widened: 0 of 14 found, 1 false | as above |
| `extract_fresh.txt` | the same on 20 messages written after the rules were final | as above |

## Results

**Scope (hybrid search, MRR).** Quoted (56): whole base 0.870; planner scope fused with the whole base 0.841 (fired on
6 questions, right document 4); the answering document as scope 0.903. Reworded (142): 0.194; 0.192 (fired 5, right 1);
0.267. A correct scope helps, but the rule that guesses it from a document's distinctive terms does not find it, so
`search` scopes only to a document whose exact title the question contains, and scoping is otherwise an explicit
choice (`find` with a scope, or a folder on the Files page).

**Assembly (answer phrase in the prompt; characters of the passage block).**

| Set | Four whole passages | Assembler 2,400 | Assembler 3,600 | Assembler 4,800 |
|---|---|---|---|---|
| Quoted (56) | 0.964 / 2,759 | 0.857 / 2,333 | 0.946 / 3,540 | 0.964 / 4,674 |
| Reworded (142) | 0.225 / 2,828 | 0.162 / 2,334 | 0.239 / 3,528 | 0.254 / 4,684 |

At the same size the assembler loses answers; it gains only with 25 to 70 % more text, which costs prefill time on four
cores (Phase 27). The chat prompt keeps whole passages; the assembler is offered through `/v1/fs/assemble`.

**Memory extraction by rule.** Written with the rules: 25 of 25, no false memory. Held out: 0 of 14 before the rules
were widened, 14 of 14 after (the set was then no longer held out). Written after the rules were final: 13 of 14
found, 1 false (precision and recall 0.929).

**End to end (live).** A conversation committed as a Markdown file with three memory files drawn from it (profile,
preference, entity), each in the change log; undo restored a file; a stated preference ("answer in Indonesian") applied
to the next answer; `find` scoped to one document answered in 0 ms once the question was embedded; the rule planner put
a skill, a memory file and passages in one result list.
