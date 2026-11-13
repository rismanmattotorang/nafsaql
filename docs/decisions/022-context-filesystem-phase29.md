# Decision 022: One context filesystem for documents, memory, skills and sessions (Phase 29)

Date 2026-11-13. Status: **adopted.** Research [`openviking_gaps.md`](../research/openviking_gaps.md). Specification
[`context_fs_spec.md`](../spec/context_fs_spec.md). Prompt [`CONTEXT_FS_PROMPT.md`](../prompts/CONTEXT_FS_PROMPT.md).
Measurements: `benchmarks/kb_phase29/`.

## Decisions

1. **One namespace, no second store.** `nafs://kb/...`, `nafs://memory/...`, `nafs://skills/...` and
   `nafs://sessions/...` name items where they already live (SurrealDB for documents, SQLite for the rest).
   `ls`, `tree`, `read`, `stat`, `glob`, `grep`, `write` and `rm` work across them (`/v1/fs/*`), and the app has a
   Files page. The design follows OpenViking; its core is AGPL-3.0, so it is re-implemented in Rust and nothing is
   copied (crate `nafs-context`, Apache-2.0).
2. **Layers without a model.** A document's summary (L0) and overview (L1) are made from what the reader and the
   index hold (title, first sentence, heading tree, captions, distinctive terms), and a base's from its documents'.
   OpenViking spends one model call per file and per directory; on four cores that would be hours for a base of
   this size. A model-written abstract stays an option for later.
3. **Scoped search is a tool, not a guess.** Search can be limited to a base, a document, a memory kind or the
   skills, masked inside the in-process index. The answering document as scope would lift reworded questions
   (MRR 0.194 to 0.267), but a rule that guesses the document from its distinctive terms picked the right one 5 times
   in 11 and lowered MRR, so the planned `search` scopes only to a document named by its exact title.
4. **Whole passages stay in the chat prompt.** A summaries-first assembler with quotas (OpenViking's context
   assembler) loses answers at the same prompt size (quoted 0.857 against 0.964) and gains only with 25 to 70 % more
   text. It is available through `/v1/fs/assemble`.
5. **Memories are files.** Typed Markdown files keyed by what they are about (profile, preferences, people and
   things, events, notes), merged by kind, every write logged with its reason and undoable. Saving a chat writes its
   transcript and extracts memories by rule (13 of 14 found and 1 false on messages written after the rules were
   final). The profile and preferences go into every personal conversation; matching entities, events and notes
   when the recall gate opens. A "remember that …" message also becomes a memory file.
6. **Skills** are `SKILL.md` files under `nafs://skills`, found by their description.
7. **Compile** renders a base as a wiki, a term graph with evidence URIs, or a report, with no model call.
8. **Not ported:** the recursive directory beam (removed upstream in commit `69e71e4ff`), model summaries at ingest,
   agent-run compilation, a model planner every turn, multi-tenant accounts, LoCoMo's lenient judge as a target.

## Validation

533 Rust tests pass; `cargo fmt --check` and `cargo clippy -D warnings` (all targets, local and the newer nightly
clippy) are clean. Live sweep on the final build, 2026-11-13 09:00 to 09:41 (load 7 to 27): contract passed, security
26/26, Rooms 10/10, end to end 74/74, knowledge 55 passed (1 skipped), knowledge security 33/33, harness 53/53,
acceptance 9/9 (first token after 151 ms; a verified reasoning answer in 182 s against a 240 s target).
