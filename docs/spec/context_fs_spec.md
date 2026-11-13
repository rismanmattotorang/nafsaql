# Context filesystem specification (Phase 29)

One namespace for documents, memories, skills and sessions, with layered summaries, scoped search, a budgeted
context assembler, sessions committed as files and compilation into a wiki, a graph or a report. Research:
`docs/research/openviking_gaps.md`. Decision 022.

## 1. URIs

`nafs://{space}/{path}[#p{ord}]`, lower-case space, path segments separated by `/`, no `..`, no empty segment,
percent-encoding for anything outside `[A-Za-z0-9._~-]` in a segment.

| URI | What |
|---|---|
| `nafs://` | the root: the four spaces |
| `nafs://kb` | knowledge bases |
| `nafs://kb/{base}` | one base (a directory: its documents) |
| `nafs://kb/{base}/{doc}` | one document (a directory: its passages); its content is the document's Markdown |
| `nafs://kb/{base}/{doc}#p{ord}` | one passage |
| `nafs://memory` | memory kinds: `profile`, `preferences`, `entities`, `events`, `notes` |
| `nafs://memory/{kind}/{key}` | one memory file (Markdown with front matter) |
| `nafs://skills/{name}` | one skill (`SKILL.md`) |
| `nafs://sessions/{conversation}` | one committed session: its transcript and what was extracted from it |

Ids are the stores' own (SurrealDB record ids, SQLite UUIDs); a memory key is a slug of its identity (§5).

## 2. Operations

| Route | Does |
|---|---|
| `GET /v1/fs/ls?uri=&layers=0` | children of a directory; `layers=1` adds each child's L0 |
| `GET /v1/fs/tree?uri=&depth=2` | the subtree to a depth, with L0 lines |
| `GET /v1/fs/read?uri=&level=0\|1\|2` | L0, L1 or the content (L2) |
| `GET /v1/fs/stat?uri=` | kind, size, times, counts, layer freshness |
| `GET /v1/fs/glob?pattern=` | URIs matching a pattern (`*` within a segment, `**` across) |
| `POST /v1/fs/grep` `{pattern, scope}` | lines matching a literal or regex within a scope, with URIs |
| `POST /v1/fs/find` `{query, scope?, k?, layers?}` | the query run directly within the scope |
| `POST /v1/fs/search` `{query, conversation?, k?}` | a planned search: scopes and queries chosen from the question and the session (§4) |
| `PUT /v1/fs/write?uri=` | write a memory file or a skill (other spaces are read-only here) |
| `DELETE /v1/fs/rm?uri=` | remove a memory file, a skill or a committed session (a document is removed on the Knowledge page) |
| `POST /v1/sessions/{conversation}/commit` | archive the conversation and extract memories (§6) |
| `GET /v1/memory/diff` / `POST /v1/memory/diff/{id}/undo` | the log of memory writes; undo one |
| `POST /v1/compile` `{from, to: wiki\|graph\|report}` | render a scope into Markdown files (§8) |

All routes need the owner's token and pass the same-origin guard. Writes are bounded: 64 KB per memory file, 256 KB per
skill, names `[a-z0-9-]{1,64}`.

## 3. Layers

Every document and base has three levels: **L0** an abstract of at most 300 characters, **L1** an overview of at most
2,000 characters, **L2** the content. They are made **without a model** from the structure the reader found:

- **Document L1**: the title; the heading tree (at most 24 headings, indented by level); the first paragraph of the
  body (at most 400 characters); table and figure captions (at most 6); the 12 most distinctive terms (term frequency
  in the document times inverse document frequency in its base, stems shown in their commonest surface form).
- **Document L0**: the title, then the first sentence of the body that is not a byline, then up to 6 distinctive terms.
- **Base L1**: the base's name and description, then one line per document with its L0, in title order; above 32
  documents, 32 evenly spaced documents (`i·(n−1)/(k−1)`) and a line saying how many are not shown.
- **Base L0**: the name, document count and the base's 8 most distinctive terms.
- **Memory file**: L0 is its title and first line; L1 is the file. **Skill**: L0 is its name and description.

Freshness: a document's layers are made when it is read (no extra cost); a base's layers are made again only when a
document's L0 changed and, for a base of more than 32 documents, when a tenth of its documents changed since. Layers
are stored with a hash of their inputs. An optional model-written L0 (`[context] model_abstracts`, off) runs in the
background on the quick model, one call per document.

## 4. Search

**Scope.** A scope is a URI prefix. `nafs://kb/{base}` searches one base; `nafs://kb/{base}/{doc}` one document;
`nafs://memory/{kind}` one memory kind; `nafs://` everything. Inside the in-process index a document scope masks the
passages before ranking, so words and meaning both rank only inside it.

**find** runs the query as given within the scope and returns hits with their URI, level and score; document L0/L1
are searched beside passages (a document whose overview matches is returned as `level: 1`).

**search** plans first, by rule: (1) a question that contains a document's exact title is scoped to it (a rule from
distinctive terms was measured and lowered MRR; §9), (2) a question that leans on the previous turn is joined to it (the
Phase 26 follow-up rule), (3) a personal question (first-person words, "my", "I prefer") adds the memory space, (4) a
question that asks how to do a task adds the skills space. It then runs `find` per scope and merges by reciprocal
rank. A model planner is not called by default.

## 5. Memories

A memory file is Markdown with front matter:

```
---
kind: preferences
key: answer-language
title: Answer language
updated: 2026-11-13T07:00:00Z
sources: [nafs://sessions/2f1c...]
---
Answer in Indonesian unless the question is in English.
```

| Kind | Key | Merge when the key exists |
|---|---|---|
| `profile` | `profile` (one file) | per line `field: value`, the new value replaces the old |
| `preferences` | slug of the topic | replace the body |
| `entities` | slug of the name | append new lines not already present |
| `events` | `{yyyy-mm-dd}-{slug}` | never: a new key per event (a suffix `-2`, `-3` when taken) |
| `notes` | slug of the first words | append |

Every write is a **diff** row: time, session, operation (`add`, `update`, `delete`, `skip`), URI, body before and
after, reason. Undo restores the body before. Events score with a half-life of 30 days after a protection window of
7 days in the recall ranking; the other kinds do not decay.

## 6. Session commit

Commit (on request, or when a conversation has been idle for `[context] commit_idle_minutes`, off by default):
1. Archive: the transcript as Markdown under `nafs://sessions/{id}` (messages with roles and times, sources cited).
2. Extract by rule: "remember that …", "I prefer / I like / I don't like …", "my {x} is …", "call me …", dates with an
   event verb. Each candidate becomes a memory write with its kind and key; duplicates of an existing body are skipped
   with the reason.
3. Optionally (`[context] model_extraction`, off), one grammar-constrained call to the quick model per commit that
   returns `[{kind, title, body}]`; the same keys and merge rules apply.
4. Write the diff rows and mark the session committed; a later commit only reads messages after the last one.

## 7. Skills

`SKILL.md`: front matter `name` (`[a-z0-9-]{1,64}`), `description` (≤ 300 characters), `tags`, `tools`; a Markdown body.
A skill is found by `find` over its name, description and tags (L0) and read in full (L2) only when used. The chat
prompt may carry the L0 of at most two skills whose description matches the question.

## 8. Compile

- **wiki**: `index.md` (the base's L1) and one page per document (its L1 and a list of its sections with links to the
  passages' URIs), plus one page per distinctive term shared by two or more documents, linking them.
- **graph**: `entities/{slug}.md` for each entity of the model-built graph (or each term of the text graph) with its
  type and mentions, and `relations.jsonl` lines `{from, relation, to, evidence: [uri]}`.
- **report**: one Markdown file with the base L1 and each document's L1.

The output is returned as a list of `{path, content}` and can be downloaded as a ZIP; nothing is written to disk.

**Assembly.** `/v1/fs/assemble` returns the budgeted block; the chat prompt keeps four whole passages, because at the
same size the assembler put the answer in the prompt less often (§9).

## 9. Measures

Each mechanism is compared with what it replaces, on the existing question sets and the 50-question answer set:
scoped `find` against the whole base on questions about one document; the assembler against four whole passages
(answers right, prompt tokens, time to answer, interleaved); rule extraction on a set of written conversations
(precision and recall of the memory writes); layers' use in ranking (document-level questions).

Results (`benchmarks/kb_phase29/README.md`): the answering document as scope lifts reworded MRR 0.194 to 0.267, a rule
guessing it from distinctive terms lowers it; the assembler loses answers at equal size (quoted 0.857 against 0.964)
and gains only with 25 to 70 % more text; rule extraction finds 13 of 14 memories with 1 false on messages written
after the rules were final.
