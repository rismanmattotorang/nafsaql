# Decision 027: Forms, answers, documents, the knowledge graph and the code agent as a person meets them (Phase 35)

Date 2026-11-22. Status: **adopted.** Prompt [`PRODUCT_QUALITY_PROMPT.md`](../prompts/PRODUCT_QUALITY_PROMPT.md).
Audit and scorecard [`uiux_audit_phase35.md`](../research/uiux_audit_phase35.md); literature
[`harness_ux_research.md`](../research/harness_ux_research.md). Measurements: `benchmarks/harness_phase35/`.

## Decisions

1. **Settings explain themselves.** Each setting says what it does and what it is when untouched, and can be put back
   alone or with the rest; three presets; a search box; a link (`#/settings`). The balance between speed and care is
   sent with a message only when the person chose one: before, the browser's value silently replaced the service's.
2. **A dialog is a container, not a form.** The form swallowed any form placed inside it (renaming a knowledge base
   never saved) and closed on Enter in a text box (a half-written file was lost). A dialog that holds unsaved text
   asks before closing.
3. **Segmented controls say which button is chosen**, on every page, through one observer rather than a change at
   each of nine call sites.
4. **Answers.** Tables sort by a column (numbers as numbers), copy as Markdown and save as CSV, with actions that are
   always visible; code blocks wrap, save and number their lines from six lines; footnotes; a figure opened larger is a
   dialog that keeps the keyboard inside it and zooms.
5. **A message can be edited and sent again, and Regenerate replaces the answer.** The stored conversation is rewound
   to the message first (`rewind_to`), so what followed is replaced, not added after. A room's conversation, which
   several people share, is never rewound.
6. **A screen reader is told that an answer is being written and that it is ready**, once each. The streamed words
   are no longer a live region (WCAG 2.2 SC 4.1.3).
7. **Documents in.** The knowledge base reads Excel (a heading and Markdown tables per sheet, the header repeated
   every 40 rows so that a passage cut from the middle of a sheet still says what its columns are) and CSV/TSV; Word
   lists and tables keep their structure. A file can also be attached to one question: its text goes beside the
   question, 8,000 characters at most, and the page says what was cut and that a whole document belongs in a base.
8. **Documents out.** An answer or a chat is saved as Markdown, plain text, a web page, LaTeX, CSV, Excel, Word,
   PowerPoint or PDF. The model writes Markdown; a converter in the gateway makes the file. The web page, LaTeX, CSV
   and Excel writers are our own (what may appear in the output is a closed list); Word and PowerPoint come from
   `office_oxide` and PDF from `pdf_oxide`, both already in the build. Raw HTML is shown as text, a link that is not
   `http`, `https` or `mailto` loses its target, a cell that a spreadsheet would run as a formula is stored as text.
9. **The knowledge graph** saves as SVG or PNG, filters by kind, fits what is drawn, moves and zooms from the keyboard
   and labels the relations of the chosen thing. Files can be dropped anywhere on the Knowledge page.
10. **The code agent has a page** (Code): a task and its tests, each step as it ends, Stop, the checked program, the
    earlier runs. The loop notifies its host after each step and can be stopped, which also ends the model call.
11. **The agent reads what small models write.** A code block ends at a fence line indented at most three spaces, not
    at the first three backticks; `python3` and similar tags are Python; of two blocks the one that defines something
    is the file; a run block that names no file says what a run block is for; a whole-file reply written from a
    shortened view is refused.
12. **A turn with nothing to run does not offer a run.**
    - Before: on 12 tasks that start with no file, the quick model's first step tried to run the system prompt's
      example file 11 times, and one task passed on that step.
    - After: 12 of 12 first steps write the program and all 12 pass on that step (median 7.9 s). A task that went to
      the reasoning model for minutes is answered by the quick model in seconds.
13. **Configuration is checked when it is read.** A value that names nothing or a count out of range stops the start
    with the key and what is allowed; a key this version does not know is reported at start and on the Status page;
    options the active engine cannot honour are named (`harness.constrain`, `adaptive_exit` and a four-candidate
    cascade with the default engine). The schema is generated again: it had been missing two sections, so the shipped
    configurations failed it.

## Review of both phases, and what it changed

An independent read of the two commits (engine and product) found no way for a request to start from a cache that
does not match its tokens, and nine things to correct. All nine were corrected before the release.

| Finding | What was wrong | What it is now |
|---|---|---|
| A rewind could lose a turn | Editing a message or asking for an answer again deleted the old exchange before the new answer existed. With the model down, a time-out or Stop, the exchange was gone. | The turn is answered from the conversation as it was at that point, and the stored conversation goes back in the same transaction that stores the new turn (`Store::append_turn`). A failed request changes nothing; the page carries the point until an answer is stored, so a retry replaces the same tail. |
| Rows left behind by a rewind | Only messages and replies were deleted: the audit, decision, feedback, reviews, preference pairs, the entry in the memory of checked answers and the recall entries stayed, so an edited question could still be recalled in its old wording. | One routine removes what a reply left elsewhere, for a rewind and for deleting a conversation alike; recall entries written since the first removed message go too. |
| A small spreadsheet could ask for any amount of memory | A cell that refers to a shared string costs a few bytes in the file and the whole string in memory. | A cell keeps 2,000 characters, a workbook 16 MB of cell text, and the text says where reading stopped. A column reference with too many letters is not a column. |
| A run that died stayed "going" | A panic inside a run left the list of runs occupied; every later run was refused until a restart. | A guard takes the run off the list however its task ends. Runs beyond the newest 100 are deleted. |
| The new configuration checks refused files that used to work | Another kind of server, a judge named loosely, and the settings of a feature that is switched off stopped the start. | Those are said, not refused. What the active engine cannot honour is worked out when the Status page asks, since the engine can be changed while the service runs. |
| A formula went into the LaTeX file as written | `\input`, `\write`, `\def` and TeX's `^^` spelling of any character passed through. | A formula keeps its mathematics; 44 commands that read or write files or define things are shown as their names, `^^` is broken up, `%` cannot comment out a line, and `^` in a link is encoded. |
| Maths delimiters were rewritten inside code | `` `\(\d+\)` `` was saved as `` `$\d+$` ``, an escaped bracket in a sentence became a display formula, and the Markdown file was not the text as written. | Code spans are left alone, `\[` opens a formula only at the start of a line, the Markdown file is saved as written, and the writers that know no maths drop the dollar signs of a formula but not of a price. |
| The page could misjudge where a message is stored | The same question asked twice, the first time without an answer, was taken for the stored one. | A question counts as stored only when the answer under it finished. Regenerate inside a comparison no longer replaces the comparison. |
| Smaller things | A footnote text nothing pointed at vanished; opening another run while one was being read left two readers; an unusable run block beside SUBMIT was refused; `<<<` in an attached file was rewritten; an attached spreadsheet could make a question "maths"; two files read at once could exceed the room beside a question; the API description lacked the new routes. | Each corrected; the kind of question is read from the question, not from the attached text. |

Left as designed, with the reviewer's note recorded: of two Python blocks the one that defines something is the file
even when a later block is a script; several engine tests compare greedy tokens between passes of different batch
shape, which holds on the two models shipped and could differ in the last bits on another.

## Not done, and why

- **More diagram kinds** (class, entity-relationship, Gantt, mind map): the prompts ask the models for flowcharts,
  sequences, state diagrams and pies, which render; a kind that is not drawn is shown as its source, never blank.
- **A search/replace edit format for long files**: the research favours it over patches for small models; it needs a
  task set with files over 80 lines, which the measured sets do not have.
- **Versions of a regenerated answer**: the old answer is replaced. Keeping both needs a tree of messages in the
  store.
- **Reading scanned PDFs and pictures**: needs a recognition model.
- **Server-side settings per person**: settings stay in the browser.
