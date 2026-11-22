# Nafs-Aql user guide

Nafs-Aql answers in two ways. A quick model answers at once. A thorough model thinks a question through before answering, when it needs it. In **Auto** mode Nafs-Aql decides which to use, and answers that can be checked exactly are marked **Checked**.

## Chatting

- **Ask anything.** Press Enter to send and Shift+Enter for a new line. Press `?` for keyboard shortcuts.
- **Mode** (under the message box):
  - *Auto* answers quickly, and thinks it through when the question needs it.
  - *Quick* always answers at once.
  - *Thorough* always thinks first. It is slower: a minute or two on this computer.
- **Add a check:** give the expected answer, Python tests or a JSON format, and the reply is checked against it exactly. An answer that passes is marked *Checked*. A JSON format that is not valid yet is said beside the field as you type.
- **Attach a file** to a question with the paperclip, by dropping it on the conversation, or by pasting it: PDF, Word, PowerPoint, Excel, CSV, LaTeX, web pages, Markdown and text. Its text is read on this computer and goes beside your question. One question carries about 8,000 characters of attached text; a longer file is cut and the page says so. To ask about the whole of a long document, add it to a knowledge base.
- **Change your mind.** Point at a message of yours and press the pencil to edit it and send it again: what followed it is replaced. **Regenerate** under an answer writes that answer again in its place.
- **Save an answer or a chat as a file.** *Save as…* under an answer, or *Save chat as a file…* in the chat's menu (the three dots), offers Markdown, Word, PDF, PowerPoint, a web page, LaTeX, plain text, and Excel or CSV when there is a table. The file is made on this computer. The chat's menu also renames the chat.
- **Settings** (the gear at the bottom left, or the sliders icon on a phone). Every setting says what it does and how it is when untouched; *Put back* returns one to that, *Put everything back* returns all of them, and the box at the top finds a setting by a word. They are kept in this browser.
  - *How Auto should answer* is a starting point: *Quicker*, *Balanced* (how this service is set up) or *More careful*.
  - *Balance between speed and care* decides how readily Auto takes the thorough path. Leave *As this service is set up* ticked unless you want your own.
  - *Thinking time* sets how long a thorough answer may think; *Automatic* uses the measured setting for the topic.
  - *Show the thinking* displays the thorough model's reasoning while it writes.
  - *Reuse earlier checked answers* lets checked answers to similar questions inform the reply; they are shown as sources.
  - *Appearance* switches between light, dark and the computer's own setting.
  - *Access key* is needed only when your administrator has set one up.

## Reading an answer

- **Badges** show how the answer was made and how far to trust it:
  - *Quick* or *Thorough*, and how long it took.
  - *Checked*: an exact check passed (tests, the expected answer, or the required format).
  - *Cross-checked*: independent attempts agree, or a review found it sound. A good sign, not a proof.
  - *Not sure*: no answer passed the checks in the time allowed; a note explains why.
  - Open questions have no exact check, so they carry no check badge.
- **Why?** explains in one sentence why the answer took the path it did.
- **Next** suggests a follow-up: *Think harder* (or *Quick answer*), *Explain step by step*, or *Check this answer*.
- **👍 / 👎** records your rating. Ratings help reviewers; they never change the answers directly.

## Finding your way

Press **Ctrl/Cmd + K** on any page to jump to a page or do something (new chat, settings, light or dark, Advanced mode). Type to narrow the list, **Enter** to choose, **Esc** to close. A new chat is **Ctrl/Cmd + Shift + O**. When you scroll up in a long chat, a **Latest** button brings you back to the newest message.

## Tables, formulas, diagrams and code

Answers are drawn the way you would read them.

- **A table** sorts by a column when you click its heading (again to reverse, a third time to put the rows back); numbers sort as numbers. Above it: *Copy table* (pastes into a spreadsheet), *Copy Markdown* and *Save CSV*.
- **A formula** such as $x^2$ is typeset; a displayed formula has *Copy LaTeX*.
- **A diagram** (flowchart, sequence, states or pie) and **a chart** have *Source*, *Save SVG*, *Save PNG* and *Expand*. Expanded, a diagram zooms with + and − (or Ctrl/Cmd and the wheel) and closes with Esc. A chart also shows its data as a table and saves it as CSV.
- **Code** has *copy*, *wrap* for long lines and *save* (a file with the right ending); from six lines its lines are numbered, and long code folds with **Show all lines**.
- **A footnote** such as `[^1]` becomes a small number, with the notes listed under the answer. While an answer is still being written, a half-finished formula or diagram shows as plain text and turns into the picture when it is complete.

In **Knowledge**, the graph shows what your documents name: bigger circles are mentioned more, a colour is a group. Point at a circle to light up what it connects to, click it to read the sentences that say so, double-click to open what is around it, drag to move it, scroll to zoom, and use the arrows button for full screen.

## Comparing answers

Turn on *Show Insights pages and technical details* in **Settings** (the *Advanced* section) to add:

- *Compare* mode: a quick and a thorough answer side by side; pick the better one. Your choices help decide when thinking longer is worth the wait.
- *Details* on every answer: how it was routed, what each check found, and where the time went.
- The **Insights** pages (see below).

## Rooms

A room is a shared chat. Everyone connected sees each question and the assistant's answer as it is written, and the transcript is kept until the room is deleted. Open **Rooms** in the left bar.

- **Create** a room with *New room*. Names are unique without regard to case; spaces are tidied and the name is cut to 80 characters.
- **Join** by clicking a room, or from a link: *Link* in the room's header copies an address of the form `…/#/rooms/<id>`. The page reopens the last room you were in.
- **Your name** (bottom left) is shown to everyone. If someone in the room already uses it, you appear as *Name 2*, and a note tells you so. *Assistant* and *System* are reserved.
- **Ask** with Enter (Shift+Enter for a new line). *Auto*, *Quick* and *Thorough* work as in Chat. The assistant answers one question at a time; up to three more wait, and the header says whose question is being answered. A message may hold 8,000 characters; a counter appears near the limit and a longer message is refused, not cut. *Regenerate* and *Think harder* under an answer ask the same question again as a new question in the room.
- **See who is here**: avatars in the header, the sidebar's *n online*, a dot for new activity since you last looked, and "… is typing". Date separators and join and leave notes mark the flow. Long histories show the latest 60 messages with *Show earlier*.
- **Rename** with the pencil beside the title. Everyone in the room sees the new name at once.
- **Export** downloads a Markdown transcript or the full JSON records.
- **Delete** removes the room *for everyone* and cannot be undone: the room, its messages, and the saved decisions, feedback and retrieval-memory entries made from it. The dialog says how many questions and people are affected. Others in the room are told and disconnected; an answer being written is stopped.
- **If the connection drops** (a restart, a laptop that slept) the header shows *Reconnecting…* with a countdown and *Retry now*; your draft stays in the box. When the connection returns the transcript is rebuilt from the server, so nothing is missing or doubled. A room that was deleted meanwhile is reported as such.
- **On a phone** the room list moves into a drawer behind the *Rooms* button, and the header actions become icons.

Rooms are not removed by the 90-day retention purge: they stay until someone deletes them. There are no accounts, so anyone who can reach the service can read, rename or delete any room; run it with access keys (see the operator runbook) when that is not acceptable.

## Your documents

Open **Knowledge** in the left bar to give the assistant documents of your own. Everything stays on this machine.

- **Create a knowledge base** with *New*, then drop files anywhere on the page or choose them: PDF, Word, PowerPoint, Excel, CSV, LaTeX, web pages, Markdown and text, up to 50 MB each. A spreadsheet becomes one table per sheet under the sheet's name, with the column headings repeated so a passage from the middle of a long sheet still says what its columns are; a formula is read as its last value. A scanned PDF or a picture cannot be read. Each file gets its own row with its stage, a *Cancel*, and, if it is refused, the reason and a *Try again*. A document is *Ready* within seconds and searchable by its words at once; its meaning index follows in the background (the row says *Words only* until then), and a bigger upload never holds a smaller one up. PDFs are read by their layout: columns in reading order, section titles as headings, tables as rows, running headers and page numbers left out. A scanned PDF with no text layer cannot be read. Each passage is also searched sentence by sentence; a document stored before this version gets both the new reading and the sentence search when it is read again (the circular arrow under *Passages*).
- **Search** the base by *Meaning and words* (the default), *Meaning* alone, or *Words* alone. Results appear as you type. Each hit shows its document, heading and page, with your words highlighted; click it to open the passage where it sits in the document, or *Ask about this* to start a chat on it. Search runs in memory, so it stays fast while the machine is busy.
- **Passages** opens a document's passages, each with the heading it sits under; the circular arrow reads a document again (after a parser or chunking change; passages that did not change keep their meaning index), and the cross removes it with its passages.
- **The knowledge graph** card shows a map of your documents as soon as they are read: the names, acronyms and identifiers they share, linked when passages mention both (drawn from the text, no waiting). Scroll to zoom, drag to move, click a legend entry to show one group or a kind (person, method, system) to show one kind. *Fit* brings everything into view; with the keyboard, the arrow keys move the view, + and − zoom and 0 fits. *Save SVG* and *Save PNG* keep the picture. For relations with the sentence that states each, build the full graph with *Build the graph* (it reads every passage with the quick model, so it takes about a minute per passage on a 4-core CPU and runs in the background). Pick a thing to see its relations, each with the sentence that states it; a relation is kept only when that sentence really occurs in the passage. The list of types the graph is read under can be changed through the API (`PUT /v1/kb/{kb}/ontology`).
- **Ask about a base** from Chat: pick it in the box under the composer, then choose *Passages* (the answer cites numbered passages), *Passages and graph* (relations are shown above the passages, and a question that needs two steps follows the chain), or *Choose for me*. Every answer says which searches ran and why any was skipped, and lists the passages it used, so a claim can be checked against its source: click a number such as [2] in the answer to read that passage. If nothing in the base matches the question well, the answer says *No strong match*. You can leave the Chat page while an answer is being written; it keeps going and is there when you return.
- **Is the answer in the documents?** Under an answer from documents you see either *Every sentence of this answer is in the passages* or how many are not, with those sentences listed. The comparison is by words and numbers, with no model: it catches a number or a claim the passages do not contain; it does not prove an answer right.
- **Follow-up questions** such as "And how fast is it?" are searched together with your previous question, and the answer says so.
- **Typing slips** in a question ("converetr") are matched to the nearest word your documents contain.
- **Memory** modes (experimental, *Advanced* only) use a module trained on the base's documents; they appear once a module has been trained and activated from the *Memory* card. A module that gives the same memory whatever is asked is recognised and not used; the answer then comes from passages and says why.


## Code

**Code** in the left bar has a small program written and checked for you. Say what the function should do and give the tests it must pass, one Python `assert` per line. The program is written to `solution.py`, run against your tests in a sandbox that has no network access, and tried again from what failed until the tests pass or the attempts are used up.

- Each step appears as it ends: who wrote it (the quick model first, the careful model after), what it did, whether the tests passed, and what came back. **Stop** ends the run after the step in progress.
- At the end you get the program, with *copy* and *save*, and what changed from any starting code you gave. A run that did not pass shows the best version and how many of the tests it passes.
- *Earlier runs* keeps the runs on this computer; *Use these again* puts a run's task and tests back in the form.
- It is for one function or a small file. One run is served at a time.

## Speed

**Settings → Speed.** *Get the careful model ready while you read*: after each answer, while you read, the careful
(reasoning) model reads the conversation, so if your next question needs it, it starts in a couple of seconds instead of
half a minute; anything you send stops it at once. The model that just answered reads the conversation too, so your
next question to the same model starts in under a second instead of several. *Between messages* is the default; *also while drafting* makes every
quick answer slower on this kind of computer and is off unless you choose it. *Keep the start of a conversation the
same* puts the sources and notes for a question next to the question, so both models can reuse what they read; it is on
by default and changed no answers in testing.

## Files

**Files** shows everything the assistant knows in one tree, each item with a one-line summary: your knowledge bases and
their documents and passages, your memory, your skills and the chats you saved as files. Every item has an address
such as `nafs://memory/preferences/answer-language`.

- **Summary, Overview, Full**: an item opens at its overview (for a document: its first paragraph, its sections, its
  tables and figures and its key terms); *Summary* is one line, *Full* is the whole text.
- **Search here** searches only the folder you are in (a knowledge base, one document, your memory); *Everywhere* lets
  the assistant choose where to look.
- **Memory** holds files you can read, edit, add (*New*) and delete: who you are, how you like things done, people and
  things in your life, events, and notes. Your profile and preferences are used in every chat. *Recent changes* lists
  every change and why, and *Undo* puts a file back.
- **Sessions → Save as file** writes a chat out in full and keeps what you said about yourself ("call me …", "I prefer
  …", "my manager is …", "remember that …") as memory files.
- **Skills** are instructions for tasks, written in Markdown; the assistant finds one by its description.
- A knowledge base can be **compiled** into a wiki, a map of its terms or one report, downloaded as Markdown.

## What it remembers

The assistant keeps a short note of each finished turn of your saved chats, so that something you told it in one chat can be found in another. Nothing is sent anywhere; the notes are in the same local database as your chats.

- **When it looks.** Only when your question is about you or points back: it says *my*, *we*, *last time*, *you said*, *remember*. A general question ("what is a B-tree?") is answered without the notes, so it is not slowed down by them. When a knowledge base is selected, the documents answer and the notes are left out.
- **Keep something on purpose.** Start a message with *Remember that …* ("Remember that the staging server is on port 8443"). Such a note is never removed by the size limit.
- **See and delete.** *Settings → Memory → Show what is remembered* lists every note with a *Forget* button, *Merge repeated notes*, and *Forget everything* (your chats themselves are kept). Deleting a chat deletes its notes.
- **What is never remembered.** Private chats, rooms (what is said there is shared by its members) and the Lab.
- When an answer used notes it says so under the answer (*Used 2 notes from earlier conversations*) and shows them.

Turn the whole feature off with `[recall] enabled = false` in the configuration.

## Privacy

Chats may be used to improve answers, after personal details are removed and a person has reviewed them. To keep a chat out, open the chat's menu (the three dots above it) and choose **Don't use to improve answers**. Chats are kept for 90 days by default.

## The name

Nafs-Aql joins two classical terms, *nafs* (the impulse that acts first) and *ʿaql* (the reason that restrains and corrects it), as a naming analogy for a quick answer and a considered one. Nothing in how the assistant works depends on the analogy. Click the logo for the same note in the app.

## Insights (Advanced mode)

The Insights pages are for the people who run and study Nafs-Aql. On a phone they sit behind *More*.

- **Quality**: how Auto decides when to think longer, how trustworthy each check is, reliability by topic, what people prefer in Compare, whether quality is changing, warnings and spot checks.
- **Learning**: which checked answers the quick model can learn from, the review queue, the models in use, and the operator steps to train and replace a model.
- **Experiments:** ready-made experiments (routing, speed against care, independence, thinking time, the full service, strong code tests, early stopping). *Which strategy wins?* shows the best strategy per topic at each speed weight, pooled over every experiment. *Release readiness* tracks the release milestones. Reports download as Markdown and CSV.
- **Speed:** load tests, the standard load test with 1, 2 and 4 users, and *Where the time goes*: the share of time per step of recent answers, with how fast the models read and write and how much of each question they could reuse.
- **Status:** whether both models are ready, the settings in force, a reliability summary, and *Try a check* for running the exact checks on any answer. Three model servers can be switched here: Smiley Core (the default, the project's own Rust engine, started by `make up`), llama.cpp (`make up-llama`; it serves several people at once better) and vLLM; the switch warms the new servers before it takes effect, and answers already being written finish on the old one.

## For developers

- **Math and code harness:** `GET /v1/harness` says what this server offers.
  - *Formal proofs.* Send `verification.lean = {"statement": "<a .lean file whose theorem is proved by sorry>"}` with a chat request; the answer is **verified** only if Lean accepts a proof that keeps the statement byte for byte, with no `sorry`, no axioms beyond Lean's three standard ones, and no commands that run code. The server needs a Lean toolchain (`verifiers.lean`); otherwise the request is refused rather than answered unverified.
  - *Repairs.* When System 2's answer fails its tests, the next attempt repairs the candidate that passed the most tests (`repair_policy = "best"`), and is shown what earlier repairs of it already tried. Set `"patch_repair": true` on a code request to let System 2 answer a repair with a small patch instead of the whole program (experimental; off by default).
  - *Code agent.* `POST /v1/harness/code` with `{"task": ..., "files": {"solution.py": ...}, "verification": {"tests": [...]}}` lets the models run their own code in the sandbox and edit it until the hidden tests pass (at most `max_steps` model calls). The first step goes to System 1 as four candidates in one request, each checked by the tests; System 2 takes over only when none passes (`"cascade": false` turns that off). The reply lists every step and returns the passing files, or the best ones it found, labelled *unverified*.
  - *Parallel work on the four cores (Phase 13).* Nothing to switch on: when a checkable task's first answer fails its exact check, Nafs-Aql tries four more quick answers at once (each checked as it finishes) before it asks the thorough model, which saves an escalation of several minutes in about one failed draft in six. For power users, a request can set `rescue` (0 to 4 quick extra tries), `parallel` (candidates the thorough model tries at once; 1 by default because it did not pay on this machine), `repair_rounds` and `hedge_after_failure`; `GET /v1/harness` shows the server's settings.

