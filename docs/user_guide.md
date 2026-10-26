# Nafs-Aql user guide

Nafs-Aql answers in two ways. A quick model answers at once. A thorough model thinks a question through before answering, when it needs it. In **Auto** mode Nafs-Aql decides which to use, and answers that can be checked exactly are marked **Checked**.

## Chatting

- **Ask anything.** Press Enter to send and Shift+Enter for a new line. Press `?` for keyboard shortcuts.
- **Mode** (under the message box):
  - *Auto* answers quickly, and thinks it through when the question needs it.
  - *Quick* always answers at once.
  - *Thorough* always thinks first. It is slower: a minute or two on this computer.
- **Add a check:** give the expected answer, Python tests or a JSON format, and the reply is checked against it exactly. An answer that passes is marked *Checked*.
- **Settings** (the gear at the bottom left, or the sliders icon on a phone):
  - *Balance between speed and care* decides how readily Auto takes the thorough path.
  - *Thinking time* sets how long a thorough answer may think; *Automatic* uses the measured setting for the topic.
  - *Show the thinking* displays the thorough model's reasoning while it writes.
  - *Reuse earlier checked answers* lets checked answers to similar questions inform the reply; they are shown as sources.
  - *Appearance* switches between light, dark and the system setting.
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

## Privacy

Chats may be used to improve answers, after personal details are removed and a person has reviewed them. To keep a chat out, open the chat's menu (the three dots above it) and choose **Don't use to improve answers**. Chats are kept for 90 days by default.

## The name

Nafs-Aql borrows classical terms of the self as an **analogy**, not as a religious claim:

- *nafs*: the first impulse (the quick answer);
- *lawwāma*: self-reproach (a quick answer flagged for a second look);
- *ʿaql*: reflection (thinking it through);
- *muṭmaʾinna*: at rest (an accepted answer);
- *tawaqquf*: withholding judgement (*Not sure*);
- *muḥāsaba*: self-accounting (the Quality page);
- *tazkiya*: purification (the Learning page).

Click the logo for the notes on the name and their sources. A scholar's review of this text is part of the release process.

## Insights (Advanced mode)

The Insights pages are for the people who run and study Nafs-Aql. On a phone they sit behind *More*.

- **Quality** (*muḥāsaba*): how Auto decides when to think longer, how trustworthy each check is, reliability by topic, what people prefer in Compare, whether quality is changing, warnings and spot checks.
- **Learning** (*tazkiya*): which checked answers the quick model can learn from, the review queue, the models in use, and the operator steps to train and replace a model.
- **Experiments:** ready-made experiments (routing, speed against care, independence, thinking time, the full service, strong code tests, early stopping). *Which strategy wins?* shows the best strategy per topic at each speed weight, pooled over every experiment. *Release readiness* tracks the release milestones. Reports download as Markdown and CSV.
- **Speed:** load tests, the standard load test with 1, 2 and 4 users, and *Where the time goes*: the share of time per step of recent answers, with how fast the models read and write and how much of each question they could reuse.
- **Status:** whether both models are ready, the settings in force, a reliability summary, and *Try a check* for running the exact checks on any answer.

## For developers

- **Math and code harness:** `GET /v1/harness` says what this server offers.
  - *Formal proofs.* Send `verification.lean = {"statement": "<a .lean file whose theorem is proved by sorry>"}` with a chat request; the answer is **verified** only if Lean accepts a proof that keeps the statement byte for byte, with no `sorry`, no axioms beyond Lean's three standard ones, and no commands that run code. The server needs a Lean toolchain (`verifiers.lean`); otherwise the request is refused rather than answered unverified.
  - *Repairs.* When System 2's answer fails its tests, the next attempt repairs the candidate that passed the most tests (`repair_policy = "best"`), and is shown what earlier repairs of it already tried. Set `"patch_repair": true` on a code request to let System 2 answer a repair with a small patch instead of the whole program (experimental; off by default).
  - *Code agent.* `POST /v1/harness/code` with `{"task": ..., "files": {"solution.py": ...}, "verification": {"tests": [...]}}` lets System 2 run its own code in the sandbox and edit it with patches until the hidden tests pass (at most `max_steps` model calls). The reply lists every step and returns the passing files, or the best ones it found, labelled *unverified*.

