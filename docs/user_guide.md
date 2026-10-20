# Nafs-Aql user guide

Nafs-Aql answers in two ways. A fast model (System 1) drafts an answer at once. A careful model (System 2) reasons before answering, when a question needs it. The gate between them (the "Aql Gate") decides which to use, and exact checks decide what counts as verified. The app is also a laboratory for studying when deliberation is worth its cost.

## Chatting

- **Ask anything.** Press Enter to send and Shift+Enter for a new line. Press `?` to see the shortcuts.
- **Path:**
  - *Auto* lets the gate decide.
  - *Fast* uses System 1 only.
  - *Deliberate* always uses System 2.
  - *Both* is a lab trial: both answer side by side, and you pick the better one.
- **Care–Speed slider (λ):** the price of extra waiting. Move it towards Care and the gate consults System 2 more often.
- **Options:**
  - *Thinking budget* sets how long System 2 may reason; "per domain" uses the measured setting.
  - *Show reasoning* displays System 2's thinking.
  - *Use verified memory* lets earlier verified answers inform the reply. They are shown as sources.
- **Early exit from reasoning:** on maths and factual questions System 2 is asked for its answer part-way through its reasoning; if an exact check passes (or two probes agree), it stops reasoning and writes the answer. The Inspector shows each probe and how the reasoning ended.
- **Verify:** attach a reference answer, Python tests or a JSON Schema, and the answer is checked exactly. Verified means an exact check passed.

## Reading an answer

- **Badges:** each answer shows the path, the assurance level and the timing.
  - *Verified* means an exact check passed.
  - *Consensus* means independent agreement or a passed rubric judge. It is a softer signal.
  - *Unverified* means no check applied.
  - *Abstained* means no candidate earned acceptance; you get a notice explaining why.
- **Strip and one-line reason:** the strip shows the answer's journey (Draft → Flagged → Deliberate → Accepted, or Abstain). The line beneath it gives the gate's reason.
- **Inspector:** shows the full decision (estimated error, cost, threshold), verifier and judge scores, a timeline, and **where the time went**: each stage (draft, verification, deliberation, judge) with its prefill and decode time and how much of the prompt came from the cache.
- **Compare:** sets the System 1 draft beside the final answer.
- **👍 / 👎:** records your rating. Ratings are shown in the ledger and help reviewers; they never change the gate directly.

## Rooms

A room is a shared chat. Everyone connected sees each question and the assistant's answer as it is written, and the transcript is kept until the room is deleted. Open **Rooms** in the left bar.

- **Create** a room with *New room*. Names are unique without regard to case; spaces are tidied and the name is cut to 80 characters.
- **Join** by clicking a room, or from a link: *Link* in the room's header copies an address of the form `…/#/rooms/<id>`. The page reopens the last room you were in.
- **Your name** (bottom left) is shown to everyone. If someone in the room already uses it, you appear as *Name 2*, and a note tells you so. *Assistant* and *System* are reserved.
- **Ask** with Enter (Shift+Enter for a new line). *Auto*, *Fast* and *Deliberate* work as in Chat. The assistant answers one question at a time; up to three more wait, and the header says whose question is being answered. A message may hold 8,000 characters; a counter appears near the limit and a longer message is refused, not cut. *Regenerate*, *Think harder* and *Fast only* under an answer ask the same question again as a new question in the room.
- **See who is here**: avatars in the header, the sidebar's *n online*, a dot for new activity since you last looked, and "… is typing". Date separators and join and leave notes mark the flow. Long histories show the latest 60 messages with *Show earlier*.
- **Rename** with the pencil beside the title. Everyone in the room sees the new name at once.
- **Export** downloads a Markdown transcript or the full JSON records.
- **Delete** removes the room *for everyone* and cannot be undone: the room, its messages, and the saved decisions, feedback and retrieval-memory entries made from it. The dialog says how many questions and people are affected. Others in the room are told and disconnected; an answer being written is stopped.
- **If the connection drops** (a gateway restart, a laptop that slept) the header shows *Reconnecting…* with a countdown and *Retry now*; your draft stays in the box. When the connection returns the transcript is rebuilt from the server, so nothing is missing or doubled. A room that was deleted meanwhile is reported as such.
- **On a phone** the room list moves into a drawer behind the *Rooms* button, and the header actions become icons.

Rooms are not removed by the 90-day retention purge: they stay until someone deletes them. There are no accounts, so anyone who can reach the gateway can read, rename or delete any room; run the gateway with tokens (see the operator runbook) when that is not acceptable.

## Privacy

Conversations may be used, after removing personal data, for evaluation and for training the fast model. Turn off **Use for research** in a conversation's header to exclude it. Conversations are kept for 90 days by default.

## The design vocabulary

The interface borrows classical terms of the self as an **analogy**, not as a religious claim:

- *nafs*: the first impulse;
- *lawwāma*: self-reproach;
- *ʿaql*: reflection;
- *muṭmaʾinna*: at rest;
- *tawaqquf*: withholding judgement;
- *muḥāsaba*: self-accounting;
- *tazkiya*: purification.

Click the logo for the design notes and their sources. A scholar's review of this text is part of the release process.

## For researchers

- **Lab:** experiment presets (routing, λ frontier, coupling, budget curve, the full system, strong code tests). The **Insights** table shows which policy is best per domain and price of latency, pooled over every run. Reports export to Markdown and CSV.
- **Muhasaba:** gate mode and calibration, verifier quality, the value of deliberation from Both trials, drift and alerts.
- **Tazkiya:** what would become training data, the review queue and the model registry.
- **Performance:** load tests, the recorded 1/2/4-session protocol, and **Where the time goes**: the share of wall time per stage over recent replies, with the engine's prefill and decode speed and KV-cache reuse. Tick *include lab runs* to add evaluation traffic.
