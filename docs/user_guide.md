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
