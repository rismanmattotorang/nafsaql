# Phase 30: ShadowPrefill (keeping the next model warm)

Reference machine (Intel i3-8100, 4 cores), quiet during these runs (load 2 to 7) unless stated. Spec:
`docs/spec/shadow_prefill_spec.md`. Research: `docs/research/shadow_prefill.md`. Decision 023.

## Files

| File | What | Tool |
|---|---|---|
| `probe_baseline_engine.json` | a 925-token prompt, 3 rounds: the quick model's draft, then the reasoning model's first token, cold; with the reasoning model reading the same prompt at full speed during the draft; and with it read beforehand | `scripts/shadow_probe.py` |
| `session_server.json` | four-turn conversations (three quick turns answered by the quick model, then a switch), 3 rounds x 4 arms interleaved, 20 s of reading between turns, measured at the model servers: the reasoning model's time to read the conversation at the switch (a one-token request), and what each quick turn read | `scripts/shadow_session_server.py` |
| `stable_prefix_ab.json` | 50 questions answered from documents by the quick model, the stable layout against the usual one, order alternating | `scripts/stable_prefix_ab.py` |

`scripts/shadow_session.py` (the same study through the gateway) was run and set aside: the gateway reports only the last
chunk of the reasoning model's work, and the time to the first answer token includes a varying stretch of reasoning
and a judge call, so it could not isolate the switch.

## Results

**Contention (probe, medians of 3).** Cold switch: draft 19.0 s, reasoning model's first token 46 s, 65 s in all. A
shadow at full speed during the draft: draft 33.3 s (+75 %), first token 0.2 s, 33.5 s in all. Read beforehand: draft
19.0 s, first token 0.24 s. A shadow beside the draft pays only if more than about 31 % of questions escalate; real
chats escalate about 25 % of the time (ledger), so the arrival shadow stays off.

**Sessions (medians of 3; switch = the reasoning model's first token after three quick turns).**

| Layout / shadow | Switch | Read at the switch | Quick turns read |
|---|---|---|---|
| usual, none (today) | 36.6 s | 493 tokens | 292 |
| usual, between turns | 28.0 s | 466 | 268 |
| stable, none | 41.3 s | 502 | 193 |
| **stable, between turns** | **2.0 s** | **34** | 207 |

Per round, stable + between: 2.0, 2.3 and 1.5 s against 25.8 to 51.9 s for every other arm. With the usual layout the
system prompt changes with every question, so what the shadow read never matches the next prompt; with the stable layout
the reasoning model restores 493 to 526 tokens and reads only the new question. The stable layout also lets the quick
model reuse its own conversation (a third turn read 165 to 182 tokens with 192 to 228 cached, against 333 to 394 read
with 29 to 43 cached).

**Answers with the stable layout (50 questions from documents, quick model).** Usual 42 of 50 right, stable 42 of 50;
2 questions right only with the stable layout, 2 only with the usual one (McNemar p = 1.0). Median time 21.7 s against
22.4 s; prompts the same size (1,140 tokens). These are single questions, so the stable layout's reuse across turns does
not show here.
