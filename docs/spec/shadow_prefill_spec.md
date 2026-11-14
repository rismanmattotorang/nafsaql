# ShadowPrefill specification (Phase 30)

Keep the model a conversation may switch to warm, so a switch reads only what is new. After Qiu, Mahapatra, Fonseca
and Russinovich, *ShadowPrefill: Keeping Your Next Model Warm for Per-Step Model Routing in Agentic LLM Serving*
(arXiv 2611.13176, 2026). Research note: `docs/research/shadow_prefill.md`. Decision 023.

## 1. The switch in Nafs-Aql

The gate sends a question to the quick model (S1); when the draft does not pass, the reasoning model (S2) answers it
from the same messages. S2 has never read that conversation, so it reads the whole prompt before its first token: about
22 tokens a second on the reference machine, 46 s for a 925-token prompt (`benchmarks/engine_phase30/`). The reverse
switch (S1 after an S2 answer) reads S2's answer. The paper's cluster has idle GPUs and SLO slack; here both models share
four CPU cores, and the idle time is the time a person spends reading an answer.

## 2. Engine (Smiley Core)

| Item | Behaviour |
|---|---|
| Shadow class | `SlotCfg::shadow`: the slot reads its prompt (all but the last token) and is never decoded; its cache goes to the prefix store when it is read |
| Priority | a real prompt is always read first; a shadow chunk runs only when no real prompt waits, and while real requests decode only every `--shadow-every` steps (0 = only when nothing else is served) |
| Take-over | a real request that shares at least 16 tokens with a shadow's read part cancels it before its prefix lookup, so it restores what the shadow read; a full engine gives up its oldest shadow for a real request |
| Abandonment | a caller that has gone stops its slot even while the prompt is being read (cancelling keeps what was read) |
| Duty | `"shadow": {"duty": d}`: while shadows are the only work, the engine rests in proportion `(1-d)/d` to each step; an arriving request wakes it at once |
| Open end | a shadow's messages may end with the assistant's turn: then the conversation so far is read, as the next turn's prompt will begin |
| Report | `timings.smiley.shadow` = shadows read to the end, taken over, positions read |

## 3. Gateway

`[shadow]` in the configuration, overridable per request (`ChatRequest.shadow`, `ChatRequest.stable_prefix`):

| Key | Meaning | Default |
|---|---|---|
| `between_turns` | after a reply, once no request has been in flight for `idle_delay_ms`, the model that did not answer reads the conversation including the reply; the next request stops it | set by measurement (decision 023) |
| `idle_delay_ms` | wait before starting | 1500 |
| `at_arrival` | while S1 drafts, S2 reads the same prompt at `arrival_duty`; stopped once the gate has decided | off |
| `arrival_duty` | share of time the arrival shadow may compute | 0.25 |
| `stable_prefix` | the system prompt holds only the assistant's own text; the domain instruction, earlier answers, passages and notes go before the question in the last user message, so each turn's prompt begins with the previous turn's | set by measurement |

A between-turns shadow counts as work in flight, so the idle-time workers (audits, the label stream) wait for it. Only
Smiley Core servers take shadows (`Backend::supports_shadow`); other engines are unaffected.

## 4. Interface

Settings → **Speed** (`PREFS.shadow`, `PREFS.stablePrefix`, sent with each request):

- *Get the careful model ready while you read* — Off / Between messages / Also while drafting.
- *Keep the start of a conversation the same* — on/off, with the hint that it lets both models reuse what they read.

The Status page (Advanced) shows the `[shadow]` settings in force and the S2 server's shadow counters.

## 5. Measures

1. Contention probe (`scripts/shadow_probe.py`): draft time, S2 first-token time and their sum, for a cold switch, a
   shadow during the draft (full speed and capped duty) and a shadow done beforehand.
2. Session study (`scripts/shadow_session.py`): four-turn conversations with a reasoning turn last, for each layout and
   shadow setting, interleaved by round, with 30 s between turns.
3. Answer quality with the stable layout: the RAG answer set and the live sweep's acceptance and harness suites.
