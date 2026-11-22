# FreeToken for Smiley Core (Phase 34)

Source: Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica, *FreeToken: Efficient Edge-Native MoE Serving
with Bandwidth-Adaptive Execution*, arXiv 2608.16157 (17 Aug 2026), and its code at `third_party/FreeToken` (revision
c84946b). Measurements: `benchmarks/engine_phase34/`. Decision 026.

This note assesses what of FreeToken applies to a CPU-only engine serving dense models, what Smiley Core did before,
the strategy that followed, and what the measurements said.

## 1. What FreeToken does

FreeToken serves Mixture-of-Experts models whose expert pool is larger than the GPU's memory, on a personal machine.
It treats the GPU, the CPU, host memory and the PCIe link as one platform and divides work by measured bandwidth.

| Mechanism | Paper | In the code |
|---|---|---|
| Full-layer double buffering | While the GPU computes layer l, a second stream copies all of layer l+1's experts into the other of two staging buffers. Prefill becomes bound by the slower of copy and compute, not their sum (19 to 26 % at 4K to 16K tokens). | `layers/moe.py` asks for layers l and l+1, then waits for l; two buffers by `layer % 2`; falls back to serial when the cache cannot hold two layers. |
| Semantic anchors | Recurrent-state checkpoints at the special tokens where a harness edits context (thinking, tool calls, tool output, turns), so an edit re-prefills only the new suffix. | Narrower than the paper: one anchor (the tool-call opener, when it is a single token), off by default (`--enable-special-token-ckpt`), and only for models with recurrent or sliding-window layers. For a model of full-attention layers it does nothing, because every position of a radix prefix is already a point to resume from. The general checkpoint grid is every 64 tokens. |
| Radix prefix tree, one residency | Full-attention KV is shared by prefix; pages a request computed join the tree when it finishes. | On commit the request's page table is re-pointed at the tree's pages (no copy); eviction is least recently used over unlocked leaves; the end of a prompt is pinned so a turn that drops its reasoning still matches. |
| q* policy | Of m expert misses at a decode step, q* = m B_P / B_H are copied to the GPU and the rest run on the CPU, where B_P and B_H are the measured link and host bandwidths, so both branches finish together. | `moe/bench_profile.py`; measured once by `ft bench bw`, never automatically. |
| CPU executor | A persistent pool pinned to physical cores, SIMD kernels that dequantise inside the dot product, exact merge with the GPU's partial sum. | Workers claim 32-row blocks from a shared atomic counter (no fixed shares); spin 50 ms, then park, and spin only when every worker has its own core and two are spare; prefetch capped at two rows so a worker does not fetch its neighbour's block. No thread sweep, no NUMA or huge-page handling. |
| Elastic memory | The split between expert cache and KV can be changed while serving, at a scheduler safe point. | `POST /v1/cache/rebuild`, only when the engine is idle; new requests are refused meanwhile. |
| Fast bootstrap | Weights are read straight into their final layout with parallel direct I/O and pinned afterwards; the first request is served cold. | 8 MiB chunks on 8 threads; a memory-map fallback with sequential read-ahead advice. The engine still runs two warm-up prefills. |

## 2. What applies to a CPU engine serving dense models

Nafs-Aql runs Qwen3-1.7B and Qwen3-4B (dense, full attention) on four CPU cores with no GPU. There are no experts, no
second device, no link, and no recurrent state. The mechanisms do not transfer; the reasoning behind them does.

| FreeToken's reasoning | On this machine | Outcome |
|---|---|---|
| Moving weights and computing with them use different resources, so they should overlap, not alternate. | The two resources are the disk (through the page cache) and the cores. A freshly started server has the model file mapped and none of it in memory, and the first pass faults it in a page at a time. | **Read-ahead**: a thread walks the weights in the order a pass reads them (section 4.4). |
| A harness edits context at special-token boundaries, and state kept there survives the edit. | The chat template is the harness. Qwen3 generates a reply after a reasoning block and replays it without one; the gateway sends a question with its passages and replays the question alone. The answering model's own cache stops matching at the question of the turn before. | **The answering model reads the conversation between turns** in its next-turn form (4.2), and turn tokens decide when a kept cache can be taken without a copy (4.1). |
| One residency for prefill and decode: pages join the prefix tree by re-pointing, not copying. | The prefix store copied a cache out at the end of every request and back in at the start of the next. | **Zero-copy prefix store** (4.1). |
| Unclaimed work is anybody's (the executor's block claiming). | Every matmul was split into fixed shares, and a fork-join waits for its slowest worker. The machine is shared: a second engine, the gateway, a browser. | **Claimed blocks** (4.3). |
| Memory is not dedicated; the budget must change while serving. | The prefix store's budget was fixed at start. | **`/v1/cache`** (4.5). |
| No phase hand-off between prefill and decode. | The prompt's last token always had a pass over the weights of its own. | **First token from the last prompt chunk's pass** (4.1). |
| q*: split work between two paths by measured bandwidth. | There is one path. The nearest analogue, choosing thread counts from measured bandwidth, was done by hand in Phases 17 and 27 (three threads to write, four to read). | Not ported. |
| A shared least-recently-used expert cache follows routing locality. | No experts. | Not applicable. |
| Checkpoints of recurrent state. | No recurrent layers; every cached position is a resume point. | Not applicable. |

## 3. Smiley Core as found

- **Loop.** One engine thread; a step is one chunk of one prompt (64 tokens) or one token for every ready slot. Two
  pools: three threads to write tokens, four to read prompts.
- **Prefix store.** Finished conversations kept as packed copies under 768 MB per model, least recently used evicted.
  A new request restored the longest shared beginning by copying it into a fresh cache.
- **What a request paid for that** (`handoff_before.txt`, a conversation extended by 8 tokens per request):

  | Model | Depth | Copy in | Copy out | Share of the time to the first token |
  |---|---|---|---|---|
  | Qwen3-1.7B | 512 | 27 to 30 ms | about 53 ms | 13 % |
  | Qwen3-1.7B | 2,048 | 49 to 62 ms | about 165 ms | 17 % |
  | Qwen3-4B | 512 | 36 ms | about 65 ms | 7 % |
  | Qwen3-4B | 2,048 | 65 to 74 ms | about 205 ms | 11 % |

  A conversation larger than the budget (5,300 tokens on the 4B model) was not kept at all: the next step of a long
  agent transcript read the whole transcript again.
- **First token.** The prompt's last token was never part of a prompt chunk; a request with a 7-token suffix read the
  weights twice (a 7-row pass, then a 1-row pass).
- **Turns.** After a reply only the model that did *not* answer read the conversation between turns (Phase 30). The
  model that answered kept the turn as generated, which the next turn does not replay.
- **Shares.** `pool::share` gives each worker a fixed contiguous range of rows.
- **Start.** The model file is mapped; nothing reads it before the first request.

## 4. Strategy, and what was built

Order: the pieces that remove work on every request first, then the one that removes work on every turn, then the
ones that matter when the machine is busy or cold. Each has a switch, so it was measured against the engine without
it in the same run.

### 4.1 A prefix store that moves caches (engine)

`prefix::CacheStore` holds the caches themselves. A finished slot's cache moves into the store; a request that
continues the conversation takes it out again. Nothing is copied on either side.

- **When a request takes a cache** (`prefix::supersedes`): its prompt holds every token of the kept conversation
  (the next step of an agent, a continued turn, a prompt a shadow has read), or the kept conversation ends inside
  one turn that the prompt has rewritten and gone past (a reply replayed without its reasoning block, followed by a
  new turn). Turn boundaries are the template's own token (`<|im_start|>`), given to the engine by the server.
- **When it copies instead:** another conversation under the same system prompt (the kept one holds whole turns the
  prompt does not), or a second sample of the same prompt (the first sample's reply may be the one the conversation
  continues from). A prompt with memory vectors always copies.
- **A request still being served is a source too.** The positions a slot has written never change, so a second
  request that shares its beginning copies them while it runs. Before, two users under one system prompt who arrived
  together each read it.
- **The newest conversation is always kept.** The budget bounds what is kept beside it. Its cache was the slot's own
  memory a moment before, so keeping it adds nothing, and the long-transcript cliff of section 3 is gone.
- **Accounting is by the memory a cache occupies** (the most positions it has held), not by what it holds now.
- **An error gives the cache back.** A pass is refused before it writes, or fails after writing whole positions the
  slot has not booked; the slot's cache is still what the slot says it is, and it returns to the store.
- **The first token comes from the last prompt chunk's pass.** The chunk that reaches the end of the prompt carries
  the prompt's last token and asks for that row's logits. `--no-fuse-first-token` restores the separate pass.

### 4.2 The answering model reads the conversation between turns (gateway)

`[shadow] answering_model = true`. After a reply, once nothing has run for `idle_delay_ms`, the model that answered
reads the conversation as the next turn will show it (the question as written, the reply without its reasoning
block), and then the other model does, as in Phase 30. A request that arrives first stops both; the server keeps what
was read. `"shadow": "other"` on a request gives the earlier behaviour, for comparison.

### 4.3 Claimed blocks (engine)

`pool::Claims`: each worker claims blocks of rows from its own contiguous share, front to back, and when its share is
done claims from the others'. On a quiet machine the reads are the same streams as with fixed shares and the cost is
one uncontended atomic add per block. A worker the system has descheduled no longer holds a matmul back by more than
the block it was in. Used by the matmul groups (64 rows per claim against one activation, 16 from four, 8 from
sixteen) and by attention's tasks. `SMILEY_CORE_CLAIM=0` restores fixed shares.

### 4.4 Read-ahead (engine and server)

`Model::read_ahead` walks the weights in the order a forward pass reads them: for each tensor it tells the system
the whole range is wanted and touches every page. The server runs it on its own thread at start and answers requests
meanwhile (`--no-read-ahead` turns it off).

### 4.5 The prefix store's budget can change while serving (server)

`GET /v1/cache` reports the store. `POST /v1/cache {"prefix_cache_mb": n}` changes its budget and `{"flush": true}`
drops what it holds. Both are applied by the engine thread between two steps, where no pass over the weights is in
progress; requests being served keep their own caches. `/metrics` carries the same counters.

## 5. Results

The machine was shared for every run (a second session building and benchmarking another project, a model download,
a browser; load averages from 6 to over 20). Every comparison was interleaved, in short paired rounds where the
quantity allowed it; the files in `benchmarks/engine_phase34/` carry the load.

| Question | Answer | Run |
|---|---|---|
| What does a request pay to take a kept conversation over and give it back? | Nothing, where it was 84 ms (512 tokens) and 240 ms (2,048 tokens) on the 1.7B model, 116 ms and 457 ms on the 4B model. | `handoff_ab_system1.json`, `handoff_ab.json` |
| What does that do to the first token of a request that extends a conversation by 8 tokens? | With the first token taken from the last chunk's pass as well: 244 to 184 ms and 329 to 229 ms on the 1.7B model; 602 to 448 ms and 1,075 to 568 ms on the 4B model. The fused pass alone is worth about 30 ms on the 1.7B model and 90 ms on the 4B model. | same |
| Did the kernels change speed? | No: Q4_K 95 ns a row in the old and the new build. Reading a prompt on four threads was at or above the old build in three of three pairs. | `kbench_old_new.txt`, `prefill_old_new.txt` |
| Do claimed blocks pay? | On four threads, yes: the 1.7B model wrote 19 % faster (12 of 12 paired rounds) and read 14 % faster (12 of 12); the 4B model wrote 33 % faster (8 of 8) and read 10 % faster (7 of 8). On three threads the 1.7B model was 2 % faster (7 of 12, inside noise) and the 4B model 8 % and 15 %. With one thread spinning beside the engine: 8 % faster writing at either width. | `claim_system1.txt`, `claim_system2.txt` |
| Does it matter where an idle worker takes work from? | Yes. Taking from the front of a neighbour's share, where the neighbour is reading, was 5 % slower writing and 10 % slower reading than fixed shares on three threads. Taking from the back removed the loss. | `claim_system1_run1.txt` |
| Does read-ahead pay? | On a model file the system had not cached, the first request of a fresh server took 9.2 s with it and 14.6 s without (three rounds: 9.2, 9.2, 9.8 against 12.9, 14.6, 15.1); a warm request of the same size takes 5 s. | `cold_start_system1.json` |
| Does the answering model's reading between turns pay? | Quick conversations: the first token of a later turn after 0.71 s instead of 4.25 s (median of 9 turns each way), 41 prompt tokens read instead of 188, the whole turn 9.3 s instead of 11.3 s. Conversations answered by the reasoning model: 2.4 s instead of 18.6 s to the first token (median of 6), the turn 62 s instead of 104 s. | `turn_reuse_fast.json`, `turn_reuse_slow.json` |
| Does the budget follow a change while serving? | Yes: set to 1 MB, a store of 20 conversations and 718 MB went to the newest one alone; flushed and set back, it was empty at 768 MB; a value that is not a number of megabytes is refused. | session of 2026-11-22 |

What the numbers do not show:

- The between-turns gain needs the pause between turns to be longer than the reading (3 to 5 s for the quick model, 7
  to 17 s for the reasoning model on a shared machine). One reasoning turn of six arrived before the reading had
  finished and waited 18.6 s, as before.
- A fourth decode thread was not compared with three head to head after the change; the default stays three threads
  to write and four to read.
- In the gateway's real traffic most reuses are copies, not takes (216 against 19 on the quick model after a morning
  of test traffic): many short conversations share only the system prompt. Those copies are short. The take matters
  for the long transcript, where the copy was long.
- A kept cache is accounted by the most positions it has ever held. A conversation that once reasoned at length keeps
  that memory until it is evicted.

## 6. Not adopted

| Idea | Why not |
|---|---|
| A paged cache with shared prefix pages (the radix tree proper) | Sharing a system prompt between kept conversations would save memory, not time, and a page table costs an indirection on the hottest loop of the engine. The copy that remains is the short one. |
| Checkpoints at special tokens | Every cached position of a full-attention model is already a place to resume from. The turn token is used for something else: to decide when a kept cache may be taken. |
| A measured split between two execution paths (q*) | There is one path. |
| Choosing thread counts from a measured bandwidth at start | Three to write and four to read were measured by hand (Phases 17 and 27) and claimed blocks removed most of the penalty of the wider pool; an automatic choice would need a measurement at every start on a machine whose load changes by the minute. |
| Changing the replayed form of a reply so that it matches what was generated | The models were trained on the template that drops the reasoning block from earlier turns; reading the conversation in idle time gets the same saving without changing the prompt. |
