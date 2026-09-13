# read-sweep

Harnesses that read packed-expert files the way the engine does, so the drive
can be priced without the engine in the loop. They exist because the `io`
window is the largest single span in a decode step, and because the engine's
own counters cannot separate "the drive is slow here" from "the engine is slow
at reading here" — a distinction that decided the whole 3.6-versus-3.8
investigation (`docs/OPTIMIZATION_PLAN.md` §2.1,
[IO-16/IO-17/IO-18](../experiments/summaries/01-model-install-and-expert-io.md#io-16)).

Two pairs, in the order they were built:

| | Synthetic offsets | The engine's real offsets |
| --- | --- | --- |
| One file / one depth — the device | `read_sweep` (C) | — |
| The layer walk — the read shape | `read_batch` (C) | `replay_dest.py` (Python) |
| Engine beside replay, same hour | — | `paired-engine-replay.sh` |

`capture-traces.sh` produces the traces the last two consume.

None is wired into the runtime and none should be. They are measuring
instruments, and everything they produced is a negative result: request size,
read depth, batch structure, destination memory, dispatch chain, file walk,
batch barrier, access pattern, inter-batch idle, memory pressure and the PLE
stream were each priced here or alongside here and each was refuted.

## Build

```sh
clang -O2 -Wall -Wextra -o read_sweep read_sweep.c
clang -O2 -Wall -Wextra -fblocks -o read_batch read_batch.c   # -fblocks: dispatch
```

## read_sweep — one file, one depth, one cell

```
read_sweep <file> <chunk> <stride> <depth> <totalMiB> <seed> [label] [startSlot] [spanSlots]
```

Reads `chunk` bytes at stride-aligned offsets from a single layer file with
`depth` threads and reports achieved bandwidth plus the per-read latency
distribution. Offsets are drawn *without replacement* from the slot grid, so a
cell never re-reads its own bytes and its rate is not flattered by its own
reuse — the failure [METH-14](../experiments/summaries/09-validation-and-measurement-lessons.md#meth-14)
records.

The optional `startSlot`/`spanSlots` confine the cell to a window of the slot
grid. Two cells on disjoint windows of one file touch disjoint bytes, so both
are cold without a purge.

`chunk` must be a multiple of 16384; `stride` must be too.

## read_batch — the layer walk, without the engine

```
read_batch <listfile> <chunk> <stride> <cycle> <rounds> <seed> [label]
```

Walks a list of files in order (one path per line, `#` comments allowed),
issuing `m` concurrent reads from each behind a barrier before moving to the
next file. That is the engine's read shape: one `concurrentPerform` batch per
layer, 40–48 strictly serialized batches per step, each of 3–5 reads from that
layer's own file.

- `cycle` is a comma-separated reads-per-file list cycled across the walk
  (`3,3,4`), because the engine's per-layer miss count is not an integer —
  3.37 on Qwen 3.6, 5.12 on 3.8.
- `SORT=1` sorts each batch's offsets ascending before dispatch, which is the
  order a plan carries if the router hands its experts over in expert-id order.
- Every slot touched is marked in a per-file bitmap and each round reports how
  many of its reads were **novel**, so reuse is visible rather than assumed.
- Offsets are distinct within a batch and free to repeat across rounds, which
  is the engine's own reuse shape.
- One warm read per file at offset 0 takes open/first-read cost out of the
  first counted batch. It does not make round 1 warm — one slot of 512 is 0.4%
  of a file.

The barrier is `dispatch_apply` on a concurrent queue, which is what the
engine's `DispatchQueue.concurrentPerform` resolves to. This is not a detail.
A hand-rolled spin/atomic handshake was written first and its own wake-up cost
showed up as the device's — about 27% on identical offsets (3.6 at 1.572
against 2.054 GB/s, 3.8 at 1.825 against 2.406), which is the size of several
effects this harness exists to measure. Use the engine's primitive, or the
comparison is between two barriers rather than between two read shapes.

Build a list from an install with:

```sh
jq -r '.layers[] | .file' models/<install>.finch/packed_experts/layout.json \
  | sed 's|^|models/<install>.finch/packed_experts/|' > /tmp/list.txt
```

(layout.json's `file` field is relative to `packed_experts/`; `expertStride`
is the stride.)

## capture-traces.sh — the engine's own offsets

```sh
tools/read-sweep/capture-traces.sh [output-dir] [label-prefix]
```

Runs both installs through the engine with `FQ_EXPERT_TRACE` set, four times
interleaved (3.6, 3.8, 3.6, 3.8) on one prompt, and writes `trace-<label>.txt`
plus the io counters to `summary.txt`. Defaults to `/tmp/fq-trace`.

Capture and replay must come from the **same session** — the traces are only
half of a comparison, and the counters are the other half. Traces captured
here are committed under `traces/` (two per install, `t1`/`t2`), which is what
`replay_dest.py` reads by default.

## replay_dest.py — the real trace, both installs, one process

```sh
tools/read-sweep/replay_dest.py [--rounds N] [--only 3.6|3.8] [--conds 1,3]
                                [--depth36 N] [--depth38 N] [--gap-ms MS]
                                [--ballast-gib GiB] [--evict-gib GiB]
                                [--tag NAME] [--trace36 PATH] [--trace38 PATH]
```

Replays the committed traces at each install's own engine read concurrency
(3.26 → 3, 5.30 → 5), one `preadv` per read, summing the per-batch
`pool.map` spans — that sum is exactly what `io_read_wall_ms` measures
(`fanout + span + drain`; `io_read_identity` reports `exact` when it tiles).
Both installs run inside each round and the arm order alternates by round
parity, so a drift within the session cannot be mistaken for a difference
between the installs.

The five conditions are crossed on two axes — page cache allowed or bypassed,
and destination warm or engine-shaped:

```
1  allowed,  warm dest            IO-17's condition, reproduced
2  bypassed, warm dest            IO-17's condition, page cache removed
3  allowed,  slot dest            the engine's own condition AND its shape
4  bypassed, slot dest            both ablations together
5  bypassed, slot dest, depth 3   pool width, held apart from the above
```

Condition 3 is the one that matters and the one the older table never ran: the
engine does not bypass the page cache, so pricing a 1.98 GiB rotating slot
destination with the cache *off* is not the engine's condition. The slot
destination is built the way the engine builds it — `PreadExpertStreamer` is
constructed inside the layer loop (`Model.swift:689`), so it is one ring per
layer, `slotCount` page-aligned anonymous buffers each, pre-faulted once at
init: 40x16x1.69 MiB = 1.05 GiB on 3.6, 48x16x2.64 MiB = 1.98 GiB on 3.8.
Minor faults are counted per pass, so pages evicted and re-faulted *during* a
pass show up directly rather than being inferred from a millisecond delta.

## paired-engine-replay.sh — the engine beside a replay of its own trace

```sh
tools/read-sweep/paired-engine-replay.sh <install> <trace> <depth> [rounds] [conds]
```

Alternates one engine window and one replay, round by round, so every engine
number is read against a replay that ran within a minute of it. The engine run
includes model load and takes ~10 s on 3.6, ~30 s on 3.8, so the alternation
can be tight. Condition order flips with round parity: without that, the first
condition in the list is always the cold pass on 3.6 and the second always the
warm one, which is 25–56 against 11.7 ms/step of pure method error.

It reports the engine's own split — `io_read_wall_ms/step`,
`io_thread_wall_ms/step`, `io_conc` — so **per-read service time**
(`io_thread_wall / misses`) can be compared and not just the window. Two arms
at the same in-flight bytes and different service times differ in the read
path, not in the queueing.

## Output

`read_sweep` prints per-cell bandwidth and latency percentiles and ends with
`win=<start>+<span>`. `read_batch` prints one line per round —
`reads`, `novel`, `MiB`, `ms`, `GBps`, mean batch span and mean read latency —
and a `TOTAL` line carrying `r1_GBps`/`r1_batch` and `rN_GBps`/`rN_batch`,
because on this box round 1 is the only cold one and later rounds are not.

## What these measured

Recorded in full at
[IO-16](../experiments/summaries/01-model-install-and-expert-io.md#io-16) and
[IO-17](../experiments/summaries/01-model-install-and-expert-io.md#io-17).
The short version:

| Question | Answer |
| --- | --- |
| Is the per-layer barrier the cost? | No — about 7% (48 batches against one file: 2.759 against a continuous reader's 2.966 GB/s). |
| Is the 48-file walk the cost? | No — fewer files is *slower*: 3.163 (48×5), 3.003 (24×10), 2.883 (16×15) at fixed reads and bytes per round. |
| Does the engine's batch structure reproduce the read penalty? | No — it reproduces the shape (133/249 reads per round against 134.8/245.7) and **inverts the ordering**: 1.917–2.482 GB/s where the engine reads 4.736/3.167. |
| Is the access pattern the explanation? | No — replayed from the engine's own `FQ_EXPERT_TRACE`, 3.8's trace is *faster* than 3.6's by 1.34–1.49x, the opposite of what the engine does. |
| Does the 1.98 GiB slot destination account for it? | No — +1.5 ms/step in the engine's own condition, 1.7% of the 86 ms/step it would have to explain ([IO-18](../experiments/summaries/01-model-install-and-expert-io.md#io-18)). |
| Does inter-batch idle (the drive re-ramping)? | No — a 3 ms idle costs the next batch 0.28 ms against the 1.3 ms needed. |
| Does memory pressure? | No — flat at 145–164 ms/step with 0–6 GiB of ballast held resident; the ring's ~80,000 re-faults/pass are hidden inside the drive wait. |
| Does the 3.8-only PLE stream? | No — 3.84 ms/step, open-dominated, too few bytes to matter. |

## Read this before quoting a number

Six things bit this harness, and each one is a way to be wrong:

1. **Hold the denominator.** The file-count sweep was first read as showing a
   per-file cost. It does not: bytes per round was held fixed and reads per
   round was not, so the reading came from the ratio rather than the walk.
2. **A replay pass is warm unless you force it cold, and on 3.6 that is 6.3x.**
   `F_NOCACHE` is not honoured on this volume — the "bypassed" arm still reads
   at 27.8 GB/s, which is RAM — and `purge` needs a password. So the only
   cold pass is the first one, and one same-file control gave 2.759, 4.733 and
   7.454 GB/s on rounds 1, 2 and 3. The size of the effect is the size of the
   trap: on 3.6 the *same condition* measures **56.71 cold against 8.99 warm**
   in one session, which is larger than every effect measured here. Pass
   `--evict-gib 17` to stream unrelated data through the cache before each
   pass. IO-17's headline — "the engine beats its own replay by 1.20–1.33x on
   3.6" — was that trap, and the engine is at **parity** with a cold replay.
3. **The drive drifts within a session, by more than most effects here.**
   One identical synthetic cell gave 1.825 and 3.163 GB/s twenty minutes apart
   while the engine agrees with itself to 0.3% inside a run. Interleave the
   arms, run each round in both orders, and never subtract a rate measured in
   a different session — see
   [METH-15](../experiments/summaries/09-validation-and-measurement-lessons.md#meth-15).
4. **Synthetic offsets are not the workload.** Uniform-random offsets are
   slower than the router's real choices, so a synthetic harness comparing an
   install against the device will acquit the engine for free. Capture
   `FQ_EXPERT_TRACE` and replay that instead — the offsets are the one thing a
   harness must not invent.
5. **Alternate the condition order.** With `--rounds 1` and no parity flip,
   the first condition is always the session's cold pass and the second always
   a warm one. On 3.6 that alone is 25–56 against 11.7 ms/step, which is
   larger than the effect being measured. Both drivers here flip on round
   parity; if you add a third, flip it too.
6. **A window is not a queue depth.** `io_conc` is misses per layer — the
   batch width — not reads in flight. `executeExpertCachePlan` runs once per
   layer behind a `concurrentPerform` barrier, so a step is 40–48 strictly
   serialized batches. Compare two arms at equal `io_conc` or the comparison
   is between two queueing regimes, not two read paths.

The C tools open their files without `F_NOCACHE`, matching how
`PreadExpertStreamer` opens them. That is deliberate; it is the engine's
condition, not the fastest one available.
