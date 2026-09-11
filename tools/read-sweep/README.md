# read-sweep

Two harnesses that read packed-expert files the way the engine does, so the
drive can be priced without the engine in the loop. They exist because the
`io` window is the largest single span in a decode step, and because the
engine's own counters cannot separate "the drive is slow here" from "the
engine is slow at reading here" — a distinction that decided the whole
3.6-versus-3.8 investigation (`docs/OPTIMIZATION_PLAN.md` §2.1,
[IO-16/IO-17](../experiments/summaries/01-model-install-and-expert-io.md#io-16)).

Neither is wired into the runtime and neither should be. They are measuring
instruments, and everything they produced is a negative result: request size,
read depth, batch structure and destination memory were each priced here or
alongside here and each was refuted.

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

## Read this before quoting a number

Four things bit this harness, and each one is a way to be wrong:

1. **Hold the denominator.** The file-count sweep was first read as showing a
   per-file cost. It does not: bytes per round was held fixed and reads per
   round was not, so the reading came from the ratio rather than the walk.
2. **Round 1 is the only cold round, and it is not fully cold.** A 1352 MiB
   file stays resident across rounds — one same-file control gave 2.759, 4.733
   and 7.454 GB/s on rounds 1, 2 and 3 — and even round 1 is partly served by
   the batches that preceded it.
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

Both tools open their files without `F_NOCACHE`, matching how
`PreadExpertStreamer` opens them. That is deliberate; it is the engine's
condition, not the fastest one available.
