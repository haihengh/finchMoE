# ple2bit HumanEval sweep — why it crawls, and what the harness does about it

Findings from 2026-10-12/13 while driving the Qwen3.8-125B-ple2bit cell on the
16 GB M4 mini, extended 2026-10-15/16 when the sweep finished. The durable
content here is the I/O budget of a 3.8 install, the two misdiagnoses it caused,
the harness changes that came out of it, the finished run's numbers, and the
**era regression** that has dominated every measurement since 2026-10-01.

**The sweep is DONE — 2026-10-15 16:44, 164/164, scored 17:13: base pass@1
0.835 / base+extra 0.799.** The partial everyone was reading (0.955 / 0.894 at
n=66) did not survive the full set. See §9 (the run) and §10 (the era).

## 1. Summary

- A "stalled" request is almost always a **slow** one. The engine streams routed
  experts off disk for every token; when the working set is not resident it runs
  at whatever the SSD feeds it and logs nothing — while burning 35-48% CPU.
- The harness read that silence as a hang, killed the request at 900 s, and
  restarted the engine. Each restart re-read `model_weights.bin` (3.88 GB) and
  evicted the expert cache, so the retry started cold and slow too. That loop
  manufactured two fake "poison tasks" and kept the box in its slow regime.
- Fixed by making the stall verdict a **progress** verdict (CPU burn over a
  window, not silence) with a 4 h hard ceiling, and by never restarting the
  engine to "fix" slowness. Sweep rate immediately became variable but always
  forward: 0.12-0.19 tok/s slow, 3.4-3.6 tok/s fast, interleaved.
- **The external SSD is not the villain.** Cache-bypassed, its random-read
  ceiling for the engine's own pattern is 1.8-1.9 GB/s; the *internal* NVMe
  manages 2.3 GB/s — only ~1.25x better. Both are at or below the ~2.6 GB/s the
  engine needs for full speed, which is why performance swings so widely: cache
  hits are what put it over the line.

## 2. What it looked like

Two HumanEval problems were written off as deterministically broken:

- **HumanEval/38** — eleven consecutive killed attempts on 2026-10-10 across five
  full `MAX_STALLS_PER_TASK` cycles, all "parked" in
  `-[_MTLCommandBuffer waitUntilCompleted]` under `encodeQwen38DecodeLayer`.
- **HumanEval/41** — six consecutive killed attempts on 2026-10-12, same shape,
  once in prefill (`encodeQwen38PrefillLayer`).

Both verdicts were wrong. With a quiet box, 38 streams at 3.7 tok/s and 41
finishes in 135 s (464 tokens, `prepared prompt=201`). The sweep itself
completed 41 and 42 while "failing" 41.

## 3. The evidence, in the order it arrived

1. **Standalone reproduction.** `scratch/diag-task38.py` sends one request
   exactly as the harness does and streams it. 41 completed in 134.8 s.
   *Parity matters*: evalplus does not send the bare dataset prompt, it builds
   `instruction_prefix + f"\n```python\n" + prompt.strip() + "\n```"` (see
   `evalplus/provider/openai.py:23`; `humaneval_gen.py` adds only the system
   message on top). The bare prompt prefills as **177 tokens**, the wrapped one
   as **201** — the first version of the diagnostic "completed cleanly" against
   the wrong request. Verify with the server log's `prepared prompt=N`.
2. **A task that could not be poisoned.** Task 38 — the original poison task —
   streamed at 3.7 tok/s once given a quiet box; the `SKIP_TASKS` escape that had
   been built for it now defaults to empty.
3. **The stall was alive.** During the 1060 s that a 300 s watchdog took to fire,
   the server burned CPU continuously (38-50%), and system I/O was jammed badly
   enough that `iostat` needed 14 minutes to emit four samples — including the
   watchdog's own file reads. A parked decode burns ~0%.
4. **The fast/slow split is real and per-request.** Same server, back to back:
   212 s for 768 tokens (3.62 tok/s) then 1077 s for 406 tokens (0.38), then
   149 s for 531 tokens (3.57). Disk sampling during slow requests shows
   1.0-1.7 GB/s of ~364 KB reads; during fast ones, tens of MB/s.
5. **Memory is at the edge but self-inflicted.** With the engine loaded: ~85 MB
   free, ~6.7 GB wired (`FINCHMOE_MLOCK_RESIDENT=1` pins `model_weights.bin`),
   1.2-1.9 GB in the compressor, no swap. The same state exists during fast runs,
   so it does not by itself explain the split.

## 4. The I/O budget of a 3.8 install

From `manifest.json` and `packed_experts/layout.json` of this install:

| quantity | value |
|---|---|
| layers | 48 |
| experts per layer | 512 |
| top-k routed per layer | 10 |
| bytes per expert (`expertStride`) | 1,540,096 (1.47 MiB) |
| **expert bytes read per token** | 48 x 10 x 1.47 MiB ≈ **0.74 GB** |
| read size the streamer issues | ~364 KB chunks (matches `iostat` KB/t) |
| **sustained rate for 3.5 tok/s** | ≈ **2.6 GB/s** |

> An earlier estimate in this investigation said ~1.2 GB/token and ~4.3 GB/s.
> That came from assuming 256 experts per layer at 3 MB each; the manifest says
> 512 experts at 1.47 MiB. The corrected figure is ~0.74 GB/token, ~2.6 GB/s —
> and it matters, because 2.6 GB/s is *within reach* of the hardware, which is
> exactly why the system oscillates instead of always being slow.

Measured healthy rate: 768 tokens in 212.4 s = 3.62 tok/s; 531 in 148.8 s = 3.57.
Measured starved rate: 474 tokens in 4054 s = 0.12 tok/s; 576 in 4875 s = 0.12;
406 in 1077 s = 0.38.

Note the amplification: a starved request at 0.12 tok/s with ~1.2 GB/s of disk
traffic implies **~10 GB read per token** — roughly 13x the 0.74 GB minimum. That
is not "cold cache"; something is re-reading or over-reading (see §8).

## 5. Device ceilings and the internal-storage A/B

Cache-bypassed (`F_NOCACHE`) random 375 KB reads, the engine's access shape,
both devices idle, same seed:

| threads | external SSD (disk6) | internal NVMe (disk0) | ratio |
|---|---|---|---|
| 1 | 0.87 GB/s (2266 IOPS) | 1.04 GB/s (2713 IOPS) | 1.20x |
| 8 | 1.90 GB/s | 2.33 GB/s | 1.22x |
| 16 | 1.78 GB/s | 2.31 GB/s | 1.30x |
| 32 | 1.83 GB/s | 2.27 GB/s | 1.24x |

Sequential reads on the external drive sustain **3.4-3.5 GB/s** (three passes of
19.7 GB, no degradation), so it is not thermally collapsing; it is simply a
~1.8-1.9 GB/s device for this random pattern. Neither device reaches the
~2.6 GB/s full-speed requirement on its own — cache residency is what closes the
gap, which is the whole explanation for fast vs slow requests.

A full copy of the install went to `/Users/hai/finch-models/` for this A/B:
186/186 files copied and **verified byte-identical against the receipt's own
SHA-256s** (`scratch/verify-internal-copy.py`, 68.74 GiB hashed). The copy's
`verified-install.json` needed its `modelDirectoryPath` rewritten to the new
location — `--verify trusted-install` compares that field against the resolved
physical path (`Sources/FinchMoE/Infrastructure/ModelIO/VerifiedInstallReceipt.swift:218`);
the original receipt is preserved as `.from-external-disk`. The internal copy
loads and reports `trusted-install` integrity.

Expectation for running from internal storage: **~1.25x on the read path**, i.e.
a slow regime of maybe 0.2-0.3 tok/s instead of 0.12-0.19. Worth having, not a
cure.

**Measured outcome (2026-10-13/14, the sweep's 97-problem internal leg):** the
expectation did not materialize — slow requests came in at 0.11-0.28 tok/s
against 0.12-0.19 before, fast ones at the same 2.4-3.3, and the sweep's slowest
request of all (7,529 s) ran *on internal storage*. The A/B is flat, which also
retires the "external drive at 93% full" hypothesis: a 55%-full internal NVMe
behaves identically. The drive is not the lever (§9, §10).

## 6. Harness changes

In `archive/humaneval_evalplus/run_server_cell_resilient.sh` (archive/ is
gitignored — these live on disk only):

- **`is_stalled` now decides on progress.** Past `STALL_SECS`, it samples the
  server's CPU time (`cpu_seconds`) over `PROGRESS_WINDOW` (20 s) and treats
  `PROGRESS_MIN_CPU` (2 s) of burn as alive. A parked decode sits at ~0% and is
  still caught.
- **`ALIVE_CEILING_SECS` (default 4 h)** so nothing is "alive" forever. Sized
  from the worst observed case: 768 tokens at 0.07 tok/s ≈ 11,000 s. It is read
  once per *driver process* — a new round inside the same driver keeps the old
  value — so the header line prints `ceiling=Ns` to make a stale value visible.
- **`SKIP_TASKS`** (default empty) steps over an id that genuinely cannot be
  generated. Any id it names is carried into scoring as an explicit exclusion.
- Verdict notes are rate-limited to one per 5 min; a slow request can hold that
  state for an hour and a line every 30 s buries the log.

In `archive/humaneval_evalplus/patch_evalplus.py`:

- **Missing-problem patch (v2).** evalplus asserts every dataset problem has a
  sample. With `FINCHMOE_EXPECT_MISSING` naming the ids (bare `38` or
  `HumanEval/38`), scoring proceeds and records `finchmoe_excluded_problems` in
  the results json; any *other* gap still aborts. Both branches verified.

Supporting tools (all in `scratch/`):

- `watch-ple2bit-rates.sh` -> `ple2bit-rates.csv`: per-request duration,
  completion tokens, tok/s. This is the data whose absence let the "poison task"
  story stand for two days.
- `watch-ple2bit-io.sh` -> `ple2bit-io.csv`: the external SSD's read rate
  (`iostat -d disk6`; the volume is `/dev/disk7s1` and `iostat -d disk7` fails —
  use the physical disk).
- `diag-task38.py`: one-request reproducer with prompt parity checking and
  streaming deltas (`--no-stream`, `--no-mlock`, `--task N`).
- `resume-ple2bit.sh` / `resume-ple2bit-internal.sh`: resume against the
  external install, or the internal copy.
- `verify-internal-copy.py`: hash a copied install against its receipt.

## 7. Operating rules

1. **Never restart the engine to "fix" a slow request.** Each restart re-reads
   3.88 GB of weights and evicts the expert cache, guaranteeing the retry is
   cold too. This is the single rule that matters; everything else is detail.
2. **Judge progress by CPU burn, not by log silence.** `sample <pid>` helps:
   a parked engine sits in `-[_MTLCommandBuffer waitUntilCompleted]`.
3. **Do not score the live sample jsonl in place** — its `_eval_results.json`
   is the until-done runner's DONE marker and triggers the HF upload. Score a
   copy (see `scratch/ple2bit-partial/`).
4. Expect the sweep to swing between ~0.12 and ~3.6 tok/s per request, and never
   to be predictable from the task id.

## 8. Open questions

- **Read amplification in the starved regime** (13x over the minimum, §4). Two
  candidates: the pending-routed pipeline re-fetching experts whose slots were
  reused before the deferred read completed, and over-eager readahead advice
  (`adviseExpertMisses` / `F_RDADVISE`) pulling in data the token never uses.
  Next measurement: count preads and advised ranges per token and compare with
  0.74 GB.
- **What decides residency.** Whether the model's own LFU slot cache (16 slots
  per layer, `PreadExpertStreamer`) or the OS page cache is doing the work when
  requests run fast is unknown, and it decides whether tuning slots helps.
- **The external drive is at 93% full** (135 GiB free of 1.8 TiB). **Retired
  2026-10-14:** the internal-storage A/B (§5) ran the same install off a
  55%-full NVMe and produced identical slow-regime numbers, so fill level buys
  nothing measurable for reads.
- **The era regression's root cause** (§10). The decisive control — the current
  engine running the *int4* ple4bit install, ~10 HumanEval problems against its
  Sep 15 baseline of 147 s/request — had not been run when this was written.
  Fast today would implicate the int3-era engine commits; slow would point at
  the OS/engine-wide change.
- **Product exposure.** The app runs the same engine on the same class of
  machine; a request that takes 80 minutes to answer is a hang from the user's
  point of view — and the sweep's worst was 125 minutes (7,529 s). If 3.8
  installs are to be used on 16 GB machines, the slow regime is a product
  question, not just a benchmark one.

## 9. The full sweep, finished (2026-10-15)

| | ple2bit | ple3bit | ple4bit |
|---|---:|---:|---:|
| wall clock | **154 h** | 47 h | 6.4 h |
| per problem | 56 min | 17 min | 2.3 min |
| HumanEval base / plus | **0.835 / 0.799** | 0.933 / 0.909 | 0.945 / 0.909 |
| requests ≥ 30 min | 30/70 | 2/11 | 0/144 |

Same protocol and same 164 problems throughout; per-request stats come from
each run's surviving server log (the 3.6 and ple3bit logs cover the final round,
so their means are samples).

The last 97 problems ran off the internal SSD (10-13 08:08 → 10-15 16:44,
56.6 h) with **zero stall-kills and one engine process for the whole leg** — the
corrected watchdog (§6) never had to fire. The final score is 0.835 base /
0.799 base+extra: the 66-problem partial that had been circulating (0.955 /
0.894) was optimistic, and the "keep the 2-bit variant" call it supported no
longer holds on quality grounds alone (it sits ~10 pp below 3-bit, the same
penalty 2-bit cost on the 3.6 family).

Why 154 h when the same family took 47 h at 3 bits and 6.4 h at 4 bits — two
compounding causes, in order of size:

1. **The era regression (§10).** Every run after 2026-10-01 crawls; this run
   lived entirely inside the degraded era.
2. **The 2-bit model does not stop.** Raw completions average 1,746 chars
   against ple3bit's 910; 47% exceed 2,000 chars; 46% of requests ran to the
   768-token cap — and every marathon request in the sweep is a cap-hit
   (768 tokens at 0.12-0.13 tok/s ≈ 100+ minutes). The scored jsonl hides this:
   evalplus strips the trailing ramble, so ple2bit's stored solutions look
   *shorter* than ple3bit's (median 568 vs 710 chars). **Never compare runs by
   the scored files — use the raw ones.**

Fast/slow texture on the internal leg (25 logged requests): 10 fast (≥ 2.5
tok/s), 6 mid, 9 slow (< 0.3). At least six times a fast streak began
immediately after a long (≥ 30 min) stall — consistent with the stall's
over-read pulling the working set into cache, and worth testing against §4's
amplification question. The slow mode hits *prefill* too (2,940 tokens in
1,500 s = 2.0 tok/s against 18.5 healthy): it is the whole streaming path, not
decode-specific.

## 10. The era regression — every run since 2026-10-01 crawls

While dating the sweep's slowness, every HumanEval run on this box was profiled
from its surviving server log. The split is total:

| run | dates | span | mean request | ≥ 30 min |
|---|---|---:|---:|---:|
| qwen38 baseline | Sep 14 | — | 146 s | 0 |
| ple4bit | Sep 15 | 6.4 h | 147 s | 0 |
| qwen38 int8gemm | Sep 17 | 5.4 h | 119 s | 0 |
| qwen36 abliterated | Sep 26-27 | 11.1 h | 49 s | 0 |
| qwen38 abliterated | Sep 27 | 6.7 h | 157 s | 0 |
| *macOS 27.0.1 installs Oct 1, 16:01* | | | | |
| qwen36 3-bit | Oct 4-5 | 28.9 h | ~635 s | yes |
| qwen36 2-bit | Oct 6 | 20.9 h | ~187 s | yes |
| ple3bit | Oct 7-8 | 47 h | 550 s | yes |
| ple2bit | Oct 9-15 | 154 h | 2,062 s | 30/70 |

Five consecutive runs with **zero** requests past 30 minutes, then never again.
The onset window (Sep 27 17:25 → Oct 4 17:18) contains exactly two changes: the
macOS update, and the engine's int3-codec-era commits (Oct 4 13:06, four hours
before the first slow run).

**The codec is the weaker suspect.** Routing — which experts a token reads — is
decided by the resident router and is identical across codecs; the narrower
codecs read *fewer* bytes per token (0.74 / 1.04 GB against 1.33); and the fast
mode still reaches 2.2-2.7 tok/s under the new codecs, so the decode kernels
are not slow. What changed is the cache/IO *regime*, not compute.

**The OS change is the sharper suspect**, because the engine's speed is
measurably a page-cache phenomenon — F_NOCACHE *loses* 12% tok/s (§3), the
streamer's reads are 41.7% repeats with a median reuse distance of ~4 GiB — and
because the regression is device-independent: a 55%-full internal NVMe is
exactly as slow as the 93%-full external one (§5). A VM or readahead policy
change would look precisely like this. Ruled out: drive fill (§5), Time Machine
(off, no local snapshots), Spotlight (disabled on the model volume), swap (0),
background processes (nothing above 0.5%).

The decisive control has not been run: the current engine on the *int4* ple4bit
install, ~10 problems against its Sep 15 baseline (147 s/request). Fast today
would implicate the int3-era engine commits; slow would point at the
OS/engine-wide change. It is the first entry in §8 and it gates any claim about
the cause.
