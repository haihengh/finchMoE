#!/usr/bin/env python3
"""Price the slot destination working set on the engine's own traces.

IO-17 (`docs/experiments/summaries/01-model-install-and-expert-io.md`) replayed
both installs' real `FQ_EXPERT_TRACE` sequences through one harness, interleaved
and round-alternated, and found the engine beats its own replay by 1.20-1.33x on
3.6 and *loses* to it by 1.31-1.43x on 3.8. Both arms of that comparison ran
into warm, never-re-faulting destinations, and the engine does not:

    the engine holds one PreadExpertStreamer per layer (`Model.swift:689` is
    inside the layer loop), each with its own `slotCount` slots, allocated once
    at init with posix_memalign and wrapped `makeBuffer(bytesNoCopy:)`. At 16
    slots that is

        3.6   40 x 16 x 1,769,472 B = 1.05 GiB
        3.8   48 x 16 x 2,768,896 B = 1.98 GiB

    of rotating anonymous pages on a 16 GB box.

The plan priced part of this already (`docs/OPTIMIZATION_PLAN.md` §2.1, the
older 9,083-read trace): engine-shaped destinations with the page cache
**bypassed** cost 149.6 against 130.3 ms/step warm, +19.3 ms/step. But the
engine does not bypass the page cache -- it opens its layer files without
`F_NOCACHE` -- so the two effects were never priced together. That matters
here and not on the old trace, because the interaction is the whole point:
2 GiB of *resident destinations* competes with the page cache, and 3.8's
11 GiB working set already does not fit (same section). The cell that matches
what the engine actually does is `allowed + slot dest`, and it is the one cell
the earlier table never ran.

Conditions, per install, at that install's own engine read concurrency:

  1  allowed,  warm dest            -- IO-17's condition, reproduced
  2  bypassed, warm dest            -- IO-17's condition, page cache removed
  3  allowed,  slot dest            -- the engine's own condition AND its shape
  4  bypassed, slot dest            -- both ablations together
  5  bypassed, slot dest, depth 3   -- pool width, held apart from the above

Condition 3 is the one that answers the question. On 3.6 conditions 1 and 5
are near-duplicates (depth 3 IS its native width, and its ring is 1.05 GiB);
they are kept as a within-pass noise control -- the spread between them is
what a null result has to be read against.

Scored on the **sum of per-batch read spans** -- what `io_read_wall_ms`
measures. The engine's own split is `fanout + span + drain` (`io_read_identity`
reports `exact` when it tiles), so the sum of per-layer batch walls is the
right analogue, not "span" alone; this harness times each layer's `pool.map`
and sums, which is that quantity. The replay has no GPU work between batches,
so span-sum and wall should agree; both are printed, and the gap between them
is Python loop overhead, not the drive.

Minor faults are counted per pass (`ru_minflt`). The ring is pre-faulted before
timing, so a nonzero count in a slot-dest condition means pages were evicted
and re-faulted *during* the pass -- which is the proposed mechanism showing up
directly rather than being inferred from a millisecond delta.

The ring is allocated and pre-faulted per (round, install), outside every timed
span, and freed before the next install runs, so only one ring is live at a
time -- 1.98 GiB at worst, which is the condition being tested, not an artifact
of the harness holding two.

Everything is round-alternated (METH-06) and both installs run inside each
round, because this drive drifts within a session by more than the effect
being measured (METH-15). Report per-round values; do not subtract a figure
measured in another round ordering.

  usage: replay_dest.py [--rounds N] [--depth36 N] [--depth38 N]
                        [--tag NAME] [--trace36 PATH] [--trace38 PATH]
                        [--only 36|38] [--model36 PATH] [--model38 PATH]

Pass --trace38 /tmp/trace38.txt --depth38 10 to reproduce the published table.
"""
import argparse, fcntl, json, mmap, os, resource, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PAGE = 16384
SLOTS = 16          # RuntimeConfiguration.expertCacheSlots, as captured
STEPS = 31          # decode forwards in the captured runs
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/Volumes/samsung 2t/code/finchMoE/models"

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--depth36", type=int, default=3)
ap.add_argument("--depth38", type=int, default=5)
ap.add_argument("--tag", default="new")
ap.add_argument("--only", default=None)
ap.add_argument("--conds", default=None,
                help="comma-separated 1-based condition indices; default all")
ap.add_argument("--ballast-gib", type=float, default=0.0,
                help="hold this much anonymous memory resident for the whole "
                     "run, to price memory pressure as a variable")
ap.add_argument("--evict-gib", type=float, default=0.0,
                help="before every pass, stream this much unrelated data "
                     "through the page cache to force the next pass cold")
ap.add_argument("--evict-from", default=None,
                help="packed_experts dir to evict with; default the other install")
ap.add_argument("--gap-ms", type=float, default=0.0,
                help="idle this long between layer-batches, standing in for the "
                     "engine's GPU work, to test whether the drive re-ramps")
ap.add_argument("--model36", default=f"{BASE}/Qwen3.6-35B-A3B-4bit.finch")
ap.add_argument("--model38", default=f"{BASE}/Qwen3.8-Flash-Next-125B.finch")
ap.add_argument("--trace36", default=f"{HERE}/traces/trace-36-t1.txt")
ap.add_argument("--trace38", default=f"{HERE}/traces/trace-38-t1.txt")
A = ap.parse_args()

# Engine read concurrency, measured: io_conc 3.26 on 3.6, 5.29 on 3.8.
ARMS = [("3.6", A.model36, A.trace36, A.depth36),
        ("3.8", A.model38, A.trace38, A.depth38)]
if A.only:
    want = A.only.replace(".", "")          # accept "38" and "3.8"
    ARMS = [a for a in ARMS if a[0].replace(".", "") == want]
    if not ARMS:
        sys.exit(f"replay_dest: --only {A.only} matched no arm")

CONDS = [
    ("allowed,  warm dest",          (0, False, None)),
    ("bypassed, warm dest",          (1, False, None)),
    ("allowed,  slot dest",          (0, True,  None)),
    ("bypassed, slot dest",          (1, True,  None)),
    ("bypassed, slot dest, depth 3", (1, True,  3)),
]
if A.conds:
    keep = {int(x) for x in A.conds.split(",")}
    CONDS = [c for i, c in enumerate(CONDS, 1) if i in keep]
    if not CONDS:
        sys.exit(f"replay_dest: --conds {A.conds} selected nothing")


def load(model, trace_path):
    d = os.path.join(model, "packed_experts")
    layout = json.load(open(os.path.join(d, "layout.json")))
    stride, nlayers = layout["expertStride"], layout["numLayers"]
    offs = [[e["offset"] for e in layout["layers"][L]["experts"]]
            for L in range(nlayers)]
    paths = [os.path.join(d, layout["layers"][L]["file"]) for L in range(nlayers)]
    raw = [int(x) for x in open(trace_path)]
    batches, i = [], 0
    while i < len(raw):
        c = raw[i + 1]
        batches.append((raw[i], raw[i + 2:i + 2 + c]))
        i += 2 + c
    return stride, nlayers, offs, paths, batches


EVICT_FDS = []
if A.evict_gib > 0:
    src = A.evict_from or os.path.join(
        BASE, "Qwen3.6-35B-A3B-4bit.finch", "packed_experts")
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".bin"))
    if not files:
        sys.exit(f"replay_dest: no .bin files under {src} to evict with")
    EVICT_FDS = [os.open(f, os.O_RDONLY) for f in files]


def evict(gib, stride):
    """Stream unrelated bytes through the cache so the next pass is cold.

    `sudo purge` is not usable (needs a password) and `F_NOCACHE` is not
    honoured on this volume, so the only way to get a genuinely cold read is to
    read enough other data to push the trace's pages out. 17 GB against 16 GB
    of RAM leaves nothing behind.
    """
    want = int(gib * 2**30)
    buf = bytearray(stride)
    done = 0
    for fd in EVICT_FDS:
        end = os.fstat(fd).st_size
        pos = 0
        while pos < end and done < want:
            got = os.preadv(fd, [buf], pos)
            if got <= 0:
                break
            pos += got
            done += got
        if done >= want:
            break


def make_ring(stride, nlayers):
    """The engine's slot ring: one page-aligned anonymous buffer per (layer,
    slot), all resident before timing starts."""
    ring = {}
    for L in range(nlayers):
        for s in range(SLOTS):
            b = mmap.mmap(-1, stride)
            for p in range(0, stride, PAGE):   # pre-fault
                b[p] = 0
            ring[(L, s)] = b
    return ring


def free_ring(ring):
    for b in ring.values():
        b.close()


def replay(stride, nlayers, offs, paths, batches, depth, nocache, use_ring,
           ring, alt_depth=None):
    """One pass. Returns (span_sum, wall, faults, ring_bytes)."""
    opened = [os.open(p, os.O_RDONLY) for p in paths]
    for fd in opened:
        fcntl.fcntl(fd, F_NOCACHE, nocache)

    tl = threading.local()

    def dest_for(L, s):
        if use_ring:
            return ring[(L, s % SLOTS)]
        buf = getattr(tl, "buf", None)
        if buf is None:
            buf = bytearray(stride)      # warm: touched once, never re-faulted
            tl.buf = buf
        return buf

    width = alt_depth if alt_depth else depth

    def read_one(job):
        L, e, s = job
        if os.preadv(opened[L], [dest_for(L, s)], offs[L][e]) != stride:
            raise RuntimeError("short read")

    span_sum = 0.0
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    wall0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=width) as pool:
            for L, miss in batches:
                if not miss:
                    continue
                t0 = time.perf_counter()
                list(pool.map(read_one, [(L, e, s) for s, e in enumerate(miss)]))
                span_sum += time.perf_counter() - t0
                if A.gap_ms > 0:
                    time.sleep(A.gap_ms / 1000.0)   # outside the span on purpose:
                                                    # io_read_wall is the sum of
                                                    # per-batch spans, not a
                                                    # window around the step
    finally:
        wall = time.perf_counter() - wall0
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0
        for fd in opened:
            os.close(fd)
    return span_sum, wall, faults


BALLAST = None
if A.ballast_gib > 0:
    n = int(A.ballast_gib * 2**30)
    BALLAST = mmap.mmap(-1, n)
    for p in range(0, n, PAGE):        # resident, not just reserved
        BALLAST[p] = 0
    del n

print(f"# replay_dest  tag={A.tag}  rounds={A.rounds}  ballast={A.ballast_gib} GiB")
print("loading traces and layouts...")
loaded = {}
for name, model, trace, depth in ARMS:
    stride, nlayers, offs, paths, batches = load(model, trace)
    n = sum(len(b) for _, b in batches)
    loaded[name] = (stride, nlayers, offs, paths, batches, depth)
    ring_b = nlayers * SLOTS * stride
    print(f"  {name}: {n} reads, {n/STEPS:.1f}/step, {n*stride/STEPS/2**20:.1f} MiB/step, "
          f"depth {depth}, stride {stride/2**20:.3f} MiB, "
          f"slot ring {nlayers}x{SLOTS} = {ring_b/2**30:.2f} GiB")

print()
res = {(n, c[0]): [] for n, *_ in ARMS for c in CONDS}
for r in range(A.rounds):
    arms = ARMS if r % 2 == 0 else list(reversed(ARMS))
    for name, model, trace, depth in arms:
        stride, nlayers, offs, paths, batches, depth = loaded[name]
        n = sum(len(b) for _, b in batches)
        conds = CONDS if r % 2 == 0 else list(reversed(CONDS))
        ring = make_ring(stride, nlayers)          # outside every timed span
        try:
            for cname, (nc, use_ring, alt) in conds:
                if EVICT_FDS and A.evict_gib > 0:
                    evict(A.evict_gib, stride)
                ss, wall, faults = replay(stride, nlayers, offs, paths, batches,
                                          depth, nc, use_ring, ring, alt)
                ms = ss / STEPS * 1000
                res[(name, cname)].append(ms)
                print(f"  r{r} {name}  {cname:<30} span {ms:8.2f} ms/step  "
                      f"wall {wall/STEPS*1000:8.2f}  minflt {faults:>10,}  "
                      f"{n*stride/ss/1e9:5.2f} GB/s")
        finally:
            free_ring(ring)

print()
print("=" * 78)
for name, *_ in ARMS:
    print(f"\n{name}")
    for cname, _ in CONDS:
        v = res[(name, cname)]
        print(f"  {cname:<30} " + "  ".join(f"{x:7.2f}" for x in v) +
              f"   min {min(v):7.2f}")
have = {n for n, *_ in ARMS}
if have == {"3.6", "3.8"}:
    print()
    for cname, _ in CONDS:
        a = min(res[("3.6", cname)])
        b = min(res[("3.8", cname)])
        print(f"  {cname:<30} 3.6 {a:7.2f}   3.8 {b:7.2f}   ratio {b/a:5.3f}x")
