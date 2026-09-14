// read_batch — the engine's layer walk, without the engine.
//
// read_sweep reads one file for a whole cell. The engine does not: it issues
// one layer's misses as a single `concurrentPerform` batch of 3-5 reads from
// THAT layer's file, waits for the batch, computes the layer, and moves to the
// next file. Forty-eight files, 3-5 reads each, is the engine's read shape, and
// it is the last structural difference a single-file harness cannot hold.
//
// So: walk a list of files in order, issue `m` concurrent reads from each,
// barrier, next file. `m` cycles through a small list because the engine's own
// per-layer miss count is not an integer (3.37 on Qwen 3.6, 5.12 on 3.8).
//
// The barrier is dispatch_apply on a concurrent queue -- the same primitive the
// engine's `DispatchQueue.concurrentPerform` resolves to. A hand-rolled
// spin/atomic handshake was tried first and its own wake-up cost showed up as
// the device's (2.05 GB/s against a known-cold 3.26 on the same file), which
// would have been my harness measured as the drive. Use the engine's primitive
// or the comparison is between two barriers, not between two read shapes.
//
// Offsets are drawn per (round, file) with replacement ACROSS rounds and
// without replacement WITHIN one, which is the engine's own reuse shape: a plan
// never lists the same expert twice, but the same expert comes back a step or
// two later. Every slot touched is marked in a per-file bitmap and each round
// reports how many of its reads were NOVEL, so that reuse is visible rather
// than assumed -- §2.1's own prior was a reuse claim, and [METH-14] says a
// replay is only as faithful as the reuse it cannot see.
//
// Usage:
//   read_batch <listfile> <chunk> <stride> <cycle> <rounds> <seed> [label]
//
//   listfile : one file path per line, in walk order
//   cycle    : comma-separated reads-per-file, e.g. 3,3,4
//   rounds   : walks over the whole list
//
// SORT=1 in the environment sorts each batch's offsets ascending before
// dispatch, which is the order the engine's plan carries if the router hands
// its experts over in expert-id order.

#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAXFILES 256
#define MAXBATCH 32

static int g_fd[MAXFILES];
static uint64_t g_slots[MAXFILES];
static uint64_t *g_seen[MAXFILES];   // one bit per slot, the reuse record
static int g_nfiles;

static size_t g_chunk;
static uint64_t g_stride;

static dispatch_queue_t g_q;
static uint8_t *g_buf[MAXBATCH];     // one aligned destination per batch slot
static double g_lat[MAXBATCH];       // this batch's per-read latency, ns

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// One batch: m concurrent preads from one file, and no return until all m land.
// Same shape as executeExpertCachePlan's fan-out for a single layer.
static void run_batch(int fd, int count, uint64_t *off) {
    if (getenv("SORT") && atoi(getenv("SORT")) != 0)
        qsort(off, count, sizeof(uint64_t), cmp_u64);
    dispatch_apply((size_t)count, g_q, ^(size_t i) {
        uint64_t t0 = now_ns();
        ssize_t got = pread(fd, g_buf[i], g_chunk, (off_t)off[i]);
        uint64_t t1 = now_ns();
        if (got != (ssize_t)g_chunk) {
            fprintf(stderr, "pread(%llu) -> %zd errno=%d\n",
                    (unsigned long long)off[i], got, errno);
            exit(1);
        }
        g_lat[i] = (double)(t1 - t0);
    });
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <listfile> <chunk> <stride> <cycle> <rounds> <seed> [label]\n",
                argv[0]);
        return 2;
    }
    const char *list = argv[1];
    g_chunk = strtoull(argv[2], NULL, 0);
    g_stride = strtoull(argv[3], NULL, 0);
    int cycle[MAXBATCH], ncycle = 0;
    {
        char *spec = strdup(argv[4]), *tok = strtok(spec, ",");
        while (tok && ncycle < MAXBATCH) { cycle[ncycle++] = atoi(tok); tok = strtok(NULL, ","); }
        free(spec);
    }
    int rounds = atoi(argv[5]);
    uint64_t seed = strtoull(argv[6], NULL, 0);
    const char *label = argc > 7 ? argv[7] : "";

    if (g_chunk == 0 || g_stride == 0 || rounds < 1 || ncycle == 0) return 2;
    if (g_stride < g_chunk) g_stride = g_chunk;
    if (g_stride % 16384) { fprintf(stderr, "stride must be a multiple of 16384\n"); return 2; }

    FILE *lf = fopen(list, "r");
    if (!lf) { perror("listfile"); return 1; }
    char line[1024];
    int maxcycle = 0;
    while (fgets(line, sizeof line, lf)) {
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (n == 0 || line[0] == '#') continue;
        if (g_nfiles >= MAXFILES) { fprintf(stderr, "too many files\n"); return 1; }
        g_fd[g_nfiles] = open(line, O_RDONLY);
        if (g_fd[g_nfiles] < 0) { perror(line); return 1; }
        struct stat st;
        if (fstat(g_fd[g_nfiles], &st) != 0) { perror("fstat"); return 1; }
        uint64_t size = (uint64_t)st.st_size;
        if (size < g_chunk) { fprintf(stderr, "%s smaller than chunk\n", line); return 1; }
        uint64_t slots = (size - g_chunk) / g_stride + 1;
        g_slots[g_nfiles] = slots;
        g_seen[g_nfiles] = calloc((slots + 63) / 64, sizeof(uint64_t));
        if (!g_seen[g_nfiles]) { perror("calloc"); return 1; }
        g_nfiles++;
    }
    fclose(lf);
    if (g_nfiles == 0) { fprintf(stderr, "empty list\n"); return 1; }
    for (int i = 0; i < ncycle; i++) if (cycle[i] > maxcycle) maxcycle = cycle[i];
    if (maxcycle > MAXBATCH) { fprintf(stderr, "cycle too wide\n"); return 1; }

    g_q = dispatch_queue_create("read_batch.concurrent", DISPATCH_QUEUE_CONCURRENT);
    for (int i = 0; i < maxcycle; i++)
        if (posix_memalign((void **)&g_buf[i], 16384, g_chunk) != 0) {
            perror("posix_memalign");
            return 1;
        }

    // One warm read per file, the same courtesy read_sweep pays a single file:
    // it takes open/first-read cost out of the first counted batch. One slot of
    // 256 (3.6) or 512 (3.8) is 0.4% of a file, so it does not make round 1 warm.
    for (int f = 0; f < g_nfiles; f++)
        if (pread(g_fd[f], g_buf[0], g_chunk, 0) != (ssize_t)g_chunk) perror("warm pread");

    printf("%s files=%d chunk=%zu stride=%llu cycle=%s rounds=%d sort=%d\n",
           label, g_nfiles, g_chunk, (unsigned long long)g_stride, argv[4], rounds,
           getenv("SORT") != NULL);
    fflush(stdout);

    uint64_t tot_reads = 0, tot_bytes = 0, tot_novel = 0;
    double tot_ms = 0, first_gbps = 0, last_gbps = 0, first_batch_ms = 0, last_batch_ms = 0;
    for (int r = 0; r < rounds; r++) {
        uint64_t rreads = 0, rnovel = 0, rbatches = 0;
        double lsum = 0;
        uint64_t lcnt = 0;
        uint64_t t0 = now_ns();
        for (int f = 0; f < g_nfiles; f++) {
            int m = cycle[f % ncycle];
            uint64_t slots = g_slots[f];
            uint64_t s = seed + (uint64_t)r * 1000003ULL + (uint64_t)f * 1000033ULL;
            uint64_t off[MAXBATCH], used[MAXBATCH];
            int nu = 0, tries = 0;
            // Distinct within the batch (a plan never asks for one expert
            // twice), free to repeat across rounds (the engine's own reuse).
            while (nu < m && tries < 1000) {
                uint64_t slot = xs(&s) % slots;
                int dup = 0;
                for (int k = 0; k < nu; k++) if (used[k] == slot) { dup = 1; break; }
                if (dup) { tries++; continue; }
                used[nu++] = slot;
                off[nu-1] = slot * g_stride;
            }
            if (nu < m) continue;
            for (int k = 0; k < m; k++) {          // mark before reading
                uint64_t w = used[k] >> 6, b = used[k] & 63;
                if (!(g_seen[f][w] & (1ULL << b))) { g_seen[f][w] |= 1ULL << b; rnovel++; }
            }
            run_batch(g_fd[f], m, off);
            for (int k = 0; k < m; k++) { lsum += g_lat[k]; lcnt++; }
            rbatches++;
            rreads += m;
        }
        uint64_t t1 = now_ns();
        double ms = (double)(t1 - t0) / 1e6;
        double rbytes = (double)rreads * (double)g_chunk;
        double gbps = rbytes / (ms / 1000.0) / 1e9;
        double batch_ms = rbatches ? ms / (double)rbatches : 0;
        double mean_us = lcnt ? lsum / (double)lcnt / 1000.0 : 0;
        printf("  r=%02d reads=%5llu novel=%5llu MiB=%7.1f ms=%8.2f GBps=%6.3f "
               "batch=%7.3fms mean=%8.1fus\n",
               r, (unsigned long long)rreads, (unsigned long long)rnovel,
               rbytes / 1048576.0, ms, gbps, batch_ms, mean_us);
        fflush(stdout);
        tot_reads += rreads; tot_bytes += rreads * (uint64_t)g_chunk; tot_ms += ms;
        tot_novel += rnovel;
        if (r == 0) { first_gbps = gbps; first_batch_ms = batch_ms; }
        last_gbps = gbps; last_batch_ms = batch_ms;
    }

    double tot_gbps = (double)tot_bytes / (tot_ms / 1000.0) / 1e9;
    printf("%s TOTAL files=%d reads=%llu MiB=%.1f ms=%.1f GBps=%.3f "
           "r1_GBps=%.3f r1_batch=%.3fms rN_GBps=%.3f rN_batch=%.3fms novel=%llu/%llu\n",
           label, g_nfiles, (unsigned long long)tot_reads,
           (double)tot_bytes / 1048576.0, tot_ms, tot_gbps,
           first_gbps, first_batch_ms, last_gbps, last_batch_ms,
           (unsigned long long)tot_novel, (unsigned long long)tot_reads);
    fflush(stdout);
    return 0;
}
