// The specificity control for gpu_load.
//
// gpu_load shows that GPU reads of a 1.98 GiB ring, at the engine's own dose,
// shift the offline replay's read distribution by a full bucket (p50 1.05 ->
// 2.10, thread time 2.25 -> 2.69 ms) on 3.8. That is a causal result, but on
// its own it does not say *why*: a Metal kernel reading a shared buffer and a
// CPU loop reading ordinary anonymous memory both put DRAM traffic on the same
// memory controller, and if this program reproduces the shift then the finding
// is "the drive loses bandwidth to memory traffic" rather than anything about
// the GPU, Metal, or the ring being mapped.
//
// So this reads the same bytes, in the same burst/period shape, from the same
// kind of allocation, using CPUs instead of the GPU. Same dose, different
// engine. Whatever survives the comparison is the part that is actually about
// the GPU.
//
//   cc -O2 mem_load.c -o mem_load
//   ./mem_load --gib 1.98 --burst-mb 648.8 --period-ms 147 --threads 4 --seconds 30
//
// Reports delivered GB/s so the dose can be checked against gpu_load's.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

static size_t   g_bytes;          // ring size
static size_t   g_burst;          // bytes read per burst
static long     g_period_ns;      // burst start to burst start
static int      g_threads;
static uint8_t *g_base;
static volatile uint64_t g_sink;  // keeps the loads from being dead code

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// FNV-1a over 8-byte words. Cheap enough not to be the bottleneck, and the
// result is stored, so the reads cannot be optimised away.
static uint64_t sum_span(const uint8_t *p, size_t n) {
    const uint64_t *w = (const uint64_t *)p;
    size_t nw = n / 8;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < nw; i++) {
        h ^= w[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct arg { int id; size_t offset; size_t count; };

static void *worker(void *vp) {
    struct arg *a = vp;
    size_t per = a->count / g_threads / 8 * 8;
    size_t start = a->offset + (size_t)a->id * per;
    if (start + per > a->offset + a->count) per = a->offset + a->count - start;
    if (per > 0) g_sink ^= sum_span(g_base + start, per);
    return NULL;
}

int main(int argc, char **argv) {
    double gib = 1.98, burst_mb = 648.8, period_ms = 147.0, seconds = 30.0;
    g_threads = 4;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gib")       && i + 1 < argc) gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--burst-mb")  && i + 1 < argc) burst_mb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--period-ms") && i + 1 < argc) period_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seconds")   && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads")   && i + 1 < argc) g_threads = atoi(argv[++i]);
        else { fprintf(stderr, "mem_load: unknown flag %s\n", argv[i]); return 2; }
    }

    long page = sysconf(_SC_PAGESIZE);
    g_bytes = (size_t)(gib * 1024 * 1024 * 1024);
    g_bytes = (g_bytes + page - 1) / page * page;
    g_burst = (size_t)(burst_mb * 1024 * 1024);
    if (g_burst > g_bytes) g_burst = g_bytes;
    g_period_ns = (long)(period_ms * 1e6);

    if (posix_memalign((void **)&g_base, page, g_bytes) != 0) {
        fprintf(stderr, "mem_load: allocation of %zu bytes failed\n", g_bytes);
        return 2;
    }
    memset(g_base, 0xA5, g_bytes);   // fault every page before timing

    fprintf(stderr, "mem_load: %.3f GiB, %.1f MB/burst, %.0f ms period,"
                    " %d threads, %.0fs\n",
            g_bytes / 1073741824.0, g_burst / 1e6, period_ms, g_threads, seconds);

    struct arg args[64];
    pthread_t tid[64];
    double t0 = now_s();
    double deadline = t0 + seconds;
    long bursts = 0;
    size_t offset = 0;
    double read_ns = 0;

    while (now_s() < deadline) {
        double bs = now_s();
        size_t count = g_burst;
        if (offset + count > g_bytes) count = g_bytes - offset;
        for (int i = 0; i < g_threads; i++) {
            args[i].id = i; args[i].offset = offset; args[i].count = count;
            pthread_create(&tid[i], NULL, worker, &args[i]);
        }
        for (int i = 0; i < g_threads; i++) pthread_join(tid[i], NULL);
        read_ns += (now_s() - bs) * 1e9;
        bursts++;
        offset = (offset + count >= g_bytes) ? 0 : offset + count;

        // Idle out the rest of the period, same as gpu_load: the engine's GPU
        // reads 648.8 MB in a 38.8 ms window once per step, not continuously.
        double used = now_s() - bs;
        double rest = g_period_ns / 1e9 - used;
        if (rest > 0) usleep((useconds_t)(rest * 1e6));
    }

    double wall = now_s() - t0;
    fprintf(stderr, "mem_load: %ld bursts, %.1f MB/burst, %.1fs wall,"
                    " %.2f GB/s delivered, %.1f%% duty\n",
            bursts, g_burst / 1e6, wall,
            (double)g_burst * bursts / wall / 1e9,
            read_ns / (wall * 1e9) * 100);
    // `g_sink` is only read so the compiler cannot drop the loop that fills it.
    if (g_sink == 0x1234567890ABCDEFULL) fprintf(stderr, " (unreachable)\n");
    return 0;
}
