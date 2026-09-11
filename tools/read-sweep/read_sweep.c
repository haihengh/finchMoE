// read_sweep — engine-shaped device characterization for the packed-expert read.
//
// Reads `chunk` bytes at stride-aligned pseudo-random offsets from one layer
// file with `depth` threads, and reports achieved bandwidth and the per-read
// latency distribution. The engine's shape is reproduced faithfully at the
// level that matters for a batch: all concurrent reads are inside one file,
// destinations are page-aligned and reused (the engine preads into slot pages
// and we know from the staging A/B that the destination is not a factor), and
// the file is opened without F_NOCACHE, exactly as the streamer opens it.
//
// The one thing it does NOT reproduce is the layer walk: the engine reads 48
// different files per step, one batch each, and this reads one file for the
// whole cell. If the two agree at the engine's own operating points, that gap
// is not material; if they do not, it is the next suspect.
//
// Usage:
//   read_sweep <file> <chunk> <stride> <depth> <totalMiB> <seed> [label]
//
// Offsets are drawn without replacement from the stride grid, so no cell ever
// re-reads its own bytes and a cell's rate is not flattered by its own reuse.

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PAGE 16384

static int g_fd;
static uint64_t g_file_size;
static size_t g_chunk;
static uint64_t g_stride;
static uint64_t g_nreads;
static uint64_t g_start;   // first slot of the window, 0 unless asked for
static uint64_t g_span;    // slots in the window, 0 means the whole file
static uint64_t *g_offsets;
static atomic_ullong g_next;

typedef struct {
    int tid;
    uint8_t *buf;
    uint64_t *lat;
    uint64_t count;
} worker_t;

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static inline uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

static void *worker(void *arg) {
    worker_t *w = arg;
    for (;;) {
        uint64_t i = atomic_fetch_add(&g_next, 1);
        if (i >= g_nreads) break;
        uint64_t t0 = now_ns();
        ssize_t got = pread(g_fd, w->buf, g_chunk, (off_t)g_offsets[i]);
        uint64_t t1 = now_ns();
        if (got != (ssize_t)g_chunk) {
            fprintf(stderr, "pread(%llu, %zu) -> %zd errno=%d\n",
                    (unsigned long long)g_offsets[i], g_chunk, got, errno);
            exit(1);
        }
        w->lat[w->count++] = t1 - t0;
    }
    return NULL;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <file> <chunk> <stride> <depth> <totalMiB> <seed> [label] "
                "[startSlot] [spanSlots]\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[1];
    g_chunk = strtoull(argv[2], NULL, 0);
    g_stride = strtoull(argv[3], NULL, 0);
    int depth = atoi(argv[4]);
    uint64_t total = strtoull(argv[5], NULL, 0) * 1024ULL * 1024ULL;
    uint64_t seed = strtoull(argv[6], NULL, 0);
    const char *label = argc > 7 ? argv[7] : "";
    g_start = argc > 8 ? strtoull(argv[8], NULL, 0) : 0;
    g_span = argc > 9 ? strtoull(argv[9], NULL, 0) : 0;

    if (g_chunk == 0 || g_stride == 0 || depth < 1) return 2;
    if (g_stride < g_chunk) g_stride = g_chunk;
    if (g_stride % PAGE) {
        fprintf(stderr, "stride must be a multiple of %d\n", PAGE);
        return 2;
    }

    g_fd = open(path, O_RDONLY);
    if (g_fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(g_fd, &st) != 0) { perror("fstat"); return 1; }
    g_file_size = (uint64_t)st.st_size;

    uint64_t slots = (g_file_size >= g_chunk) ? (g_file_size - g_chunk) / g_stride + 1 : 0;
    if (slots == 0) { fprintf(stderr, "file smaller than chunk\n"); return 1; }

    // Optional window onto the slot grid. Two cells confined to disjoint
    // windows of one file touch disjoint bytes, so both are cold without a
    // purge and the chunk axis can be crossed with a fixed file.
    if (g_span == 0) g_span = slots;
    if (g_start > slots) g_start = slots;
    if (g_span > slots - g_start) g_span = slots - g_start;
    if (g_span == 0) { fprintf(stderr, "empty slot window\n"); return 1; }

    g_nreads = total / g_chunk;
    if (g_nreads > g_span) g_nreads = g_span;   // never re-read our own bytes

    // Draw offsets without replacement: walk the stride grid with a step
    // coprime to `span`, then permute the first g_nreads entries. A full
    // Fisher-Yates over `span` is unnecessary; a stride walk with a random
    // start is uniform enough for a device measurement and O(nreads).
    g_offsets = malloc(g_nreads * sizeof(uint64_t));
    if (!g_offsets) { perror("malloc"); return 1; }
    uint64_t step = 1;
    for (uint64_t s = 1 + (xs(&seed) % (g_span - 1 ? g_span - 1 : 1)); ; s++) {
        if (s >= g_span) { step = 1; break; }
        uint64_t a = s, b = g_span;
        while (b) { uint64_t t = a % b; a = b; b = t; }
        if (a == 1) { step = s; break; }
    }
    uint64_t pos = xs(&seed) % g_span;
    for (uint64_t i = 0; i < g_nreads; i++) {
        g_offsets[i] = (g_start + pos) * g_stride;
        pos = (pos + step) % g_span;
    }

    worker_t *w = calloc(depth, sizeof(worker_t));
    pthread_t *th = calloc(depth, sizeof(pthread_t));
    for (int i = 0; i < depth; i++) {
        w[i].tid = i;
        w[i].lat = malloc(g_nreads * sizeof(uint64_t) / 1 + 64);
        if (posix_memalign((void **)&w[i].buf, PAGE, g_chunk) != 0) {
            perror("posix_memalign");
            return 1;
        }
    }

    // Touch the file once so the open/first-read cost is not in the cell.
    if (pread(g_fd, w[0].buf, g_chunk, 0) != (ssize_t)g_chunk) { perror("warm pread"); }

    uint64_t t0 = now_ns();
    for (int i = 0; i < depth; i++) pthread_create(&th[i], NULL, worker, &w[i]);
    for (int i = 0; i < depth; i++) pthread_join(th[i], NULL);
    uint64_t t1 = now_ns();

    uint64_t n = 0;
    for (int i = 0; i < depth; i++) n += w[i].count;
    uint64_t *all = malloc(n * sizeof(uint64_t));
    uint64_t k = 0, sum_lat_ns = 0;
    for (int i = 0; i < depth; i++)
        for (uint64_t j = 0; j < w[i].count; j++) {
            all[k++] = w[i].lat[j];
            sum_lat_ns += w[i].lat[j];
        }
    qsort(all, k, sizeof(uint64_t), cmp_u64);

    double wall = (double)(t1 - t0) / 1e9;
    double bytes = (double)g_nreads * (double)g_chunk;
    double gbps = bytes / wall / 1e9;
    double mean_us = (double)sum_lat_ns / (double)k / 1000.0;
    uint64_t mid = k / 2;
    uint64_t p95i = (uint64_t)((double)k * 0.95);

    printf("%-22s chunk=%7zu stride=%7llu depth=%2d reads=%6llu MiB=%7.1f "
           "wall=%7.3f GBps=%6.3f mean=%8.1fus p50=%8.1f p95=%9.1f max=%10.1f "
           "win=%llu+%llu\n",
           label, g_chunk, (unsigned long long)g_stride, depth,
           (unsigned long long)g_nreads, bytes / 1048576.0, wall, gbps,
           mean_us, (double)all[mid] / 1000.0,
           (double)all[p95i] / 1000.0,
           (double)all[k - 1] / 1000.0,
           (unsigned long long)g_start, (unsigned long long)g_span);
    fflush(stdout);
    return 0;
}
