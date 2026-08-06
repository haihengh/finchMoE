// io_planner.h
// ─────────────────────────────────────────────────────────────────────────────
// IOPlanner — parallel pread() I/O for expert slab loading.
//
// Design goals:
//   - No speculative prefetching (Mac unified memory; bandwidth is precious).
//   - Demand-driven: only load slabs that were actually selected by the router.
//   - Parallel: a fixed thread pool issues concurrent pread() calls so that
//     multiple experts for the same layer can be loaded simultaneously.
//   - Zero-copy: pread() writes directly into the MTLBuffer.contents pointer
//     (MTLStorageModeShared pages are CPU-writable without any blit).
//
// The IOPlanner does NOT own any buffers. It only reads bytes from the .finchmoe
// file and places them into caller-supplied memory regions.
//
// Thread-safety: fully re-entrant; multiple threads may call load_slab()
// concurrently on different keys.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"
#include "finchmoe_format.h"
#include "expert_cache.h"

#include <functional>
#include <future>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace finchmoe {

// ── IOJob ─────────────────────────────────────────────────────────────────────

struct IOJob {
    ExpertKey key;
    void*     dst;
    size_t    nbytes;
    off_t     file_offset;
    std::promise<Status> promise;
};

// ── IOPlanner ─────────────────────────────────────────────────────────────────

class IOPlanner {
public:
    explicit IOPlanner(const format::FinchMoEFile& file,
                       int num_threads = 4);
    ~IOPlanner();

    // ── synchronous load ──────────────────────────────────────────────────────

    // Load a single expert slab into `dst` (already-allocated MTLBuffer.contents).
    // Blocks until the pread completes.
    Status load_slab(ExpertKey key, void* dst, size_t nbytes) const;

    // ── async batch load ──────────────────────────────────────────────────────

    // Submit a batch of (key, dst, nbytes) loads to the thread pool.
    // Returns a future per key; collect with wait_all().
    std::vector<std::future<Status>> submit_batch(
        const std::vector<ExpertKey>& keys,
        const std::vector<void*>&     dsts,
        const std::vector<size_t>&    sizes);

    // Block until all futures in `batch` are ready.
    static std::vector<Status> wait_all(std::vector<std::future<Status>>& batch);

    // ── size query ────────────────────────────────────────────────────────────

    // Returns the on-disk slab size for (layer, expert), or 0 if not found.
    size_t slab_bytes(ExpertKey key) const;

    // ── factory: ExpertCache::LoadFn ──────────────────────────────────────────
    // Returns a LoadFn closure suitable for ExpertCache.
    // Convention: when dst == nullptr && nbytes == 0, sets g_last_slab_bytes
    // and returns Status::OK (size query path used by ExpertCache::acquire).

    ExpertCache::LoadFn make_load_fn();

    // ── stats ─────────────────────────────────────────────────────────────────

    uint64_t total_bytes_read()  const { return bytes_read_.load(); }
    uint64_t total_reads()       const { return read_count_.load(); }
    double   mean_latency_us()   const;

    void print_stats(FILE* out = stderr) const;

private:
    // ── worker thread loop ────────────────────────────────────────────────────

    void worker_loop();

    // ── raw I/O ───────────────────────────────────────────────────────────────

    Status do_pread(void* dst, size_t nbytes, off_t offset) const;

    // ── state ─────────────────────────────────────────────────────────────────

    const format::FinchMoEFile& file_;

    // Thread pool
    std::vector<std::thread>          workers_;
    std::queue<std::unique_ptr<IOJob>> job_queue_;
    std::mutex                         queue_mutex_;
    std::condition_variable            queue_cv_;
    std::atomic<bool>                  stop_flag_{false};

    // Stats
    mutable std::atomic<uint64_t> bytes_read_{0};
    mutable std::atomic<uint64_t> read_count_{0};
    mutable std::atomic<uint64_t> total_us_{0};   // total I/O microseconds
};

} // namespace finchmoe
