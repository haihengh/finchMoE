// io_planner.cpp
// ─────────────────────────────────────────────────────────────────────────────

#include "io_planner.h"

#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <cassert>
#include <numeric>

namespace aeromoe {

// ── construction / destruction ────────────────────────────────────────────────

IOPlanner::IOPlanner(const format::AeroMoEFile& file, int num_threads)
    : file_(file)
{
    int n = std::max(1, num_threads);
    workers_.reserve(n);
    for (int i = 0; i < n; ++i) {
        workers_.emplace_back([this]{ worker_loop(); });
    }
    fprintf(stderr, "[io_planner] Started with %d I/O thread(s)\n", n);
}

IOPlanner::~IOPlanner() {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_flag_.store(true, std::memory_order_relaxed);
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

// ── worker loop ───────────────────────────────────────────────────────────────

void IOPlanner::worker_loop() {
    while (true) {
        std::unique_ptr<IOJob> job;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this]{
                return !job_queue_.empty()
                    || stop_flag_.load(std::memory_order_relaxed);
            });
            if (stop_flag_.load(std::memory_order_relaxed) && job_queue_.empty())
                return;
            if (job_queue_.empty()) continue;
            job = std::move(job_queue_.front());
            job_queue_.pop();
        }

        auto t0  = std::chrono::steady_clock::now();
        Status s = do_pread(job->dst, job->nbytes, job->file_offset);
        auto t1  = std::chrono::steady_clock::now();

        if (ok(s)) {
            bytes_read_.fetch_add(job->nbytes,  std::memory_order_relaxed);
            read_count_.fetch_add(1,             std::memory_order_relaxed);
            uint64_t us = (uint64_t)std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count();
            total_us_.fetch_add(us, std::memory_order_relaxed);
        }

        job->promise.set_value(s);
    }
}

// ── raw pread ────────────────────────────────────────────────────────────────

Status IOPlanner::do_pread(void* dst, size_t nbytes, off_t offset) const {
    char*   buf   = static_cast<char*>(dst);
    size_t  total = 0;
    while (total < nbytes) {
        ssize_t n = ::pread(file_.fd(), buf + total,
                            nbytes - total, offset + (off_t)total);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "[io_planner] pread error at offset %lld: %s\n",
                    (long long)(offset + total), strerror(errno));
            return Status::IOError;
        }
        if (n == 0) {
            fprintf(stderr, "[io_planner] Unexpected EOF at offset %lld "
                    "(read %zu of %zu bytes)\n",
                    (long long)(offset + total), total, nbytes);
            return Status::IOError;
        }
        total += (size_t)n;
    }
    return Status::OK;
}

// ── synchronous load ─────────────────────────────────────────────────────────

Status IOPlanner::load_slab(ExpertKey key, void* dst, size_t nbytes) const {
    const format::ExpertRecord* rec = file_.expert(key.layer, key.expert);
    if (!rec) {
        fprintf(stderr, "[io_planner] Expert (%u,%u) not found in index\n",
                key.layer, key.expert);
        return Status::NotFound;
    }
    assert(nbytes >= rec->nbytes - (64*1024) && "dst too small");
    // Use the actual stored size (which includes alignment padding) but only
    // copy the meaningful weight bytes (rec->nbytes may include alignment pad).
    // The caller passes nbytes == slab_bytes() which is rec->nbytes rounded up.
    return do_pread(dst, nbytes, (off_t)rec->offset);
}

// ── async batch ───────────────────────────────────────────────────────────────

std::vector<std::future<Status>> IOPlanner::submit_batch(
    const std::vector<ExpertKey>& keys,
    const std::vector<void*>&     dsts,
    const std::vector<size_t>&    sizes)
{
    assert(keys.size() == dsts.size() && keys.size() == sizes.size());

    std::vector<std::future<Status>> futures;
    futures.reserve(keys.size());

    std::vector<std::unique_ptr<IOJob>> jobs;
    jobs.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        const format::ExpertRecord* rec = file_.expert(keys[i].layer, keys[i].expert);
        if (!rec) {
            // Return an immediately-resolved failed future
            std::promise<Status> p;
            p.set_value(Status::NotFound);
            futures.push_back(p.get_future());
            continue;
        }
        auto job = std::make_unique<IOJob>();
        job->key         = keys[i];
        job->dst         = dsts[i];
        job->nbytes      = sizes[i];
        job->file_offset = (off_t)rec->offset;
        futures.push_back(job->promise.get_future());
        jobs.push_back(std::move(job));
    }

    // Enqueue all at once to minimise lock contention
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        for (auto& j : jobs) job_queue_.push(std::move(j));
    }
    queue_cv_.notify_all();

    return futures;
}

std::vector<Status> IOPlanner::wait_all(
    std::vector<std::future<Status>>& batch)
{
    std::vector<Status> results;
    results.reserve(batch.size());
    for (auto& f : batch) results.push_back(f.get());
    return results;
}

// ── size query ────────────────────────────────────────────────────────────────

size_t IOPlanner::slab_bytes(ExpertKey key) const {
    const format::ExpertRecord* rec = file_.expert(key.layer, key.expert);
    return rec ? rec->nbytes : 0;
}

// ── LoadFn factory ────────────────────────────────────────────────────────────

ExpertCache::LoadFn IOPlanner::make_load_fn() {
    return [this](ExpertKey key, void* dst, size_t nbytes) -> Status {
        // Size query path (dst==nullptr, nbytes==0)
        if (dst == nullptr && nbytes == 0) {
            extern thread_local size_t g_last_slab_bytes;
            g_last_slab_bytes = slab_bytes(key);
            return Status::OK;
        }
        return load_slab(key, dst, nbytes);
    };
}

// ── stats ─────────────────────────────────────────────────────────────────────

double IOPlanner::mean_latency_us() const {
    uint64_t count = read_count_.load(std::memory_order_relaxed);
    uint64_t us    = total_us_.load(std::memory_order_relaxed);
    return count > 0 ? (double)us / count : 0.0;
}

void IOPlanner::print_stats(FILE* out) const {
    fprintf(out,
        "[io_planner] reads=%llu  bytes=%.2f GB  mean_latency=%.1f µs\n",
        (unsigned long long)total_reads(),
        (double)total_bytes_read() / 1e9,
        mean_latency_us());
}

} // namespace aeromoe
