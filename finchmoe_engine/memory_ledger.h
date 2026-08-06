// memory_ledger.h
// ─────────────────────────────────────────────────────────────────────────────
// MemoryLedger — atomic accounting of active unified-memory allocations.
//
// All Metal buffers (dense weights, expert slabs, KV-cache, activations)
// must be registered here before use and deregistered on eviction/free.
// The ledger enforces the hard <4 GB budget and provides headroom queries
// so the ExpertCache can make eviction decisions before calling into Metal.
//
// Thread-safety: all public methods are lock-free (std::atomic).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cassert>

namespace finchmoe {

class MemoryLedger {
public:
    // ── construction ─────────────────────────────────────────────────────────

    explicit MemoryLedger(size_t hard_budget = BUDGET_4GB,
                          size_t soft_budget = BUDGET_3_5GB)
        : hard_budget_(hard_budget)
        , soft_budget_(soft_budget)
        , used_bytes_(0)
        , peak_bytes_(0)
        , alloc_count_(0)
        , evict_count_(0)
    {
        assert(soft_budget < hard_budget);
    }

    // ── registration ─────────────────────────────────────────────────────────

    // Try to reserve `bytes`. Returns OK if the reservation fits under
    // hard_budget; returns BudgetExceeded if not.
    // Call this BEFORE allocating the Metal buffer.
    Status reserve(size_t bytes) noexcept {
        size_t prev = used_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        size_t after = prev + bytes;
        if (after > hard_budget_) {
            // Roll back and refuse
            used_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
            return Status::BudgetExceeded;
        }
        // Update peak
        size_t peak = peak_bytes_.load(std::memory_order_relaxed);
        while (after > peak) {
            if (peak_bytes_.compare_exchange_weak(peak, after,
                    std::memory_order_relaxed, std::memory_order_relaxed))
                break;
        }
        alloc_count_.fetch_add(1, std::memory_order_relaxed);
        return Status::OK;
    }

    // Release bytes previously reserved. Call after the buffer is freed.
    void release(size_t bytes) noexcept {
        used_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
        evict_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── headroom queries ──────────────────────────────────────────────────────

    // How many bytes are currently accounted for.
    size_t used() const noexcept {
        return used_bytes_.load(std::memory_order_relaxed);
    }

    // Bytes available before hitting the hard ceiling.
    size_t remaining_hard() const noexcept {
        size_t u = used();
        return u < hard_budget_ ? hard_budget_ - u : 0;
    }

    // Bytes available before hitting the soft ceiling.
    // The ExpertCache uses this to decide whether to evict before loading.
    size_t remaining_soft() const noexcept {
        size_t u = used();
        return u < soft_budget_ ? soft_budget_ - u : 0;
    }

    // Returns true if we are over the soft limit (time to start evicting).
    bool pressure() const noexcept {
        return used() > soft_budget_;
    }

    // Returns true if `bytes` can be reserved without exceeding hard budget.
    bool can_fit(size_t bytes) const noexcept {
        return (used() + bytes) <= hard_budget_;
    }

    // Returns how many bytes must be freed before `bytes` can fit.
    size_t evict_needed(size_t bytes) const noexcept {
        size_t total = used() + bytes;
        return total > hard_budget_ ? total - hard_budget_ : 0;
    }

    // Soft-limit variant: headroom before we start feeling pressure.
    size_t soft_evict_needed(size_t bytes) const noexcept {
        size_t total = used() + bytes;
        return total > soft_budget_ ? total - soft_budget_ : 0;
    }

    // ── diagnostics ───────────────────────────────────────────────────────────

    size_t peak() const noexcept {
        return peak_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t alloc_count() const noexcept {
        return alloc_count_.load(std::memory_order_relaxed);
    }
    uint64_t evict_count() const noexcept {
        return evict_count_.load(std::memory_order_relaxed);
    }
    size_t hard_budget() const noexcept { return hard_budget_; }
    size_t soft_budget() const noexcept { return soft_budget_; }

    void print_stats(FILE* out = stderr) const {
        size_t u = used();
        fprintf(out,
            "[ledger] used=%.2f GB  peak=%.2f GB  hard=%.2f GB  soft=%.2f GB"
            "  allocs=%llu  evictions=%llu\n",
            (double)u      / 1e9,
            (double)peak() / 1e9,
            (double)hard_budget_ / 1e9,
            (double)soft_budget_ / 1e9,
            (unsigned long long)alloc_count(),
            (unsigned long long)evict_count());
    }

private:
    const size_t hard_budget_;
    const size_t soft_budget_;

    std::atomic<size_t>   used_bytes_;
    std::atomic<size_t>   peak_bytes_;
    std::atomic<uint64_t> alloc_count_;
    std::atomic<uint64_t> evict_count_;
};

} // namespace finchmoe
