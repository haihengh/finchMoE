// finchmoe_format.h
// ─────────────────────────────────────────────────────────────────────────────
// .finchmoe file parsing: header, dense index, expert index.
// All reads are done with pread() so the fd can be shared safely across
// the IOPlanner's worker threads.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <stdexcept>

namespace finchmoe {
namespace format {

// ── on-disk constants (match Python converter exactly) ────────────────────────

static constexpr uint8_t  MAGIC[8]         = {'F','I','N','C','H','M','O','E'};
static constexpr uint32_t FORMAT_VERSION   = 0x0001'0000u;
static constexpr size_t   HEADER_SIZE      = 512;
static constexpr size_t   ALIGN            = 64 * 1024;
static constexpr size_t   DENSE_IDX_ENTRY  = 48;
static constexpr size_t   EXPERT_IDX_ENTRY = 24;

// ── on-disk structs (packed, little-endian) ───────────────────────────────────
#pragma pack(push, 1)

struct FileHeader {
    uint8_t  magic[8];
    uint32_t version;
    uint32_t dtype_code;
    // model config
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t moe_intermediate_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t num_shared_experts;
    uint8_t  norm_topk_prob;
    uint8_t  tie_word_embeddings;
    uint8_t  _pad0[2];
    float    rms_norm_eps;
    float    rope_theta;
    char     rope_type[8];
    // section pointers
    uint64_t dense_idx_offset;
    uint32_t dense_idx_count;
    uint32_t _pad1;
    uint64_t expert_idx_offset;
    uint32_t expert_idx_count;
    uint32_t _pad2;
    uint64_t dense_data_offset;
    uint64_t expert_data_offset;
    uint8_t  reserved[384];
    // Total = 8+4+4+28+4+4+4+2+2+4+4+8+8+4+4+8+4+4+8+8+384 = 512 ✓
};
static_assert(sizeof(FileHeader) == 512, "FileHeader must be exactly 512 bytes");

struct DenseIndexEntry {
    uint64_t name_hash;
    uint64_t offset;
    uint64_t nbytes;
    uint32_t ndim;
    int32_t  shape[4];
    uint32_t dtype_code;
    // 8+8+8+4+16+4 = 48 bytes
};
static_assert(sizeof(DenseIndexEntry) == 48, "DenseIndexEntry size mismatch");

struct ExpertIndexEntry {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t offset;
    uint64_t nbytes;
    // 4+4+8+8 = 24 bytes
};
static_assert(sizeof(ExpertIndexEntry) == 24, "ExpertIndexEntry size mismatch");

#pragma pack(pop)

// ── in-memory index entries ───────────────────────────────────────────────────

struct DenseRecord {
    uint64_t name_hash;
    uint64_t offset;    // byte offset from file start
    uint64_t nbytes;
    uint32_t ndim;
    int32_t  shape[4];
    DType    dtype;
};

struct ExpertRecord {
    uint32_t layer;
    uint32_t expert;
    uint64_t offset;
    uint64_t nbytes;    // includes alignment padding
};

// ── FNV-1a (must match Python side) ──────────────────────────────────────────

inline uint64_t fnv1a_64(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ── FinchMoEFile: open, validate, expose indexes ───────────────────────────────

class FinchMoEFile {
public:
    FinchMoEFile() = default;
    ~FinchMoEFile() { close(); }

    // Non-copyable, movable
    FinchMoEFile(const FinchMoEFile&) = delete;
    FinchMoEFile& operator=(const FinchMoEFile&) = delete;
    FinchMoEFile(FinchMoEFile&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

    // ── open & parse ──────────────────────────────────────────────────────────

    Status open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            fprintf(stderr, "[finchmoe] Failed to open %s: %s\n",
                    path.c_str(), strerror(errno));
            return Status::IOError;
        }

        // file size
        struct stat st{};
        if (fstat(fd_, &st) != 0) return Status::IOError;
        file_size_ = (size_t)st.st_size;

        if (auto s = read_header();  !ok(s)) return s;
        if (auto s = read_dense_index(); !ok(s)) return s;
        if (auto s = read_expert_index(); !ok(s)) return s;

        fprintf(stderr, "[finchmoe] Opened %s  (%zu GB)\n"
                "          Dense tensors : %zu\n"
                "          Expert slots  : %zu\n",
                path.c_str(), file_size_ >> 30,
                dense_by_hash_.size(), expert_map_.size());
        return Status::OK;
    }

    void close() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    int fd() const { return fd_; }

    // ── accessors ─────────────────────────────────────────────────────────────

    const ModelConfig& config() const { return cfg_; }

    // Look up a dense tensor by name; returns nullptr if not found.
    const DenseRecord* dense(const std::string& name) const {
        auto it = dense_by_hash_.find(fnv1a_64(name));
        return it != dense_by_hash_.end() ? &it->second : nullptr;
    }

    // Look up an expert slab by (layer, expert).
    const ExpertRecord* expert(uint32_t layer, uint32_t expert) const {
        auto key = ExpertKey{layer, expert};
        auto it  = expert_map_.find(encode_key(key));
        return it != expert_map_.end() ? &it->second : nullptr;
    }

    const std::vector<DenseRecord>&  all_dense()   const { return dense_list_; }
    const std::vector<ExpertRecord>& all_experts()  const { return expert_list_; }

    size_t file_size() const { return file_size_; }

    // ── raw I/O helper — used by IOPlanner ────────────────────────────────────
    // Thread-safe: pread does not modify file position.

    ssize_t pread_exact(void* buf, size_t nbytes, off_t offset) const {
        size_t total = 0;
        char*  dst   = static_cast<char*>(buf);
        while (total < nbytes) {
            ssize_t n = ::pread(fd_, dst + total, nbytes - total,
                                offset + (off_t)total);
            if (n <= 0) return n == 0 ? (ssize_t)total : -1;
            total += (size_t)n;
        }
        return (ssize_t)total;
    }

private:
    int     fd_        = -1;
    size_t  file_size_ = 0;
    ModelConfig cfg_;

    std::unordered_map<uint64_t, DenseRecord>  dense_by_hash_;
    std::vector<DenseRecord>                   dense_list_;
    std::unordered_map<uint64_t, ExpertRecord> expert_map_;
    std::vector<ExpertRecord>                  expert_list_;

    static uint64_t encode_key(ExpertKey k) {
        return ((uint64_t)k.layer << 32) | k.expert;
    }

    // ── private parsers ───────────────────────────────────────────────────────

    Status read_header() {
        FileHeader hdr{};
        ssize_t n = pread_exact(&hdr, sizeof(hdr), 0);
        if (n != (ssize_t)sizeof(hdr)) return Status::IOError;

        if (memcmp(hdr.magic, MAGIC, 8) != 0) {
            fprintf(stderr, "[finchmoe] Bad magic bytes\n");
            return Status::FormatError;
        }
        if (hdr.version != FORMAT_VERSION) {
            fprintf(stderr, "[finchmoe] Version mismatch: got 0x%08X want 0x%08X\n",
                    hdr.version, FORMAT_VERSION);
            return Status::FormatError;
        }

        cfg_.hidden_size             = hdr.hidden_size;
        cfg_.intermediate_size       = hdr.intermediate_size;
        cfg_.moe_intermediate_size   = hdr.moe_intermediate_size;
        cfg_.num_hidden_layers       = hdr.num_hidden_layers;
        cfg_.num_attention_heads     = hdr.num_attention_heads;
        cfg_.num_key_value_heads     = hdr.num_key_value_heads;
        cfg_.vocab_size              = hdr.vocab_size;
        cfg_.max_position_embeddings = hdr.max_position_embeddings;
        cfg_.num_experts             = hdr.num_experts;
        cfg_.num_experts_per_tok     = hdr.num_experts_per_tok;
        cfg_.num_shared_experts      = hdr.num_shared_experts;
        cfg_.norm_topk_prob          = (hdr.norm_topk_prob != 0);
        cfg_.tie_word_embeddings     = (hdr.tie_word_embeddings != 0);
        cfg_.rms_norm_eps            = hdr.rms_norm_eps;
        cfg_.rope_theta              = hdr.rope_theta;
        cfg_.storage_dtype           = static_cast<DType>(hdr.dtype_code);
        memcpy(cfg_.rope_scaling_type, hdr.rope_type, 8);

        dense_idx_offset_  = hdr.dense_idx_offset;
        dense_idx_count_   = hdr.dense_idx_count;
        expert_idx_offset_ = hdr.expert_idx_offset;
        expert_idx_count_  = hdr.expert_idx_count;
        return Status::OK;
    }

    Status read_dense_index() {
        size_t total = (size_t)dense_idx_count_ * DENSE_IDX_ENTRY;
        std::vector<uint8_t> buf(total);
        ssize_t n = pread_exact(buf.data(), total, (off_t)dense_idx_offset_);
        if (n != (ssize_t)total) return Status::IOError;

        dense_list_.reserve(dense_idx_count_);
        for (uint32_t i = 0; i < dense_idx_count_; ++i) {
            const DenseIndexEntry* e = reinterpret_cast<const DenseIndexEntry*>(
                buf.data() + i * DENSE_IDX_ENTRY);
            DenseRecord rec{};
            rec.name_hash = e->name_hash;
            rec.offset    = e->offset;
            rec.nbytes    = e->nbytes;
            rec.ndim      = e->ndim;
            memcpy(rec.shape, e->shape, sizeof(rec.shape));
            rec.dtype     = static_cast<DType>(e->dtype_code);
            dense_list_.push_back(rec);
            dense_by_hash_[rec.name_hash] = rec;
        }
        return Status::OK;
    }

    Status read_expert_index() {
        size_t total = (size_t)expert_idx_count_ * EXPERT_IDX_ENTRY;
        std::vector<uint8_t> buf(total);
        ssize_t n = pread_exact(buf.data(), total, (off_t)expert_idx_offset_);
        if (n != (ssize_t)total) return Status::IOError;

        expert_list_.reserve(expert_idx_count_);
        for (uint32_t i = 0; i < expert_idx_count_; ++i) {
            const ExpertIndexEntry* e = reinterpret_cast<const ExpertIndexEntry*>(
                buf.data() + i * EXPERT_IDX_ENTRY);
            ExpertRecord rec{};
            rec.layer  = e->layer_idx;
            rec.expert = e->expert_idx;
            rec.offset = e->offset;
            rec.nbytes = e->nbytes;
            expert_list_.push_back(rec);
            expert_map_[encode_key({rec.layer, rec.expert})] = rec;
        }
        return Status::OK;
    }

    // stored during header parse, used by index parsers
    uint64_t dense_idx_offset_  = 0;
    uint32_t dense_idx_count_   = 0;
    uint64_t expert_idx_offset_ = 0;
    uint32_t expert_idx_count_  = 0;
};

} // namespace format
} // namespace finchmoe
