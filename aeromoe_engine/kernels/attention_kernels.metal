// attention_kernels.metal
// ─────────────────────────────────────────────────────────────────────────────
// GQA (Grouped Query Attention) + Paged KV-Cache for Qwen3.6-35B-A3B
//
// Qwen3.6-35B-A3B config:
//   n_heads    = 64     (query heads)
//   n_kv_heads = 4      (key/value heads)
//   head_dim   = 64     (hidden / n_heads)
//   GQA ratio  = 16     (each KV head serves 16 query heads)
//
// Paged KV-cache layout:
//   The KV cache is divided into fixed-size pages (blocks).
//   A page table maps (layer, sequence, page_idx) → block_idx.
//   Each block holds PAGE_SIZE tokens for all KV heads:
//     k_cache: [num_blocks, PAGE_SIZE, n_kv_heads, head_dim]  bf16
//     v_cache: [num_blocks, PAGE_SIZE, n_kv_heads, head_dim]  bf16
//
// Kernels:
//   kv_cache_append     — write new K/V into the cache (prompt or decode step)
//   gqa_attention_decode — single-token decode attention (flash-style, GQA)
//   gqa_attention_prefill — multi-token prefill (chunked, causal mask)
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── constants ─────────────────────────────────────────────────────────────────
constant uint PAGE_SIZE = 16;   // tokens per KV cache block (must match CPU side)

// ── bf16 helpers ──────────────────────────────────────────────────────────────
inline float bf16_to_f32(ushort u) {
    return as_type<float>((uint)u << 16);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// ── kernel: kv_cache_append ───────────────────────────────────────────────────
//
// Writes new K and V slices into the paged cache.
//
// k_in / v_in:    [seq_len, n_kv_heads, head_dim]   bf16 (new tokens)
// k_cache:        [num_blocks, PAGE_SIZE, n_kv_heads, head_dim]  bf16
// page_table:     [num_layers, max_pages_per_seq]    uint32  (block indices)
// layer:          current transformer layer
// seq_start_pos:  position of the first new token in the sequence
//
// grid:  (seq_len * n_kv_heads, 1, 1)
// threadgroup: (head_dim, 1, 1)

kernel void kv_cache_append(
    device const ushort* __restrict__ k_in         [[ buffer(0) ]],
    device const ushort* __restrict__ v_in         [[ buffer(1) ]],
    device       ushort* __restrict__ k_cache      [[ buffer(2) ]],
    device       ushort* __restrict__ v_cache      [[ buffer(3) ]],
    device const uint*   __restrict__ page_table   [[ buffer(4) ]],
    constant     uint&                n_kv_heads   [[ buffer(5) ]],
    constant     uint&                head_dim     [[ buffer(6) ]],
    constant     uint&                layer        [[ buffer(7) ]],
    constant     uint&                seq_start    [[ buffer(8) ]],
    constant     uint&                max_pages    [[ buffer(9) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]]
) {
    uint tok_id  = tgid / n_kv_heads;
    uint kv_head = tgid % n_kv_heads;
    uint abs_pos = seq_start + tok_id;

    uint page_idx  = abs_pos / PAGE_SIZE;
    uint page_slot = abs_pos % PAGE_SIZE;

    // Look up physical block index from page table
    uint block_idx = page_table[layer * max_pages + page_idx];

    // Source: [tok_id, kv_head, head_dim]
    uint src_base = (tok_id * n_kv_heads + kv_head) * head_dim + tid;
    // Dest:   [block_idx, page_slot, kv_head, head_dim]
    uint dst_base = ((block_idx * PAGE_SIZE + page_slot) * n_kv_heads + kv_head)
                    * head_dim + tid;

    if (tid < head_dim) {
        k_cache[dst_base] = k_in[src_base];
        v_cache[dst_base] = v_in[src_base];
    }
}

// ── kernel: gqa_attention_decode ─────────────────────────────────────────────
//
// Flash-attention style single-token decode with GQA and paged KV cache.
//
// One threadgroup per (query_head).  All threads in the group collaborate
// to compute the attention score vector over the entire context, then reduce
// to a single weighted-sum output vector.
//
// Layout:
//   q:          [1, n_heads,    head_dim]  bf16  (single decode token)
//   k_cache:    [num_blocks, PAGE_SIZE, n_kv_heads, head_dim]  bf16
//   v_cache:    [num_blocks, PAGE_SIZE, n_kv_heads, head_dim]  bf16
//   page_table: [num_layers, max_pages]   uint32
//   out:        [1, n_heads, head_dim]    bf16
//   seq_lens:   [1]  uint32              total context length (kv tokens)
//
// grid:  (n_heads, 1, 1)
// threadgroup: (head_dim, 1, 1)    head_dim=64, fits in one SIMD group

kernel void gqa_attention_decode(
    device const ushort* __restrict__ q           [[ buffer(0) ]],
    device const ushort* __restrict__ k_cache     [[ buffer(1) ]],
    device const ushort* __restrict__ v_cache     [[ buffer(2) ]],
    device const uint*   __restrict__ page_table  [[ buffer(3) ]],
    device       ushort* __restrict__ out         [[ buffer(4) ]],
    constant     uint&                n_heads     [[ buffer(5) ]],
    constant     uint&                n_kv_heads  [[ buffer(6) ]],
    constant     uint&                head_dim    [[ buffer(7) ]],
    constant     uint&                layer       [[ buffer(8) ]],
    constant     uint&                kv_len      [[ buffer(9) ]],
    constant     uint&                max_pages   [[ buffer(10) ]],
    uint head_id [[ threadgroup_position_in_grid ]],
    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_sz   [[ threads_per_threadgroup ]]
) {
    // GQA: query head `head_id` attends to KV head `head_id / gqa_ratio`
    uint kv_head = head_id / (n_heads / n_kv_heads);
    float scale  = rsqrt((float)head_dim);

    // Load query vector into registers (each thread loads its element)
    float q_val = (tid < head_dim)
        ? bf16_to_f32(q[head_id * head_dim + tid])
        : 0.0f;

    // ── online softmax: running max and denominator ───────────────────────────
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Accumulator for weighted value sum
    threadgroup float acc_buf[64];   // head_dim ≤ 64
    if (tid < head_dim) acc_buf[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over all KV positions
    for (uint pos = 0; pos < kv_len; pos++) {
        uint page_idx  = pos / PAGE_SIZE;
        uint page_slot = pos % PAGE_SIZE;
        uint block_idx = page_table[layer * max_pages + page_idx];

        uint kv_base = ((block_idx * PAGE_SIZE + page_slot) * n_kv_heads + kv_head)
                       * head_dim;

        // Compute Q·K for this position
        float dot = 0.0f;
        if (tid < head_dim) {
            float k_val = bf16_to_f32(k_cache[kv_base + tid]);
            dot = q_val * k_val;
        }
        // SIMD reduce dot product across head_dim threads
        dot = simd_sum(dot) * scale;

        // Online softmax update (running max + exp correction)
        float prev_max  = running_max;
        running_max     = max(running_max, dot);
        float exp_score = exp(dot - running_max);
        float correction= exp(prev_max - running_max);
        running_sum     = running_sum * correction + exp_score;

        // Accumulate weighted V
        if (tid < head_dim) {
            float v_val = bf16_to_f32(v_cache[kv_base + tid]);
            // Correct previous acc by rescaling, add new contribution
            acc_buf[tid] = acc_buf[tid] * correction + exp_score * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    if (tid < head_dim) {
        float result = acc_buf[tid] / running_sum;
        out[head_id * head_dim + tid] = f32_to_bf16(result);
    }
}

// ── kernel: gqa_attention_prefill ────────────────────────────────────────────
//
// Multi-token causal self-attention for the prefill phase.
// Uses a tiled approach: each threadgroup computes attention for one
// query token against chunks of the KV sequence.
//
// For Qwen3.6 with seq_len up to 131072 and n_heads=64:
//   - We process one (q_token, q_head) pair per threadgroup.
//   - Each threadgroup iterates over KV tiles of size TILE_K.
//   - Causal mask: query at position `q_pos` only attends to KV positions ≤ q_pos.
//
// This kernel is suitable for prompt processing up to ~4K tokens.
// For very long prompts (>4K), use chunked prefill on the CPU side
// (call this kernel multiple times with sub-ranges).
//
// grid:  (q_len * n_heads, 1, 1)
// threadgroup: (64, 1, 1)   (head_dim)

constant uint TILE_K = 64;   // KV tile size per iteration

kernel void gqa_attention_prefill(
    device const ushort* __restrict__ q          [[ buffer(0) ]],  // [q_len, n_heads, hd]
    device const ushort* __restrict__ k_cache    [[ buffer(1) ]],
    device const ushort* __restrict__ v_cache    [[ buffer(2) ]],
    device const uint*   __restrict__ page_table [[ buffer(3) ]],
    device       ushort* __restrict__ out        [[ buffer(4) ]],  // [q_len, n_heads, hd]
    constant     uint&                n_heads    [[ buffer(5) ]],
    constant     uint&                n_kv_heads [[ buffer(6) ]],
    constant     uint&                head_dim   [[ buffer(7) ]],
    constant     uint&                layer      [[ buffer(8) ]],
    constant     uint&                kv_len     [[ buffer(9) ]],
    constant     uint&                q_start    [[ buffer(10) ]],
    constant     uint&                max_pages  [[ buffer(11) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    uint tg_sz [[ threads_per_threadgroup ]]
) {
    uint q_len_local = tgid / n_heads;   // within current chunk
    uint head_id     = tgid % n_heads;
    uint q_pos       = q_start + q_len_local;   // absolute position
    uint kv_head     = head_id / (n_heads / n_kv_heads);
    float scale      = rsqrt((float)head_dim);

    // Load Q
    float q_val = 0.0f;
    if (tid < head_dim) {
        uint q_base = (q_len_local * n_heads + head_id) * head_dim;
        q_val = bf16_to_f32(q[q_base + tid]);
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    threadgroup float acc[64];
    if (tid < head_dim) acc[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal: only attend to positions 0..q_pos (inclusive)
    uint attend_len = min(kv_len, q_pos + 1u);

    for (uint pos = 0; pos < attend_len; pos++) {
        uint page_idx  = pos / PAGE_SIZE;
        uint page_slot = pos % PAGE_SIZE;
        uint block_idx = page_table[layer * max_pages + page_idx];
        uint kv_base   = ((block_idx * PAGE_SIZE + page_slot) * n_kv_heads + kv_head)
                         * head_dim;

        float dot = 0.0f;
        if (tid < head_dim) {
            float k_val = bf16_to_f32(k_cache[kv_base + tid]);
            dot = q_val * k_val;
        }
        dot = simd_sum(dot) * scale;

        float prev_max  = running_max;
        running_max     = max(running_max, dot);
        float exp_score = exp(dot - running_max);
        float correction= exp(prev_max - running_max);
        running_sum     = running_sum * correction + exp_score;

        if (tid < head_dim) {
            float v_val = bf16_to_f32(v_cache[kv_base + tid]);
            acc[tid] = acc[tid] * correction + exp_score * v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        uint out_base = (q_len_local * n_heads + head_id) * head_dim;
        out[out_base + tid] = f32_to_bf16(acc[tid] / running_sum);
    }
}

// ── kernel: copy_kv_to_output ────────────────────────────────────────────────
//
// Helper: after attention, copy the raw K/V tensors into contiguous output
// buffers for downstream use (e.g., debug / export).
// Not used in the hot inference path.

kernel void copy_kv_to_output(
    device const ushort* __restrict__ k_cache   [[ buffer(0) ]],
    device const ushort* __restrict__ v_cache   [[ buffer(1) ]],
    device       ushort* __restrict__ k_out     [[ buffer(2) ]],
    device       ushort* __restrict__ v_out     [[ buffer(3) ]],
    device const uint*   __restrict__ page_table[[ buffer(4) ]],
    constant     uint&                n_kv_heads[[ buffer(5) ]],
    constant     uint&                head_dim  [[ buffer(6) ]],
    constant     uint&                layer     [[ buffer(7) ]],
    constant     uint&                seq_len   [[ buffer(8) ]],
    constant     uint&                max_pages [[ buffer(9) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]]
) {
    uint tok_id  = tgid / n_kv_heads;
    uint kv_head = tgid % n_kv_heads;
    if (tok_id >= seq_len) return;

    uint page_idx  = tok_id / PAGE_SIZE;
    uint page_slot = tok_id % PAGE_SIZE;
    uint block_idx = page_table[layer * max_pages + page_idx];
    uint src_base  = ((block_idx * PAGE_SIZE + page_slot) * n_kv_heads + kv_head)
                     * head_dim;
    uint dst_base  = (tok_id * n_kv_heads + kv_head) * head_dim;

    if (tid < head_dim) {
        k_out[dst_base + tid] = k_cache[src_base + tid];
        v_out[dst_base + tid] = v_cache[src_base + tid];
    }
}
