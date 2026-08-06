// moe_kernels.metal
// ─────────────────────────────────────────────────────────────────────────────
// MoE routing and expert computation for Qwen3.6-35B-A3B
//
// Qwen3.6 MoE specifics:
//   num_experts        = 128   (routed, per layer)
//   num_experts_per_tok= 8     (top-k)
//   norm_topk_prob     = true  (renormalize selected probabilities to sum=1)
//   shared experts     = 1     (always-active, separate from routing)
//
// Kernels:
//   moe_router_topk      — softmax over gate logits → top-k indices + weights
//   moe_expert_ffn_bf16  — SiLU-gated FFN for a single expert (gate·up→silu→down)
//   moe_reduce_bf16      — weighted sum of expert outputs → hidden
//   moe_shared_ffn_bf16  — always-active shared expert (same SiLU-gated FFN)
//   moe_fused_gate_up    — fused gate+up projection for one expert (single GEMV)
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── bf16 helpers ──────────────────────────────────────────────────────────────
inline float bf16_to_f32(ushort u) {
    return as_type<float>((uint)u << 16);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ── kernel: moe_router_topk ───────────────────────────────────────────────────
//
// Computes softmax over gate logits and selects top-k experts with optional
// renormalization (norm_topk_prob=true for Qwen3.6).
//
// gate_logits:   [seq_len, num_experts]   float32  (output of gate.weight @ hidden)
// expert_ids:    [seq_len, top_k]         uint32   (OUTPUT: selected expert indices)
// expert_weights:[seq_len, top_k]         float32  (OUTPUT: normalized weights)
//
// grid:       (seq_len, 1, 1)
// threadgroup: (num_experts, 1, 1)   — up to 128 threads; one per expert
//
// Strategy:
//   1. Each thread loads one logit.
//   2. SIMD tree-reduce to find max → stable softmax.
//   3. SIMD sum for normalization denominator.
//   4. Each thread has its own softmax probability.
//   5. Parallel top-k selection: threads write (prob, idx) to threadgroup
//      memory; a single thread performs the top-k sort.
//   For top_k=8 and num_experts=128 this is fast enough on-device.

constant uint MAX_EXPERTS = 128;
constant uint TOP_K       = 8;

kernel void moe_router_topk(
    device const float*  __restrict__ gate_logits    [[ buffer(0) ]],  // [seq, E]
    device       uint*   __restrict__ expert_ids     [[ buffer(1) ]],  // [seq, K]
    device       float*  __restrict__ expert_weights [[ buffer(2) ]],  // [seq, K]
    constant     uint&                num_experts    [[ buffer(3) ]],
    constant     uint&                top_k          [[ buffer(4) ]],
    constant     uint&                norm_topk      [[ buffer(5) ]],  // bool
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    uint tg_sz [[ threads_per_threadgroup ]]
) {
    uint seq_pos  = tgid;
    uint base     = seq_pos * num_experts;

    // Load logit
    float logit = (tid < num_experts) ? gate_logits[base + tid] : -INFINITY;

    // ── stable softmax ────────────────────────────────────────────────────────
    // Step 1: find max across all expert logits
    float max_logit = simd_max(logit);
    // Multi-group reduce for num_experts > 32
    threadgroup float tg_max[4];   // max 4 SIMD groups (128/32)
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_lane == 0 && simd_id < 4) tg_max[simd_id] = max_logit;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_max[0];
    for (uint g = 1; g < (num_experts + 31u) / 32u; ++g)
        global_max = max(global_max, tg_max[g]);

    // Step 2: compute exp and sum
    float exp_val = (tid < num_experts) ? exp(logit - global_max) : 0.0f;
    float exp_sum = simd_sum(exp_val);
    threadgroup float tg_sum[4];
    if (simd_lane == 0 && simd_id < 4) tg_sum[simd_id] = exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = 0.0f;
    for (uint g = 0; g < (num_experts + 31u) / 32u; ++g)
        global_sum += tg_sum[g];

    float prob = (tid < num_experts) ? exp_val / global_sum : 0.0f;

    // ── top-k selection (done by thread 0 from shared memory) ─────────────────
    threadgroup float tg_probs[MAX_EXPERTS];
    threadgroup uint  tg_topk_ids[TOP_K];
    threadgroup float tg_topk_ws[TOP_K];

    if (tid < num_experts) tg_probs[tid] = prob;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Simple selection sort for top_k (k=8, E=128 — O(k*E)=1024 ops)
        bool selected[MAX_EXPERTS] = {};
        float topk_sum = 0.0f;
        for (uint k = 0; k < top_k; ++k) {
            float best_p = -1.0f;
            uint  best_i = 0;
            for (uint e = 0; e < num_experts; ++e) {
                if (!selected[e] && tg_probs[e] > best_p) {
                    best_p = tg_probs[e];
                    best_i = e;
                }
            }
            selected[best_i]    = true;
            tg_topk_ids[k]      = best_i;
            tg_topk_ws[k]       = best_p;
            topk_sum           += best_p;
        }
        // Renormalize if requested
        if (norm_topk && topk_sum > 0.0f) {
            for (uint k = 0; k < top_k; ++k)
                tg_topk_ws[k] /= topk_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    uint out_base = seq_pos * top_k;
    if (tid < top_k) {
        expert_ids[out_base + tid]     = tg_topk_ids[tid];
        expert_weights[out_base + tid] = tg_topk_ws[tid];
    }
}

// ── kernel: moe_expert_ffn_bf16 ───────────────────────────────────────────────
//
// Computes one expert's SiLU-gated FFN:
//   gate_out = gate_proj @ x           [moe_inter]
//   up_out   = up_proj   @ x           [moe_inter]
//   hidden   = silu(gate_out) * up_out [moe_inter]
//   out      = down_proj @ hidden      [hidden_size]
//
// One kernel call per active expert per token.
//
// x:         [hidden_size]     bf16
// gate_w:    [moe_inter, hidden_size]  bf16  (gate_proj.weight, row-major)
// up_w:      [moe_inter, hidden_size]  bf16
// down_w:    [hidden_size, moe_inter]  bf16
// out:       [hidden_size]     float32  (accumulate into caller's buffer)
//
// grid:       (moe_inter, 1, 1)    one threadgroup per output row of gate/up
// threadgroup: (32, 1, 1)          SIMD-width for dot product
//
// Note: the final down_proj GEMV is a separate kernel dispatch (moe_down_proj)
// because its output dimension (hidden_size=4096) differs from moe_inter(768).
// We split into two kernels to avoid allocating a large intermediate tensor.
// Intermediate (hidden): [moe_inter] float32 stored in a small tmp buffer.

kernel void moe_gate_up_silu_bf16(
    device const ushort* __restrict__ x        [[ buffer(0) ]],  // [hidden]
    device const ushort* __restrict__ gate_w   [[ buffer(1) ]],  // [moe_inter, hidden]
    device const ushort* __restrict__ up_w     [[ buffer(2) ]],  // [moe_inter, hidden]
    device       float*  __restrict__ tmp_out  [[ buffer(3) ]],  // [moe_inter] float32
    constant     uint&                hidden   [[ buffer(4) ]],
    constant     uint&                moe_inter[[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    // tgid = output row index in gate/up [0, moe_inter)
    uint row  = tgid;
    uint base = row * hidden;

    float gate_dot = 0.0f;
    float up_dot   = 0.0f;

    for (uint i = tid; i < hidden; i += tg_sz) {
        float xi = bf16_to_f32(x[i]);
        gate_dot += xi * bf16_to_f32(gate_w[base + i]);
        up_dot   += xi * bf16_to_f32(up_w  [base + i]);
    }

    // SIMD reduce
    gate_dot = simd_sum(gate_dot);
    up_dot   = simd_sum(up_dot);

    if (tid == 0) {
        tmp_out[row] = silu(gate_dot) * up_dot;
    }
}

// ── kernel: moe_down_proj_bf16 ────────────────────────────────────────────────
//
// Computes: out += weight * down_proj @ tmp  (accumulate into float32 output)
//
// tmp:    [moe_inter]  float32
// down_w: [hidden, moe_inter]  bf16
// out:    [hidden]     float32  (accumulated; zeroed before first expert)
// weight: scalar expert routing weight (float32)
//
// grid:       (hidden, 1, 1)
// threadgroup: (32, 1, 1)

kernel void moe_down_proj_bf16(
    device const float*  __restrict__ tmp     [[ buffer(0) ]],  // [moe_inter]
    device const ushort* __restrict__ down_w  [[ buffer(1) ]],  // [hidden, moe_inter]
    device       float*  __restrict__ out     [[ buffer(2) ]],  // [hidden]  accumulated
    constant     uint&                hidden  [[ buffer(3) ]],
    constant     uint&                moe_inter[[ buffer(4) ]],
    constant     float&               weight  [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    uint row  = tgid;  // output hidden dimension index
    uint base = row * moe_inter;

    float dot = 0.0f;
    for (uint i = tid; i < moe_inter; i += tg_sz) {
        dot += tmp[i] * bf16_to_f32(down_w[base + i]);
    }
    dot = simd_sum(dot);

    if (tid == 0) {
        // Atomic add to handle concurrent expert writes
        // (only needed when multiple experts are dispatched to same output buffer)
        // For serial expert dispatch: plain add is fine.
        out[row] += weight * dot;
    }
}

// ── kernel: moe_reduce_to_bf16 ────────────────────────────────────────────────
//
// Final step: convert the float32 accumulated expert output back to bf16
// and add residual.
//
// accum:    [hidden]  float32  (sum of weighted expert outputs)
// residual: [hidden]  bf16
// out:      [hidden]  bf16
//
// grid:       (1, 1, 1)
// threadgroup: (256, 1, 1)

kernel void moe_reduce_to_bf16(
    device const float*  __restrict__ accum    [[ buffer(0) ]],
    device const ushort* __restrict__ residual [[ buffer(1) ]],
    device       ushort* __restrict__ out      [[ buffer(2) ]],
    constant     uint&                hidden   [[ buffer(3) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    for (uint i = tid; i < hidden; i += tg_sz) {
        float val = accum[i] + bf16_to_f32(residual[i]);
        out[i] = f32_to_bf16(val);
    }
}

// ── kernel: moe_zero_accum ────────────────────────────────────────────────────
//
// Zero the float32 accumulator before dispatching expert kernels.
// Faster than a CPU-side memset since it avoids a PCIe round-trip.

kernel void moe_zero_accum(
    device float*  __restrict__ accum  [[ buffer(0) ]],
    constant uint& hidden              [[ buffer(1) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    for (uint i = tid; i < hidden; i += tg_sz) accum[i] = 0.0f;
}

// ── kernel: moe_shared_ffn_bf16 ───────────────────────────────────────────────
//
// Shared (always-active) expert FFN. Identical computation to the routed
// experts but uses the shared weight tensors and writes to a separate
// accumulator (added to the routed-expert accumulator on the CPU before
// moe_reduce_to_bf16).
//
// Uses the same moe_gate_up_silu_bf16 + moe_down_proj_bf16 kernels with
// different weight buffers and inter_size = intermediate_size (not moe_inter).
// This is dispatched from the CPU as a separate pair of kernel calls.
// No separate Metal kernel is needed — we reuse the two above.
// This comment block documents the calling convention.
//
// CPU dispatch sequence for one MoE layer (one token):
//
//   1. moe_zero_accum(accum_routed, hidden)
//   2. For each active expert e in top_k:
//        a. moe_gate_up_silu_bf16(x, expert_e.gate, expert_e.up, tmp, hidden, moe_inter)
//        b. moe_down_proj_bf16(tmp, expert_e.down, accum_routed, hidden, moe_inter, w_e)
//   3. moe_zero_accum(accum_shared, hidden)
//   4. moe_gate_up_silu_bf16(x, shared.gate, shared.up, tmp_shared, hidden, inter_size)
//   5. moe_down_proj_bf16(tmp_shared, shared.down, accum_shared, hidden, inter_size, 1.0)
//   6. Element-wise add accum_shared into accum_routed  [see moe_add_accum below]
//   7. moe_reduce_to_bf16(accum_routed, residual, out, hidden)

// ── kernel: moe_add_accum ────────────────────────────────────────────────────
// Add shared-expert accumulator into routed accumulator.

kernel void moe_add_accum(
    device       float*  __restrict__ dst [[ buffer(0) ]],
    device const float*  __restrict__ src [[ buffer(1) ]],
    constant     uint&                n   [[ buffer(2) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    for (uint i = tid; i < n; i += tg_sz) dst[i] += src[i];
}
