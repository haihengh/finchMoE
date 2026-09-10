#include <metal_stdlib>
using namespace metal;

// ============================================================================
// gdn_prefill — batched Gated-DeltaNet (linear-attention) prefill unit.
//
// Prefill runs the SAME sequential per-token recurrence as the decode unit
// (`gdn.metal`), but batched over the chunk's T tokens: each kernel's work is
// independent across tokens except the causal-conv, which carries the 3 prior
// inputs across the chunk. The recurrence is inherently sequential within a
// value head (state S evolves token by token), so the per-value-head threadgroup
// loops over the T tokens in order, applying the exact decode step each time.
// This is correctness-by-construction: it reuses the validated decode math
// (`gdn_recurrent`), just unrolled over the chunk.
//
// Math is pinned to `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
// (see `docs/QWEN36_PORT.md`), and bit-mirrors the decode kernels in
// `gdn.metal`:
//   * torch_causal_conv1d_fn   (prefill, kernel 4, silu)
//   * l2norm + torch_recurrent_gated_delta_rule  (per token, in order)
//   * Qwen3_5MoeRMSNormGated   (mean-based, gated by silu(z))
//
// Layouts match decode:
//   recurrent state  fp32 [V][D][D], v-major: state[(hv*D + v)*D + k]
//   conv state       fp16 [C][3]  (persistent; updated via newState + blit)
//   conv in/out      fp16 [T][C]  (the conv output IS the [q,k,v] block per
//                                  token: q at [0,keyDim), k at [keyDim,2*keyDim),
//                                  v at [2*keyDim, 2*keyDim+valueDim))
//   z (in_proj_z)    fp16 [T][valueDim]
//   gate a|b         fp16 [T][2V]  (a rows then b rows per token — QMM output)
//   gate g/beta      fp32 [T][V]
//
// GQA: value head `hv` uses key head `hv / 2` (repeat_interleave(2)).
//
// Dispatch:
//   prefill_gdn_conv_chunk     — one thread per (channel, token); the last
//                                token's row commits the post-chunk conv state
//                                to a SEPARATE newState buffer (the wrapper
//                                blits it into the persistent state).
//   prefill_gdn_recurrent_seq  — one 256-thread threadgroup per value head,
//                                looping the chunk's T tokens sequentially.
//   prefill_gdn_gate           — one thread per (token, value head).
//   prefill_gdn_rmsnorm_gated  — one 256-thread threadgroup per (token, value head).
// ============================================================================

// Compile-time bound for per-head threadgroup scratch (see gdn.metal).
constant constexpr uint kGdnPrefillMaxHeadDim = 256;
constant constexpr uint kGdnPrefillMaxSimdGroups = 8;

// Reuse the decode unit's stable softplus (same module library, same MSL).
// Defined here to keep this module self-contained; identical op-tree to
// gdn.metal's gdn_softplus.
static inline float gdn_prefill_softplus(float x) {
    if (x > 0.0f) return x + log(1.0f + exp(-x));
    return log(1.0f + exp(x));
}

static inline float gdn_prefill_silu(float x) { return x / (1.0f + exp(-x)); }

// Two-stage SIMD-group block sum (mirror of gdn_block_sum in gdn.metal).
static inline void gdn_prefill_block_sum(
    float acc,
    uint  simd_lane,
    uint  simd_group,
    uint  simdgroups,
    threadgroup float* partial
) {
    acc = simd_sum(acc);
    if (simd_lane == 0) partial[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float v = (simd_lane < simdgroups) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ----------------------------------------------------------------------------
// Batched causal conv1d over the chunk (kernel 4, silu).
//
// For token t, channel c:
//   out[t][c]    = silu(w0*x[t-3][c] + w1*x[t-2][c] + w2*x[t-1][c] + w3*x[t][c])
// where x[t-1..t-3] are the raw (pre-silu) inputs of prior tokens, seeded from
// the persistent conv `state` for the tokens before the chunk.
//
// x is [T][C] raw inputs; out is [T][C] silu'd; w is [C,4]; state [C,3],
// newState [C,3]. One thread per (c,t). Threads at the same t are
// independent, but threads at different t READ x — so `out` must NOT alias
// `x`: thread (t+3, c) reads the raw tap x[t][c] while thread (t, c) writes
// out[t][c], and cross-threadgroup ordering is not guaranteed. (The decode
// conv is single-token, where the read-then-write happens in one thread and
// in-place is safe; the batched form is not.)
//
// newState is a SEPARATE buffer (chunk scratch, blit-copied into the
// persistent state by the wrapper): the only threads reading `state` are the
// t < 3 seeding rows, while the t == T-1 row writes newState, so an in-place
// newState == state would race across threads.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_gdn_conv_chunk(
    device const half* w          [[buffer(0)]],   // [C, 4]
    device const half* state      [[buffer(1)]],   // [C, 3] (pre-chunk, read)
    device const half* x          [[buffer(2)]],   // [T][C] raw inputs
    device       half* out        [[buffer(3)]],   // [T][C] silu'd outputs
    device       half* newState   [[buffer(4)]],   // [C, 3] (post-chunk, written)
    constant     uint&  C         [[buffer(5)]],
    constant     uint&  T         [[buffer(6)]],
    uint2  gid                    [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C) return;

    // Gather the 4 taps for this (c, t): 3 prior raw inputs + current.
    // state[c][j] holds the (j+1)-th-most-recent input before the chunk
    // (state[c][0] oldest, state[c][2] newest = x[-1]).
    float s0, s1, s2;
    if (t >= 3u) {
        s0 = float(x[(t - 3u) * C + c]);
        s1 = float(x[(t - 2u) * C + c]);
        s2 = float(x[(t - 1u) * C + c]);
    } else {
        s0 = (t < 3u) ? float(state[c*3 + t])      : float(x[(t - 3u) * C + c]);
        s1 = (t < 2u) ? float(state[c*3 + t + 1u]) : float(x[(t - 2u) * C + c]);
        s2 = (t < 1u) ? float(state[c*3 + t + 2u]) : float(x[(t - 1u) * C + c]);
    }
    const float cur = float(x[t * C + c]);

    const float acc = float(w[c*4 + 0]) * s0
                    + float(w[c*4 + 1]) * s1
                    + float(w[c*4 + 2]) * s2
                    + float(w[c*4 + 3]) * cur;
    out[t * C + c] = half(gdn_prefill_silu(acc));

    // The last token's row commits the post-chunk state = the last 3 raw
    // inputs: newState[j] = x[T-3+j], with negative indices seeded from the
    // pre-chunk state (x[idx] lives at state[c*3 + 3 + idx]). Handles all T
    // (1, 2, or >= 3).
    if (t + 1u == T) {
        for (uint j = 0; j < 3u; ++j) {
            const int idx = int(T) - 3 + int(j);
            newState[c*3 + j] = (idx >= 0)
                ? x[idx * C + c]
                : state[c*3 + (3 + idx)];
        }
    }
}

// ----------------------------------------------------------------------------
// Batched recurrent gated-delta-rule: the decode step, applied sequentially to
// each of the chunk's T tokens for one value head (one threadgroup).
//
// For token t (in order), with q_t/k_t/v_t the t-th rows of the conv output
// (q at element offset 0, k at kOff, v at vOff; row stride C):
//   qn    = l2norm(q[hv/2]_t) * scale  // scale AFTER the norm, per the torch
//                                      // chunk rule (scaling inside cancels)
//   kn    = l2norm(k[hv/2]_t)
//   decay = exp(g[t][hv])
//   S     = S * decay
//   r[v]  = sum_k S[v,k]*kn[k]
//   delta = beta[t][hv] * (v_t[v] - r[v])
//   S     = S + outer(kn, delta)
//   o[t][hv,v] = sum_k S[v,k]*qn[k]
//
// S is fp32 in `state` (shared across all tokens — the whole point of the
// recurrence). conv is fp16 [T][C]; g/beta are fp32 [T][V]; out is fp16
// [T][V][D] (= [T][valueDim] contiguous).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_gdn_recurrent_seq(
    device       float* state      [[buffer(0)]],   // [V][D][D] fp32, v-major (in/out)
    device const half*  conv       [[buffer(1)]],   // [T][C] fp16 (q|k|v fused)
    device const float* g          [[buffer(2)]],   // [T][V] fp32
    device const float* beta       [[buffer(3)]],   // [T][V] fp32
    device       half*  out        [[buffer(4)]],   // [T][V][D] fp16
    constant     uint&  D          [[buffer(5)]],
    constant     uint&  C          [[buffer(6)]],   // conv row stride (elements)
    constant     uint&  kOff       [[buffer(7)]],   // k block offset (elements)
    constant     uint&  vOff       [[buffer(8)]],   // v block offset (elements)
    constant     uint&  V          [[buffer(9)]],   // number of value heads
    constant     uint&  T          [[buffer(10)]],
    constant     float& scale      [[buffer(11)]],  // 1/sqrt(head_dim)
    constant     float& l2eps      [[buffer(12)]],
    uint  hv                       [[threadgroup_position_in_grid]],
    uint  lid                      [[thread_position_in_threadgroup]],
    uint  lsize                    [[threads_per_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_group               [[simdgroup_index_in_threadgroup]],
    uint  simdgroups               [[simdgroups_per_threadgroup]]
) {
    if (hv >= V) return;
    const uint headStateElems = D * D;    // per-head state element count
    const uint kh = hv / 2;              // key head for this value head

    device float* S = state + hv * headStateElems;  // state is [V][D][D], hv-major

    threadgroup float qn[kGdnPrefillMaxHeadDim];
    threadgroup float kn[kGdnPrefillMaxHeadDim];
    threadgroup float pQ[8];
    threadgroup float pK[8];

    for (uint t = 0; t < T; ++t) {
        device const half* qh = conv + t * C + kh * D;        // q[t][kh][:]
        device const half* kh_ = conv + t * C + kOff + kh * D; // k[t][kh][:]
        device const half* vh = conv + t * C + vOff + hv * D;  // v[t][hv][:]
        device half* oh = out + t * (V * D) + hv * D;          // out[t][hv][:]
        const float decay = exp(g[t * V + hv]);
        const float betaV = beta[t * V + hv];

        // Pass 1: l2norm of q and k into threadgroup scratch (scale applied
        // to the NORMED q below — after the l2norm, matching the torch oracle).
        float accq = 0.0f, acck = 0.0f;
        for (uint i = lid; i < D; i += lsize) {
            float qv = float(qh[i]);
            float kv = float(kh_[i]);
            accq = fma(qv, qv, accq);
            acck = fma(kv, kv, acck);
        }
        gdn_prefill_block_sum(accq, simd_lane, simd_group, simdgroups, pQ);
        gdn_prefill_block_sum(acck, simd_lane, simd_group, simdgroups, pK);
        const float qinv = rsqrt(pQ[0] + l2eps);
        const float kinv = rsqrt(pK[0] + l2eps);
        for (uint i = lid; i < D; i += lsize) {
            qn[i] = float(qh[i]) * qinv * scale;
            kn[i] = float(kh_[i]) * kinv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // kn·qn constant across v — compute once for the readout shortcut.
        float dotknqn = 0.0f;
        for (uint i = lid; i < D; i += lsize) {
            dotknqn = fma(kn[i], qn[i], dotknqn);
        }
        dotknqn = simd_sum(dotknqn);
        if (simd_lane == 0) pQ[simd_group] = dotknqn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float tt = (simd_lane < simdgroups) ? pQ[simd_lane] : 0.0f;
            tt = simd_sum(tt);
            if (simd_lane == 0) pQ[0] = tt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float knq = pQ[0];

        // Pass 2: per-v decay, residual read, rank-1 write, updated-state readout.
        for (uint vIdx = lid; vIdx < D; vIdx += lsize) {
            device float* row = S + vIdx * D;

            for (uint kk = 0; kk < D; kk++) row[kk] *= decay;

            float r = 0.0f;
            float base = 0.0f;
            for (uint kk = 0; kk < D; kk++) {
                r    = fma(row[kk], kn[kk], r);
                base = fma(row[kk], qn[kk], base);
            }

            const float delta = betaV * (float(vh[vIdx]) - r);

            for (uint kk = 0; kk < D; kk++) row[kk] += kn[kk] * delta;

            oh[vIdx] = half(base + delta * knq);
        }
        // Barrier so the next token's l2norm pass doesn't race this token's
        // threadgroup-scratch reuse. (S itself is device memory, but the
        // threadgroup qn/kn/pQ/pK buffers are reused next iteration.)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ----------------------------------------------------------------------------
// Batched gate over the chunk: one thread per (token, value head).
//
//   beta[t][v] = sigmoid(b)
//   g[t][v]    = -exp(A_log[v]) * softplus(a + dt_bias[v])
//
// `ab` is the fp16 QMM output [T][2V] (a rows then b rows per token);
// A_log/dt_bias are [V] fp32; g/beta are [T][V] fp32 outputs. The formula is
// applied in fp32 from the fp16-rounded a/b, mirroring
// `Qwen3_5MoeGatedDeltaNet.forward`.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_gdn_gate(
    device const half*  ab         [[buffer(0)]],   // [T][2V] fp16 (a|b rows)
    device const float* A_log      [[buffer(1)]],   // [V] fp32
    device const float* dt_bias    [[buffer(2)]],   // [V] fp32
    device       float* g          [[buffer(3)]],   // [T][V] fp32 out
    device       float* beta       [[buffer(4)]],   // [T][V] fp32 out
    constant     uint&  V          [[buffer(5)]],
    constant     uint&  T          [[buffer(6)]],
    uint2  gid                     [[thread_position_in_grid]]
) {
    const uint v = gid.x;
    const uint t = gid.y;
    if (v >= V || t >= T) return;

    const float av   = float(ab[t * (2u * V) + v]);
    const float bv   = float(ab[t * (2u * V) + V + v]);
    const float dt   = dt_bias[v];
    const float Alog = A_log[v];
    g[t * V + v]    = -exp(Alog) * gdn_prefill_softplus(av + dt);
    beta[t * V + v] = 1.0f / (1.0f + exp(-bv));
}

// ----------------------------------------------------------------------------
// Batched gated RMSNorm over each (token, value head) vector:
//   y[t][hv][i] = x[t][hv][i] * rsqrt(mean_i(x[t][hv][i]^2) + eps)
//                 * weight[i] * act(z[t][hv][i])
// One 256-thread threadgroup per (token, value head) — linearized 1D grid,
// `t = head / V`, `hv = head % V`. `weight` is shared across all heads
// (Qwen3_5MoeRMSNormGated(head_v_dim)); x/z are [T][V][D] fp16. Mean-based
// (not sum), matching `Qwen3_5MoeRMSNormGated.forward`.
//
// `act` is silu (Qwen 3.5/3.6) or sigmoid (Qwen 3.8 Flash-Next, qwen4exp
// `build_norm_gated`), selected by the same function constant the decode
// kernel uses — FC_GDN_RMSNORM_GATE_SIGMOID at index 66, declared once in
// `gdn.metal`. Both files land in one merged source, so declaring it here
// again is a redefinition and a duplicate index; one declaration is what
// makes the two forms unable to disagree about the family. The constant is
// absent for a 3.6 install, which leaves this path bit-identical to before.
// ----------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_gdn_rmsnorm_gated(
    device const half*   x          [[buffer(0)]],   // [T][V][D] fp16
    device const half*   z          [[buffer(1)]],   // [T][V][D] fp16
    device const bfloat* weight     [[buffer(2)]],   // [D] bf16, shared per head
    device       half*   out        [[buffer(3)]],   // [T][V][D] fp16
    constant     uint&   D          [[buffer(4)]],
    constant     uint&   V          [[buffer(5)]],
    constant     uint&   T          [[buffer(6)]],
    constant     float&  eps        [[buffer(7)]],
    uint  head                     [[threadgroup_position_in_grid]],
    uint  lid                      [[thread_position_in_threadgroup]],
    uint  lsize                    [[threads_per_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_group               [[simdgroup_index_in_threadgroup]],
    uint  simdgroups               [[simdgroups_per_threadgroup]]
) {
    const uint t = head / V;
    const uint hv = head % V;
    if (t >= T || hv >= V) return;

    threadgroup float partial[kGdnPrefillMaxSimdGroups];
    const uint rowBase = (t * V + hv) * D;
    device const half* xh = x   + rowBase;
    device const half* zh = z   + rowBase;
    device       half* oh = out + rowBase;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float vv = float(xh[i]);
        acc = fma(vv, vv, acc);
    }
    gdn_prefill_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(D) + eps);

    // Uniform per-pipeline branch, as in `gdn_rmsnorm_gated`.
    const bool sigmoidGate =
        is_function_constant_defined(FC_GDN_RMSNORM_GATE_SIGMOID) &&
        FC_GDN_RMSNORM_GATE_SIGMOID;

    for (uint i = lid; i < D; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        float zv = float(zh[i]);
        float act = sigmoidGate ? 1.0f / (1.0f + exp(-zv)) : gdn_prefill_silu(zv);
        oh[i] = half(xv * inv * wv * act);
    }
}
