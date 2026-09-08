#include <metal_stdlib>
using namespace metal;

// ============================================================================
// gdn — Gated-DeltaNet (linear attention) decode unit.
//
// Math is pinned to `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:
//   * torch_causal_conv1d_update  (decode, kernel 4)
//   * l2norm                       (sum-based, eps 1e-6)
//   * torch_recurrent_gated_delta_rule (decode recurrence)
//
// Recurrent state convention: v-major, `state[(hv*D + v)*D + k]`, so a matvec
// `sum_k S[v,k]*x[k]` and the rank-1 write `S[v,k] += k[k]*delta[v]` are both
// contiguous over `k`. The state is fp32 (64 KB per head) and lives in a
// persistent device buffer — it exceeds the 32 KB threadgroup memory, so it
// is read/written through a buffer, not staged in threadgroup memory.
//
// GQA: value head `hv` uses key head `hv / 2` (repeat_interleave(2)).
//
// Dispatch:
//   gdn_conv_update  — one thread per channel (32-thread groups).
//   gdn_recurrent    — one 256-thread threadgroup per value head.
// ============================================================================

static inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }

// Compile-time bound for the per-head threadgroup scratch (see gdn_recurrent).
constant constexpr uint kGdnMaxHeadDim = 256;

// Threadgroup partial slots for a 256-thread group (256 / 32 SIMD-groups).
constant constexpr uint kGdnMaxSimdGroups = 8;

// Stable softplus: `log(1 + exp(x))` branched so `exp` never overflows.
// MSL has no `log1p`, so this uses only `log`/`exp` — and is a different
// op-tree than the CPU reference's `log1p` form, which keeps the comparison
// meaningful.
static inline float gdn_softplus(float x) {
    if (x > 0.0f) return x + log(1.0f + exp(-x));
    return log(1.0f + exp(x));
}

// ----------------------------------------------------------------------------
// Causal conv1d decode update (kernel 4):
//   out[c]    = silu(w0*s0 + w1*s1 + w2*s2 + w3*x)
//   new_state = [s1, s2, x]
// One thread per channel; w is [C,4], state/newState [C,3], x/out [C].
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void gdn_conv_update(
    device const half* w          [[buffer(0)]],   // [C, 4]
    device const half* state      [[buffer(1)]],   // [C, 3]
    device const half* x          [[buffer(2)]],   // [C]
    device       half* out        [[buffer(3)]],   // [C]
    device       half* newState   [[buffer(4)]],   // [C, 3]
    constant     uint&  C         [[buffer(5)]],
    uint  lid                     [[thread_position_in_threadgroup]],
    uint  lsize                   [[threads_per_threadgroup]],
    uint  tgx                     [[threadgroup_position_in_grid]]
) {
    uint c = tgx * lsize + lid;
    if (c >= C) return;

    float acc = float(w[c*4 + 0]) * float(state[c*3 + 0])
              + float(w[c*4 + 1]) * float(state[c*3 + 1])
              + float(w[c*4 + 2]) * float(state[c*3 + 2])
              + float(w[c*4 + 3]) * float(x[c]);
    out[c] = half(gdn_silu(acc));

    newState[c*3 + 0] = state[c*3 + 1];
    newState[c*3 + 1] = state[c*3 + 2];
    newState[c*3 + 2] = x[c];
}

// Two-stage SIMD-group block sum of a per-thread `acc` into `partial[0]`.
static inline void gdn_block_sum(
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
// Recurrent gated-delta-rule decode step for one value head (one threadgroup).
//
//   qn    = l2norm(q[hv/2]) * scale   // scale AFTER the norm — torch l2norms
//                                      // then applies 1/sqrt(head_dim); scaling
//                                      // inside the norm would cancel it out
//   kn    = l2norm(k[hv/2])
//   decay = exp(g[hv])
//   S     = S * decay
//   r[v]  = sum_k S[v,k]*kn[k]
//   delta = beta[hv] * (v[v] - r[v])
//   S     = S + outer(kn, delta)          // S[v,k] += kn[k]*delta
//   o[v]  = sum_k S[v,k]*qn[k]            // read from the UPDATED state
//
// State is fp32 in a buffer; q/k/v are fp16; g/beta are per-head fp32 scalars.
// The only cross-thread reductions are the two l2norm sums (pass 1). Pass 2
// is a pure per-thread dot product + rank-1 write per `v` (one thread per v).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void gdn_recurrent(
    device       float* state      [[buffer(0)]],   // [V][D][D] fp32, v-major
    device const half*  q          [[buffer(1)]],   // [K][D] fp16
    device const half*  k          [[buffer(2)]],   // [K][D] fp16
    device const half*  v          [[buffer(3)]],   // [V][D] fp16
    device const float* g          [[buffer(4)]],   // [V] fp32
    device const float* beta       [[buffer(5)]],   // [V] fp32
    device       half*  out        [[buffer(6)]],   // [V][D] fp16
    constant     uint&  D          [[buffer(7)]],
    constant     float& scale      [[buffer(8)]],   // 1/sqrt(head_dim)
    constant     float& l2eps      [[buffer(9)]],
    uint  hv                       [[threadgroup_position_in_grid]],
    uint  lid                      [[thread_position_in_threadgroup]],
    uint  lsize                    [[threads_per_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_group               [[simdgroup_index_in_threadgroup]],
    uint  simdgroups               [[simdgroups_per_threadgroup]]
) {
    const uint K = D * D;                 // per-head state element count
    const uint kh = hv / 2;              // key head for this value head
    device const half* qh = q + kh * D;
    device const half* kh_ = k + kh * D;
    device const half* vh = v + hv * D;
    device float* S = state + hv * K;
    device half* oh = out + hv * D;
    const float decay = exp(g[hv]);
    const float betaV = beta[hv];

    // Head dim is a runtime buffer constant; threadgroup arrays need a
    // compile-time bound. GDN heads are 128; 256 leaves headroom.
    threadgroup float qn[kGdnMaxHeadDim];
    threadgroup float kn[kGdnMaxHeadDim];
    threadgroup float pQ[8];
    threadgroup float pK[8];

    // Pass 1: l2norm of q and k into threadgroup scratch (scale applied to
    // the NORMED q below — after the l2norm, matching the torch oracle).
    float accq = 0.0f, acck = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float qv = float(qh[i]);
        float kv = float(kh_[i]);
        accq = fma(qv, qv, accq);
        acck = fma(kv, kv, acck);
    }
    gdn_block_sum(accq, simd_lane, simd_group, simdgroups, pQ);
    gdn_block_sum(acck, simd_lane, simd_group, simdgroups, pK);
    const float qinv = rsqrt(pQ[0] + l2eps);
    const float kinv = rsqrt(pK[0] + l2eps);
    for (uint i = lid; i < D; i += lsize) {
        qn[i] = float(qh[i]) * qinv * scale;
        kn[i] = float(kh_[i]) * kinv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // kn·qn is constant across v — compute once for the readout shortcut.
    float dotknqn = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        dotknqn = fma(kn[i], qn[i], dotknqn);
    }
    dotknqn = simd_sum(dotknqn);
    if (simd_lane == 0) pQ[simd_group] = dotknqn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float t = (simd_lane < simdgroups) ? pQ[simd_lane] : 0.0f;
        t = simd_sum(t);
        if (simd_lane == 0) pQ[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float knq = pQ[0];

    // Pass 2: per-v decay, residual read, rank-1 write, updated-state readout.
    for (uint vIdx = lid; vIdx < D; vIdx += lsize) {
        device float* row = S + vIdx * D;

        // decay the row in place
        for (uint kk = 0; kk < D; kk++) row[kk] *= decay;

        // r = row · kn   (decayed)
        float r = 0.0f;
        float base = 0.0f;              // row · qn (decayed), for the readout
        for (uint kk = 0; kk < D; kk++) {
            r    = fma(row[kk], kn[kk], r);
            base = fma(row[kk], qn[kk], base);
        }

        const float delta = betaV * (float(vh[vIdx]) - r);

        // rank-1 write
        for (uint kk = 0; kk < D; kk++) row[kk] += kn[kk] * delta;

        // o = sum_k (row + kn*delta)·qn = base + delta·(kn·qn)
        oh[vIdx] = half(base + delta * knq);
    }
}

// ----------------------------------------------------------------------------
// Per-value-head gate: `beta = sigmoid(b)` and
//   `g = -exp(A_log) * softplus(a + dt_bias)`
// One thread per value head; a/b/dt_bias/A_log are per-head fp32 vectors,
// g/beta are per-head fp32 outputs. Pinned to Qwen3_5MoeGatedDeltaNet.forward:
//   beta = b.sigmoid()
//   g    = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
// (a/b come from the in_proj_a/in_proj_b GEMVs; fp32 here for a clean,
// isolated formula check.)
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void gdn_gate(
    device const float* a          [[buffer(0)]],   // [V] fp32
    device const float* b          [[buffer(1)]],   // [V] fp32
    device const float* A_log      [[buffer(2)]],   // [V] fp32
    device const float* dt_bias    [[buffer(3)]],   // [V] fp32
    device       float* g          [[buffer(4)]],   // [V] fp32 out
    device       float* beta       [[buffer(5)]],   // [V] fp32 out
    constant     uint&  V          [[buffer(6)]],
    uint  lid                      [[thread_position_in_threadgroup]]
) {
    if (lid >= V) return;
    float av   = a[lid];
    float bv   = b[lid];
    float dt   = dt_bias[lid];
    float Alog = A_log[lid];
    g[lid]    = -exp(Alog) * gdn_softplus(av + dt);
    beta[lid] = 1.0f / (1.0f + exp(-bv));
}

// ----------------------------------------------------------------------------
// Fused in_proj_a/in_proj_b GEMVs + per-head gate. One 256-thread threadgroup
// computes all 2V rows (V a-rows then V b-rows, V <= 32) as int4-affine GEMV
// rows — 8 SIMD groups, 8 sequential passes — staging the fp32 accs in
// threadgroup memory, then the gate formula from
// Qwen3_5MoeGatedDeltaNet.forward:
//   beta = sigmoid(b)
//   g    = -exp(A_log) * softplus(a + dt_bias)
// W is [2V, N/2] nibbles with BF16 [2V, N/64] scales/biases (MLX affine
// layout, same as dequant_int4.metal); x is [N] fp16; A_log/dt_bias [V] fp32;
// g/beta [V] fp32 outputs. The GEMV loop mirrors
// dequant_int4_gemv_simd_body's group-64 scalar path (that body sinks into
// device halves, so the loop is restated with a threadgroup fp32 sink).
// ----------------------------------------------------------------------------
constant constexpr uint kGdnGroupSize = 64;

[[kernel, max_total_threads_per_threadgroup(256)]]
void gdn_gate_gemv(
    device const uint8_t* W        [[buffer(0)]],   // [2V, N/2] nibbles
    device const bfloat*  scales  [[buffer(1)]],   // [2V, N/64]
    device const bfloat*  biases  [[buffer(2)]],   // [2V, N/64]
    device const half*    x       [[buffer(3)]],   // [N] fp16
    device const float*   A_log   [[buffer(4)]],   // [V] fp32
    device const float*   dt_bias [[buffer(5)]],   // [V] fp32
    device       float*   g       [[buffer(6)]],   // [V] fp32 out
    device       float*   beta    [[buffer(7)]],   // [V] fp32 out
    constant     uint&    V       [[buffer(8)]],
    constant     uint&    N       [[buffer(9)]],
    uint  sg_idx [[simdgroup_index_in_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]]
) {
    const uint n_groups = N / kGdnGroupSize;
    const uint rowBytes = N / 2u;
    threadgroup float ab[64]; // 2V accs, V <= 32 → at most 64 rows

    // 8 SIMD groups x 8 passes cover the 2V (<= 64) rows.
    for (uint pass = 0; pass < 8u; ++pass) {
        const uint row = pass * 8u + sg_idx;
        if (row >= 2u * V) break;
        device const uint8_t* W_row = W      + row * rowBytes;
        device const bfloat*  s_row = scales + row * n_groups;
        device const bfloat*  b_row = biases + row * n_groups;
        float acc = 0.0f;
        for (uint gidx = 0; gidx < n_groups; ++gidx) {
            const float s = float(s_row[gidx]);
            const float b = float(b_row[gidx]);
            const uint8_t byte = W_row[gidx * (kGdnGroupSize / 2u) + lane];
            const float x0 = float(x[gidx * kGdnGroupSize + lane * 2u]);
            const float x1 = float(x[gidx * kGdnGroupSize + lane * 2u + 1u]);
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            const float sum = x0 + x1;
            acc = fma(s, dot, acc);
            acc = fma(b, sum, acc);
        }
        acc = simd_sum(acc);
        if (lane == 0) ab[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < V) {
        const float av   = ab[lane];
        const float bv   = ab[V + lane];
        const float dt   = dt_bias[lane];
        const float Alog = A_log[lane];
        g[lane]    = -exp(Alog) * gdn_softplus(av + dt);
        beta[lane] = 1.0f / (1.0f + exp(-bv));
    }
}

// ----------------------------------------------------------------------------
// Gated RMSNorm over each value head's vector:
//   y[i] = x[i] * rsqrt(mean(x[i]^2) + eps) * weight[i] * silu(z[i])
// One 256-thread threadgroup per value head. `weight` is shared across the V
// heads (Qwen3_5MoeRMSNormGated(head_v_dim)); x/z are [V][D] fp16. Mean-based
// (not sum), matching `Qwen3_5MoeRMSNormGated.forward`.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void gdn_rmsnorm_gated(
    device const half*   x          [[buffer(0)]],   // [V][D] fp16
    device const half*   z          [[buffer(1)]],   // [V][D] fp16
    device const bfloat* weight     [[buffer(2)]],   // [D] bf16, shared per head
    device       half*   out        [[buffer(3)]],   // [V][D] fp16
    constant     uint&   D          [[buffer(4)]],
    constant     float&  eps        [[buffer(5)]],
    uint  head                     [[threadgroup_position_in_grid]],
    uint  lid                      [[thread_position_in_threadgroup]],
    uint  lsize                    [[threads_per_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_group               [[simdgroup_index_in_threadgroup]],
    uint  simdgroups               [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kGdnMaxSimdGroups];
    device const half* xh = x   + head * D;
    device const half* zh = z   + head * D;
    device       half* oh = out + head * D;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(xh[i]);
        acc = fma(v, v, acc);
    }
    gdn_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(D) + eps);

    for (uint i = lid; i < D; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        float zv = float(zh[i]);
        oh[i] = half(xv * inv * wv * gdn_silu(zv));
    }
}
