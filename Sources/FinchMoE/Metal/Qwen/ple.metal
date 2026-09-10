#include <metal_stdlib>
using namespace metal;

// ============================================================================
// ple.metal — Qwen 3.8 Flash-Next PLE n-gram head (decode).
//
// Math pinned to `archive/llama.cpp/src/models/qwen4exp.cpp` `build_ple`
// (:1122-1213) and `build_conv_state_at` (:1052-1104); the locked math is in
// `docs/QWEN38_PORT.md` "PLE n-gram math (locked)".
//
//   ple_gate:          gate[c] = sigmoid(sgn(s)·√max(|s|,1e-6)) with
//                      s = (1/√D)·Σ_d key[c·D+d]·query[c·D+d]   (:1144-1148)
//   ple_gated_value:   gated[c·D+d] = value[d]·gate[c]           (:1152-1156)
//   ple_conv_update:   out[c] = silu(Σ_k w[c·K+k]·x_tap)         (:1183-1206)
//                      and the dilated history roll                (:1087-1097)
//   ple_plane_add:     plane += gated + conv_out                 (:1213)
//
// The grouped RMSNorm both `ple_gate` inputs and the conv input pass through
// is `hc_grouped_rms` (same op: one stream's RMS under a whole-plane gamma),
// and the two projections are the existing int8 GEMV path — neither is
// duplicated here.
//
// Host-side work (the n-gram hash and the ≤16-row gather) is *not* here: the
// table is 102.4 GB and lives in the part files, so the rows are chosen and
// read on the CPU. This file starts from the gathered [n_heads·160] vector.
//
// Storage is FP16 for activations with FP32 kernel internals (the decode
// convention). The one exception is `ple_gate`'s output: it is FP32. The gate
// is a *multiplier* on the value stream, computed from a dot product of two
// normed vectors, and its four values cost nothing to keep exact.
// ============================================================================

constant constexpr uint kPleThreads = 256;
constant constexpr uint kPleMaxSimdGroups = kPleThreads / 32;  // 8

static inline float ple_silu(float x) { return x / (1.0f + exp(-x)); }

// ============================================================================
// ple_gate — one threadgroup per stream.
//
// s is the per-stream dot of the normed key and query, scaled by 1/√D, and
// the gate is the *signed square root* of it under a sigmoid: the sign is
// carried outside the root, so a negative s gates below 0.5 and s == 0 gives
// exactly 0.5. The clamp floor keeps √0 finite without meaning it — sgn(0)
// is 0, so the floor never reaches the sigmoid.
//
// Grid = HC threadgroups.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_gate(
    device const half*  key      [[buffer(0)]],   // [HC*D] normed key
    device const half*  query    [[buffer(1)]],   // [HC*D] normed query
    device       float* gate     [[buffer(2)]],   // [HC] out, FP32
    constant     uint&  D        [[buffer(3)]],   // one stream's width
    constant     float& invSqrtD [[buffer(4)]],   // 1/√D
    uint  stream        [[threadgroup_position_in_grid]],
    uint  lid           [[thread_position_in_threadgroup]],
    uint  lsize         [[threads_per_threadgroup]],
    uint  simd_lane     [[thread_index_in_simdgroup]],
    uint  simd_group    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kPleMaxSimdGroups];
    device const half* ks = key   + stream * D;
    device const half* qs = query + stream * D;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        acc = fma(float(ks[i]), float(qs[i]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) partial[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float v = (simd_lane < simdgroups) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) {
            const float s = v * invSqrtD;
            const float mag = sqrt(max(fabs(s), 1e-6f));
            const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
            gate[stream] = 1.0f / (1.0f + exp(-(sgn * mag)));
        }
    }
}

// ============================================================================
// ple_gated_value — gated[c*D + d] = value[d] * gate[c]. The [D] value vector
// is broadcast down the streams, each stream scaled by its own gate (:1152).
// Grid = HC*D threads.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_gated_value(
    device const half*  value [[buffer(0)]],   // [D]
    device const float* gate  [[buffer(1)]],   // [HC]
    device       half*  gated [[buffer(2)]],   // [HC*D]
    constant     uint&  D     [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    const uint c = tid / D;
    const uint d = tid % D;
    gated[tid] = half(float(value[d]) * gate[c]);
}

// ============================================================================
// ple_conv_update — dilated depthwise causal conv, one thread per channel.
//
//   out[c] = silu( Σ_k w[c*K + k] · x[c, t − (K−1−k)·dil] )
//
// Tap k reaches back (K−1−k)·dil positions (`:1183-1206`) — with K 4 and
// dil 3 the taps are t, t−3, t−6, t−9 — so `state` holds the last
// (K−1)·dil normed rows, oldest first, laid out row-major [hist, C]. The
// shift is done here rather than by a bulk copy because the row that falls
// off the end is different for every thread's channel and each new state row
// is exactly the token just processed.
//
// A history row is the *normed gated value*, not a conv output: the reference
// appends `normalized` to the state (`:1087-1097`). Grid = ceil(C/tg) groups.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_conv_update(
    device const half* w        [[buffer(0)]],   // [C, K]
    device const half* state    [[buffer(1)]],   // [hist, C] oldest first
    device const half* x        [[buffer(2)]],   // [C] this token's normed
    device       half* out      [[buffer(3)]],   // [C] silu'd conv output
    device       half* newState [[buffer(4)]],   // [hist, C]
    constant     uint& C        [[buffer(5)]],
    constant     uint& K        [[buffer(6)]],
    constant     uint& dil      [[buffer(7)]],
    uint  lid                   [[thread_position_in_threadgroup]],
    uint  lsize                 [[threads_per_threadgroup]],
    uint  tgx                   [[threadgroup_position_in_grid]]
) {
    const uint hist = (K - 1) * dil;
    const uint c = tgx * lsize + lid;
    if (c >= C) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        const uint back = (K - 1 - k) * dil;
        const float tap = (back == 0) ? float(x[c]) : float(state[(hist - back) * C + c]);
        acc = fma(float(w[c * K + k]), tap, acc);
    }
    out[c] = half(ple_silu(acc));

    // Roll the history one row: every row moves up, the new row is this token.
    for (uint row = 0; row + 1 < hist; ++row) {
        newState[row * C + c] = state[(row + 1) * C + c];
    }
    newState[(hist - 1) * C + c] = x[c];
}

// ============================================================================
// ple_plane_add — plane[i] += gated[i] + conv_out[i]. Both PLE terms land in
// the layer's HC plane before its attention mixer (`:1213`), in place.
// Grid = HC*D threads.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_plane_add(
    device       half*  plane [[buffer(0)]],   // [N] in/out
    device const half*  gated [[buffer(1)]],   // [N]
    device const half*  conv  [[buffer(2)]],   // [N]
    constant     uint&  N     [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    plane[tid] = half(float(plane[tid]) + float(gated[tid]) + float(conv[tid]));
}
