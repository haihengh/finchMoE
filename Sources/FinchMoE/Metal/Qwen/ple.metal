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
//
// The chunked prefill path calls this with `N = T · HC·D`: the op is
// elementwise over a flat range with no token structure, so one dispatch
// covers a whole chunk exactly as it covers one token.
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

// ============================================================================
// Chunked (prefill) forms.
//
// llama runs PLE through the same graph for a whole ubatch
// (`qwen4exp.cpp:1022-1047` window, `:1122-1213` chain) — every tensor simply
// gains a token axis, and nothing branches on `n_tokens == 1`. These three
// kernels are that token axis; the gate/value/silu reductions are unchanged.
// `ple_plane_add` and `hc_grouped_rms` need no twin (see their notes).
// ============================================================================

// ============================================================================
// ple_seq_gate — `ple_gate` over a chunk: one threadgroup per (stream, token),
// laid out flat as `tgid = token·HC + stream` over an HC·T-wide 1-D grid (a
// uint2 grid axis cannot be mixed with the scalar thread indices the reduction
// needs). Each token's gate is a dot of its own normed key and query, which is
// why the gate is a per-(token, stream) [T][HC] array rather than [HC].
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_seq_gate(
    device const half*  key      [[buffer(0)]],   // [T, HC*D] normed key
    device const half*  query    [[buffer(1)]],   // [T, HC*D] normed query
    device       float* gate     [[buffer(2)]],   // [T, HC] out, FP32
    constant     uint&  D        [[buffer(3)]],   // one stream's width
    constant     uint&  HC       [[buffer(4)]],
    constant     float& invSqrtD [[buffer(5)]],   // 1/√D
    uint  tgid            [[threadgroup_position_in_grid]],
    uint  lid             [[thread_position_in_threadgroup]],
    uint  lsize           [[threads_per_threadgroup]],
    uint  simd_lane       [[thread_index_in_simdgroup]],
    uint  simd_group      [[simdgroup_index_in_threadgroup]],
    uint  simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kPleMaxSimdGroups];
    const uint base = tgid * D;
    device const half* ks = key   + base;
    device const half* qs = query + base;

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
            gate[tgid] = 1.0f / (1.0f + exp(-(sgn * mag)));
        }
    }
}

// ============================================================================
// ple_seq_gated_value — `ple_gated_value` over a chunk:
//   gated[t][c*D + d] = value[t][d] · gate[t][c]
// Each token broadcasts its own [D] value down its own streams. Grid = T·HC·D.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_seq_gated_value(
    device const half*  value [[buffer(0)]],   // [T, D]
    device const float* gate  [[buffer(1)]],   // [T, HC]
    device       half*  gated [[buffer(2)]],   // [T, HC*D]
    constant     uint&  D     [[buffer(3)]],
    constant     uint&  HC    [[buffer(4)]],
    uint tid                  [[thread_position_in_grid]]
) {
    const uint d = tid % D;
    const uint c = (tid / D) % HC;
    const uint t = tid / (D * HC);
    gated[tid] = half(float(value[t * D + d]) * gate[t * HC + c]);
}

// ============================================================================
// ple_seq_conv — the dilated depthwise causal conv over a whole chunk.
//
// llama builds `padded = concat(history, chunk)` once per ubatch and then
// takes a shifted view per tap: `start = hist − (kern−1−k)·dil`, `n_seq_tokens`
// wide (`qwen4exp.cpp:1182-1206`). Both halves of that are the same expression
// here — tap k of token t sits at extended index `hist + t − (kern−1−k)·dil`,
// which is a history row when it is below `hist` and the chunk's own row
// otherwise. So one kernel serves a chunk of any length, and the history a
// tap reaches back into is the same one decode reads.
//
// The history roll is a separate kernel (`ple_seq_conv_roll`) rather than a
// tail of this one: the new history is the last `hist` entries of the extended
// timeline, which is a `hist · C`-wide write that has nothing to do with the
// token row a given thread happens to be processing.
//
// The grid is flat — `tgid = token·nGroups + channelGroup` — and the (channel
// group, token) pair is recovered from `lsize`, because a uint2 grid axis
// cannot be mixed with the scalar thread indices (Metal requires the input
// declarations to be all-scalar or all-matching-vector).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_seq_conv(
    device const half* w     [[buffer(0)]],   // [C, K]
    device const half* state [[buffer(1)]],   // [hist, C] oldest first
    device const half* x     [[buffer(2)]],   // [T, C] normed gated value
    device       half* out   [[buffer(3)]],   // [T, C] silu'd conv output
    constant     uint& C     [[buffer(4)]],
    constant     uint& K     [[buffer(5)]],
    constant     uint& dil   [[buffer(6)]],
    uint  tgid         [[threadgroup_position_in_grid]],
    uint  lid          [[thread_position_in_threadgroup]],
    uint  lsize        [[threads_per_threadgroup]]
) {
    const uint hist = (K - 1) * dil;
    const uint groups = (C + lsize - 1) / lsize;
    const uint c = (tgid % groups) * lsize + lid;
    const uint t = tgid / groups;
    if (c >= C) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        const uint back = (K - 1 - k) * dil;
        const uint idx = hist + t - back;
        const float tap = (idx < hist) ? float(state[idx * C + c])
                                       : float(x[(idx - hist) * C + c]);
        acc = fma(float(w[c * K + k]), tap, acc);
    }
    out[t * C + c] = half(ple_silu(acc));
}

// ============================================================================
// ple_seq_conv_roll — the chunk's history write-back:
//   newState[row] = extended[T + row], the last `hist` entries of
//   `state ++ x`. A row still inside the pre-chunk history comes from `state`
//   (which happens when the chunk is shorter than the receptive field), the
//   rest from the chunk. Grid = hist·C threads.
//
// `newState` must not alias `state`: every row of `ple_seq_conv` reads
// `state`, and those reads are still in flight when this lands. The caller
// blits `newState` back into `state` afterwards.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kPleThreads)]]
void ple_seq_conv_roll(
    device const half* state    [[buffer(0)]],   // [hist, C]
    device const half* x        [[buffer(1)]],   // [T, C]
    device       half* newState [[buffer(2)]],   // [hist, C]
    constant     uint& C        [[buffer(3)]],
    constant     uint& T        [[buffer(4)]],
    constant     uint& hist     [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]]
) {
    const uint c = tid % C;
    const uint row = tid / C;
    if (row >= hist) return;
    // extended index T + row: a pre-chunk history row while it is below hist,
    // else the chunk row `T + row − hist`.
    newState[row * C + c] = (T + row < hist)
        ? state[(T + row) * C + c]
        : x[(T + row - hist) * C + c];
}
