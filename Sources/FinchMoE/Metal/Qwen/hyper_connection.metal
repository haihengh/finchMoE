#include <metal_stdlib>
using namespace metal;

// ============================================================================
// hyper_connection.metal — Qwen 3.8 Flash-Next hyper-connection kernels.
//
// Math pinned to `archive/llama.cpp/src/models/qwen4exp.cpp`:
//   hc_plane_init:  res_hc = hc identical copies of the embedding
//                   (:324-327) — the per-token residual plane starts wide.
//   hc_grouped_rms: xn = per-stream RMS over the hc parallel planes of width
//                   D, scaled by the [hc*D] gamma (build_hc_mix :230-234)
//   hc_silu_scale:  lo = silu(down·xn · 1/hc) — the ÷hc sits BEFORE the silu
//                   (:238). Applied to the down-projection GEMV output.
//   hc_gate_mul:    gated = xn · sigmoid(up·lo) (:239, :242). Applied to the
//                   up-projection GEMV output (its own read gate).
//   hc_stream_mean: blockInput = mean over the hc streams of gated (:246-255)
//   hc_combine:     plane += blockOut · 2·sigmoid(inject · 1/hc) (:266-286)
//                   — one stream-group per plane stream, weight computed once.
//
// Hyper-connections keep hc parallel residual streams of width D (one plane,
// laid out stream-major: [hc*D]) in place of layer norms; every mixer reads
// the whole plane and produces a one-stream-wide [D] block input, and the
// block output scatters back into the plane. The root mixer (no inject) is
// the final norm: its stream-mean output feeds lm_head.
//
// Activation storage is FP16 everywhere, with FP32 kernel internals (the
// decode convention of the rest of the engine). The grouped-RMS gamma is raw
// BF16 — HC norms are never 1+w-baked (HyperConnectionRef.groupedRMS).
// ============================================================================

static inline float hc_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float hc_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

constant constexpr uint kHcThreads = 256;
constant constexpr uint kHcMaxSimdGroups = kHcThreads / 32;  // 8

// ============================================================================
// hc_plane_init — replicate the [D] hidden stream into hc identical stream
// copies: plane[c*D + i] = hidden[i]. The decode-step residual starts wide.
// Grid = D threads (one per hidden element).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_plane_init(
    device const half* hidden  [[buffer(0)]],  // [D] FP16
    device       half* plane   [[buffer(1)]],  // [hc*D] FP16
    constant     uint& D       [[buffer(2)]],
    constant     uint& HC      [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if (tid >= D) return;
    const half v = hidden[tid];
    for (uint c = 0; c < HC; ++c) {
        plane[c * D + tid] = v;
    }
}

// ============================================================================
// hc_grouped_rms — per-stream RMSNorm over the hc streams of the plane:
//     inv_c = rsqrt(mean over i of x[c*D+i]^2 + eps)
//     out[c*D+i] = x[c*D+i] * inv_c * gamma[c*D+i]
// FP32 accumulator; the gamma is a plain BF16 scale (no 1+w fold). One
// threadgroup per stream; two-stage block reduction (rmsnorm convention).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_grouped_rms(
    device const half*   x      [[buffer(0)]],   // [hc*D] FP16 plane
    device const bfloat* gamma  [[buffer(1)]],   // [hc*D] BF16
    device       half*   out    [[buffer(2)]],   // [hc*D] FP16 xn
    constant     uint&   D      [[buffer(3)]],   // one stream's width
    constant     float&  eps    [[buffer(4)]],
    uint  stream          [[threadgroup_position_in_grid]],
    uint  lid             [[thread_position_in_threadgroup]],
    uint  lsize           [[threads_per_threadgroup]],
    uint  simd_lane       [[thread_index_in_simdgroup]],
    uint  simd_group      [[simdgroup_index_in_threadgroup]],
    uint  simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kHcMaxSimdGroups];
    device const half* xs = x + stream * D;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(xs[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) partial[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float v = (simd_lane < simdgroups) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partial[0] = rsqrt(v / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];

    device const bfloat* ws = gamma + stream * D;
    device       half*   os = out   + stream * D;
    for (uint i = lid; i < D; i += lsize) {
        os[i] = half(float(xs[i]) * inv * float(ws[i]));
    }
}

// ============================================================================
// hc_silu_scale — lo[i] = silu(z[i] · 1/hc). The ÷hc is applied before the
// silu and separately from it — the down GEMV emits the raw dot product.
// Grid = N threads (lowrank).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_silu_scale(
    device const half* z       [[buffer(0)]],  // [N] FP16 down·xn
    device       half* out     [[buffer(1)]],  // [N] FP16 lo
    constant     uint& N       [[buffer(2)]],
    constant     float& invHc  [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    out[tid] = half(hc_silu(float(z[tid]) * invHc));
}

// ============================================================================
// hc_gate_mul — gated[i] = xn[i] · sigmoid(z[i]). The up GEMV emits the raw
// gate dot product over the whole plane; sigmoid + the xn product fuse here.
// Grid = N threads (hc*D).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_gate_mul(
    device const half* xn   [[buffer(0)]],  // [N] FP16
    device const half* z    [[buffer(1)]],  // [N] FP16 up·lo
    device       half* out  [[buffer(2)]],  // [N] FP16 gated
    constant     uint& N    [[buffer(3)]],
    uint tid                [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    out[tid] = half(float(xn[tid]) * hc_sigmoid(float(z[tid])));
}

// ============================================================================
// hc_stream_mean — blockInput[i] = (1/hc) · Σ_c gated[c*D + i]. The mean over
// the streams collapses the [hc*D] gated plane to the [D] block input (and,
// at the root mixer, to the model output). One thread per output element.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_stream_mean(
    device const half* gated  [[buffer(0)]],  // [hc*D] FP16
    device       half* out    [[buffer(1)]],  // [D] FP16
    constant     uint& D      [[buffer(2)]],
    constant     uint& HC     [[buffer(3)]],
    constant     float& invHc [[buffer(4)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid >= D) return;
    float acc = 0.0f;
    for (uint c = 0; c < HC; ++c) {
        acc += float(gated[c * D + tid]);
    }
    out[tid] = half(acc * invHc);
}

// ============================================================================
// hc_combine — plane[c*D + i] += blockOut[i] · 2·sigmoid(inject[c] · 1/hc),
// in place. 2·sigmoid centres the scatter weight on 1 (:274-275), so a zero
// injection is a plain residual add. One threadgroup per stream: the stream
// weight is computed once into threadgroup memory, then the group scatters
// its stream slice.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kHcThreads)]]
void hc_combine(
    device       half* plane   [[buffer(0)]],  // [hc*D] FP16 in place
    device const half* block   [[buffer(1)]],  // [D] FP16 block output
    device const half* inject  [[buffer(2)]],  // [hc] FP16
    constant     uint& D       [[buffer(3)]],
    constant     float& invHc  [[buffer(4)]],
    uint  stream       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_position_in_threadgroup]],
    uint  lsize        [[threads_per_threadgroup]]
) {
    threadgroup float wTg;
    if (lid == 0) {
        wTg = 2.0f * hc_sigmoid(float(inject[stream]) * invHc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float w = wTg;

    device half* ps = plane + stream * D;
    for (uint i = lid; i < D; i += lsize) {
        ps[i] = half(float(ps[i]) + float(block[i]) * w);
    }
}
