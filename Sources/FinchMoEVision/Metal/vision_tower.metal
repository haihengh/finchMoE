#include <metal_stdlib>
using namespace metal;

// Vision tower primitives for the Qwen3-VL BF16 tower.
//
// Activations are FP16 ([rows, columns], row-major). Weights stay BF16 in the
// mapped safetensors payload and are converted while they are staged into
// threadgroup memory, so loading the tower does not copy ~0.9 GB.

constant constexpr uint kVisionTileM = 16;
constant constexpr uint kVisionTileN = 64;
constant constexpr uint kVisionTileK = 16;

// GEMM flag bits. Kept in one word so the whole epilogue is one dispatch.
constant constexpr uint kVisionFlagBias       = 1u << 0;
constant constexpr uint kVisionFlagResidual   = 1u << 1;
constant constexpr uint kVisionFlagGeluTanh   = 1u << 2;
constant constexpr uint kVisionFlagGeluExact  = 1u << 3;

static inline float vision_gelu_tanh(float x) {
    // gelu_pytorch_tanh (the tower's hidden_act). The argument is clamped
    // because Metal's fast-math tanh is implemented through exp and returns
    // NaN once |2·arg| overflows Float (observed above ~44); tanh is ±1 for
    // every |arg| above ~10 anyway.
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float inner = c * (x + 0.044715f * x * x * x);
    const float t = tanh(clamp(inner, -15.0f, 15.0f));
    return 0.5f * x * (1.0f + t);
}

static inline float vision_erf(float x) {
    // Abramowitz & Stegun 7.1.26; max absolute error 1.5e-7. MSL has no erf().
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly = (((((1.061405429f * t - 1.453152027f) * t)
                          + 1.421413741f) * t - 0.284496736f) * t
                        + 0.254829592f) * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

static inline float vision_gelu_exact(float x) {
    // nn.GELU() in the merger.
    return 0.5f * x * (1.0f + vision_erf(x * 0.7071067811865476f));
}

/// C[M,N] = A[M,K] * W[N,K]^T (+ bias) (+ residual), with optional GELU.
/// A and C are FP16; W and bias are BF16.
kernel void vision_gemm_bf16w(
    device const half*   A        [[buffer(0)]],
    device const bfloat* W        [[buffer(1)]],
    device const bfloat* bias     [[buffer(2)]],
    device const half*   residual [[buffer(3)]],
    device half*         C        [[buffer(4)]],
    constant uint&       M        [[buffer(5)]],
    constant uint&       N        [[buffer(6)]],
    constant uint&       K        [[buffer(7)]],
    constant uint&       flags    [[buffer(8)]],
    uint3                tgid     [[threadgroup_position_in_grid]],
    uint3                lid3     [[thread_position_in_threadgroup]]
) {
    const uint tid = lid3.x;
    threadgroup half aTile[kVisionTileM * kVisionTileK];
    threadgroup half bTile[kVisionTileN * kVisionTileK];

    // Each thread owns four neighbouring output columns in one row:
    // 16 rows x 64 columns = 1024 outputs for 256 threads.
    const uint outRow = tid / 16;              // 0..15
    const uint outColBase = (tid % 16) * 4;    // 0,4,...,60

    float acc[4] = {0, 0, 0, 0};

    const uint kTiles = (K + kVisionTileK - 1) / kVisionTileK;
    for (uint kt = 0; kt < kTiles; ++kt) {
        const uint kBase = kt * kVisionTileK;

        // Stage A: 16x16.
        {
            const uint aRow = tid / kVisionTileK;
            const uint aK = tid % kVisionTileK;
            const uint gRow = tgid.y * kVisionTileM + aRow;
            const uint gK = kBase + aK;
            half value = half(0.0f);
            if (gRow < M && gK < K) {
                value = A[gRow * K + gK];
            }
            aTile[aRow * kVisionTileK + aK] = value;
        }

        // Stage B: 64x16, four elements per thread.
        for (uint item = 0; item < 4; ++item) {
            const uint idx = tid + item * 256u;
            const uint bRow = idx / kVisionTileK;
            const uint bK = idx % kVisionTileK;
            const uint gCol = tgid.x * kVisionTileN + bRow;
            const uint gK = kBase + bK;
            half value = half(0.0f);
            if (gCol < N && gK < K) {
                value = half(float(W[gCol * K + gK]));
            }
            bTile[bRow * kVisionTileK + bK] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < kVisionTileK; ++kk) {
            const float a = float(aTile[outRow * kVisionTileK + kk]);
            for (uint j = 0; j < 4; ++j) {
                const float b = float(bTile[(outColBase + j) * kVisionTileK + kk]);
                acc[j] = fma(a, b, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < 4; ++j) {
        const uint gRow = tgid.y * kVisionTileM + outRow;
        const uint gCol = tgid.x * kVisionTileN + outColBase + j;
        if (gRow >= M || gCol >= N) { continue; }
        float value = acc[j];
        if ((flags & kVisionFlagBias) != 0u) {
            value += float(bias[gCol]);
        }
        if ((flags & kVisionFlagResidual) != 0u) {
            value += float(residual[gRow * N + gCol]);
        }
        if ((flags & kVisionFlagGeluTanh) != 0u) {
            value = vision_gelu_tanh(value);
        } else if ((flags & kVisionFlagGeluExact) != 0u) {
            value = vision_gelu_exact(value);
        }
        C[gRow * N + gCol] = half(value);
    }
}

/// Elementwise FP16 add, used to add the interpolated position embedding
/// after the patch projection (the reference's patch_embed hook excludes it).
kernel void vision_add_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half*       out [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    out[gid] = half(float(a[gid]) + float(b[gid]));
}

/// Row-wise LayerNorm over FP16 activations with BF16 weight and bias.
kernel void vision_layernorm_bf16w(
    device const half*   X       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device half*         Y       [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    constant float&      eps     [[buffer(6)]],
    uint                 row     [[threadgroup_position_in_grid]],
    uint                 tid     [[thread_position_in_threadgroup]],
    uint                 lsize   [[threads_per_threadgroup]]
) {
    threadgroup float partialSum[32];
    threadgroup float partialSq[32];
    threadgroup float meanShared;
    threadgroup float invShared;

    if (row >= rows) { return; }
    device const half* x = X + row * D;

    float sum = 0.0f;
    float sumSq = 0.0f;
    for (uint d = tid; d < D; d += lsize) {
        const float v = float(x[d]);
        sum += v;
        sumSq += v * v;
    }
    const uint lane = tid % 32u;
    const uint simdGroup = tid / 32u;
    sum = simd_sum(sum);
    sumSq = simd_sum(sumSq);
    if (lane == 0u) {
        partialSum[simdGroup] = sum;
        partialSq[simdGroup] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0u) {
        const uint groups = (lsize + 31u) / 32u;
        float v = lane < groups ? partialSum[lane] : 0.0f;
        float q = lane < groups ? partialSq[lane] : 0.0f;
        v = simd_sum(v);
        q = simd_sum(q);
        if (lane == 0u) {
            const float mean = v / float(D);
            const float variance = max(q / float(D) - mean * mean, 0.0f);
            meanShared = mean;
            invShared = rsqrt(variance + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = meanShared;
    const float inv = invShared;
    device half* y = Y + row * D;
    for (uint d = tid; d < D; d += lsize) {
        const float normed = (float(x[d]) - mean) * inv;
        y[d] = half(normed * float(weight[d]) + float(bias[d]));
    }
}

/// 2D rotary embedding on Q and K in the packed [token, 3, head, 72] layout.
/// Each pair (i, 36 + i) uses one frequency; the cosine table is per token.
kernel void vision_rope_2d(
    device half*         qkv  [[buffer(0)]],
    device const float*  cosT [[buffer(1)]],
    device const float*  sinT [[buffer(2)]],
    constant uint&       tokens [[buffer(3)]],
    uint3                gid  [[thread_position_in_grid]]
) {
    const uint pair = gid.x;   // 0..35
    const uint head = gid.y;   // 0..15
    const uint token = gid.z;  // 0..tokens-1
    if (token >= tokens || pair >= 36u || head >= 16u) { return; }

    const uint headDim = 72u;
    const uint tokenStride = 3u * 16u * headDim; // 3456
    const uint qBase = token * tokenStride + head * headDim;
    const uint kBase = qBase + 16u * headDim;    // q block is 1152 elements

    const float c = cosT[token * 36u + pair];
    const float s = sinT[token * 36u + pair];

    device half* q = qkv + qBase;
    const float q0 = float(q[pair]);
    const float q1 = float(q[36u + pair]);
    q[pair] = half(q0 * c - q1 * s);
    q[36u + pair] = half(q0 * s + q1 * c);

    device half* k = qkv + kBase;
    const float k0 = float(k[pair]);
    const float k1 = float(k[36u + pair]);
    k[pair] = half(k0 * c - k1 * s);
    k[36u + pair] = half(k0 * s + k1 * c);
}

constant constexpr uint kVisionQTile = 16;
constant constexpr uint kVisionKTile = 32;
constant constexpr uint kVisionHeads = 16;
constant constexpr uint kVisionHeadDim = 72;

/// Full (bidirectional) attention for one image. One threadgroup per
/// (query tile, head). Q/K/V are read from the packed QKV buffer.
kernel void vision_attention(
    device const half* qkv       [[buffer(0)]],
    device half*       out       [[buffer(1)]],
    constant uint&     tokens    [[buffer(2)]],
    constant float&    scale     [[buffer(3)]],
    uint3              tgid      [[threadgroup_position_in_grid]],
    uint3              lid3      [[thread_position_in_threadgroup]]
) {
    const uint tid = lid3.x;
    threadgroup half  qTile[kVisionQTile * kVisionHeadDim];
    threadgroup half  kTile[kVisionKTile * kVisionHeadDim];
    threadgroup half  vTile[kVisionKTile * kVisionHeadDim];
    threadgroup float scores[kVisionQTile * kVisionKTile];
    threadgroup float outAcc[kVisionQTile * kVisionHeadDim];
    threadgroup float rowMax[kVisionQTile];
    threadgroup float rowSum[kVisionQTile];

    const uint head = tgid.y;
    const uint qStart = tgid.x * kVisionQTile;
    const uint tokenStride = 3u * kVisionHeads * kVisionHeadDim;
    const uint headOffset = head * kVisionHeadDim;
    const uint kBase = kVisionHeads * kVisionHeadDim;
    const uint vBase = 2u * kVisionHeads * kVisionHeadDim;

    for (uint idx = tid; idx < kVisionQTile * kVisionHeadDim; idx += 128u) {
        const uint qi = idx / kVisionHeadDim;
        const uint d = idx % kVisionHeadDim;
        const uint token = qStart + qi;
        half value = half(0.0f);
        if (token < tokens) {
            value = qkv[token * tokenStride + headOffset + d];
        }
        qTile[idx] = value;
    }
    for (uint idx = tid; idx < kVisionQTile * kVisionHeadDim; idx += 128u) {
        outAcc[idx] = 0.0f;
    }
    if (tid < kVisionQTile) {
        rowMax[tid] = -INFINITY;
        rowSum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kTiles = (tokens + kVisionKTile - 1) / kVisionKTile;
    for (uint kt = 0; kt < kTiles; ++kt) {
        const uint kStart = kt * kVisionKTile;
        for (uint idx = tid; idx < kVisionKTile * kVisionHeadDim; idx += 128u) {
            const uint ki = idx / kVisionHeadDim;
            const uint d = idx % kVisionHeadDim;
            const uint token = kStart + ki;
            half kv = half(0.0f);
            half vv = half(0.0f);
            if (token < tokens) {
                kv = qkv[token * tokenStride + kBase + headOffset + d];
                vv = qkv[token * tokenStride + vBase + headOffset + d];
            }
            kTile[idx] = kv;
            vTile[idx] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scores[q][k] = q . k * scale
        for (uint s = tid; s < kVisionQTile * kVisionKTile; s += 128u) {
            const uint qi = s / kVisionKTile;
            const uint ki = s % kVisionKTile;
            const uint token = kStart + ki;
            float dot = 0.0f;
            if (token < tokens) {
                threadgroup const half* qRow = qTile + qi * kVisionHeadDim;
                threadgroup const half* kRow = kTile + ki * kVisionHeadDim;
                for (uint d = 0; d < kVisionHeadDim; ++d) {
                    dot += float(qRow[d]) * float(kRow[d]);
                }
                dot *= scale;
            } else {
                dot = -INFINITY;
            }
            scores[s] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update, one thread per query row.
        if (tid < kVisionQTile) {
            const uint qi = tid;
            float tileMax = -INFINITY;
            for (uint ki = 0; ki < kVisionKTile; ++ki) {
                tileMax = max(tileMax, scores[qi * kVisionKTile + ki]);
            }
            const float previousMax = rowMax[qi];
            const float newMax = max(previousMax, tileMax);
            const float rescale = isinf(previousMax) ? 0.0f : exp(previousMax - newMax);
            float sum = rowSum[qi] * rescale;
            for (uint ki = 0; ki < kVisionKTile; ++ki) {
                float p = 0.0f;
                if (scores[qi * kVisionKTile + ki] != -INFINITY) {
                    p = exp(scores[qi * kVisionKTile + ki] - newMax);
                }
                scores[qi * kVisionKTile + ki] = p;
                sum += p;
            }
            rowMax[qi] = newMax;
            rowSum[qi] = sum;
            for (uint d = 0; d < kVisionHeadDim; ++d) {
                outAcc[qi * kVisionHeadDim + d] *= rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // outAcc += P * V
        for (uint idx = tid; idx < kVisionQTile * kVisionHeadDim; idx += 128u) {
            const uint qi = idx / kVisionHeadDim;
            const uint d = idx % kVisionHeadDim;
            float sum = 0.0f;
            for (uint ki = 0; ki < kVisionKTile; ++ki) {
                sum += scores[qi * kVisionKTile + ki] * float(vTile[ki * kVisionHeadDim + d]);
            }
            outAcc[idx] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint idx = tid; idx < kVisionQTile * kVisionHeadDim; idx += 128u) {
        const uint qi = idx / kVisionHeadDim;
        const uint d = idx % kVisionHeadDim;
        const uint token = qStart + qi;
        if (token >= tokens) { continue; }
        const float denominator = max(rowSum[qi], 1e-20f);
        out[token * (kVisionHeads * kVisionHeadDim) + headOffset + d] =
            half(outAcc[idx] / denominator);
    }
}

/// Concatenate each group of four consecutive patch rows into one merger row.
kernel void vision_concat_merge(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     patches [[buffer(2)]],
    constant uint&     width   [[buffer(3)]],
    uint               gid     [[thread_position_in_grid]]
) {
    const uint mergeWidth = width * 4u;
    const uint groups = patches / 4u;
    const uint total = groups * mergeWidth;
    if (gid >= total) { return; }
    const uint group = gid / mergeWidth;
    const uint offset = gid % mergeWidth;
    const uint patch = offset / width;
    const uint d = offset % width;
    output[gid] = input[(group * 4u + patch) * width + d];
}
