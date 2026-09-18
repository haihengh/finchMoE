#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

constant constexpr uint kPrefillGroupSize = 64;
constant constexpr uint kPrefillRmsMaxSimdGroups = 8;
constant constexpr uint kPrefillPostMaxD = 4096;
// Qwen 3.6 routes over 256 experts, Qwen 3.8 over 512. The kernel clamps with
// `min(num_experts, kPrefillRouterMaxExperts)`, so this must be at least the
// real expert count or the router silently considers only a prefix of them —
// which is exactly why PrefillRouter's Swift side asserts instead. Keep the two
// in sync, and keep this matching MoE.swift's decode-side 512.
constant constexpr uint kPrefillRouterMaxExperts = 512;
constant constexpr uint kPrefillRouterMaxTopK = 64;
constant constexpr uint kPrefillAttentionMaxSimdGroups = 16;
constant constexpr uint kPrefillMaxTileExperts = 16;
constant constexpr float kPrefillGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kPrefillGeluCubicCoeff = 0.044715f;
constant uint FC_PREFILL_KV_RING_CAP [[function_constant(76)]];
// Qwen 3.6 silu routed-expert activation (mirrors moe.metal's FC_MOE_ACT_SILU
// on the decode path); defaults to gelu_pytorch_tanh for Gemma.
constant bool FC_PREFILL_MOE_ACT_SILU [[function_constant(77)]];

static inline float prefill_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kPrefillGeluSqrt2OverPi * (x + kPrefillGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

static inline float prefill_silu(float x) { return x / (1.0f + exp(-x)); }

static inline float prefill_expert_activation(float x) {
    return (is_function_constant_defined(FC_PREFILL_MOE_ACT_SILU) &&
            FC_PREFILL_MOE_ACT_SILU)
        ? prefill_silu(x)
        : prefill_gelu_pytorch_tanh(x);
}
kernel void prefill_embed_lookup_int4_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    const uint groups_per_row = D / kPrefillGroupSize;
    device const uint8_t* row_q = table  + token * (D / 2u);
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint8_t byte = row_q[d >> 1];
    const uint q = (d & 1u) == 0u ? uint(byte & 0x0Fu) : uint(byte >> 4);
    const float s = float(row_s[d / kPrefillGroupSize]);
    const float b = float(row_b[d / kPrefillGroupSize]);
    out[t * D + d] = half((float(q) * s + b) * out_scale);
}

static inline float prefill_rms_block_inv(
    device const half* x,
    uint D,
    float eps,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(v / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_block(
    device const half*   x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device half*         out     [[buffer(2)]],
    constant uint&       T       [[buffer(3)]],
    constant uint&       D       [[buffer(4)]],
    constant float&      eps     [[buffer(5)]],
    uint                 row     [[threadgroup_position_in_grid]],
    uint                 lid     [[thread_position_in_threadgroup]],
    uint                 lsize   [[threads_per_threadgroup]],
    uint                 lane    [[thread_index_in_simdgroup]],
    uint                 sg      [[simdgroup_index_in_threadgroup]],
    uint                 sgs     [[simdgroups_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xr = x + row * D;
    device half* yr = out + row * D;
    const float inv = prefill_rms_block_inv(xr, D, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < D; i += lsize) {
        yr[i] = half(float(xr[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device const bfloat* weight              [[buffer(1)]],
    device half*         out                 [[buffer(2)]],
    constant uint&       T                   [[buffer(3)]],
    constant uint&       head_dim            [[buffer(4)]],
    constant uint&       num_heads           [[buffer(5)]],
    constant uint&       token_stride_elems  [[buffer(6)]],
    constant float&      eps                 [[buffer(7)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_no_scale_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device half*         out                 [[buffer(1)]],
    constant uint&       T                   [[buffer(2)]],
    constant uint&       head_dim            [[buffer(3)]],
    constant uint&       num_heads           [[buffer(4)]],
    constant uint&       token_stride_elems  [[buffer(5)]],
    constant float&      eps                 [[buffer(6)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_post_attn_setup_block(
    device       half*   hidden                [[buffer(0)]],
    device const half*   attn                  [[buffer(1)]],
    device       half*   dense_x               [[buffer(2)]],
    device       half*   routed_x              [[buffer(3)]],
    device       half*   router_x              [[buffer(4)]],
    device const bfloat* w_post_attn           [[buffer(5)]],
    device const bfloat* w_pre_ffn             [[buffer(6)]],
    device const bfloat* w_pre_ffn2            [[buffer(7)]],
    constant uint&       T                     [[buffer(8)]],
    constant uint&       D                     [[buffer(9)]],
    constant uint&       hidden_stride_elems   [[buffer(10)]],
    constant uint&       attn_stride_elems     [[buffer(11)]],
    constant uint&       dense_stride_elems    [[buffer(12)]],
    constant uint&       routed_stride_elems   [[buffer(13)]],
    constant uint&       router_stride_elems   [[buffer(14)]],
    constant float&      rms_eps               [[buffer(15)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half attn_norm_tg[kPrefillPostMaxD];
    threadgroup half hidden_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device half* hidden_row = hidden + row * hidden_stride_elems;
    device const half* attn_row = attn + row * attn_stride_elems;
    device half* dense_row = dense_x + row * dense_stride_elems;
    device half* routed_row = routed_x + row * routed_stride_elems;
    device half* router_row = router_x + row * router_stride_elems;

    const float attn_inv = prefill_rms_block_inv(attn_row, D, rms_eps,
                                                 lid, lsize, lane, sg, sgs,
                                                 partial);
    for (uint i = lid; i < D; i += lsize) {
        attn_norm_tg[i] = half(float(attn_row[i]) * attn_inv * float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        half h = half(float(hidden_row[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden_row[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float hidden_inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_row[i] = half(h * float(w_pre_ffn[i]));
        routed_row[i] = half(h * float(w_pre_ffn2[i]));
        router_row[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_layer_tail_block(
    device const half*   h2                    [[buffer(0)]],
    device const half*   h1                    [[buffer(1)]],
    device       half*   hidden                [[buffer(2)]],
    device const bfloat* w_postffn2            [[buffer(3)]],
    device const bfloat* w_postffn             [[buffer(4)]],
    constant uint&       T                     [[buffer(5)]],
    constant uint&       D                     [[buffer(6)]],
    constant uint&       h2_stride_elems       [[buffer(7)]],
    constant uint&       h1_stride_elems       [[buffer(8)]],
    constant uint&       hidden_stride_elems   [[buffer(9)]],
    constant float&      rms_eps               [[buffer(10)]],
    constant float&      layer_scalar          [[buffer(11)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half tmp_tg[kPrefillPostMaxD];
    threadgroup half h12_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device const half* h2_row = h2 + row * h2_stride_elems;
    device const half* h1_row = h1 + row * h1_stride_elems;
    device half* hidden_row = hidden + row * hidden_stride_elems;

    const float inv_h2 = prefill_rms_block_inv(h2_row, D, rms_eps,
                                               lid, lsize, lane, sg, sgs,
                                               partial);
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h2_row[i]) * inv_h2 * float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        h12_tg[i] = h1_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(h12_tg[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_h12 = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h12_tg[i]) * inv_h12 * float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half h_scale = half(layer_scalar);
    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] * h_scale;
    }
}

struct PrefillTokenExpertPairMSL {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct PrefillStreamedRoutedBlobsMSL {
    device const uint8_t* blob[kPrefillMaxTileExperts];
};

struct PrefillGroupedRoutedMoEStreamedParamsMSL {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride_elements;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint local_expert_8;
    uint local_expert_9;
    uint local_expert_10;
    uint local_expert_11;
    uint local_expert_12;
    uint local_expert_13;
    uint local_expert_14;
    uint local_expert_15;
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

static inline uint prefill_streamed_local_expert_id(
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p,
    uint slot
) {
    switch (slot) {
        case 0: return p.local_expert_0;
        case 1: return p.local_expert_1;
        case 2: return p.local_expert_2;
        case 3: return p.local_expert_3;
        case 4: return p.local_expert_4;
        case 5: return p.local_expert_5;
        case 6: return p.local_expert_6;
        case 7: return p.local_expert_7;
        case 8: return p.local_expert_8;
        case 9: return p.local_expert_9;
        case 10: return p.local_expert_10;
        case 11: return p.local_expert_11;
        case 12: return p.local_expert_12;
        case 13: return p.local_expert_13;
        case 14: return p.local_expert_14;
        default: return p.local_expert_15;
    }
}

static inline float prefill_moe_int4_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        device const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

static inline float prefill_moe_int4_gemv_row_tg(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    threadgroup const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        threadgroup const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

kernel void prefill_router_gemma4_block(
    device const uint8_t* W                [[buffer(0)]],
    device const bfloat*  scales           [[buffer(1)]],
    device const bfloat*  biases           [[buffer(2)]],
    device const half*    hidden           [[buffer(3)]],
    device const bfloat*  effective_scale  [[buffer(4)]],
    device const bfloat*  per_expert_scale [[buffer(5)]],
    device uint*          out_indices      [[buffer(6)]],
    device half*          out_weights      [[buffer(7)]],
    constant uint&        T                [[buffer(8)]],
    constant uint&        num_experts      [[buffer(9)]],
    constant uint&        D                [[buffer(10)]],
    constant uint&        top_k            [[buffer(11)]],
    constant uint&        hidden_stride    [[buffer(12)]],
    uint                  row              [[threadgroup_position_in_grid]],
    uint                  tid              [[thread_position_in_threadgroup]],
    uint                  tg_size          [[threads_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float scores[kPrefillRouterMaxExperts];
    const uint NE = min(num_experts, kPrefillRouterMaxExperts);
    const uint KK = min(top_k, kPrefillRouterMaxTopK);
    device const half* row_hidden = hidden + row * hidden_stride;

    for (uint e = tid; e < NE; e += tg_size) {
        const uint n_groups = D / kPrefillGroupSize;
        device const uint8_t* W_row = W + e * D;
        device const bfloat* s_row = scales + e * n_groups;
        device const bfloat* b_row = biases + e * n_groups;

        float acc = 0.0f;
        for (uint g = 0; g < n_groups; ++g) {
            float s = float(s_row[g]);
            float b = float(b_row[g]);
            device const uint8_t* Wg = W_row + g * kPrefillGroupSize;
            device const half* xg = row_hidden + g * kPrefillGroupSize;
            device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
            float dot_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < kPrefillGroupSize; ++k) {
                float q = float(uint(Wg[k]));
                float xv = float(xg[k]) * float(eg[k]);
                dot_qx = fma(q, xv, dot_qx);
                sum_x += xv;
            }
            acc = fma(s, dot_qx, acc);
            acc = fma(b, sum_x, acc);
        }
        scores[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint top_idx[kPrefillRouterMaxTopK];
        float top_score[kPrefillRouterMaxTopK];
        for (uint i = 0; i < kPrefillRouterMaxTopK; ++i) {
            top_idx[i] = 0u;
            top_score[i] = -INFINITY;
        }

        for (uint e = 0; e < NE; ++e) {
            float s = scores[e];
            if (KK > 0 && s <= top_score[KK - 1]) continue;
            uint pos = KK;
            for (uint i = 0; i < KK; ++i) {
                if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                    pos = i;
                    break;
                }
            }
            if (pos >= KK) continue;
            for (uint i = KK - 1; i > pos; --i) {
                top_idx[i] = top_idx[i - 1];
                top_score[i] = top_score[i - 1];
            }
            top_idx[pos] = e;
            top_score[pos] = s;
        }

        float max_s = top_score[0];
        float sum_exp = 0.0f;
        float exps[kPrefillRouterMaxTopK];
        for (uint i = 0; i < KK; ++i) {
            float e = fast::exp(top_score[i] - max_s);
            exps[i] = e;
            sum_exp += e;
        }
        for (uint i = 0; i < KK; ++i) {
            const uint expert_idx = top_idx[i];
            const float w = exps[i] / sum_exp;
            const float gain = float(per_expert_scale[expert_idx]);
            out_indices[row * top_k + i] = expert_idx;
            out_weights[row * top_k + i] = half(w * gain);
        }
    }
}

kernel void prefill_moe_reduce_token_major(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights  [[buffer(1)]],
    device half*       h2             [[buffer(2)]],
    constant uint&     T              [[buffer(3)]],
    constant uint&     top_k          [[buffer(4)]],
    constant uint&     D              [[buffer(5)]],
    uint2              gid            [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    float acc = 0.0f;
    for (uint r = 0; r < top_k; ++r) {
        const uint partial_index = (t * top_k + r) * D + d;
        acc = fma(float(route_weights[t * top_k + r]),
                  float(route_partials[partial_index]),
                  acc);
    }
    h2[t * D + d] = half(acc);
}

kernel void prefill_grouped_routed_moe_batched_phase1(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         gate_up_act_scratch  [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint f = gid.x;
    const uint pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    device const uint8_t* gate_W = expert + p.gate_W_off;
    device const bfloat* gate_s = reinterpret_cast<device const bfloat*>(expert + p.gate_s_off);
    device const bfloat* gate_b = reinterpret_cast<device const bfloat*>(expert + p.gate_b_off);
    device const uint8_t* up_W = expert + p.up_W_off;
    device const bfloat* up_s = reinterpret_cast<device const bfloat*>(expert + p.up_s_off);
    device const bfloat* up_b = reinterpret_cast<device const bfloat*>(expert + p.up_b_off);

    const float gate = prefill_moe_int4_gemv_row_dev(gate_W, gate_s, gate_b, x, f, p.D);
    const float up = prefill_moe_int4_gemv_row_dev(up_W, up_s, up_b, x, f, p.D);
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(gate);
    gate_up_act_scratch[row_elements + index] = half(up);
    gate_up_act_scratch[2u * row_elements + index] =
        half(prefill_expert_activation(gate) * up);
}

kernel void prefill_grouped_routed_moe_batched_down(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device const half*                                   gate_up_act_scratch  [[buffer(7)]],
    device half*                                         down_scratch         [[buffer(8)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const uint8_t* down_W = expert + p.down_W_off;
    device const bfloat* down_s = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* down_b = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);
    device const half* act = gate_up_act_scratch + 2u * p.pair_count * p.F + pair_local * p.F;
    const half value = half(prefill_moe_int4_gemv_row_dev(down_W, down_s, down_b, act, d, p.F));
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

kernel void prefill_dequant_int4_qmm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]]
) {
    const uint n = tgid.x * 8u + tid.x;
    const uint t = tgid.y * 8u + tid.y;
    if (t >= T || n >= N) return;

    const uint groups = K / kPrefillGroupSize;
    const uint row_bytes = K / 2u;
    device const uint8_t* w_row = W + n * row_bytes;
    device const bfloat* s_row = scales + n * groups;
    device const bfloat* b_row = biases + n * groups;
    device const half* x_row = X + t * K;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        const uint group_base = g * kPrefillGroupSize;
        for (uint kk = 0; kk < kPrefillGroupSize; ++kk) {
            const uint k = group_base + kk;
            const uint8_t packed = w_row[k >> 1];
            const uint q = (k & 1u) == 0u ? uint(packed & 0x0Fu) : uint(packed >> 4);
            const float w = fma(float(q), scale, bias);
            acc = fma(w, float(x_row[k]), acc);
        }
    }
    Y[t * N + n] = half(acc);
}

static inline void prefill_rope_apply_neox_pair(
    device half* head_ptr,
    uint i,
    uint half_dim,
    uint freq_divisor,
    float position,
    float theta_base
) {
    const float exponent = -float(2u * i) / float(freq_divisor);
    const float freq = pow(theta_base, exponent);
    const float angle = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);

    const uint i0 = i;
    const uint i1 = half_dim + i;
    const float x0 = float(head_ptr[i0]);
    const float x1 = float(head_ptr[i1]);
    head_ptr[i0] = half(x0 * c - x1 * s);
    head_ptr[i1] = half(x0 * s + x1 * c);
}

kernel void prefill_rope_default_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_dim = head_dim / 2u;
    if (i >= half_dim) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

kernel void prefill_rope_proportional_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotated_pairs       [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (i >= rotated_pairs) return;
    if (h >= num_heads) return;

    const uint half_dim = head_dim / 2u;
    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

struct PrefillAttentionParams {
    uint startPosition;
    uint queryCount;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    uint kvValidCount;
    uint slidingWindow;
    uint kvTokenStrideElements;
    uint qTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
};

static inline uint prefill_kv_slot(uint logical) {
    return (is_function_constant_defined(FC_PREFILL_KV_RING_CAP) &&
            FC_PREFILL_KV_RING_CAP != 0u)
        ? (logical % FC_PREFILL_KV_RING_CAP)
        : logical;
}

static inline float prefill_attention_tg_sum(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    float s = simd_sum(value);
    if (lane == 0u) {
        partial[simd_group] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        float v = lane < simdgroups ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

static inline float prefill_attention_tg_sum_single_bank(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    const float result = prefill_attention_tg_sum(
        value, lane, simd_group, simdgroups, partial);
    // A single scratch bank needs an explicit reader-to-next-writer edge.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

[[kernel, max_total_threads_per_threadgroup(512)]]
kernel void attention_prefill_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float partial[2u * kPrefillAttentionMaxSimdGroups];

    const uint d = tid.x;
    const bool owns = d < p.headDim;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + t;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * p.headDim;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (uint key = first; key < last_exclusive; ++key) {
        const uint phys_key = prefill_kv_slot(key);
        device const half* k_row = K + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
        const float qv = owns ? float(q_row[d]) : 0.0f;
        const float kv = owns ? float(k_row[d]) : 0.0f;
        const uint bank = key & 1u;
        const float score = prefill_attention_tg_sum(
            qv * kv,
            lane,
            simd_group,
            simdgroups,
            partial + bank * kPrefillAttentionMaxSimdGroups) * p.scale;

        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (owns) {
            device const half* v_row = V + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
            acc = fma(new_scale, float(v_row[d]), acc * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * p.headDim;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

#if defined(__HAVE_TENSOR__)

constant constexpr int kPrefillTensorOpsOutputs = 8;
constant constexpr int kPrefillTensorOpsKeys = 64;
constant constexpr int kPrefillTensorOpsHeadDim = 512;

static inline void attention_prefill_full_tensorops_2d_validity_v2_impl(
    device const half* Q,
    device half* K,
    device half* V,
    device half* O,
    constant PrefillAttentionParams& p,
    uint3 tg,
    uint lid,
    uint threads,
    threadgroup half* query_tile,
    threadgroup float* score_tile,
    threadgroup float* weight_tile,
    threadgroup float* row_max,
    threadgroup float* row_sum,
    threadgroup float* row_old_scale
) {
    constexpr auto qk_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsKeys,
        kPrefillTensorOpsHeadDim,
        false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsHeadDim,
        kPrefillTensorOpsKeys,
        false, false, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;

    using device_half_tensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint query_start = tg.x;
    const uint qh_start = tg.y * uint(kPrefillTensorOpsOutputs);
    const uint valid_query_rows =
        min(1u, p.queryCount - min(query_start, p.queryCount));
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh_start / q_per_kv;

    for (uint linear = lid;
         linear < uint(kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim);
         linear += threads) {
        const uint output_row =
            linear / uint(kPrefillTensorOpsHeadDim);
        const uint d = linear % uint(kPrefillTensorOpsHeadDim);
        if (valid_query_rows != 0u) {
            query_tile[linear] = Q[
                query_start * p.qTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d];
        } else {
            query_tile[linear] = half(0.0f);
        }
    }
    if (lid < uint(kPrefillTensorOpsOutputs)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_half_tensor query_tensor(
        query_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsHeadDim,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsHeadDim}));
    threadgroup_float_tensor weight_tensor(
        weight_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsKeys,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsKeys}));
    device_half_tensor key_tensor(
        K + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));
    device_half_tensor value_tensor(
        V + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));

    auto query_slice = query_tensor.slice(0, 0);
    auto first_value_slice = value_tensor.slice(0, 0);
    auto output_accumulator =
        pv_op.get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(first_value_slice), float>();
    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (output_accumulator.is_valid_element(element)) {
            output_accumulator[element] = 0.0f;
        }
    }

    // This full-attention kernel starts at key zero and ignores slidingWindow.
    // The Swift selector must dispatch it only when every prior key is visible.
    const uint last =
        min(p.kvValidCount, p.startPosition + query_start + valid_query_rows);
    for (uint key_start = 0u;
         key_start < last;
         key_start += uint(kPrefillTensorOpsKeys)) {
        auto key_slice = key_tensor.slice(0, int32_t(key_start));
        auto score_product =
            qk_op.get_destination_cooperative_tensor<
                decltype(query_slice), decltype(key_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (score_product.is_valid_element(element)) {
                score_product[element] = 0.0f;
            }
        }
        qk_op.run(query_slice, key_slice, score_product);

        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (!score_product.is_valid_element(element)) continue;
            const auto position =
                score_product.get_multidimensional_index(element);
            const uint key_column = uint(position[0]);
            const uint output_row = uint(position[1]);
            score_tile[
                output_row * uint(kPrefillTensorOpsKeys) + key_column] =
                score_product[element] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(kPrefillTensorOpsOutputs)) {
            const uint output_row = lid;
            const uint causal_last = valid_query_rows != 0u
                ? min(p.kvValidCount, p.startPosition + query_start + 1u)
                : 0u;
            const uint visible =
                causal_last > key_start
                    ? min(uint(kPrefillTensorOpsKeys), causal_last - key_start)
                    : 0u;

            float tile_max = -INFINITY;
            for (uint key = 0u; key < visible; ++key) {
                tile_max = max(
                    tile_max,
                    score_tile[
                        output_row * uint(kPrefillTensorOpsKeys) + key]);
            }
            const float next_max = max(row_max[output_row], tile_max);
            const float old_scale = row_sum[output_row] > 0.0f
                ? fast::exp(row_max[output_row] - next_max)
                : 0.0f;
            float tile_sum = 0.0f;
            for (uint key = 0u;
                 key < uint(kPrefillTensorOpsKeys);
                 ++key) {
                const float weight = key < visible
                    ? fast::exp(
                        score_tile[
                            output_row * uint(kPrefillTensorOpsKeys) + key]
                        - next_max)
                    : 0.0f;
                weight_tile[
                    output_row * uint(kPrefillTensorOpsKeys) + key] = weight;
                tile_sum += weight;
            }
            row_old_scale[output_row] = old_scale;
            row_sum[output_row] =
                row_sum[output_row] * old_scale + tile_sum;
            row_max[output_row] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = value_tensor.slice(0, int32_t(key_start));
        auto output_product =
            pv_op.get_destination_cooperative_tensor<
                decltype(weight_tensor), decltype(value_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_product.get_capacity();
             ++element) {
            if (output_product.is_valid_element(element)) {
                output_product[element] = 0.0f;
            }
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_accumulator.get_capacity();
             ++element) {
            if (!output_accumulator.is_valid_element(element)
                || !output_product.is_valid_element(element)) {
                continue;
            }
            const auto position =
                output_accumulator.get_multidimensional_index(element);
            const uint output_row = uint(position[1]);
            output_accumulator[element] =
                fma(
                    1.0f,
                    output_product[element],
                    output_accumulator[element] * row_old_scale[output_row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (!output_accumulator.is_valid_element(element)) continue;
        const auto position =
            output_accumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint output_row = uint(position[1]);
        if (valid_query_rows != 0u) {
            const float denominator = row_sum[output_row];
            O[
                query_start * p.oTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d] = denominator > 0.0f
                    ? half(output_accumulator[element] / denominator)
                    : half(0.0f);
        }
    }
}

kernel void attention_prefill_full_tensorops_2d_validity_v2(
    device const half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]
) {
    threadgroup half query_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim];
    threadgroup float score_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float weight_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float row_max[kPrefillTensorOpsOutputs];
    threadgroup float row_sum[kPrefillTensorOpsOutputs];
    threadgroup float row_old_scale[kPrefillTensorOpsOutputs];
    attention_prefill_full_tensorops_2d_validity_v2_impl(
        Q, K, V, O, p, tg, lid, threads3.x,
        query_tile, score_tile, weight_tile,
        row_max, row_sum, row_old_scale);
}

#endif

// MARK: - Row-level fingerprints (diagnostic)

// One FNV-1a hash per row of a [rows][rowStrideBytes] tensor, computed on the
// GPU into a side buffer that the host reads once, after the run.
//
// Why it exists: the engine's long-prompt run-to-run nondeterminism
// (docs/experiments/summaries/05-attention-and-kv-cache.md, KV-15) is known to
// start inside a prefill *chunk* but has no instrument that reaches below it.
// A chunk's rows all encode into one command buffer, so a host read of an
// intermediate row would need a commit and a wait per chunk -- which changes the
// timing that is itself the correlate under investigation. Hashing on the GPU
// costs one small dispatch per (layer, stage) and leaves the run's shape alone.
//
// 64-bit FNV-1a over the row's bytes, the same algorithm and constants as the
// QSA selection fingerprint in FQ_QSA_DUMP, so the two instruments report
// comparable values.
constant constexpr ulong kRowHashOffsetBasis = 0xcbf29ce484222325UL;
constant constexpr ulong kRowHashPrime = 0x100000001b3UL;

kernel void row_hash_fnv1a_64(
    device const uchar* src           [[buffer(0)]],
    device ulong*       dst           [[buffer(1)]],
    constant uint&      rowBytes      [[buffer(2)]],
    constant uint&      rowStrideBytes [[buffer(3)]],
    constant uint&      rowCount      [[buffer(4)]],
    constant uint&      dstRowBase    [[buffer(5)]],
    constant uint&      srcRowBase    [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rowCount) { return; }
    if (rowBytes == 0 || rowStrideBytes < rowBytes) { return; }
    // `srcRowBase` and `dstRowBase` differ by design: the plane and the other
    // attention scratch are chunk-local, so their row 0 is the chunk's first
    // row, while the indexer's key timeline and its pooled blocks are indexed by
    // position and by block. Without the source offset a chunk other than the
    // first would fingerprint the timeline's *first* rows again — which is what
    // the idxk stage did before this existed, and why it read as identical.
    const device uchar* row = src + (size_t)(srcRowBase + gid) * (size_t)rowStrideBytes;
    ulong h = kRowHashOffsetBasis;
    for (uint i = 0; i < rowBytes; ++i) {
        h = (h ^ (ulong)row[i]) * kRowHashPrime;
    }
    // `dstRowBase` exists because every chunk writes the same plane rows: the
    // row's identity is its position in the sequence, not its index in the
    // chunk, or each chunk would overwrite the last one's fingerprints.
    dst[dstRowBase + gid] = h;
}

// MARK: - Batched int8 projection (prefill)

// The linear-attention (GDN) projections are int8 on every shipped Qwen 3.8
// install, and the prefill path fed them through `dequant_int8_gemv_simd` once
// per token -- one dispatch per token, each walking the whole weight matrix
// again. At T=426 that is 426 re-reads of 57 MB per layer, ~46,000 dispatches
// per chunk, and it measured 11.9 s of a 29.5 s prefill across the GDN stack's
// four projection stages.
//
// This is the batch the GEMV could not express: the weight tile is dequantized
// once into threadgroup memory and reused across the token dimension, and W, X
// and Y are all loaded and stored coalesced. The math is the same affine
// dequantization (w = q * scale + bias, per 64-wide group along K, one pair per
// row per group) that `dequant_int8_gemv_simd` applies, so the two agree to
// floating-point reassociation -- accumulation order differs, values do not.
//
// Layout notes, because they are where the speed comes from:
//   * `ws` is transposed to [k][row] so the compute loop walks k in the outer
//     step and rows contiguously, and it is padded by 4 halves so that walking
//     k for a fixed row does not stride into the same bank for every thread.
//   * one BK step is exactly one int8 group, so each row needs exactly one
//     scale and one bias per step, read once.
constant constexpr uint kInt8GemmBN = 64;   // output rows per threadgroup
constant constexpr uint kInt8GemmBT = 32;   // tokens per threadgroup
constant constexpr uint kInt8GemmBK = kInt8GroupSize;  // K per step == one group
constant constexpr uint kInt8GemmRowPad = 4;           // halves; see above
constant constexpr uint kInt8GemmRowsPerThread = 4;
constant constexpr uint kInt8GemmTokensPerThread = 4;

kernel void prefill_dequant_int8_gemm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        M      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]]
) {
    // Guard the shapes this kernel is dispatched with. K must be a multiple of
    // the group size, which every projection in this model is; the tile edges
    // are handled by zero-filling below rather than by a special case.
    if (K == 0 || (K % kInt8GemmBK) != 0) { return; }

    // The weight tile is fp32, not half: a dequantized weight here is q * scale
    // + bias and can reach ~120, where fp16 carries 0.03 of absolute error, and
    // that is 0.8% of a small output. The GEMV this replaces keeps its weights
    // in fp32 registers, so storing them at half width would make the prefill
    // less accurate than the decode it has to agree with. The input tile stays
    // half because that is how x is stored in the buffer — no loss is added.
    threadgroup float ws[kInt8GemmBK][kInt8GemmBN + kInt8GemmRowPad];
    threadgroup half  xs[kInt8GemmBK][kInt8GemmBT + kInt8GemmRowPad];

    const uint n0 = tgid.x * kInt8GemmBN;
    const uint t0 = tgid.y * kInt8GemmBT;
    const uint lane = tid.y * 16u + tid.x;   // 16 x 8 = 128 threads
    const uint groups = K / kInt8GemmBK;

    const uint rowBase = tid.x * kInt8GemmRowsPerThread;
    const uint tokBase = tid.y * kInt8GemmTokensPerThread;

    float acc[kInt8GemmRowsPerThread][kInt8GemmTokensPerThread];
    for (uint i = 0; i < kInt8GemmRowsPerThread; ++i)
        for (uint j = 0; j < kInt8GemmTokensPerThread; ++j)
            acc[i][j] = 0.0f;

    for (uint g = 0; g < groups; ++g) {
        const uint kBase = g * kInt8GemmBK;

        // Weight tile: dequantize once, transposed into shared memory.
        // 64 rows x 64 k = 4096 elements over 128 threads = 32 each.
        for (uint j = 0; j < 32u; ++j) {
            const uint l = lane + j * 128u;      // 0..4095
            const uint row = l / kInt8GemmBK;
            const uint kk = l % kInt8GemmBK;
            const uint grow = n0 + row;
            float value = 0.0f;
            if (grow < M) {
                const uint8_t q = W[grow * K + kBase + kk];
                const float s = float(scales[grow * groups + g]);
                const float b = float(biases[grow * groups + g]);
                value = fma(float(q), s, b);
            }
            ws[kk][row] = value;
        }

        // Input tile: 32 tokens x 64 k = 2048 elements over 128 threads = 16 each.
        for (uint j = 0; j < 16u; ++j) {
            const uint l = lane + j * 128u;      // 0..2047
            const uint token = l / kInt8GemmBK;
            const uint kk = l % kInt8GemmBK;
            const uint gt = t0 + token;
            xs[kk][token] = gt < T ? X[gt * K + kBase + kk] : half(0.0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scalar loads here, deliberately: hand-unrolling them into float4/half4
        // was measured and moved the projection stages by nothing (2423 ms
        // against 2420-2437 before it), so the compiler was already coalescing
        // them and the unrolled form was only harder to read. The 2:1
        // FMA-to-load ratio this loop looks like it has is therefore not what
        // limits the kernel.
        float wv[kInt8GemmRowsPerThread];
        half  xv[kInt8GemmTokensPerThread];
        for (uint kk = 0; kk < kInt8GemmBK; ++kk) {
            for (uint i = 0; i < kInt8GemmRowsPerThread; ++i)
                wv[i] = ws[kk][rowBase + i];
            for (uint j = 0; j < kInt8GemmTokensPerThread; ++j)
                xv[j] = xs[kk][tokBase + j];
            for (uint i = 0; i < kInt8GemmRowsPerThread; ++i) {
                for (uint j = 0; j < kInt8GemmTokensPerThread; ++j)
                    acc[i][j] = fma(wv[i], float(xv[j]), acc[i][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < kInt8GemmTokensPerThread; ++j) {
        const uint gt = t0 + tokBase + j;
        if (gt >= T) { continue; }
        for (uint i = 0; i < kInt8GemmRowsPerThread; ++i) {
            const uint gn = n0 + rowBase + i;
            if (gn >= M) { continue; }
            Y[gt * M + gn] = half(acc[i][j]);
        }
    }
}
