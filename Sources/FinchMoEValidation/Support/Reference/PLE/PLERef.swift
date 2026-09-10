import Foundation

/// FP32 reference for the Qwen 3.8 Flash-Next PLE n-gram head, grounded in
/// `archive/llama.cpp/src/models/qwen4exp.cpp`: the host hash in
/// `llm_graph_input_ple::set_input` (:990-1047), the forward in `build_ple`
/// (:1122-1213), and the conv-state handling in `build_conv_state_at`
/// (:1052-1104). The locked math lives in `docs/QWEN38_PORT.md`
/// "PLE n-gram math (locked)".
///
/// The head runs on the layer `ple_layer_ids` names (index 1 on the real
/// checkpoint, before that layer's attention mixer) and is the one part of
/// the model whose *routing* is decided on the host: the checkpoint ships a
/// 102.4 GB table of 160-wide rows and no embedding matrix, so "which 16 rows"
/// is a 64-bit hash of the recent token n-grams, computed CPU-side because
/// ggml has no int64 xor (`:962-964`).
///
/// Two pieces, and they fail differently:
///
///   - `contextWindow` + `rowIndices` are *exact integer* work. There is no
///     tolerance to hide behind — a wrong index reads a different row and the
///     output is silently wrong, not slightly off. Every operation is
///     unsigned 64-bit wrap, matching C++'s `uint64_t` conversions, and the
///     head order is load-bearing: heads 0..<perGram are the bigrams, the
///     next `perGram` the trigrams.
///   - `forward` is ordinary float work, written as explicit scalar loops so
///     that comparing it against the Metal kernels is meaningful.
///
/// The PLE head's device-side inputs are the gathered rows (one 160-wide
/// slice per head, concatenated in head order) and the layer's HC plane; its
/// output is added back into that plane. Everything else — the projections,
/// the three grouped norms, the gate, the dilated conv — is here.
public enum PLERef {
    /// Model RMS epsilon — the same hyperparameter the HC and GDN norms use
    /// (`f_norm_rms_eps`, `build_ple:1135`; the real config is 1e-6).
    public static let rmsEps: Float = 1e-6

    /// `LLAMA_TOKEN_NULL`: a predecessor that does not exist because the
    /// window reaches past the start of the sequence.
    public static let nullToken: Int32 = -1

    // MARK: - Host n-gram hash

    /// The n-gram context window at `position` (`:1030-1037`).
    ///
    /// `ctx[0]` is the token itself and is *never* cut — a token's own EOS
    /// does not end its context, which is what lets the EOS-position gather
    /// stay meaningful. Walking back, the first EOS or missing predecessor
    /// **freezes the rest of the window as EOS**: it is a sticky cut, not a
    /// per-slot substitution, so `[a, EOS, b]` at `b` reads `[b, EOS, EOS]`
    /// and not `[b, EOS, a]`.
    public static func contextWindow(
        tokens: [Int32], position: Int, ngramSize: Int, eos: Int32
    ) -> [Int32] {
        precondition(position >= 0 && position < tokens.count,
                     "position \(position) outside the sequence")
        precondition(ngramSize >= 2, "the n-gram head needs at least one predecessor")
        var ctx = [Int32](repeating: eos, count: ngramSize)
        ctx[0] = tokens[position]
        var cut = false
        for s in 1..<ngramSize {
            // A slot behind an already-cut one is null in llama (`:1033`),
            // which only ever re-affirms the cut — its value is never read.
            var t = nullToken
            if !cut {
                let back = position - s
                t = back >= 0 ? tokens[back] : nullToken
            }
            cut = cut || t < 0 || t == eos
            ctx[s] = cut ? eos : t
        }
        return ctx
    }

    /// The `nHeads` gather rows for one token (`:1038-1046`), in head order.
    ///
    /// Gram size `n` contributes the multiplier of every position `0..<n` and
    /// fills `perGram` consecutive heads starting at `(n-2)·perGram` — so the
    /// first `perGram` heads are bigrams and the rest trigrams, and head `h`
    /// has its own vocabulary size and offset. All three arrays are int64 in
    /// the checkpoint and all of this is unsigned 64-bit wrap: the multiply
    /// wraps, the xor wraps, the modulo is unsigned, and the offset add wraps
    /// before the truncation to int32. Doing any of it in a signed or wider
    /// type gives a different row.
    public static func rowIndices(
        context: [Int32],
        multipliers: [UInt64],
        vocabSizes: [UInt64],
        offsets: [UInt64],
        headsPerNGram: Int,
        ngramSize: Int? = nil
    ) -> [Int] {
        let nGram = ngramSize ?? context.count
        precondition(nGram >= 2 && nGram <= context.count,
                     "gram size must fit the context window")
        precondition(multipliers.count >= nGram, "one multiplier per gram position")
        let nHeads = (nGram - 1) * headsPerNGram
        precondition(vocabSizes.count >= nHeads && offsets.count >= nHeads,
                     "vocab sizes and offsets are indexed per head")

        var rows = [Int](repeating: 0, count: nHeads)
        for n in 2...nGram {
            var mixed = UInt64(context[0]) &* multipliers[0]
            for j in 1..<n {
                mixed ^= UInt64(context[j]) &* multipliers[j]
            }
            let base = (n - 2) * headsPerNGram
            for g in 0..<headsPerNGram {
                let h = base + g
                let row = mixed % vocabSizes[h] &+ offsets[h]
                rows[h] = Int(Int32(truncatingIfNeeded: row))
            }
        }
        return rows
    }

    // MARK: - Forward

    /// `build_ple` (:1122-1213): the gated value and the dilated-conv term,
    /// both added into the layer's plane.
    ///
    /// Returns the new plane and the updated conv history. `convHistory` is
    /// the *normalized gated* value of the preceding `history` positions,
    /// oldest first, flat row-major `[history, channels]` — zero-filled at a
    /// sequence start, which is what makes the first few tokens read zeros
    /// rather than wrapping. The new history is the last `history` rows of
    /// (old history ++ this token's `normalized`), i.e. llama's
    /// `conv_input` tail (`:1087-1097`), so feeding tokens one at a time
    /// gives the same result as feeding a whole chunk.
    ///
    /// `streamCount` is the HC stream count (4), `nEmbd` is one stream's
    /// width (2560), so the plane is `streamCount · nEmbd`. `keyProj` is
    /// `[streamCount·nEmbd, emb.count]` row-major, `valueProj` is
    /// `[nEmbd, emb.count]`, and `convWeight` is `[streamCount·nEmbd,
    /// convKernel]` — column `k` being one weight per channel, applied at tap
    /// `(convKernel-1-k)·dilation` (`:1188-1200`).
    public static func forward(
        embedding emb: [Float],
        plane: [Float],
        keyProj: [Float],
        valueProj: [Float],
        normKey: [Float],
        normQuery: [Float],
        normConv: [Float],
        convWeight: [Float],
        convHistory: [Float],
        streamCount: Int,
        convKernel: Int,
        dilation: Int,
        eps: Float = PLERef.rmsEps
    ) -> (plane: [Float], history: [Float]) {
        precondition(streamCount > 0 && plane.count % streamCount == 0,
                     "plane must split into whole streams")
        let hcDim = plane.count
        let nEmbd = hcDim / streamCount
        let nEmb = emb.count
        precondition(keyProj.count == hcDim * nEmb, "key_proj must be [hcDim, nEmb] row-major")
        precondition(valueProj.count == nEmbd * nEmb, "value_proj must be [nEmbd, nEmb] row-major")
        precondition(normKey.count == hcDim && normQuery.count == hcDim && normConv.count == hcDim,
                     "the grouped norms carry one gamma per plane element")
        precondition(convWeight.count == hcDim * convKernel,
                     "conv weight must be [hcDim, convKernel] row-major")
        let history = (convKernel - 1) * dilation
        precondition(convHistory.isEmpty || convHistory.count == history * hcDim,
                     "conv history must be [history, hcDim] or empty at a sequence start")
        // An empty history is the zero state a sequence starts from, not a
        // special case: the recurrent row is always `history` rows wide.
        let state = convHistory.isEmpty
            ? [Float](repeating: 0, count: history * hcDim) : convHistory

        // key = key_proj · emb, value = value_proj · emb (build_lora_mm, :1131-1132)
        var key = [Float](repeating: 0, count: hcDim)
        for c in 0..<hcDim {
            let rowBase = c * nEmb
            var acc: Float = 0
            for i in 0..<nEmb { acc += keyProj[rowBase + i] * emb[i] }
            key[c] = acc
        }
        var value = [Float](repeating: 0, count: nEmbd)
        for d in 0..<nEmbd {
            let rowBase = d * nEmb
            var acc: Float = 0
            for i in 0..<nEmb { acc += valueProj[rowBase + i] * emb[i] }
            value[d] = acc
        }

        // Both norms group over ONE stream: ggml_rms_norm runs over ne0, and
        // the reshape puts one stream there (:1134-1141). The gamma then
        // scales the whole plane, so it is indexed [stream·nEmbd + d].
        let keyNormed = HyperConnectionRef.groupedRMS(
            x: key, gamma: normKey, streamCount: streamCount, eps: eps)
        let queryNormed = HyperConnectionRef.groupedRMS(
            x: plane, gamma: normQuery, streamCount: streamCount, eps: eps)

        // Per-stream dot, scaled by 1/√n_embd, then the signed square root
        // before the sigmoid (:1144-1148). The clamp's lower bound matters at
        // s == 0: the sign is 0 there, so the gate is exactly 0.5.
        let invSqrt = 1.0 / Float(nEmbd).squareRoot()
        var gated = [Float](repeating: 0, count: hcDim)
        for c in 0..<streamCount {
            let base = c * nEmbd
            var s: Float = 0
            for d in 0..<nEmbd { s += keyNormed[base + d] * queryNormed[base + d] }
            s *= invSqrt
            let mag = max(abs(s), 1e-6).squareRoot()
            let sign: Float = s > 0 ? 1 : (s < 0 ? -1 : 0)
            let gate = 1.0 / (1.0 + exp(-(sign * mag)))
            // value is [nEmbd] and broadcasts across the streams (:1152-1156)
            for d in 0..<nEmbd { gated[base + d] = value[d] * gate }
        }

        let normalized = HyperConnectionRef.groupedRMS(
            x: gated, gamma: normConv, streamCount: streamCount, eps: eps)

        // Depthwise causal conv, dilated by the n-gram size: tap k reads
        // (convKernel-1-k)·dilation positions back (:1183-1206).
        var convOut = [Float](repeating: 0, count: hcDim)
        for c in 0..<hcDim {
            var acc: Float = 0
            for k in 0..<convKernel {
                let back = (convKernel - 1 - k) * dilation
                let tap: Float
                if back == 0 {
                    tap = normalized[c]
                } else if back <= history {
                    // `state` holds positions t-history .. t-1, oldest first —
                    // read the normalized copy, not the caller's array, which
                    // is empty at a sequence start.
                    let row = history - back
                    tap = state[row * hcDim + c]
                } else {
                    tap = 0                        // before the sequence start
                }
                acc += convWeight[c * convKernel + k] * tap
            }
            convOut[c] = acc
        }
        for c in 0..<hcDim { convOut[c] = silu(convOut[c]) }

        // The plane keeps its original value in the query above — the add is
        // of the original hidden, not the normed one (:1213).
        var out = [Float](repeating: 0, count: hcDim)
        for c in 0..<hcDim { out[c] = plane[c] + gated[c] + convOut[c] }

        // New state = the last `history` rows of (old ++ this token): for one
        // token that is a shift by one with `normalized` entering at the tail
        // (:1087-1097). The state is a recurrent row of exactly `history`
        // rows, zero at a sequence start, so there is no partly-filled case to
        // handle — an empty `convHistory` is just the zeros.
        var newHistory = [Float](repeating: 0, count: history * hcDim)
        for row in 0..<history {
            let dest = row * hcDim
            if row == history - 1 {
                newHistory.replaceSubrange(dest..<(dest + hcDim), with: normalized)
            } else {
                // Destination row `row` was source row `row + 1` of the window
                // before this token; `state` is that window, always full.
                let src = (row + 1) * hcDim
                newHistory.replaceSubrange(dest..<(dest + hcDim),
                                           with: state[src..<(src + hcDim)])
            }
        }
        return (out, newHistory)
    }

    private static func silu(_ x: Float) -> Float {
        x / (1.0 + exp(-x))
    }
}
