import Foundation
import Metal

/// Host-side routing for the Qwen 3.8 Flash-Next PLE n-gram head: the 64-bit
/// hash that picks which rows of the 102.4 GB table this token reads, and the
/// gather that fetches them.
///
/// This lives on the CPU because it has to: the table ships as 128 raw BF16
/// part files (ggml has no int64 xor to hash with, `qwen4exp.cpp:962-964`) and
/// the rows are chosen by unsigned 64-bit wrap arithmetic over the recent
/// tokens. The gather is 16 rows × 320 bytes = 5 KB per token, which is
/// nothing next to a decode step's expert reads — the table is large but the
/// per-token footprint is trivial, so nothing is cached and nothing is
/// resident.
///
/// **Where the predecessors come from.** llama reads them out of the KV
/// cache's cell tokens (`get_prev_tokens`), so a *missing* cell reads as
/// `LLAMA_TOKEN_NULL` and, via `cut`, as EOS. The window is only
/// `ngramSize − 1 = 2` deep, so the equivalent here is an `ngramSize`-entry
/// ring of `(position, token)` — the token being routed plus its two
/// predecessors — not a token history. Memory is O(1) and the ring has no
/// rewind: **this engine never rewinds the KV cache**, so a resumed run
/// (`ServerPromptCache` only hits where `kvPosition == kvBackedTokenIDs.count`)
/// continues from the cursor with the last tokens still in the ring. Adding a
/// rewind would need the caller to re-seed the window, because a rewind past
/// the ring's depth leaves the cache holding predecessors the ring has
/// already scrolled off.
///
/// Two behaviours this cannot reproduce, recorded because they are the
/// assumptions rather than details: if a cell were *evicted* mid-sequence (a
/// windowed or compacting cache) the oracle would read EOS where this ring
/// still has the token; and if a fresh process resumed onto a warm cache the
/// ring would read EOS where the oracle reads real tokens. Neither can arise
/// on this family — the window reaches back 2 positions, those cells are always
/// visible, and a resume stays inside the runner that fed them.
///
/// Head order is load-bearing: heads `0..<headsPerNgram` are the bigrams and
/// the next `headsPerNgram` the trigrams, and the gathered vector is
/// head-major (`[head][rowDim]`, 16 × 160 = 2560) because `ggml_get_rows`
/// lays the head dimension out slowest (`:1114-1117`).
final class PLEHost {

    /// The geometry the hash and gather need, all from the model config. The
    /// row width and per-part row count are frozen from the source tensor
    /// shape rather than the config, so they arrive with it.
    struct Geometry: Equatable {
        let ngramSize: Int
        let headsPerNGram: Int
        let rowDim: Int
        let partCount: Int
        let partRows: Int
        /// The hash's cut token (`eos_token_id`), not a generation stop token.
        let eosTokenId: Int32

        var headCount: Int { (ngramSize - 1) * headsPerNGram }
        /// Width of the gathered vector, and the `emb` the projections read.
        var gatheredWidth: Int { headCount * rowDim }
        var totalRows: Int { partCount * partRows }
    }

    let geometry: Geometry
    /// Per gram position, one 45-bit multiplier (int64 in the checkpoint).
    let multipliers: [UInt64]
    /// Per head: `row = mixed % vocabSizes[h] + offsets[h]`.
    let headOffsets: [UInt64]
    let headVocabSizes: [UInt64]

    /// When set, every row decoded from a **raw-BF16** part is quantized and
    /// decoded back before it reaches the GPU, so the engine sees exactly what
    /// the int4 PLE install would hand it while the table on disk stays
    /// untouched. `FQ_PLE_QUANT_SIM=<groupSize>` is the only thing that sets
    /// it; it exists to isolate PLE quantization error from every other source
    /// of error (`docs/PLE_QUANTIZATION_PLAN.md` Phase 5.1), not as a
    /// production path. A part that is *already* quantized is left alone —
    /// re-encoding it would be the identity, since its values are on the grid
    /// by construction.
    private let quantizationSimulation: Int?

    /// The last `ngramSize` tokens with their absolute positions, oldest
    /// first: the token being routed plus the `ngramSize − 1` predecessors its
    /// window reaches. Never longer.
    private var tail: [(position: Int, token: Int32)] = []

    /// Build the host for a Qwen 3.8 install, or return nil when the family
    /// carries no PLE (every other family). `nil` is the "no PLE" answer, not
    /// a malformed-config answer — a config that claims a PLE layer but
    /// supplies nonsense geometry traps here instead, because silently
    /// skipping the PLE would produce fluent, wrong output.
    init?(config: ArchConfig,
          multipliers: [UInt64],
          headOffsets: [UInt64],
          headVocabSizes: [UInt64],
          quantizationSimulation: Int? = nil) {
        guard config.isQwen3_8, config.pleLayerIndexes.isEmpty == false,
              config.ngramSize > 1, config.headsPerNgram > 0,
              config.ngramRowDim > 0, config.ngramPartCount > 0,
              config.ngramPartRows > 0 else { return nil }
        guard config.numLayers == 0
                || config.pleLayerIndexes.allSatisfy({ $0 >= 0 && $0 < config.numLayers })
        else { return nil }

        let geometry = Geometry(ngramSize: config.ngramSize,
                                headsPerNGram: config.headsPerNgram,
                                rowDim: config.ngramRowDim,
                                partCount: config.ngramPartCount,
                                partRows: config.ngramPartRows,
                                eosTokenId: Int32(config.pleEosTokenId))
        precondition(multipliers.count >= geometry.ngramSize,
                     "PLE needs one multiplier per gram position")
        precondition(headOffsets.count == geometry.headCount
                        && headVocabSizes.count == geometry.headCount,
                     "PLE needs one offset and vocab size per head")
        // 0 would make every window read as "the token before this one was an
        // EOS", i.e. the hash would be wrong for every token while still
        // producing a valid-looking row. The manifest has to carry it.
        precondition(geometry.eosTokenId != 0,
                     "PLE eos_token_id is 0 — the manifest is missing pleEosTokenId")

        // llama's own load-time bound (`:135`): the last row a head can reach
        // must exist. A bad constant would otherwise read a neighbouring
        // shard's bytes or past the table.
        for h in 0..<geometry.headCount {
            precondition(headVocabSizes[h] > 0, "PLE head \(h) has an empty vocabulary")
            precondition(headOffsets[h] + headVocabSizes[h] <= UInt64(geometry.totalRows),
                         "PLE head \(h) reaches row \(headOffsets[h] + headVocabSizes[h]) "
                         + "past the table's \(geometry.totalRows)")
        }

        self.geometry = geometry
        self.multipliers = Array(multipliers.prefix(geometry.ngramSize))
        self.headOffsets = headOffsets
        self.headVocabSizes = headVocabSizes
        // A group size that does not divide the row width would quantize a
        // ragged final group — a layout no install can have, since the writer
        // rejects it at plan time. Refuse it here too rather than let a
        // measurement report on a shape that cannot ship.
        if let groupSize = quantizationSimulation {
            precondition(groupSize > 0 && geometry.rowDim % groupSize == 0,
                         "PLE simulation group \(groupSize) does not divide the row "
                         + "width \(geometry.rowDim)")
        }
        self.quantizationSimulation = quantizationSimulation
    }

    // MARK: - Sequence state

    /// Record the token at `position`. llama's `apply_ubatch` has already
    /// stored the ubatch by the time the PLE input is built, so the token
    /// being produced is itself a predecessor for the next one — record
    /// before routing, as `set_input` does.
    ///
    /// Positions must be strictly increasing: the ring is a sliding window, so
    /// recording one twice (or rewinding onto one) would push a live
    /// predecessor out and the *next* token would hash the wrong n-gram. That
    /// is a silent wrong answer rather than a crash, so it traps.
    func record(position: Int, token: Int32) {
        if let last = tail.last {
            precondition(position > last.position,
                         "PLE positions must increase: recorded \(position) after \(last.position)")
        }
        tail.append((position, token))
        // Keep the token itself plus its whole window of predecessors —
        // `ngramSize` entries, since `contextWindow` reads back as far as
        // `position - (ngramSize - 1)`.
        let keep = geometry.ngramSize
        if tail.count > keep { tail.removeFirst(tail.count - keep) }
    }

    /// How many tokens the ring is holding — the one being routed plus its
    /// predecessors. Bounded by `ngramSize` however long the sequence runs.
    var windowDepth: Int { tail.count }

    /// Drop the window entirely. Runs with the KV cache's reset.
    func reset() {
        tail.removeAll(keepingCapacity: true)
    }

    // MARK: - Hash

    /// The n-gram context window at `position` (`set_input:1030-1037`).
    ///
    /// `ctx[0]` is the token itself and is never cut, so a token's own EOS
    /// still gets a full window. Going back, the first EOS — or the first
    /// predecessor that does not exist, which reads as one — freezes
    /// everything behind it as EOS. It is a sticky cut: `[a, EOS, b]` at `b`
    /// reads `[b, EOS, EOS]`, never `[b, EOS, a]`.
    ///
    /// The token at `position` must have been `record`ed; the predecessors
    /// come from the ring.
    func contextWindow(atPosition position: Int) -> [Int32] {
        var ctx = [Int32](repeating: geometry.eosTokenId, count: geometry.ngramSize)
        guard let current = tail.last(where: { $0.position == position })?.token else {
            preconditionFailure("PLE routed position \(position) before it was recorded")
        }
        ctx[0] = current
        var cut = false
        for s in 1..<geometry.ngramSize {
            // A slot behind an already-cut one is null in llama, which only
            // ever re-affirms the cut — its value is never read.
            var t: Int32 = -1
            if !cut {
                let want = position - s
                t = tail.last(where: { $0.position == want })?.token ?? -1
            }
            cut = cut || t < 0 || t == geometry.eosTokenId
            ctx[s] = cut ? geometry.eosTokenId : t
        }
        return ctx
    }

    /// The `headCount` gather rows for one token (`set_input:1038-1046`), in
    /// head order.
    ///
    /// Gram size `n` contributes the multiplier of every position `0..<n` and
    /// fills `headsPerNGram` consecutive heads starting at `(n-2)·headsPerNGram`.
    /// Every step is **unsigned 64-bit wrap** — the multiply wraps, the xor
    /// wraps, the modulo is unsigned, and the offset add wraps before the
    /// truncation to int32. Doing any of it in a wider or signed type gives a
    /// different row, and a different row is silently wrong output, not a
    /// slightly-off vector.
    func rowIndices(atPosition position: Int) -> [Int] {
        let ctx = contextWindow(atPosition: position)
        let perGram = geometry.headsPerNGram
        var rows = [Int](repeating: 0, count: geometry.headCount)
        for n in 2...geometry.ngramSize {
            var mixed = UInt64(ctx[0]) &* multipliers[0]
            for j in 1..<n { mixed ^= UInt64(ctx[j]) &* multipliers[j] }
            let base = (n - 2) * perGram
            for g in 0..<perGram {
                let h = base + g
                let row = mixed % headVocabSizes[h] &+ headOffsets[h]
                rows[h] = Int(Int32(truncatingIfNeeded: row))
            }
        }
        return rows
    }

    // MARK: - Gather

    /// Which part file holds `row`, and where in it.
    ///
    /// The parts are equal-sized (`split_ngram_parts` 128), so the split is a
    /// plain divmod — llama asserts the padding that makes this legal when it
    /// reads the table back (`:128-140`).
    func location(ofRow row: Int) -> (part: Int, rowInPart: Int) {
        precondition(row >= 0 && row < geometry.totalRows,
                     "PLE row \(row) outside the table's 0..<\(geometry.totalRows)")
        return (row / geometry.partRows, row % geometry.partRows)
    }

    /// The head-major `[headCount · rowDim]` gathered rows for one token,
    /// decoded from the part's on-disk row (raw BF16 or int4 affine) into the
    /// engine's FP16 activations.
    ///
    /// `open` is the model's cached part opener. The rows are hash-random, so
    /// they land in unrelated parts and mostly unrelated rows; this is at most
    /// `headCount` single-row preads. The decode is per-layout, decided by the
    /// streamer (which `Model` set from the manifest's `pleNgram` slot), so the
    /// output shape/type is identical for both and nothing downstream (the GPU
    /// gate/conv/plane-add kernels in `ple.metal`) changes.
    func gather(atPosition position: Int,
                open: (Int) throws -> PLEPartStreamer) throws -> [Float16] {
        let rows = rowIndices(atPosition: position)
        let rowDim = geometry.rowDim
        var out = [Float16](repeating: 0, count: rows.count * rowDim)
        for (h, row) in rows.enumerated() {
            // Consecutive heads sharing a part (they are hash-independent, so
            // this is chance, not structure) would not gain anything: the part
            // handle is cached and a row read is one pread either way.
            let (part, rowInPart) = location(ofRow: row)
            let streamer = try open(part)
            let bytes = try streamer.readRows(rowInPart..<(rowInPart + 1))
            let base = h * rowDim
            switch streamer.layout {
            case .rawBF16:
                bytes.withUnsafeBytes { raw in
                    let bits = raw.bindMemory(to: UInt16.self)
                    guard let groupSize = quantizationSimulation else {
                        for d in 0..<rowDim {
                            out[base + d] = Float16(Quantization.bf16ToFloat(bits[d]))
                        }
                        return
                    }
                    // Phase 5.1: quantize the row as a whole — the writer's
                    // unit — and decode it back, so this run's activation is
                    // exactly what the int4 install would supply for the same
                    // row. Everything else in the model is untouched, which is
                    // what makes the A/B attributable to the PLE table alone.
                    var values = [Float](repeating: 0, count: rowDim)
                    for d in 0..<rowDim { values[d] = Quantization.bf16ToFloat(bits[d]) }
                    let quantized = Quantization.quantizeInt4AffinePLE(values, groupSize: groupSize)
                    let decoded = Quantization.dequantizeInt4AffinePLE(quantized, n: rowDim)
                    for d in 0..<rowDim { out[base + d] = Float16(decoded[d]) }
                }
            case .quantized(let groupSize):
                // On-disk row (writer layout): [packed nibbles: rowDim/2]
                // [scale BF16 × nGroups] [bias BF16 × nGroups], native-endian.
                let nGroups = rowDim / groupSize
                bytes.withUnsafeBytes { raw in
                    let b = raw.bindMemory(to: UInt8.self)
                    let packedLen = rowDim / 2
                    let packed = Array(b.prefix(packedLen))
                    func u16(_ i: Int) -> UInt16 { UInt16(b[i]) | (UInt16(b[i + 1]) << 8) }
                    var scales = [UInt16](repeating: 0, count: nGroups)
                    for g in 0..<nGroups { scales[g] = u16(packedLen + 2 * g) }
                    let biasBase = packedLen + 2 * nGroups
                    var biases = [UInt16](repeating: 0, count: nGroups)
                    for g in 0..<nGroups { biases[g] = u16(biasBase + 2 * g) }
                    let r = Quantization.Int4AffinePLERow(packed: packed,
                                                          scales: scales,
                                                          biases: biases)
                    let values = Quantization.dequantizeInt4AffinePLE(r, n: rowDim)
                    for d in 0..<rowDim {
                        out[base + d] = Float16(values[d])
                    }
                }
            }
        }
        return out
    }
}
