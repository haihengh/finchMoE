import Foundation
import Darwin
import FinchMoEFormat

/// Writes the Qwen `.finch` payloads: the resident `model_weights.bin`
/// (index + quantized entries) and the per-layer expert blobs. Every source
/// tensor is bf16; the writer transforms row by row — int4/int8 affine via
/// the canonical `FinchQuantization`, `(1 + w)` norm baking, and bf16 →
/// fp16 / fp32 raw conversions — with scratch bounded by the widest row
/// batch, never whole-tensor buffers.
enum QwenQuantizedWriter {

    /// Widest single-buffer row in the model (elements, bf16). Qwen3.6's
    /// widest is 4096 (the GDN out_proj row); Qwen3.8's hyper-connection
    /// plane and PLE norms are 10240-element rows (hc_norm / norm_key /
    /// norm_query / norm_conv / in_proj_qkv). 16384 covers both with headroom
    /// and bounds scratch at ~96 KB. Wider 1-D entries (conv1d squeezes up to
    /// 40960 elements) are streamed in `maxRowElements`-sized chunks.
    static let maxRowElements = 16384

    // MARK: - Resident file

    static func writeResident(plan: QwenResidentFilePlan,
                              audit: RepackAudit,
                              cancellationCheck: () throws -> Void = {}) throws -> RepackAudit.OutputFile {
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        defer { close(fd) }
        try Posix.ftruncate(fd, path: plan.path, size: plan.totalSize)

        let records = plan.entries.map {
            ResidentIndexRecord(
                name: $0.name, dtype: $0.dtype, logicalShape4: $0.logicalShape4,
                fileOffset: $0.fileOffset, sizeBytes: $0.sizeBytes,
                scaleOffset: $0.scaleOffset, scaleSize: $0.scaleSize,
                biasOffset: $0.biasOffset, biasSize: $0.biasSize)
        }
        let indexData = try ResidentWriter.encodeIndex(
            records: records,
            stringTable: plan.stringTable,
            stringTableOffsets: plan.stringTableOffsets,
            indexSize: plan.indexSize,
            residentSize: plan.residentSize)
        if indexData.count > audit.largestScratchBytes {
            audit.largestScratchBytes = indexData.count
        }
        try indexData.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.pwriteAll(fd: fd, path: plan.path,
                                buf: base, count: indexData.count, offset: 0)
        }
        audit.recordWrite(bytes: indexData.count)

        var shardsByPath: [String: MmapHandle] = [:]
        var scratch = QwenRowScratch()
        scratch.report(audit: audit)

        for entry in plan.entries {
            try cancellationCheck()
            let shard = try mappedShard(path: entry.source.shardPath,
                                        shardsByPath: &shardsByPath)
            try writeResidentEntry(plan: plan, entry: entry, shard: shard,
                                   fd: fd, scratch: &scratch, audit: audit)
        }

        try Posix.fsync(fd, path: plan.path)
        let size = try Posix.fileSize(fd: fd, path: plan.path)
        let sha = try WriterCore.hashEntireFile(path: plan.path, size: size,
                                                audit: audit,
                                                cancellationCheck: cancellationCheck)
        let rel = plan.relativePath
        let outFile = RepackAudit.OutputFile(relativePath: rel, size: size, sha256: sha)
        audit.outputFiles.append(outFile)
        return outFile
    }

    private static func writeResidentEntry(plan: QwenResidentFilePlan,
                                           entry: QwenResidentEntry,
                                           shard: MmapHandle,
                                           fd: Int32,
                                           scratch: inout QwenRowScratch,
                                           audit: RepackAudit) throws {
        let srcBase = entry.source.absoluteOffset
        switch entry.transform {
        case .int4Affine(let rows, let cols):
            try writeAffine(rows: rows, cols: cols, bits: 4,
                            source: shard, srcBase: srcBase,
                            fd: fd, path: plan.path,
                            weightOffset: entry.fileOffset,
                            scaleOffset: entry.scaleOffset,
                            biasOffset: entry.biasOffset,
                            audit: audit)
        case .int8Affine(let rows, let cols):
            try writeAffine(rows: rows, cols: cols, bits: 8,
                            source: shard, srcBase: srcBase,
                            fd: fd, path: plan.path,
                            weightOffset: entry.fileOffset,
                            scaleOffset: entry.scaleOffset,
                            biasOffset: entry.biasOffset,
                            audit: audit)
        case .normOnePlusW(let n):
            try writeNormOnePlusW(count: n, source: shard, srcBase: srcBase,
                                  fd: fd, path: plan.path,
                                  dstOffset: entry.fileOffset,
                                  scratch: &scratch, audit: audit)
        case .normRawBf16(let n):
            try writeRawBf16(count: n, source: shard, srcBase: srcBase,
                             fd: fd, path: plan.path,
                             dstOffset: entry.fileOffset,
                             audit: audit)
        case .bf16ToFp16(let n):
            try writeBf16To(count: n, fp32Out: false,
                            source: shard, srcBase: srcBase,
                            fd: fd, path: plan.path,
                            dstOffset: entry.fileOffset,
                            scratch: &scratch, audit: audit)
        case .bf16ToFp32(let n):
            try writeBf16To(count: n, fp32Out: true,
                            source: shard, srcBase: srcBase,
                            fd: fd, path: plan.path,
                            dstOffset: entry.fileOffset,
                            scratch: &scratch, audit: audit)
        case .rawInt64(let n):
            // PLE hash metadata (int64 elements) — byte-exact copy, never
            // through a float representation.
            try writeRawBytes(byteCount: n * 8,
                              source: shard, srcBase: srcBase,
                              fd: fd, path: plan.path,
                              dstOffset: entry.fileOffset,
                              audit: audit)
        }
    }

    /// One int4/int8 affine transform: per source row, quantize into
    /// packed weights + BF16 scales/biases. Rows are processed in batches of
    /// `affineBatchRows`: each batch is one mmap slice, one quantize pass per
    /// row into batch buffers, and THREE contiguous pwrites (weights region,
    /// scales region, biases region are row-contiguous in the output) —
    /// per-row 1 KB strided pwrites to an external drive are far too slow.
    private static let affineBatchRows = 64

    private static func writeAffine(rows: Int, cols: Int, bits: Int,
                                    source: MmapHandle, srcBase: UInt64,
                                    fd: Int32, path: String,
                                    weightOffset: UInt64,
                                    scaleOffset: UInt64,
                                    biasOffset: UInt64,
                                    audit: RepackAudit) throws {
        let groups = cols / FinchQuantization.groupSize
        let packedRowBytes = cols / (8 / bits)
        let auxRowBytes = groups * 2

        var floats = [Float](repeating: 0, count: cols)
        var packedBatch = [UInt8](repeating: 0, count: affineBatchRows * packedRowBytes)
        var scalesBatch = [UInt16](repeating: 0, count: affineBatchRows * groups)
        var biasesBatch = [UInt16](repeating: 0, count: affineBatchRows * groups)
        let batchBytes = floats.count * 4 + packedBatch.count
            + (scalesBatch.count + biasesBatch.count) * 2
        if batchBytes > audit.largestScratchBytes {
            audit.largestScratchBytes = batchBytes
        }

        var row = 0
        while row < rows {
            let batch = min(affineBatchRows, rows - row)
            let srcOff = srcBase + UInt64(row * cols * 2)
            let src = source.slice(at: srcOff, count: batch * cols * 2)
            audit.recordRead(bytes: batch * cols * 2)

            for i in 0..<batch {
                // Decode the bf16 row straight into the scratch float buffer.
                let rowBase = i * cols * 2
                for k in 0..<cols {
                    let bits16 = UInt16(src[rowBase + 2 * k])
                        | UInt16(src[rowBase + 2 * k + 1]) << 8
                    floats[k] = FinchQuantization.bf16ToFloat(bits16)
                }
                if bits == 4 {
                    let q = floats.withUnsafeBufferPointer {
                        FinchQuantization.quantizeInt4Affine($0, count: cols)
                    }
                    q.packed.withUnsafeBytes { raw in
                        packedBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * packedRowBytes),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                    q.scales.withUnsafeBytes { raw in
                        scalesBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * groups * 2),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                    q.biases.withUnsafeBytes { raw in
                        biasesBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * groups * 2),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                } else {
                    let q = floats.withUnsafeBufferPointer {
                        FinchQuantization.quantizeInt8Affine($0, count: cols)
                    }
                    q.packed.withUnsafeBytes { raw in
                        packedBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * packedRowBytes),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                    q.scales.withUnsafeBytes { raw in
                        scalesBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * groups * 2),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                    q.biases.withUnsafeBytes { raw in
                        biasesBatch.withUnsafeMutableBytes { dst in
                            memcpy(dst.baseAddress!.advanced(by: i * groups * 2),
                                   raw.baseAddress!, raw.count)
                        }
                    }
                }
            }

            try packedBatch.withUnsafeBytes { raw in
                try Posix.pwriteAll(fd: fd, path: path,
                                    buf: raw.baseAddress!,
                                    count: batch * packedRowBytes,
                                    offset: weightOffset + UInt64(row * packedRowBytes))
            }
            try scalesBatch.withUnsafeBytes { raw in
                try Posix.pwriteAll(fd: fd, path: path,
                                    buf: raw.baseAddress!,
                                    count: batch * auxRowBytes,
                                    offset: scaleOffset + UInt64(row * auxRowBytes))
            }
            try biasesBatch.withUnsafeBytes { raw in
                try Posix.pwriteAll(fd: fd, path: path,
                                    buf: raw.baseAddress!,
                                    count: batch * auxRowBytes,
                                    offset: biasOffset + UInt64(row * auxRowBytes))
            }
            audit.recordWrite(bytes: batch * (packedRowBytes + 2 * auxRowBytes))
            source.adviseDontNeed(offset: srcOff, count: batch * cols * 2)
            row += batch
        }
    }

    /// `(1 + w)` baking for the Qwen RMSNorm weights, emitted as BF16.
    private static func writeNormOnePlusW(count: Int,
                                          source: MmapHandle, srcBase: UInt64,
                                          fd: Int32, path: String,
                                          dstOffset: UInt64,
                                          scratch: inout QwenRowScratch,
                                          audit: RepackAudit) throws {
        let src = source.slice(at: srcBase, count: count * 2)
        audit.recordRead(bytes: count * 2)
        let floats = try scratch.decodeBf16Row(src, count: count)
        let baked = scratch.encodeBf16OnePlusW(floats)
        try baked.withUnsafeBytes { raw in
            try Posix.pwriteAll(fd: fd, path: path,
                                buf: raw.baseAddress!, count: baked.count * 2,
                                offset: dstOffset)
        }
        audit.recordWrite(bytes: baked.count * 2)
        source.adviseDontNeed(offset: srcBase, count: count * 2)
    }

    /// Byte copy of the source's raw bf16 row (RMSNormGated weight-direct —
    /// the runtime kernel multiplies the stored values unchanged).
    private static func writeRawBf16(count: Int,
                                     source: MmapHandle, srcBase: UInt64,
                                     fd: Int32, path: String,
                                     dstOffset: UInt64,
                                     audit: RepackAudit) throws {
        try writeRawBytes(byteCount: count * 2,
                          source: source, srcBase: srcBase,
                          fd: fd, path: path,
                          dstOffset: dstOffset,
                          audit: audit)
    }

    /// Raw byte copy of `byteCount` source bytes (shared by the bf16 norm
    /// rows and the int64 PLE hash metadata).
    private static func writeRawBytes(byteCount: Int,
                                      source: MmapHandle, srcBase: UInt64,
                                      fd: Int32, path: String,
                                      dstOffset: UInt64,
                                      audit: RepackAudit) throws {
        let src = source.slice(at: srcBase, count: byteCount)
        audit.recordRead(bytes: byteCount)
        try src.withUnsafeBytes { raw in
            try Posix.pwriteAll(fd: fd, path: path,
                                buf: raw.baseAddress!, count: raw.count,
                                offset: dstOffset)
        }
        audit.recordWrite(bytes: byteCount)
        source.adviseDontNeed(offset: srcBase, count: byteCount)
    }

    /// Raw conversion (bf16 → fp16, or bf16 → fp32) for conv1d / A_log /
    /// dt_bias, streamed in scratch-bounded chunks (the conv1d entry is
    /// 32768 elements, wider than the per-row scratch).
    private static func writeBf16To(count: Int, fp32Out: Bool,
                                    source: MmapHandle, srcBase: UInt64,
                                    fd: Int32, path: String,
                                    dstOffset: UInt64,
                                    scratch: inout QwenRowScratch,
                                    audit: RepackAudit) throws {
        let elementBytes = fp32Out ? 4 : 2
        var done = 0
        while done < count {
            let chunk = min(QwenQuantizedWriter.maxRowElements, count - done)
            let src = source.slice(at: srcBase + UInt64(done * 2), count: chunk * 2)
            audit.recordRead(bytes: chunk * 2)
            let floats = try scratch.decodeBf16Row(src, count: chunk)
            if fp32Out {
                let words = floats.map { $0.bitPattern }
                try words.withUnsafeBytes { raw in
                    try Posix.pwriteAll(fd: fd, path: path,
                                        buf: raw.baseAddress!, count: raw.count,
                                        offset: dstOffset + UInt64(done * 4))
                }
            } else {
                let halves = floats.map { Float16($0) }
                try halves.withUnsafeBytes { raw in
                    // raw.count is already in bytes ([Float16] → 2 B/element).
                    try Posix.pwriteAll(fd: fd, path: path,
                                        buf: raw.baseAddress!, count: raw.count,
                                        offset: dstOffset + UInt64(done * 2))
                }
            }
            audit.recordWrite(bytes: chunk * elementBytes)
            source.adviseDontNeed(offset: srcBase + UInt64(done * 2), count: chunk * 2)
            done += chunk
        }
    }

    // MARK: - Expert layer files

    static func writeLayer(plan: LayerFilePlan,
                           audit: RepackAudit,
                           cancellationCheck: () throws -> Void = {}) throws -> RepackAudit.OutputFile {
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        defer { close(fd) }
        try Posix.ftruncate(fd, path: plan.path, size: plan.fileSize)

        var shardsByPath: [String: MmapHandle] = [:]
        var scratch = QwenRowScratch()
        scratch.report(audit: audit)

        // Only the weight slices carry source rows; scales/biases are
        // computed alongside them.
        let weightSlices = plan.subTensors.filter { $0.component == "weights" }
        for expert in 0..<plan.expertsPerLayer {
            try cancellationCheck()
            let blobBase = UInt64(plan.physicalRank(for: expert)) * plan.expertStride
            for slice in weightSlices {
                let shard = try mappedShard(path: slice.sourceTensor.shardPath,
                                            shardsByPath: &shardsByPath)
                let rows = Int(slice.logicalShape[0])
                let cols = Int(slice.logicalShape[1])
                let srcBase = slice.sourceTensor.absoluteOffset
                    + slice.sourceBaseOffset
                    + UInt64(expert) * slice.sourceOffsetPerExpert
                guard let scaleSlice = plan.subTensors.first(where: {
                    $0.role == slice.role && $0.component == "scales"
                }), let biasSlice = plan.subTensors.first(where: {
                    $0.role == slice.role && $0.component == "biases"
                }) else {
                    throw RepackError.configurationInvalid(
                        detail: "layer \(plan.layerIndex) role \(slice.role) missing scales/biases slices")
                }
                try writeAffine(rows: rows, cols: cols, bits: 4,
                                source: shard, srcBase: srcBase,
                                fd: fd, path: plan.path,
                                weightOffset: blobBase + slice.offsetInExpertBlob,
                                scaleOffset: blobBase + scaleSlice.offsetInExpertBlob,
                                biasOffset: blobBase + biasSlice.offsetInExpertBlob,
                                audit: audit)
            }
        }

        try Posix.fsync(fd, path: plan.path)
        let size = try Posix.fileSize(fd: fd, path: plan.path)
        let sha = try WriterCore.hashEntireFile(path: plan.path, size: size,
                                                audit: audit,
                                                cancellationCheck: cancellationCheck)
        let rel = plan.relativePath
        let outFile = RepackAudit.OutputFile(relativePath: rel, size: size, sha256: sha)
        audit.outputFiles.append(outFile)
        return outFile
    }

    // MARK: - PLE n-gram part files

    /// One PLE n-gram table part. Each 160-wide BF16 source row is quantized
    /// to int4 affine (group `plan.groupSize`, 32) and written per-row as
    /// `[packed nibbles: cols/2][scale BF16 × nGroups][bias BF16 × nGroups]`
    /// — a fixed `rowByteStride` (100 bytes for 160 / group 32). Rows are
    /// independent (one row decoded from its own bytes), processed in bounded
    /// batches with the same mmap-slice → transform → pwrite discipline the
    /// resident/expert writers use, and evicted from page cache as we go.
    /// Real parts are ~800 MB of BF16 → ~250 MB quantized.
    static func writePLEPart(plan: QwenPLEPartFilePlan,
                             audit: RepackAudit,
                             cancellationCheck: () throws -> Void = {}) throws -> RepackAudit.OutputFile {
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        defer { close(fd) }

        let cols = plan.cols
        let groupSize = plan.groupSize
        let nGroups = cols / groupSize
        let packedRowBytes = cols / 2
        let auxBytes = nGroups * 2
        let rowStride = plan.rowByteStride
        let totalRows = plan.rows
        let totalBytes = UInt64(totalRows) * UInt64(rowStride)
        try Posix.ftruncate(fd, path: plan.path, size: totalBytes)

        let shard = try MmapHandle(path: plan.source.shardPath)
        let srcBase = plan.source.absoluteOffset

        let batchRows = 1024
        let srcBatchBytes = batchRows * cols * 2
        let outBatchBytes = batchRows * rowStride
        let scratchBytes = outBatchBytes + cols * MemoryLayout<Float>.size
        if scratchBytes > audit.largestScratchBytes {
            audit.largestScratchBytes = scratchBytes
        }
        var floats = [Float](repeating: 0, count: cols)
        var out = [UInt8](repeating: 0, count: batchRows * rowStride)

        var row = 0
        while row < totalRows {
            try cancellationCheck()
            let batch = min(batchRows, totalRows - row)
            let srcOff = srcBase + UInt64(row * cols * 2)
            let src = shard.slice(at: srcOff, count: batch * cols * 2)
            audit.recordRead(bytes: batch * cols * 2)

            for i in 0..<batch {
                let rowBase = i * cols * 2
                for k in 0..<cols {
                    let bits = UInt16(src[rowBase + 2 * k])
                        | UInt16(src[rowBase + 2 * k + 1]) << 8
                    floats[k] = FinchQuantization.bf16ToFloat(bits)
                }
                let q = floats.withUnsafeBufferPointer {
                    FinchQuantization.quantizeInt4AffinePLE($0, count: cols, groupSize: groupSize)
                }
                let dst = i * rowStride
                q.packed.withUnsafeBytes { memcpy(&out[dst], $0.baseAddress!, $0.count) }
                q.scales.withUnsafeBytes { memcpy(&out[dst + packedRowBytes], $0.baseAddress!, $0.count) }
                q.biases.withUnsafeBytes { memcpy(&out[dst + packedRowBytes + auxBytes], $0.baseAddress!, $0.count) }
            }

            try out.withUnsafeBytes { raw in
                try Posix.pwriteAll(fd: fd, path: plan.path,
                                    buf: raw.baseAddress!, count: batch * rowStride,
                                    offset: UInt64(row) * UInt64(rowStride))
            }
            audit.recordWrite(bytes: batch * rowStride)
            shard.adviseDontNeed(offset: srcOff, count: batch * cols * 2)
            row += batch
        }

        try Posix.fsync(fd, path: plan.path)
        let size = try Posix.fileSize(fd: fd, path: plan.path)
        let sha = try WriterCore.hashEntireFile(path: plan.path, size: size,
                                                audit: audit,
                                                cancellationCheck: cancellationCheck)
        let outFile = RepackAudit.OutputFile(relativePath: plan.relativePath,
                                             size: size, sha256: sha)
        audit.outputFiles.append(outFile)
        return outFile
    }

    // MARK: - Scratch

    /// Reusable per-row scratch: a heap float buffer (max row =
    /// `maxRowElements` — 10240-wide Qwen3.8 HC/PLE norm rows included) and a
    /// word buffer for conversions. Never grows past the widest row.
    struct QwenRowScratch {
        var floats = [Float](repeating: 0, count: QwenQuantizedWriter.maxRowElements)
        var words = [UInt16](repeating: 0, count: QwenQuantizedWriter.maxRowElements)

        mutating func report(audit: RepackAudit) {
            let bytes = floats.count * MemoryLayout<Float>.size
                + words.count * MemoryLayout<UInt16>.size
            if bytes > audit.largestScratchBytes {
                audit.largestScratchBytes = bytes
            }
        }

        mutating func decodeBf16Row(_ bytes: UnsafeRawBufferPointer,
                                    count: Int) throws -> [Float] {
            guard bytes.count >= count * 2 else {
                throw RepackError.preadShort(path: "(mmap slice)", expected: count * 2,
                                             got: bytes.count, errno: 0)
            }
            guard count <= floats.count else {
                throw RepackError.scratchExceeded(requested: count,
                                                  limit: floats.count)
            }
            for i in 0..<count {
                let bits = UInt16(bytes[2 * i]) | UInt16(bytes[2 * i + 1]) << 8
                floats[i] = FinchQuantization.bf16ToFloat(bits)
            }
            return Array(floats[0..<count])
        }

        mutating func encodeBf16OnePlusW(_ values: [Float]) -> [UInt16] {
            for (i, v) in values.enumerated() {
                words[i] = FinchQuantization.bf16Bits(1.0 + v)
            }
            return Array(words[0..<values.count])
        }
    }

    // MARK: - Helpers

    private static func mappedShard(path: String,
                                    shardsByPath: inout [String: MmapHandle]) throws -> MmapHandle {
        if let h = shardsByPath[path] { return h }
        let h = try MmapHandle(path: path)
        shardsByPath[path] = h
        return h
    }
}
