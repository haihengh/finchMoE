import Foundation

/// Writes the source vision tower into a `.finch` install as a standalone
/// safetensors payload.
///
/// The Qwen checkpoints keep the tower under `model.visual.*` (333 BF16 tensors
/// for 3.6/3.8). The text runtime ignores files it does not open, so declaring
/// this payload in `manifest.files` makes it part of the download and the
/// verified receipt without changing how today's engine runs the model. A
/// future vision runtime reads `vision/vision.safetensors` plus the config
/// sidecars.
enum VisionPayloadWriter {
    static let tensorPrefix = "model.visual."
    static let relativePath = "vision/vision.safetensors"

    /// Source config files copied beside the payload, keyed by their
    /// `manifest.files` path. `video_preprocessor_config.json` is optional.
    static func configSources(snapshotDir: String) throws
        -> [(source: String, relativePath: String, required: Bool)] {
        [
            (path(snapshotDir, "config.json"), "vision/model_config.json", true),
            (path(snapshotDir, "preprocessor_config.json"),
             "vision/preprocessor_config.json", true),
            (path(snapshotDir, "video_preprocessor_config.json"),
             "vision/video_preprocessor_config.json", false),
        ]
    }

    static func missingRequiredConfig(snapshotDir: String) throws -> String? {
        for entry in try configSources(snapshotDir: snapshotDir) where entry.required {
            guard try Posix.entryKind(entry.source) == .regular else {
                return (entry.source as NSString).lastPathComponent
            }
        }
        return nil
    }

    struct Placement {
        let tensor: SourceTensor
        /// Offset relative to the safetensors payload, not the file.
        let payloadOffset: UInt64
    }

    struct Payload {
        let placements: [Placement]
        let header: Data
        let fileBytes: UInt64
    }

    /// Builds the deterministic safetensors header and computes the exact
    /// output size. No bytes are written here, so callers can include the size
    /// in the disk-space check before starting.
    static func makePayload(snapshot: QwenLocalSnapshot.Snapshot) throws -> Payload {
        var byName: [String: SourceTensor] = [:]
        for header in snapshot.shardHeaders {
            for tensor in header.tensors where tensor.name.hasPrefix(tensorPrefix) {
                byName[tensor.name] = tensor
            }
        }
        guard !byName.isEmpty else {
            throw RepackError.missingTensor(name: "\(tensorPrefix)*")
        }

        let tensors = byName.values.sorted { $0.name < $1.name }
        var placements: [Placement] = []
        placements.reserveCapacity(tensors.count)
        var payloadOffset: UInt64 = 0
        for tensor in tensors {
            // Standard safetensors writers align tensor payloads to 8 bytes.
            if payloadOffset % 8 != 0 {
                payloadOffset += 8 - (payloadOffset % 8)
            }
            placements.append(Placement(tensor: tensor, payloadOffset: payloadOffset))
            payloadOffset += tensor.sizeBytes
        }

        var headerObject: [String: Any] = ["__metadata__": ["format": "pt"]]
        for placement in placements {
            let tensor = placement.tensor
            headerObject[tensor.name] = [
                "dtype": dtypeName(tensor.dtype),
                "shape": tensor.shape.map { Int($0) },
                "data_offsets": [
                    Int(placement.payloadOffset),
                    Int(placement.payloadOffset + tensor.sizeBytes),
                ],
            ]
        }
        var header = try JSONSerialization.data(withJSONObject: headerObject,
                                                options: [.sortedKeys])
        // The header length must make the payload start 8-byte aligned. Spaces
        // are legal JSON whitespace and are what the reference writer pads with.
        while header.count % 8 != 0 {
            header.append(0x20)
        }
        let fileBytes = 8 + UInt64(header.count) + payloadOffset
        return Payload(placements: placements, header: header, fileBytes: fileBytes)
    }

    /// Writes the payload. Byte copying and the audit counters mirror
    /// `WriterCore.pwriteTensorRegion`: source pages are mmap'd and advised
    /// away tile by tile so the 0.9 GB tower does not inflate process RSS.
    static func write(_ payload: Payload,
                      to path: String,
                      audit: RepackAudit) throws {
        try Posix.mkdirP((path as NSString).deletingLastPathComponent)
        let fd = try Posix.openCreateRW(path)
        defer { close(fd) }
        try Posix.ftruncate(fd, path: path, size: payload.fileBytes)

        var headerLength = UInt64(payload.header.count).littleEndian
        try withUnsafeBytes(of: &headerLength) { raw in
            try Posix.pwriteAll(fd: fd, path: path,
                                buf: raw.baseAddress!, count: raw.count, offset: 0)
        }
        try payload.header.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.pwriteAll(fd: fd, path: path,
                                buf: base, count: raw.count, offset: 8)
        }
        audit.recordWrite(bytes: 8 + payload.header.count)

        var shards: [String: MmapHandle] = [:]
        defer { shards.removeAll() }
        let payloadBase = UInt64(8 + payload.header.count)
        for placement in payload.placements {
            let tensor = placement.tensor
            let shard: MmapHandle
            if let existing = shards[tensor.shardPath] {
                shard = existing
            } else {
                let opened = try MmapHandle(path: tensor.shardPath)
                shards[tensor.shardPath] = opened
                shard = opened
            }
            try WriterCore.pwriteTensorRegion(
                srcShard: shard,
                srcAbsoluteOffset: tensor.absoluteOffset,
                size: tensor.sizeBytes,
                dstFd: fd,
                dstPath: path,
                dstOffset: payloadBase + placement.payloadOffset,
                audit: audit)
        }
        try Posix.fsync(fd, path: path)
    }

    private static func dtypeName(_ dtype: SourceTensor.Dtype) -> String {
        switch dtype {
        case .u32: return "U32"
        case .bf16: return "BF16"
        case .fp16: return "F16"
        case .fp32: return "F32"
        case .i64: return "I64"
        }
    }

    private static func path(_ directory: String, _ name: String) -> String {
        (directory as NSString).appendingPathComponent(name)
    }
}
