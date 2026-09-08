import Foundation

/// Loads a LOCAL bf16 Qwen 3.6 snapshot (`model.safetensors.index.json` +
/// `config.json` + shard headers) for the quantizing repack. Unlike the
/// Gemma remote path (`IndexLoader`), the Qwen source has no
/// `config.json -> quantization` slot — the writer quantizes from bf16, so
/// the metadata here is just the weight map, the index hash, and the
/// resolved per-shard tensor headers.
enum QwenLocalSnapshot {

    struct Snapshot {
        let metadata: SourceMetadata
        let arch: ArchInfo
        let shardHeaders: [Safetensors.Header]
    }

    struct SourceMetadata {
        let indexPath: String
        let configPath: String
        let indexSha256Hex: String
        /// `tensor_name -> shard_filename`
        let weightMap: [String: String]
        /// Shard files referenced by the index, in encounter order.
        let shardFilenames: [String]
    }

    static func load(snapshotDir: String) throws -> Snapshot {
        let indexPath = (snapshotDir as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString)
            .appendingPathComponent("config.json")

        let weightMap: [String: String]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let m = root["weight_map"] as? [String: String] else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
            }
            weightMap = m
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "\(error)")
        }
        let indexSha = try Sha256Stream.hashFile(path: indexPath)

        var seen = Set<String>()
        var shards: [String] = []
        for k in weightMap.keys.sorted() {
            let shard = weightMap[k]!
            if !seen.contains(shard) { seen.insert(shard); shards.append(shard) }
        }

        let arch = try ArchInfo.load(configPath: configPath)

        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(shards.count)
        for shard in shards {
            let path = (snapshotDir as NSString).appendingPathComponent(shard)
            let fd = try Posix.openRead(path)
            defer { close(fd) }
            let fileSize = try Posix.fileSize(fd: fd, path: path)
            let prefix = UnsafeMutableRawBufferPointer.allocate(
                byteCount: 8, alignment: 8)
            defer { prefix.deallocate() }
            try Posix.preadAll(fd: fd, path: path, buf: prefix.baseAddress!, count: 8, offset: 0)
            var headerSize: UInt64 = 0
            for i in 0..<8 {
                headerSize |= UInt64(prefix[i]) << UInt64(i * 8)
            }
            if headerSize > Safetensors.maxHeaderBytes || headerSize > fileSize - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
            }
            let headerBytes = UnsafeMutableRawBufferPointer.allocate(
                byteCount: Int(headerSize), alignment: 8)
            defer { headerBytes.deallocate() }
            try Posix.preadAll(fd: fd, path: path,
                               buf: headerBytes.baseAddress!, count: Int(headerSize), offset: 8)
            headers.append(try Safetensors.parseHeaderBytes(
                path: path, fileSize: fileSize,
                headerBytes: Data(bytes: headerBytes.baseAddress!, count: Int(headerSize))))
        }

        return Snapshot(
            metadata: SourceMetadata(indexPath: indexPath, configPath: configPath,
                                     indexSha256Hex: indexSha,
                                     weightMap: weightMap, shardFilenames: shards),
            arch: arch,
            shardHeaders: headers)
    }
}
