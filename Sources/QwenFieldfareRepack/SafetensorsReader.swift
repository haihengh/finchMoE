import Foundation
import QwenFieldfareFormat

/// Reads a `.safetensors` file: parses the JSON header (which follows an 8-byte
/// little-endian length prefix), memory-maps the data region, and exposes a
/// `tensor(name:)` accessor returning dtype, shape, and a raw buffer pointer.
///
/// The memory mapping stays valid for the lifetime of the reader instance.
public final class SafetensorsReader {

    public struct TensorInfo: Sendable {
        public let name: String
        public let dtype: String       // safetensors dtype string, e.g. "F16", "U32", "BF16"
        public let shape: [Int]
        public let dataOffsetStart: Int // relative to start of data region
        public let dataOffsetEnd: Int
    }

    public enum ReaderError: Error, CustomStringConvertible {
        case cannotOpen(String)
        case cannotStat(String)
        case mmapFailed(String)
        case badHeader(String)
        case tensorNotFound(String)

        public var description: String {
            switch self {
            case .cannotOpen(let s): return "SafetensorsReader: cannot open \(s)"
            case .cannotStat(let s): return "SafetensorsReader: cannot stat \(s)"
            case .mmapFailed(let s): return "SafetensorsReader: mmap failed for \(s)"
            case .badHeader(let s): return "SafetensorsReader: bad header — \(s)"
            case .tensorNotFound(let s): return "SafetensorsReader: tensor not found — \(s)"
            }
        }
    }

    public let url: URL
    private let fd: Int32
    private let fileSize: Int
    private let mapBase: UnsafeMutableRawPointer
    /// Byte offset in the file where the tensor data region begins
    /// (8 + headerLength).
    private let dataRegionStart: Int
    private var infos: [String: TensorInfo] = [:]
    public private(set) var tensorNames: [String] = []

    public init(url: URL) throws {
        self.url = url
        let path = url.path

        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw ReaderError.cannotOpen(path) }
        self.fd = fd

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw ReaderError.cannotStat(path)
        }
        let size = Int(st.st_size)
        self.fileSize = size

        guard size > 8 else {
            close(fd)
            throw ReaderError.badHeader("file too small")
        }

        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            close(fd)
            throw ReaderError.mmapFailed(path)
        }
        self.mapBase = base

        // Sequential read hint — repacking scans linearly.
        madvise(base, size, MADV_SEQUENTIAL)

        // Parse the 8-byte little-endian header length.
        let headerLen = base.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        let headerLength = Int(UInt64(littleEndian: headerLen))
        guard headerLength > 0, 8 + headerLength <= size else {
            munmap(base, size); close(fd)
            throw ReaderError.badHeader("invalid header length \(headerLength)")
        }
        self.dataRegionStart = 8 + headerLength

        // Parse header JSON.
        let headerData = Data(bytes: base.advanced(by: 8), count: headerLength)
        guard let obj = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            munmap(base, size); close(fd)
            throw ReaderError.badHeader("header is not a JSON object")
        }

        for (name, value) in obj {
            if name == "__metadata__" { continue }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shapeArr = entry["shape"] as? [Any],
                  let offsets = entry["data_offsets"] as? [Any],
                  offsets.count == 2 else {
                continue
            }
            let shape = shapeArr.compactMap { ($0 as? NSNumber)?.intValue }
            guard let s0 = (offsets[0] as? NSNumber)?.intValue,
                  let s1 = (offsets[1] as? NSNumber)?.intValue else { continue }
            let info = TensorInfo(name: name, dtype: dtype, shape: shape,
                                  dataOffsetStart: s0, dataOffsetEnd: s1)
            infos[name] = info
        }
        tensorNames = infos.keys.sorted()
    }

    deinit {
        munmap(mapBase, fileSize)
        close(fd)
    }

    public func info(for name: String) -> TensorInfo? { infos[name] }

    public func contains(_ name: String) -> Bool { infos[name] != nil }

    /// Returns dtype, shape, and a raw buffer pointer into the mmap for `name`.
    /// The pointer is valid for the lifetime of this reader.
    public func tensor(name: String) throws -> (dtype: QTurboDType, shape: [Int], buffer: UnsafeRawBufferPointer) {
        guard let info = infos[name] else { throw ReaderError.tensorNotFound(name) }
        let start = dataRegionStart + info.dataOffsetStart
        let len = info.dataOffsetEnd - info.dataOffsetStart
        let ptr = UnsafeRawPointer(mapBase.advanced(by: start))
        let buf = UnsafeRawBufferPointer(start: ptr, count: len)
        return (mapDType(info.dtype), info.shape, buf)
    }

    /// Raw byte length of a tensor's payload.
    public func byteLength(name: String) -> Int? {
        guard let info = infos[name] else { return nil }
        return info.dataOffsetEnd - info.dataOffsetStart
    }

    private func mapDType(_ st: String) -> QTurboDType {
        switch st.uppercased() {
        case "F16": return .fp16
        case "BF16": return .bf16
        case "F32": return .fp32
        case "U32", "I32": return .uint32
        case "U8", "I8": return .int4Packed // MLX sometimes stores packed as uint32; keep u32 path primary
        default: return .fp16
        }
    }
}
