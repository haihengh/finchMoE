import Foundation
import Metal

/// One BF16 tensor in `vision/vision.safetensors`.
struct VisionTensor: Sendable, Equatable {
    let name: String
    let shape: [Int]
    /// Byte offset from the start of the mapped payload buffer.
    let offset: Int
    let sizeBytes: Int
}

/// The vision payload as one read-only mapping plus a name → tensor table.
///
/// The file is a standard safetensors container: an 8-byte little-endian header
/// length, a JSON header whose `data_offsets` are relative to the byte after
/// the header, then the tensors. `ResidentBuffer` maps the payload region
/// (header excluded) and wraps it in a single `MTLBuffer`, so a tensor's offset
/// in the table is also its offset in the buffer.
final class VisionWeights {
    static let relativePath = "vision/vision.safetensors"
    static let maximumHeaderBytes = 1 << 20

    let buffer: MTLBuffer
    let tensors: [String: VisionTensor]
    /// Number of BF16 bytes in the mapped payload.
    let payloadBytes: Int

    init(directory: FinchModelDirectory,
         device: MTLDevice,
         maxBytes: UInt64 = 2 << 30) throws {
        let url = directory.rootURL.appendingPathComponent(Self.relativePath,
                                                           isDirectory: false)
        let fd = try directory.openFile(Self.relativePath)
        var closeFD = true
        defer { if closeFD { close(fd) } }

        var fileStat = stat()
        guard fstat(fd, &fileStat) == 0, fileStat.st_size > 0 else {
            throw VisionError.missingPayload(url.path)
        }
        let fileSize = UInt64(fileStat.st_size)
        guard fileSize <= maxBytes else {
            throw VisionError.invalidPayload(
                "payload is \(fileSize) bytes, above the \(maxBytes)-byte cap")
        }

        var lengthBytes = [UInt8](repeating: 0, count: 8)
        try Self.readExactly(fd: fd, into: &lengthBytes, count: 8,
                             offset: 0, path: url.path)
        var headerLength: UInt64 = 0
        for i in 0..<8 {
            headerLength |= UInt64(lengthBytes[i]) << UInt64(i * 8)
        }
        guard headerLength > 0, headerLength <= UInt64(Self.maximumHeaderBytes),
              headerLength <= fileSize - 8 else {
            throw VisionError.invalidPayload(
                "header length \(headerLength) is outside the file")
        }
        var headerBytes = [UInt8](repeating: 0, count: Int(headerLength))
        try Self.readExactly(fd: fd, into: &headerBytes, count: headerBytes.count,
                             offset: 8, path: url.path)
        let header = try Self.parseHeader(Data(headerBytes), fileSize: fileSize,
                                          headerLength: headerLength, path: url.path)

        let payloadBase = 8 + headerLength
        let payloadSize = fileSize - payloadBase
        guard payloadSize <= UInt64(Int.max) else {
            throw VisionError.invalidPayload("payload is not addressable")
        }
        let mapped = try ResidentBuffer(
            fileURL: url,
            fileOffset: payloadBase,
            residentSize: payloadSize,
            device: device,
            fileDescriptor: fd,
            wirePages: false)
        closeFD = false
        close(fd)

        self.buffer = mapped.buffer
        self.tensors = header
        self.payloadBytes = Int(payloadSize)
    }

    func tensor(_ name: String) throws -> VisionTensor {
        guard let tensor = tensors[name] else {
            throw VisionError.invalidPayload("missing tensor \(name)")
        }
        return tensor
    }

    func require(_ name: String, shape: [Int], elementBytes: Int = 2) throws
        -> VisionTensor {
        let tensor = try tensor(name)
        guard tensor.shape == shape || shape.isEmpty else {
            throw VisionError.invalidPayload(
                "\(name) has shape \(tensor.shape), expected \(shape)")
        }
        guard tensor.sizeBytes == shape.reduce(1, *) * elementBytes else {
            throw VisionError.invalidPayload("\(name) has an inconsistent size")
        }
        return tensor
    }

    private static func parseHeader(_ data: Data, fileSize: UInt64,
                                    headerLength: UInt64, path: String)
        throws -> [String: VisionTensor] {
        let object: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: data)
                    as? [String: Any] else {
                throw VisionError.invalidPayload("\(path): header is not a JSON object")
            }
            object = parsed
        } catch let error as VisionError {
            throw error
        } catch {
            throw VisionError.invalidPayload("\(path): header: \(error)")
        }

        var tensors: [String: VisionTensor] = [:]
        for (name, value) in object where name != "__metadata__" {
            guard let entry = value as? [String: Any] else {
                throw VisionError.invalidPayload("\(name): entry is not an object")
            }
            guard let dtype = entry["dtype"] as? String, dtype == "BF16" else {
                throw VisionError.invalidPayload(
                    "\(name): unsupported dtype \(entry["dtype"] ?? "nil")")
            }
            guard let rawShape = entry["shape"] as? [NSNumber] else {
                throw VisionError.invalidPayload("\(name): shape is missing")
            }
            let shape = rawShape.map(\.intValue)
            guard let rawOffsets = entry["data_offsets"] as? [NSNumber],
                  rawOffsets.count == 2 else {
                throw VisionError.invalidPayload("\(name): bad data_offsets")
            }
            let begin = rawOffsets[0].intValue
            let end = rawOffsets[1].intValue
            guard begin >= 0, end >= begin else {
                throw VisionError.invalidPayload("\(name): reversed data_offsets")
            }
            let size = end - begin
            let elements = shape.reduce(1) { $0 * max(1, $1) }
            guard elements * 2 == size else {
                throw VisionError.invalidPayload(
                    "\(name): shape \(shape) does not fit \(size) bytes")
            }
            guard UInt64(end) <= fileSize - 8 - headerLength else {
                throw VisionError.invalidPayload("\(name): payload exceeds the file")
            }
            tensors[name] = VisionTensor(name: name, shape: shape,
                                         offset: begin, sizeBytes: size)
        }
        guard !tensors.isEmpty else {
            throw VisionError.invalidPayload("\(path): no tensors")
        }
        return tensors
    }

    private static func readExactly(fd: Int32, into buffer: inout [UInt8],
                                    count: Int, offset: UInt64, path: String) throws {
        var read = 0
        while read < count {
            let got = buffer.withUnsafeMutableBytes { raw -> Int in
                pread(fd, raw.baseAddress!.advanced(by: read),
                      count - read, off_t(offset) + off_t(read))
            }
            if got < 0, errno == EINTR { continue }
            guard got > 0 else {
                throw VisionError.invalidPayload(
                    "\(path): short read at \(offset + UInt64(read))")
            }
            read += got
        }
    }
}
