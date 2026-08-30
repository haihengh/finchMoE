import Foundation
import Metal

/// Owns the shared `MTLDevice`, command queue, and compiled default library
/// (built from `Kernels.metal`, bundled as a package resource).
public final class MetalContext {

    public enum MetalError: Error, CustomStringConvertible {
        case noDevice
        case noCommandQueue
        case libraryNotFound
        case libraryCompileFailed(String)

        public var description: String {
            switch self {
            case .noDevice: return "MetalContext: no Metal device available"
            case .noCommandQueue: return "MetalContext: could not create command queue"
            case .libraryNotFound: return "MetalContext: Kernels.metal library not found in bundle"
            case .libraryCompileFailed(let s): return "MetalContext: library compile failed — \(s)"
            }
        }
    }

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary

    public init() throws {
        #if arch(arm64)
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        #else
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        #endif
        self.device = dev

        guard let queue = dev.makeCommandQueue() else { throw MetalError.noCommandQueue }
        self.commandQueue = queue

        self.library = try Self.loadLibrary(device: dev)
    }

    /// Loads the Metal library. Prefers the precompiled `.metallib` from the
    /// package bundle; falls back to compiling `Kernels.metal` source at runtime.
    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        // 1. Precompiled default library in the resource bundle.
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        // 2. Locate Kernels.metal source in the bundle and compile it.
        if let url = Bundle.module.url(forResource: "Kernels", withExtension: "metal"),
           let source = try? String(contentsOf: url, encoding: .utf8) {
            do {
                let opts = MTLCompileOptions()
                return try device.makeLibrary(source: source, options: opts)
            } catch {
                throw MetalError.libraryCompileFailed(String(describing: error))
            }
        }
        // 3. Last resort: process default library.
        if let lib = device.makeDefaultLibrary() {
            return lib
        }
        throw MetalError.libraryNotFound
    }

    /// Convenience: allocate a shared-storage buffer of `length` bytes.
    public func makeBuffer(length: Int, label: String? = nil) -> MTLBuffer? {
        let buf = device.makeBuffer(length: max(1, length), options: .storageModeShared)
        buf?.label = label
        return buf
    }

    /// Convenience: allocate a shared buffer initialized from raw bytes.
    public func makeBuffer(bytes: UnsafeRawPointer, length: Int, label: String? = nil) -> MTLBuffer? {
        let buf = device.makeBuffer(bytes: bytes, length: max(1, length), options: .storageModeShared)
        buf?.label = label
        return buf
    }
}
