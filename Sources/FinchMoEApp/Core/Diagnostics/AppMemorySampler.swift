import Darwin
import Darwin.Mach
import Foundation

/// The two numbers the kernel keeps for a task. They answer different
/// questions and can differ by gigabytes:
///
/// - `footprintBytes` (`phys_footprint`) is what the memory limit is
///   enforced against: dirty, compressed, and IOKit memory. It deliberately
///   **excludes clean file-backed pages**.
/// - `residentBytes` (`resident_size`) counts every resident page, including
///   those clean file mappings.
///
/// The model's `model_weights.bin` is `mmap`ed read-only and wrapped in a
/// Metal buffer. Its pages are clean, so on an install whose weights are
/// several GB the footprint can read 1.7 GB while the process actually holds
/// 5+ GB of RAM. A readout labelled "RAM" must not use the footprint alone.
struct AppMemorySample: Equatable, Sendable {
    let footprintBytes: UInt64
    let residentBytes: UInt64

    var reportedBytes: UInt64 { max(footprintBytes, residentBytes) }
}

public final class AppMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let processMemory: @Sendable () -> AppMemorySample?

    public init() {
        self.processMemory = { Self.readProcessMemory() }
    }

    init(processMemory: @escaping @Sendable () -> AppMemorySample?) {
        self.processMemory = processMemory
    }

    public func resetPeak() {
        lock.lock()
        peak = 0
        lock.unlock()
    }

    public func sample() -> UInt64? {
        guard let current = processMemory() else { return nil }
        let bytes = current.reportedBytes
        lock.lock()
        if bytes > peak { peak = bytes }
        lock.unlock()
        return bytes
    }

    private static func readProcessMemory() -> AppMemorySample? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return AppMemorySample(footprintBytes: UInt64(info.phys_footprint),
                               residentBytes: UInt64(info.resident_size))
    }

    public var peakBytes: UInt64? {
        lock.lock()
        let value = peak
        lock.unlock()
        return value == 0 ? nil : value
    }
}
