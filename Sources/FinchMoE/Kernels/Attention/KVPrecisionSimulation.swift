import Foundation
import Metal

/// `FQ_KV_INT4_SIM=<group>`: rewrite K/V as `dequantize(quantize(x))` in place
/// before attention, with a symmetric scale per `<group>` elements.
///
/// A measurement tool, not a setting — the same shape as `FQ_PLE_QUANT_SIM`.
/// It prices the *precision* half of a 4-bit KV format while the storage, the
/// layout and the byte counts stay fp16, so a quality verdict arrives without a
/// packed format existing. Storage savings are never claimed from a run with
/// this on: it changes values only.
final class KVPrecisionSimulation {
    /// Parse `FQ_KV_INT4_SIM` and build the probe. Returns nil when the variable
    /// is unset or unusable, so an unparseable value cannot silently masquerade
    /// as "simulation on".
    static func make(
        context: MetalContext,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> KVPrecisionSimulation? {
        guard let raw = environment["FQ_KV_INT4_SIM"],
              let group = Int(raw),
              allowedGroups.contains(group) else { return nil }
        return try KVPrecisionSimulation(context: context, groupElements: group)
    }

    static let allowedGroups = [16, 32, 64, 128]

    let groupElements: Int
    let bits: Int

    private let pso: MTLComputePipelineState

    init(context: MetalContext, groupElements: Int, bits: Int = 4) throws {
        precondition(Self.allowedGroups.contains(groupElements),
                     "unsupported KV simulation group \(groupElements)")
        precondition((2...8).contains(bits), "simulation bits must be 2...8")
        self.groupElements = groupElements
        self.bits = bits
        self.pso = try context.pipeline("kv_simulate_int4_roundtrip_rows")
    }

    /// Rounds `rows` contiguous fp16 rows in place.
    func simulate(commandBuffer: MTLCommandBuffer,
                  rows: MTLBuffer, rowsOffset: Int,
                  rowCount: Int,
                  rowDim: Int) {
        precondition(rowCount > 0, "rowCount must be positive")
        precondition(rowDim % groupElements == 0,
                     "rowDim \(rowDim) is not a whole number of \(groupElements)-element groups")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(rows, offset: rowsOffset, index: 0)
        var rd = UInt32(rowDim)
        var ge = UInt32(groupElements)
        var b = UInt32(bits)
        enc.setBytes(&rd, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&ge, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
