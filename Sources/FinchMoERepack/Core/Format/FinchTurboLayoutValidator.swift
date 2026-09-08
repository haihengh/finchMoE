import Foundation
import FinchMoEFormat

enum FinchTurboLayoutValidator {
    static func validate(path: String,
                                plan: RepackPlan,
                                audit: RepackAudit? = nil) throws {
        try validate(path: path, layers: plan.layers, audit: audit)
    }

    static func validate(path: String,
                                layers: [LayerFilePlan],
                                audit: RepackAudit? = nil) throws {
        // Qwen 3.6 (256 experts × 40 layers) produces a ~22 MB layout.json.
        let data = try Posix.readBoundedData(path, maximumBytes: 64 * 1024 * 1024)
        let layout: FinchTurboPackedExpertsLayoutV1
        do { layout = try FinchTurboPackedExpertsLayoutCodec.decode(data) }
        catch {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: \(error)")
        }
        var validatedLogicalExperts = 0
        for layer in layout.layers {
            guard let planLayer = layers.first(where: { $0.layerIndex == layer.layer }) else {
                throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed layer")
            }
            guard layer.experts.count == planLayer.expertsPerLayer,
                  layout.expertStride == planLayer.expertStride else {
                throw RepackError.configurationInvalid(detail:
                    "layout.json validation failed: plan mismatch in layer \(layer.layer)")
            }
            validatedLogicalExperts += layer.experts.count
        }
        guard layout.layers.count == layers.count else {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: layer count mismatch")
        }
        audit?.packedExpertLayoutAuditLogicalIDCount = validatedLogicalExperts
        audit?.packedExpertLayoutOffsetValidationPassed = true
    }
}
