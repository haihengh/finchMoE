import FinchMoE
import Foundation
import Testing
@testable import FinchMoEAppCore

/// The app-core path for images: request validation, prompt assembly with
/// markers, image injection, and a greedy answer. Gated on the real vision
/// install and the reference fixture, like the engine-level test.
@Suite struct VisionInferenceClientTests {

    private static var modelPath: String {
        ProcessInfo.processInfo.environment["FINCH_VISION_MODEL"]
            ?? FileManager.default.currentDirectoryPath
                + "/models/Qwen3.6-35B-A3B-3bit-vision.finch"
    }

    private static var fixturePath: String {
        let reference = ProcessInfo.processInfo.environment["FINCH_VISION_REF"]
            ?? "/tmp/vision-ref-36-512"
        return reference + "/fixture-512.png"
    }

    private static var fixturesExist: Bool {
        FileManager.default.fileExists(atPath: modelPath + "/manifest.json")
            && FileManager.default.fileExists(atPath: fixturePath)
    }

    @Test(.enabled(if: fixturesExist))
    func theClientDescribesAnAttachedImage() async throws {
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let imageData = try Data(contentsOf: URL(fileURLWithPath: Self.fixturePath))
        // Exercise the same prepare step the picker uses.
        let prepared = try AppImagePreparation.prepare(data: imageData)
        let client = RealInferenceClient(promptReuseEnabled: false)
        try await client.ensureLoaded(
            modelDirectory: modelURL,
            maxContextTokens: 4_096,
            options: AppRuntimeOptions(),
            forceLogitsHead: false) { _ in }

        let request = AppGenerationRequest(
            modelDirectory: modelURL,
            prompt: "What shapes and colors are in this image? Answer in one sentence.",
            images: [AppImageAttachment(mediaType: prepared.mediaType,
                                        data: prepared.data,
                                        pixelWidth: prepared.pixelWidth,
                                        pixelHeight: prepared.pixelHeight)],
            maxNewTokens: 48,
            maxContextTokens: 4_096,
            temperature: 0,
            topK: nil,
            topP: nil,
            runtimeOptions: AppRuntimeOptions())
        var answer = ""
        var prefillTotal: Int?
        for try await event in client.generate(request) {
            switch event {
            case .prefillProgress(_, let total, _):
                prefillTotal = total
            case .token(let token):
                answer += token.textDelta
            case .finished, .cancelled, .failed:
                break
            }
        }

        print("app vision answer: \(answer) (prompt \(prefillTotal ?? -1) tokens)")
        #expect((prefillTotal ?? 0) > 256)
        let lower = answer.lowercased()
        #expect(["red", "blue", "green", "rectangle", "circle", "shape"]
            .contains { lower.contains($0) },
                "the client did not see the image: \(answer)")
    }

    private static var model38Path: String {
        ProcessInfo.processInfo.environment["FINCH_VISION_38_MODEL"]
            ?? FileManager.default.currentDirectoryPath
                + "/models/Qwen3.8-Flash-Next-abliterated-ple2bit-vision.finch"
    }

    private static var fixture38Path: String {
        let reference = ProcessInfo.processInfo.environment["FINCH_VISION_38_REF"]
            ?? "/tmp/vision-ref-38-512"
        return reference + "/fixture-512.png"
    }

    private static var fixtures38Exist: Bool {
        FileManager.default.fileExists(atPath: model38Path + "/manifest.json")
            && FileManager.default.fileExists(atPath: fixture38Path)
    }

    @Test(.enabled(if: fixtures38Exist))
    func theClientDescribesAnAttachedImageOn38() async throws {
        let modelURL = URL(fileURLWithPath: Self.model38Path)
        let imageData = try Data(contentsOf: URL(fileURLWithPath: Self.fixture38Path))
        let prepared = try AppImagePreparation.prepare(data: imageData)
        let client = RealInferenceClient(promptReuseEnabled: false)
        try await client.ensureLoaded(
            modelDirectory: modelURL,
            maxContextTokens: 4_096,
            options: AppRuntimeOptions(),
            forceLogitsHead: false) { _ in }

        let request = AppGenerationRequest(
            modelDirectory: modelURL,
            prompt: "What shapes and colors are in this image? Answer in one sentence.",
            images: [AppImageAttachment(mediaType: prepared.mediaType,
                                        data: prepared.data,
                                        pixelWidth: prepared.pixelWidth,
                                        pixelHeight: prepared.pixelHeight)],
            maxNewTokens: 24,
            maxContextTokens: 4_096,
            temperature: 0,
            topK: nil,
            topP: nil,
            runtimeOptions: AppRuntimeOptions())
        var answer = ""
        var prefillTotal: Int?
        for try await event in client.generate(request) {
            switch event {
            case .prefillProgress(_, let total, _):
                prefillTotal = total
            case .token(let token):
                answer += token.textDelta
            case .finished, .cancelled, .failed:
                break
            }
        }

        print("app vision38 answer: \(answer) (prompt \(prefillTotal ?? -1) tokens)")
        #expect((prefillTotal ?? 0) > 256)
        let lower = answer.lowercased()
        #expect(["red", "blue", "green", "rectangle", "circle", "shape"]
            .contains { lower.contains($0) },
                "the 3.8 client did not see the image: \(answer)")
    }

    @Test func imageOnlyRequestsValidate() throws {
        let image = AppImageAttachment(mediaType: "image/jpeg", data: Data([0xFF]))
        let request = AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            prompt: "",
            images: [image])
        // The directory does not need to exist for this half of validation.
        try request.validate(requireModelDirectory: false)
    }

    @Test func tooManyImagesAreRejected() {
        let images = (0..<(AppImageAttachment.maximumImagesPerTurn + 1)).map {
            AppImageAttachment(mediaType: "image/jpeg",
                               data: Data([UInt8($0)]))
        }
        let request = AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            prompt: "",
            images: images)
        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }
}
