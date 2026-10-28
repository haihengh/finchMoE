import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "hello")
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.repetitionPenalty == 1)
        #expect(request.promptReuseEnabled)
        #expect(!request.isPureGreedy)
    }

    @Test func promptReuseCanBeTurnedOffPerRequest() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           promptReuseEnabled: false)
        #expect(!request.promptReuseEnabled)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           temperature: 0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "   ")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func imageContentCoversCurrentTurnAndReplayedHistory() {
        let image = AppImageAttachment(data: Data([0xFF, 0xD8, 0xFF]))
        let textOnly = AppGenerationRequest(modelDirectory: existingDirectory,
                                            prompt: "hello")
        #expect(!textOnly.hasImageContent)

        let current = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "what is this?",
                                           images: [image])
        #expect(current.hasImageContent)

        let replayed = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "and now?",
            history: [
                AppChatTurn(role: .user, text: "look", images: [image]),
                AppChatTurn(role: .assistant, text: "a red square"),
            ])
        #expect(replayed.hasImageContent)
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               prompt: "hello", topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.finch"),
            prompt: "hello")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }
}
