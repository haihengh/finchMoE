import Foundation

public struct DecodeRuntimeOptions: Codable, Sendable, Equatable {
    public var expertCacheSlots: Int
    public var expertCachePolicy: String
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: String
    public var modelVerification: String

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: String = "lfu",
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 512,
                rdadvisePolicy: String = "off",
                modelVerification: String = "full-sha256") {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
    }
}

public struct DecodeLoadRequest: Codable, Sendable {
    public var modelPath: String
    public var maxContextTokens: Int
    public var runtimeOptions: DecodeRuntimeOptions
    public var forceLogitsHead: Bool
    public var requestID: UUID

    public init(modelPath: String, maxContextTokens: Int,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                forceLogitsHead: Bool = false,
                requestID: UUID = UUID()) {
        self.modelPath = modelPath
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        self.requestID = requestID
    }
}

/// One earlier conversation turn carried to the service, which is the side
/// that owns the tokenizer and therefore the chat template.
public struct DecodeChatTurn: Codable, Sendable, Equatable {
    public var role: String
    public var text: String

    public init(role: String, text: String) {
        self.role = role
        self.text = text
    }
}

public struct DecodeGenerationRequest: Codable, Sendable {
    public var prompt: String
    /// Older turns, oldest first. Absent on the wire means "no history",
    /// which keeps a request from a build that predates chat history decodable.
    public var history: [DecodeChatTurn]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var repetitionPenalty: Float
    public var runtimeOptions: DecodeRuntimeOptions
    public var generationID: UUID

    public init(prompt: String, history: [DecodeChatTurn] = [],
                maxNewTokens: Int, maxContextTokens: Int,
                temperature: Float, repetitionPenalty: Float = 1,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                generationID: UUID = UUID()) {
        self.prompt = prompt
        self.history = history
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
        self.generationID = generationID
    }

    private enum CodingKeys: String, CodingKey {
        case prompt, history, maxNewTokens, maxContextTokens
        case temperature, repetitionPenalty, runtimeOptions, generationID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try container.decode(String.self, forKey: .prompt)
        history = try container.decodeIfPresent(
            [DecodeChatTurn].self, forKey: .history) ?? []
        maxNewTokens = try container.decode(Int.self, forKey: .maxNewTokens)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        temperature = try container.decode(Float.self, forKey: .temperature)
        repetitionPenalty = try container.decode(Float.self, forKey: .repetitionPenalty)
        runtimeOptions = try container.decode(
            DecodeRuntimeOptions.self, forKey: .runtimeOptions)
        generationID = try container.decode(UUID.self, forKey: .generationID)
    }
}

public enum DecodeServiceCommand: Codable, Sendable {
    case load(DecodeLoadRequest)
    case generate(DecodeGenerationRequest)
    case cancel
    case unload(UUID)
    case shutdown
}

public enum DecodeServiceEventKind: String, Codable, Sendable {
    case loading
    case ready
    case prefill
    case snapshot
    case finished
    case cancelled
    case failed
    case unloaded
}

public struct DecodeRunnerDiagnostics: Codable, Sendable, Equatable {
    public var cb1MillisecondsPerToken: Double
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double,
                ioMillisecondsPerToken: Double,
                cb2MillisecondsPerToken: Double,
                headMillisecondsPerToken: Double,
                rdadviseMillisecondsPerToken: Double,
                rdadviseCallsPerToken: Double,
                rdadviseMegabytesPerToken: Double,
                rdadviseSkippedPerToken: Double,
                rdadviseFailures: UInt64) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

public struct DecodePrefillDiagnostics: Codable, Sendable, Equatable {
    public var requestedMode: String
    public var executedMode: String
    public var kvStorageMode: String?
    public var chunkCompleteness: String
    public var unsupportedReason: String?

    public init(requestedMode: String, executedMode: String,
                kvStorageMode: String?, chunkCompleteness: String,
                unsupportedReason: String?) {
        self.requestedMode = requestedMode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
        self.unsupportedReason = unsupportedReason
    }
}

public struct DecodeServiceEvent: Codable, Sendable {
    public var kind: DecodeServiceEventKind
    public var generationID: UUID
    public var sequence: UInt64
    public var textDelta: String
    public var tokenCount: Int
    public var promptTokenCount: Int?
    public var prefillDone: Int?
    public var prefillTotal: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var stopReason: String?
    public var error: String?
    public var currentMemoryBytes: UInt64?
    public var peakMemoryBytes: UInt64?
    public var prefill: DecodePrefillDiagnostics?
    public var runner: DecodeRunnerDiagnostics?
    /// Carried on the load response only, because the service is what performs
    /// the load: the app cannot see the receipt decision any other way.
    public var integrityOutcome: String?

    public init(kind: DecodeServiceEventKind, generationID: UUID,
                sequence: UInt64 = 0, textDelta: String = "",
                tokenCount: Int = 0, promptTokenCount: Int? = nil,
                prefillDone: Int? = nil, prefillTotal: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double? = nil,
                decodeSeconds: Double = 0, tokensPerSecond: Double = 0,
                stopReason: String? = nil, error: String? = nil,
                currentMemoryBytes: UInt64? = nil, peakMemoryBytes: UInt64? = nil,
                prefill: DecodePrefillDiagnostics? = nil,
                runner: DecodeRunnerDiagnostics? = nil,
                integrityOutcome: String? = nil) {
        self.kind = kind
        self.generationID = generationID
        self.sequence = sequence
        self.textDelta = textDelta
        self.tokenCount = tokenCount
        self.promptTokenCount = promptTokenCount
        self.prefillDone = prefillDone
        self.prefillTotal = prefillTotal
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
        self.error = error
        self.currentMemoryBytes = currentMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.prefill = prefill
        self.runner = runner
        self.integrityOutcome = integrityOutcome
    }
}

public enum DecodeFrameCodec {
    public static let maximumPayloadBytes = 4 * 1_024 * 1_024

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let payload = try JSONEncoder().encode(value)
        guard payload.count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        var length = UInt32(payload.count).littleEndian
        var frame = withUnsafeBytes(of: &length) { Data($0) }
        frame.append(payload)
        return frame
    }

    public static func read<T: Decodable>(_ type: T.Type, from handle: FileHandle) throws -> T {
        let header = try readExactly(4, from: handle)
        let count = header.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(as: UInt32.self).littleEndian
        }
        guard count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        let payload = try readExactly(Int(count), from: handle)
        return try JSONDecoder().decode(type, from: payload)
    }

    private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                throw DecodeFrameError.unexpectedEOF
            }
            result.append(chunk)
        }
        return result
    }
}

public enum DecodeFrameError: Error {
    case oversized
    case unexpectedEOF
}
