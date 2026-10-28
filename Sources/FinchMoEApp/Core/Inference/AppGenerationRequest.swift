import Foundation

/// One image attached to a user turn. The bytes travel with the request so the
/// in-process client and the decode service see the same image; the app keeps
/// a downscaled copy, which is what bounds the request size.
public struct AppImageAttachment: Equatable, Sendable, Codable, Identifiable {
    public static let maximumImagesPerTurn = 6
    public static let maximumBytes = 12 * 1_048_576
    /// One request carries the current turn plus the replayed history images.
    public static let maximumTotalBytes = 24 * 1_048_576

    public var id: UUID
    public var mediaType: String
    public var data: Data
    public var pixelWidth: Int?
    public var pixelHeight: Int?

    public init(id: UUID = UUID(),
                mediaType: String = "image/jpeg",
                data: Data,
                pixelWidth: Int? = nil,
                pixelHeight: Int? = nil) {
        self.id = id
        self.mediaType = mediaType
        self.data = data
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
    }
}

/// One earlier turn of a conversation, replayed to the model ahead of the new
/// prompt so a follow-up question can refer to what was already said.
public struct AppChatTurn: Equatable, Sendable {
    public var role: ChatRole
    public var text: String
    /// Images the user attached to this turn, in prompt order.
    public var images: [AppImageAttachment]

    public init(role: ChatRole,
                text: String,
                images: [AppImageAttachment] = []) {
        self.role = role
        self.text = text
        self.images = images
    }
}

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    public var prompt: String
    /// Images attached to the new user turn, in prompt order.
    public var images: [AppImageAttachment]
    /// Earlier completed turns, oldest first. Empty for the first question of
    /// a conversation.
    public var history: [AppChatTurn]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: AppRuntimeOptions
    /// Whether this turn may resume from the session's retained KV prefix.
    /// Live, unlike `runtimeOptions`: flipping it takes effect on the next
    /// message and does not force a model reload.
    public var promptReuseEnabled: Bool

    public init(modelDirectory: URL,
                prompt: String,
                images: [AppImageAttachment] = [],
                history: [AppChatTurn] = [],
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions(),
                promptReuseEnabled: Bool = true) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.images = images
        self.history = history
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
        self.promptReuseEnabled = promptReuseEnabled
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    /// True when this turn runs the vision tower: either it carries new
    /// images, or it replays an image-bearing turn from history. Those turns
    /// spend time encoding images and building the multimodal prompt before
    /// the text prefill starts.
    public var hasImageContent: Bool {
        !images.isEmpty || history.contains { !$0.images.isEmpty }
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        let hasImages = !images.isEmpty
        guard hasImages || !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt cannot be empty.")
        }
        guard images.count <= AppImageAttachment.maximumImagesPerTurn else {
            throw AppInferenceError.invalidRequest(
                "At most \(AppImageAttachment.maximumImagesPerTurn) images can be attached to one message.")
        }
        for image in images {
            guard !image.data.isEmpty else {
                throw AppInferenceError.invalidRequest("An attached image is empty.")
            }
            guard image.data.count <= AppImageAttachment.maximumBytes else {
                throw AppInferenceError.invalidRequest(
                    "An attached image is larger than \(AppImageAttachment.maximumBytes) bytes.")
            }
            guard image.mediaType.hasPrefix("image/") else {
                throw AppInferenceError.invalidRequest(
                    "Only image attachments are supported.")
            }
        }
        for turn in history {
            guard turn.images.count <= AppImageAttachment.maximumImagesPerTurn else {
                throw AppInferenceError.invalidRequest(
                    "At most \(AppImageAttachment.maximumImagesPerTurn) images can be attached to one message.")
            }
        }
        let totalImageBytes = (images + history.flatMap(\.images))
            .reduce(0) { $0 + $1.data.count }
        guard totalImageBytes <= AppImageAttachment.maximumTotalBytes else {
            throw AppInferenceError.invalidRequest(
                "The attached images are too large for one request.")
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        guard repetitionPenalty >= 1 else {
            throw AppInferenceError.invalidRequest("Repetition penalty must be at least 1.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
        }
    }
}
