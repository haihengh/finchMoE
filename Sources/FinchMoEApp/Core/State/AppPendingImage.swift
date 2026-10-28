import Foundation

/// An image the user attached to the draft but has not sent yet.
public struct AppPendingImage: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var data: Data
    public var mediaType: String
    public var pixelWidth: Int?
    public var pixelHeight: Int?

    public init(id: UUID = UUID(),
                data: Data,
                mediaType: String,
                pixelWidth: Int? = nil,
                pixelHeight: Int? = nil) {
        self.id = id
        self.data = data
        self.mediaType = mediaType
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
    }

    public var attachment: AppImageAttachment {
        AppImageAttachment(id: id,
                           mediaType: mediaType,
                           data: data,
                           pixelWidth: pixelWidth,
                           pixelHeight: pixelHeight)
    }
}
