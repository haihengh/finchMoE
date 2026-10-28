import CoreGraphics
import FinchMoE
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Decodes, orients, and downsizes a picked image before it enters a request.
///
/// The vision preprocessor resizes again for the tower, but doing the import
/// step here has three jobs the request layer needs: it bounds the bytes that
/// travel to the decode service, it strips EXIF orientation from the stored
/// copy, and it turns HEIC/PNG/GIF inputs into one predictable JPEG.
public enum AppImagePreparation {
    /// Longest side after import. The tower's own budget is 1536²; keeping the
    /// stored copy on the same order avoids re-encoding work at send time.
    public static let maximumPixelDimension = 1_536
    public static let jpegQuality: Double = 0.84
    /// Standard MIME type. (`UTType.jpeg.identifier` is the UTI
    /// `public.jpeg`, which the request validator rightly rejects.)
    public static let jpegMediaType = "image/jpeg"

    /// The import budget for a model on this device.
    ///
    /// Qwen 3.8's 1536 px grid is 48×48 = 2,304 image tokens. On a device
    /// running this app the image activations, the image-conditioned KV and
    /// the text model's resident weights are already close to the jetsam
    /// ceiling, so 3.8 uses 1024 px (32×32 = 1,024 tokens). The Mac and 3.6
    /// keep the 1536 px budget the tower was validated at.
    ///
    /// The gate stays `canImport(UIKit)` rather than an idiom check, and
    /// deliberately so: an iPad takes the 1,024 path too. Raising a budget is
    /// the risky direction — 1536 px more than doubles the image tokens and
    /// the activations along with them, and the overflow shows up as the
    /// process being killed mid-turn rather than as a slower turn. iPad's
    /// ceiling is higher than the iPhone's but unmeasured here, so the tight
    /// budget stands until `AppMemorySampler.memoryLimitBytes` gives a
    /// number to price against.
    public static func recommendedPixelDimension(
        for architecture: ArchConfig
    ) -> Int {
        #if canImport(UIKit)
        if architecture.isQwen3_8 { return 1_024 }
        #endif
        return maximumPixelDimension
    }

    public struct PreparedImage: Sendable {
        public let data: Data
        public let mediaType: String
        public let pixelWidth: Int
        public let pixelHeight: Int
    }

    public enum PreparationError: Error, CustomStringConvertible {
        case unreadable
        case encodingFailed

        public var description: String {
            switch self {
            case .unreadable: return "The file is not a readable image."
            case .encodingFailed: return "The image could not be prepared."
            }
        }
    }

    public static func prepare(data: Data,
                               maxDimension: Int = maximumPixelDimension) throws
        -> PreparedImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw PreparationError.unreadable
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0, options as CFDictionary) else {
            throw PreparationError.unreadable
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw PreparationError.encodingFailed
        }
        CGImageDestinationAddImage(destination, image, [
            kCGImageDestinationLossyCompressionQuality: jpegQuality,
        ] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw PreparationError.encodingFailed
        }
        return PreparedImage(data: output as Data,
                             mediaType: jpegMediaType,
                             pixelWidth: image.width,
                             pixelHeight: image.height)
    }
}
