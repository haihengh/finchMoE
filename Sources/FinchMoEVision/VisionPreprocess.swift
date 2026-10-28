import CoreGraphics
import Foundation
import ImageIO
import Metal

/// A preprocessed image ready for the vision tower.
struct VisionPreparedImage {
    /// `[N, 3 * temporalPatchSize * patchSize * patchSize]` FP16, in merge
    /// block order (`[C, T, P, P]` inside each patch).
    let pixels: MTLBuffer
    /// Grid in patches: `(1, h/16, w/16)`.
    let gridT: Int
    let gridH: Int
    let gridW: Int
    /// Number of LLM image tokens: every 2×2 patch block becomes one token.
    let tokenCount: Int
    /// Pixel dimensions actually fed to the tower.
    let pixelWidth: Int
    let pixelHeight: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int

    var patchCount: Int { gridT * gridH * gridW }
}

/// Qwen2VL/Qwen3VL image preprocessing: smart resize, normalize, and
/// merge-block patchification. This is CPU work (tens of milliseconds for a
/// 1536² image); the tower itself is on the GPU.
enum VisionPreprocessor {

    static func prepare(imageData: Data,
                        config: VisionConfig,
                        device: MTLDevice,
                        maxPixels: Int) throws -> VisionPreparedImage {
        let geometry = config.geometry
        let factor = geometry.patchSize * geometry.spatialMergeSize
        let cap = max(factor * factor, min(maxPixels, config.configuredMaxPixels))

        let source = try decode(imageData)
        let (targetW, targetH) = smartResize(width: source.width,
                                             height: source.height,
                                             factor: factor,
                                             minPixels: config.minPixels,
                                             maxPixels: cap)
        let rgba = try render(source: source, width: targetW, height: targetH)

        let gridW = targetW / geometry.patchSize
        let gridH = targetH / geometry.patchSize
        guard gridW > 0, gridH > 0,
              gridW % geometry.spatialMergeSize == 0,
              gridH % geometry.spatialMergeSize == 0 else {
            throw VisionError.unsupportedImage(
                "resized image \(targetW)×\(targetH) is not a multiple of "
                + "\(factor) in both dimensions")
        }
        let patches = gridW * gridH
        let tokens = patches / (geometry.spatialMergeSize * geometry.spatialMergeSize)
        let elementsPerPatch = geometry.inChannels
            * geometry.temporalPatchSize
            * geometry.patchSize * geometry.patchSize
        var pixels = [Float16](repeating: 0, count: patches * elementsPerPatch)

        // The processor groups tokens by merge block, then walks the 2×2
        // patches inside each block row-major. That order must match
        // `fast_pos_embed_interpolate` and the 2D RoPE coordinates.
        var token = 0
        let merge = geometry.spatialMergeSize
        let patch = geometry.patchSize
        let temporal = geometry.temporalPatchSize
        let channels = geometry.inChannels
        let rowBytes = targetW * 4
        rgba.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for blockY in stride(from: 0, to: gridH, by: merge) {
                for blockX in stride(from: 0, to: gridW, by: merge) {
                    for intraY in 0..<merge {
                        for intraX in 0..<merge {
                            let patchY = (blockY + intraY) * patch
                            let patchX = (blockX + intraX) * patch
                            var out = token * elementsPerPatch
                            // [C, T, P, P]: the single image frame is
                            // duplicated into both temporal slots, matching
                            // the processor's temporal patch.
                            for c in 0..<channels {
                                for _ in 0..<temporal {
                                    for y in 0..<patch {
                                        let row = (patchY + y) * rowBytes
                                        for x in 0..<patch {
                                            let index = row + (patchX + x) * 4 + c
                                            let value = Float(bytes[index]) / 255.0
                                            let normalized = (value - config.imageMean[c])
                                                / config.imageStd[c]
                                            pixels[out] = Float16(normalized)
                                            out += 1
                                        }
                                    }
                                }
                            }
                            token += 1
                        }
                    }
                }
            }
        }
        guard token == patches else {
            throw VisionError.encodingFailed(
                "patchified \(token) patches, expected \(patches)")
        }
        let byteCount = pixels.count * MemoryLayout<Float16>.stride
        guard let buffer = device.makeBuffer(bytes: pixels,
                                             length: byteCount,
                                             options: .storageModeShared) else {
            throw VisionError.encodingFailed("could not allocate the pixel buffer")
        }
        return VisionPreparedImage(pixels: buffer,
                                   gridT: 1,
                                   gridH: gridH,
                                   gridW: gridW,
                                   tokenCount: tokens,
                                   pixelWidth: targetW,
                                   pixelHeight: targetH,
                                   patchSize: patch,
                                   temporalPatchSize: temporal,
                                   mergeSize: merge)
    }

    /// Qwen2VL's `smart_resize`. Sides are multiples of `factor` and the total
    /// pixel count is kept inside `[minPixels, maxPixels]`.
    static func smartResize(width: Int, height: Int,
                            factor: Int,
                            minPixels: Int,
                            maxPixels: Int) -> (width: Int, height: Int) {
        guard width > 0, height > 0, factor > 0 else { return (factor, factor) }
        let h = Double(height)
        let w = Double(width)
        var hBar = max(Double(factor), (h / Double(factor)).rounded() * Double(factor))
        var wBar = max(Double(factor), (w / Double(factor)).rounded() * Double(factor))
        if hBar * wBar > Double(maxPixels) {
            let beta = ((h * w) / Double(maxPixels)).squareRoot()
            hBar = max(Double(factor),
                       ((h / beta) / Double(factor)).rounded(.down) * Double(factor))
            wBar = max(Double(factor),
                       ((w / beta) / Double(factor)).rounded(.down) * Double(factor))
        } else if hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = (((h * beta) / Double(factor)).rounded(.up)) * Double(factor)
            wBar = (((w * beta) / Double(factor)).rounded(.up)) * Double(factor)
        }
        let widthOut = max(factor, Int(wBar))
        let heightOut = max(factor, Int(hBar))
        return (widthOut, heightOut)
    }

    private struct SourceImage {
        let image: CGImage
        let width: Int
        let height: Int
    }

    private static func decode(_ data: Data) throws -> SourceImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw VisionError.unsupportedImage("the data is not a decodable image")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any] ?? [:]
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?
            .intValue ?? 1
        guard (1...8).contains(orientation) else {
            throw VisionError.unsupportedImage("unknown EXIF orientation \(orientation)")
        }
        // `CGImageSourceCreateThumbnailAtIndex` with the transform flag bakes
        // the orientation in; ask for the original pixel size first so the
        // resize below still has the true oriented dimensions.
        let pixelWidth = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?
            .intValue ?? 1
        let pixelHeight = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?
            .intValue ?? 1
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(pixelWidth, pixelHeight),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0, options as CFDictionary) else {
            throw VisionError.unsupportedImage("ImageIO could not decode the image")
        }
        return SourceImage(image: image, width: image.width, height: image.height)
    }

    private static func render(source: SourceImage,
                               width: Int, height: Int) throws -> Data {
        guard width > 0, height > 0,
              width <= 8_192, height <= 8_192 else {
            throw VisionError.unsupportedImage(
                "target \(width)×\(height) is outside the renderer's bounds")
        }
        let bytesPerRow = width * 4
        var data = Data(count: bytesPerRow * height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ok: Bool = data.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(source.image,
                         in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else {
            throw VisionError.encodingFailed("could not create the image context")
        }
        return data
    }
}
