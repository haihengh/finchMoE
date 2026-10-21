import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
@testable import FinchMoE

@Suite struct VisionPreprocessTests {

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finch-vision-pre-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A 64×64 PNG split into a red left half and a blue right half, at exact
    /// patch boundaries so the color of a merge block is unambiguous.
    private func makeTestImage() throws -> Data {
        let width = 64
        let height = 64
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * bytesPerRow + x * 4
                if x < width / 2 {
                    pixels[i] = 255; pixels[i + 1] = 0; pixels[i + 2] = 0
                } else {
                    pixels[i] = 0; pixels[i + 1] = 0; pixels[i + 2] = 255
                }
                pixels[i + 3] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let maybeImage: CGImage? = pixels.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue) else {
                return nil
            }
            return context.makeImage()
        }
        let image = try #require(maybeImage)
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(
            data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw VisionError.encodingFailed("could not encode the test image")
        }
        return data as Data
    }

    @Test func smartResizeKeepsTheContract() {
        // Exactly 1536² is the selected budget: 48×48 patches, 2304 tokens.
        let exact = VisionPreprocessor.smartResize(
            width: 1_536, height: 1_536, factor: 32,
            minPixels: 65_536, maxPixels: 2_359_296)
        #expect(exact.width == 1_536 && exact.height == 1_536)

        // A small image is upscaled to at least the shortest-edge bound.
        let small = VisionPreprocessor.smartResize(
            width: 100, height: 100, factor: 32,
            minPixels: 65_536, maxPixels: 2_359_296)
        #expect(small.width == 256 && small.height == 256)

        // A large image is downscaled under the cap, with both sides on the
        // patch/merge grid.
        let large = VisionPreprocessor.smartResize(
            width: 4_000, height: 2_000, factor: 32,
            minPixels: 65_536, maxPixels: 2_359_296)
        #expect(large.width % 32 == 0 && large.height % 32 == 0)
        #expect(large.width * large.height <= 2_359_296)
        let ratio = Double(large.width) / Double(large.height)
        #expect(abs(ratio - 2.0) < 0.1)
    }

    @Test func preparePatchifiesInMergeBlockOrder() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try VisionPayloadTests.writeVisionConfig(into: dir)
        let config = try VisionConfig.load(
            directory: FinchModelDirectory(rootURL: dir))
        let prepared = try VisionPreprocessor.prepare(
            imageData: try makeTestImage(), config: config, device: device,
            maxPixels: 1_536 * 1_536)

        #expect(prepared.gridT == 1)
        #expect(prepared.gridH == 16)
        #expect(prepared.gridW == 16)
        #expect(prepared.tokenCount == 64)
        #expect(prepared.pixelWidth == 256)
        #expect(prepared.pixelHeight == 256)

        let elementsPerPatch = 3 * 2 * 16 * 16
        let elements = prepared.patchCount * elementsPerPatch
        let values = prepared.pixels.contents()
            .bindMemory(to: Float16.self, capacity: elements)

        // Token 0 is the top-left merge block: red. The vector order is
        // [C, T, P, P], so the channels are 2·16·16 = 512 elements apart.
        let channelStride = 2 * 16 * 16
        #expect(values[0] == 1.0)                       // red R
        #expect(values[channelStride] == -1.0)          // green
        #expect(values[2 * channelStride] == -1.0)      // blue

        // Eight merge blocks per row; block 7 starts at x=224, well inside the
        // blue half (block 4 starts at the interpolation seam). Four patch
        // tokens live in each merge block, so token 28 is block 7's first
        // patch.
        let blueToken = 28
        let blueBase = blueToken * elementsPerPatch
        #expect(values[blueBase] == -1.0)                    // red
        #expect(values[blueBase + 2 * channelStride] == 1.0)  // blue
    }
}
