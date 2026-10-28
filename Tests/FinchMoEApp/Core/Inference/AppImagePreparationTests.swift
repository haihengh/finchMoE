import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import FinchMoEAppCore

@Suite struct AppImagePreparationTests {

    private func makePNG(width: Int, height: Int) throws -> Data {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        for index in 0..<(width * height) {
            pixels[index * 4] = 220
            pixels[index * 4 + 1] = 30
            pixels[index * 4 + 2] = 30
            pixels[index * 4 + 3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let maybeImage: CGImage? = pixels.withUnsafeMutableBytes { raw -> CGImage? in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue) else { return nil }
            return context.makeImage()
        }
        let image = try #require(maybeImage)
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(
            data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    @Test func preparesAJPEGWithAMIMETypeAndBoundedDimensions() throws {
        let prepared = try AppImagePreparation.prepare(
            data: try makePNG(width: 64, height: 32))
        #expect(prepared.mediaType == "image/jpeg")
        #expect(prepared.pixelWidth == 64)
        #expect(prepared.pixelHeight == 32)
        #expect(!prepared.data.isEmpty)
        // The prepared bytes are a decodable JPEG.
        let source = try #require(CGImageSourceCreateWithData(
            prepared.data as CFData, nil))
        #expect(CGImageSourceGetType(source) as String? == "public.jpeg")
    }

    @Test func rejectsDataThatIsNotAnImage() {
        #expect(throws: AppImagePreparation.PreparationError.self) {
            _ = try AppImagePreparation.prepare(data: Data([0x00, 0x01]))
        }
    }
}
