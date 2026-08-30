import Foundation

extension Foundation.Bundle {
    static nonisolated let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("QwenFieldfare_QwenFieldfareRuntime.bundle").path
        let buildPath = "/Volumes/samsung 2t/code/flash-qwen/.build/arm64-apple-macosx/release/QwenFieldfare_QwenFieldfareRuntime.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}