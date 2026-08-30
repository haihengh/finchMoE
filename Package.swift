// swift-tools-version: 6.0
import PackageDescription

// The codebase is written in Swift 5 language mode; tools 6.0 is only needed
// for the `.macOS(.v15)` platform declaration, so pin the language mode back
// down to v5 to avoid strict-concurrency errors.
let swiftV5: [SwiftSetting] = [.swiftLanguageMode(.v5)]

let package = Package(
    name: "QwenFieldfare",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "QwenFieldfareFormat", targets: ["QwenFieldfareFormat"]),
        .library(name: "QwenFieldfareRepack", targets: ["QwenFieldfareRepack"]),
        .library(name: "QwenFieldfareRuntime", targets: ["QwenFieldfareRuntime"]),
        .library(name: "QwenFieldfareServer", targets: ["QwenFieldfareServer"]),
        .executable(name: "qwen-fieldfare", targets: ["QwenFieldfareCLI"]),
        .executable(name: "qwen-fieldfare-server", targets: ["QwenFieldfareServerMain"]),
    ],
    targets: [
        // MARK: - Format
        .target(
            name: "QwenFieldfareFormat",
            path: "Sources/QwenFieldfareFormat",
            swiftSettings: swiftV5
        ),

        // MARK: - Repack (download + repack tooling)
        .target(
            name: "QwenFieldfareRepack",
            dependencies: ["QwenFieldfareFormat"],
            path: "Sources/QwenFieldfareRepack",
            swiftSettings: swiftV5
        ),

        // MARK: - Runtime (inference engine + Metal resources)
        .target(
            name: "QwenFieldfareRuntime",
            dependencies: ["QwenFieldfareFormat"],
            path: "Sources/QwenFieldfareRuntime",
            resources: [
                .process("Metal/Kernels.metal")
            ],
            swiftSettings: swiftV5
        ),

        // MARK: - Server (OpenAI-compatible) — library so the CLI can host it too
        .target(
            name: "QwenFieldfareServer",
            dependencies: ["QwenFieldfareRuntime"],
            path: "Sources/QwenFieldfareServer",
            swiftSettings: swiftV5
        ),

        // MARK: - CLI (repack + run + serve)
        .executableTarget(
            name: "QwenFieldfareCLI",
            dependencies: ["QwenFieldfareRuntime", "QwenFieldfareRepack", "QwenFieldfareServer"],
            path: "Sources/QwenFieldfareCLI",
            swiftSettings: swiftV5
        ),

        // MARK: - Standalone server executable
        .executableTarget(
            name: "QwenFieldfareServerMain",
            dependencies: ["QwenFieldfareServer", "QwenFieldfareRuntime"],
            path: "Sources/QwenFieldfareServerMain",
            swiftSettings: swiftV5
        ),

        // MARK: - Tests
        .testTarget(
            name: "QwenFieldfareTests",
            dependencies: ["QwenFieldfareFormat", "QwenFieldfareRepack"],
            path: "Tests/QwenFieldfareTests",
            swiftSettings: swiftV5
        ),
    ]
)
