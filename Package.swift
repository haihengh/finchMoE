// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "flash-qwen",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(name: "FlashQwen", targets: ["FlashQwen"]),
        .executable(name: "FlashQwenRepack", targets: ["FlashQwenRepack"]),
        .executable(name: "FlashQwenCLI", targets: ["FlashQwenCLI"]),
        .executable(name: "FlashQwenMac", targets: ["FlashQwenMac"]),
        .executable(name: "FlashQwenDecodeService", targets: ["FlashQwenDecodeService"]),
        .executable(name: "FlashQwenServer", targets: ["FlashQwenServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "FlashQwenFormat",
            path: "Sources/FlashQwenFormat"
        ),
        .target(
            name: "FlashQwen",
            dependencies: [
                "FlashQwenFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/FlashQwen",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "FlashQwenRepackCore",
            dependencies: ["FlashQwenFormat"],
            path: "Sources/FlashQwenRepack/Core"
        ),
        .executableTarget(
            name: "FlashQwenRepack",
            dependencies: ["FlashQwenRepackCore"],
            path: "Sources/FlashQwenRepack/Command"
        ),
        .target(
            name: "FlashQwenCLICore",
            dependencies: ["FlashQwen"],
            path: "Sources/FlashQwenCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "FlashQwenCLI",
            dependencies: ["FlashQwenCLICore"],
            path: "Sources/FlashQwenCLI/Command"
        ),
        .target(
            name: "FlashQwenAppCore",
            dependencies: ["FlashQwen", "FlashQwenRepackCore", "FlashQwenDecodeProtocol"],
            path: "Sources/FlashQwenApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "FlashQwenMacPresentation",
            dependencies: ["FlashQwenAppCore"],
            path: "Sources/FlashQwenApp/MacPresentation"
        ),
        .target(
            name: "FlashQwenDecodeProtocol",
            path: "Sources/FlashQwenDecodeProtocol"
        ),
        .executableTarget(
            name: "FlashQwenDecodeService",
            dependencies: ["FlashQwenAppCore", "FlashQwenDecodeProtocol"],
            path: "Sources/FlashQwenDecodeService"
        ),
        .target(
            name: "FlashQwenServerCore",
            dependencies: [
                "FlashQwen",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/FlashQwenServer/Core"
        ),
        .executableTarget(
            name: "FlashQwenServer",
            dependencies: ["FlashQwenServerCore"],
            path: "Sources/FlashQwenServer/Command"
        ),
        .executableTarget(
            name: "FlashQwenMac",
            dependencies: ["FlashQwenAppCore", "FlashQwenMacPresentation"],
            path: "Sources/FlashQwenApp/Mac",
            resources: [
                .copy("Resources/flashqwen-app-icon.png"),
            ]
        ),
        .target(
            name: "FlashQwenValidationSupport",
            dependencies: ["FlashQwen"],
            path: "Sources/FlashQwenValidation/Support"
        ),
        .testTarget(
            name: "FlashQwenFormatTests",
            dependencies: ["FlashQwenFormat"],
            path: "Tests/FlashQwenFormat"
        ),
        .testTarget(
            name: "FlashQwenFormatCompatibilityTests",
            dependencies: ["FlashQwenFormat", "FlashQwen", "FlashQwenRepackCore"],
            path: "Tests/FlashQwenFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "FlashQwenTestsCore",
            dependencies: ["FlashQwen", "FlashQwenValidationSupport", "FlashQwenRepackCore", "FlashQwenCLICore"],
            path: "Tests/FlashQwen/Core",
            // The fp32 model-replay probes (GDN recurrence, MoE tails) are
            // ~50x slower at -Onone; -O on the test target is a debug-build
            // convenience only (discovery still works with target-only -O).
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .testTarget(
            name: "FlashQwenRepackTests",
            dependencies: ["FlashQwenFormat", "FlashQwenRepackCore"],
            path: "Tests/FlashQwenRepack/Core"
        ),
        .testTarget(
            name: "FlashQwenAppCoreTests",
            dependencies: ["FlashQwenAppCore", "FlashQwen", "FlashQwenRepackCore", "FlashQwenDecodeProtocol"],
            path: "Tests/FlashQwenApp/Core"
        ),
        .testTarget(
            name: "FlashQwenDecodeServiceTests",
            dependencies: ["FlashQwenDecodeService", "FlashQwenAppCore", "FlashQwenDecodeProtocol"],
            path: "Tests/FlashQwenDecodeService"
        ),
        .testTarget(
            name: "FlashQwenMacPresentationTests",
            dependencies: ["FlashQwenAppCore", "FlashQwenMacPresentation"],
            path: "Tests/FlashQwenApp/MacPresentation"
        ),
        .testTarget(
            name: "FlashQwenServerTests",
            dependencies: [
                "FlashQwenServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/FlashQwenServer",
            resources: [.copy("Fixtures")]
        ),
    ]
)
