// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "finchMoE",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(name: "FinchMoE", targets: ["FinchMoE"]),
        .executable(name: "FinchMoERepack", targets: ["FinchMoERepack"]),
        .executable(name: "FinchMoECLI", targets: ["FinchMoECLI"]),
        .executable(name: "FinchMoEMac", targets: ["FinchMoEMac"]),
        .executable(name: "FinchMoEDecodeService", targets: ["FinchMoEDecodeService"]),
        .executable(name: "FinchMoEServer", targets: ["FinchMoEServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "FinchMoEFormat",
            path: "Sources/FinchMoEFormat"
        ),
        .target(
            name: "FinchMoE",
            dependencies: [
                "FinchMoEFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/FinchMoE",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "FinchMoERepackCore",
            dependencies: ["FinchMoEFormat"],
            path: "Sources/FinchMoERepack/Core"
        ),
        .executableTarget(
            name: "FinchMoERepack",
            dependencies: ["FinchMoERepackCore"],
            path: "Sources/FinchMoERepack/Command"
        ),
        .target(
            name: "FinchMoECLICore",
            dependencies: ["FinchMoE"],
            path: "Sources/FinchMoECLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "FinchMoECLI",
            dependencies: ["FinchMoECLICore"],
            path: "Sources/FinchMoECLI/Command"
        ),
        .target(
            name: "FinchMoEAppCore",
            dependencies: ["FinchMoE", "FinchMoERepackCore", "FinchMoEDecodeProtocol"],
            path: "Sources/FinchMoEApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "FinchMoEMacPresentation",
            dependencies: ["FinchMoEAppCore"],
            path: "Sources/FinchMoEApp/MacPresentation"
        ),
        .target(
            name: "FinchMoEDecodeProtocol",
            path: "Sources/FinchMoEDecodeProtocol"
        ),
        .executableTarget(
            name: "FinchMoEDecodeService",
            dependencies: ["FinchMoEAppCore", "FinchMoEDecodeProtocol"],
            path: "Sources/FinchMoEDecodeService"
        ),
        .target(
            name: "FinchMoEServerCore",
            dependencies: [
                "FinchMoE",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/FinchMoEServer/Core"
        ),
        .executableTarget(
            name: "FinchMoEServer",
            dependencies: ["FinchMoEServerCore"],
            path: "Sources/FinchMoEServer/Command"
        ),
        .executableTarget(
            name: "FinchMoEMac",
            dependencies: ["FinchMoEAppCore", "FinchMoEMacPresentation"],
            path: "Sources/FinchMoEApp/Mac",
            resources: [
                .copy("Resources/finchmoe-app-icon.png"),
            ]
        ),
        .target(
            name: "FinchMoEValidationSupport",
            dependencies: ["FinchMoE"],
            path: "Sources/FinchMoEValidation/Support"
        ),
        .testTarget(
            name: "FinchMoEFormatTests",
            dependencies: ["FinchMoEFormat"],
            path: "Tests/FinchMoEFormat"
        ),
        .testTarget(
            name: "FinchMoEFormatCompatibilityTests",
            dependencies: ["FinchMoEFormat", "FinchMoE", "FinchMoERepackCore"],
            path: "Tests/FinchMoEFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "FinchMoETestsCore",
            dependencies: ["FinchMoE", "FinchMoEValidationSupport", "FinchMoERepackCore", "FinchMoECLICore"],
            path: "Tests/FinchMoE/Core",
            // The fp32 model-replay probes (GDN recurrence, MoE tails) are
            // ~50x slower at -Onone; -O on the test target is a debug-build
            // convenience only (discovery still works with target-only -O).
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .testTarget(
            name: "FinchMoERepackTests",
            dependencies: ["FinchMoEFormat", "FinchMoERepackCore"],
            path: "Tests/FinchMoERepack/Core"
        ),
        .testTarget(
            name: "FinchMoEAppCoreTests",
            dependencies: ["FinchMoEAppCore", "FinchMoE", "FinchMoERepackCore", "FinchMoEDecodeProtocol"],
            path: "Tests/FinchMoEApp/Core"
        ),
        .testTarget(
            name: "FinchMoEDecodeServiceTests",
            dependencies: ["FinchMoEDecodeService", "FinchMoEAppCore", "FinchMoEDecodeProtocol"],
            path: "Tests/FinchMoEDecodeService"
        ),
        .testTarget(
            name: "FinchMoEMacPresentationTests",
            dependencies: ["FinchMoEAppCore", "FinchMoEMacPresentation"],
            path: "Tests/FinchMoEApp/MacPresentation"
        ),
        .testTarget(
            name: "FinchMoEServerTests",
            dependencies: [
                "FinchMoEServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/FinchMoEServer",
            resources: [.copy("Fixtures")]
        ),
    ]
)
