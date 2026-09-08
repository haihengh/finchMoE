// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FinchmoeChat",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "FinchmoeChat", targets: ["FinchmoeChat"]),
    ],
    targets: [
        .executableTarget(
            name: "FinchmoeChat",
            path: "Sources/FinchmoeChat"),
    ]
)
