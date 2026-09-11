import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.finch"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.finch")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/FinchMoEApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finch")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/FinchMoEApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finch")
    }

    @Test func qwenInstallInPackageIsPreferredWhenPresent() {
        let qwenManifest = "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json"
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            qwenManifest,
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/models/Qwen3.6-35B-A3B-4bit.finch")
    }

    /// A checkout holding only the 125B install still resolves to a Qwen
    /// rather than falling through to the Gemma download target.
    @Test func qwen38InstallInPackageIsPreferredWhenItIsTheOnlyOne() {
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.8-Flash-Next-125B.finch/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/models/Qwen3.8-Flash-Next-125B.finch")
    }

    /// With both installed, 3.6 stays the default: 3.8 is a 174 GB model that
    /// needs far more memory to run, so it must not become the app's implicit
    /// choice for a checkout that happens to hold it.
    @Test func qwen36WinsWhenBothInstallsArePresent() {
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/repo/models/Qwen3.8-Flash-Next-125B.finch/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/models/Qwen3.6-35B-A3B-4bit.finch")
    }

    @Test func absentQwenManifestKeepsGemmaTargetEvenWhenModelsDirExists() {
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models",  // directory exists, but no Qwen install inside
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finch")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/FinchMoE/gemma4.finch")
    }
}
