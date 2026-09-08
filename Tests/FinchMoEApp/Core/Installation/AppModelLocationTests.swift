import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.finchturbo"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.finchturbo")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/FinchMoEApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finchturbo")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/FinchMoEApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finchturbo")
    }

    @Test func qwenInstallInPackageIsPreferredWhenPresent() {
        let qwenManifest = "/repo/models/Qwen3.6-35B-A3B-4bit.finchturbo/manifest.json"
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
        #expect(result.path == "/repo/models/Qwen3.6-35B-A3B-4bit.finchturbo")
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
        #expect(result.path == "/repo/scratch/gemma4.finchturbo")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/FinchMoE/gemma4.finchturbo")
    }
}
