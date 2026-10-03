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

    /// The app-release monorepo keeps the Mac app under `Apps/FinchMoEMac`
    /// instead of `Sources/FinchMoEApp/Mac`, so a checkout is recognized by
    /// either marker and the `models/` install convention keeps working.
    @Test func monorepoLayoutIsAlsoRecognizedAsPackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Apps/FinchMoEMac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.finch")
    }

    @Test func qwenInstallInMonorepoPackageIsPreferredWhenPresent() {
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Apps/FinchMoEMac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/models/Qwen3.6-35B-A3B-4bit.finch")
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
    ///
    /// This is now specifically the *no-abliterated-install* case of that rule;
    /// see `abliteratedWinsWithinItsFamily` and
    /// `familyOrderingSurvivesTheAbliteratedPreference` for the cases the
    /// abliterated entries added.
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

    /// Within a family the abliterated install is tried first. It is the same
    /// architecture and the same measured quality band — EvalPlus 2026-09-27 on
    /// both families found no regression against the base weights — and a
    /// checkout that went to the trouble of repacking one means to run it.
    ///
    /// A checkout holding only the base still starts on the base, so this
    /// preference cannot change behaviour for an install that has no
    /// abliterated twin.
    @Test func abliteratedWinsWithinItsFamily() {
        let both: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: both).path
                    == "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch")

        let baseOnly: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: baseOnly).path
                    == "/repo/models/Qwen3.6-35B-A3B-4bit.finch")

        let qwen38Pair: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.8-Flash-Next-125B-ple4bit.finch/manifest.json",
            "/repo/models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: qwen38Pair).path
                    == "/repo/models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch")
    }

    /// Preferring abliterated must not disturb the 3.6-before-3.8 ordering
    /// above, which exists for memory reasons that have nothing to do with
    /// which variant of a family is installed. Every install except the 3.8
    /// base is present here, and the answer is still a 3.6 — the abliterated
    /// 125B must not win merely by being abliterated.
    @Test func familyOrderingSurvivesTheAbliteratedPreference() {
        let files: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch/manifest.json",
            "/repo/models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: files).path
                    == "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch")
    }

    private func resolve(inPackage files: Set<String>) -> URL {
        AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
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
