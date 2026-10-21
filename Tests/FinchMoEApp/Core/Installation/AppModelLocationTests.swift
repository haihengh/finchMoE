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

    /// A checkout holding only the 3-bit build — what the app's own download
    /// produces — resolves to it rather than falling through to Gemma. With
    /// both 3.6 installs present the 4-bit one keeps winning, so an existing
    /// checkout does not change which install it starts on.
    @Test func qwen36ThreeBitInstallIsFoundWhenItIsTheOnlyOneAndYieldsToTheFourBit() {
        let threeBitOnly: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-3bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: threeBitOnly).path
                    == "/repo/models/Qwen3.6-35B-A3B-3bit.finch")

        let both: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/repo/models/Qwen3.6-35B-A3B-3bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: both).path
                    == "/repo/models/Qwen3.6-35B-A3B-4bit.finch")
    }

    @Test func qwen36AbliteratedThreeBitIsFoundBehindItsFourBitTwin() {
        let threeBitOnly: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-abliterated-3bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: threeBitOnly).path
                    == "/repo/models/Qwen3.6-35B-A3B-abliterated-3bit.finch")

        let both: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch/manifest.json",
            "/repo/models/Qwen3.6-35B-A3B-abliterated-3bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: both).path
                    == "/repo/models/Qwen3.6-35B-A3B-abliterated-4bit.finch")
    }

    /// The 3.8 width rows. Each new width sits behind the sibling that used to
    /// win, so a checkout holding more than one keeps starting on what it
    /// started on before, and a checkout holding only a new width finds it
    /// rather than falling through to Gemma.
    @Test func qwen38WidthInstallsAreFoundBehindTheirFourBitSiblings() {
        let twoBitOnly: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.8-Flash-Next-125B-ple2bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: twoBitOnly).path
                    == "/repo/models/Qwen3.8-Flash-Next-125B-ple2bit.finch")

        let fourAndTwo: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.8-Flash-Next-125B-ple4bit.finch/manifest.json",
            "/repo/models/Qwen3.8-Flash-Next-125B-ple2bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: fourAndTwo).path
                    == "/repo/models/Qwen3.8-Flash-Next-125B-ple4bit.finch")

        // An abliterated width still beats a base install (the family rule
        // the abliterated entries established), and the raw-BF16-PLE
        // fallback stays last.
        let abliteratedThreeBitAndBaseFourBit: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/FinchMoEApp/Mac",
            "/repo/models/Qwen3.8-Flash-Next-abliterated-ple3bit.finch/manifest.json",
            "/repo/models/Qwen3.8-Flash-Next-125B-ple4bit.finch/manifest.json",
        ]
        #expect(resolve(inPackage: abliteratedThreeBitAndBaseFourBit).path
                    == "/repo/models/Qwen3.8-Flash-Next-abliterated-ple3bit.finch")
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

    @Test func standaloneAppStartsOnTheDefaultQwenRow() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path
                    == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-3bit.finch")
    }

    // MARK: - App-owned destinations (no package checkout)

    private func resolveStandalone(_ files: Set<String>) -> URL {
        AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
    }

    @Test func appInstallURLKeepsGemmaAtLegacySupportPath() {
        let url = AppModelLocation.appInstallURL(
            for: .gemma4,
            applicationSupportURL: URL(fileURLWithPath: "/support"))
        #expect(url.path == "/support/FinchMoE/gemma4.finch")
    }

    @Test func appInstallURLPlacesEveryQwenRowUnderModels() {
        for choice in AppModelChoice.allCases where choice != .gemma4 {
            let url = AppModelLocation.appInstallURL(
                for: choice,
                applicationSupportURL: URL(fileURLWithPath: "/support"))
            #expect(url.path == "/support/FinchMoE/Models/\(choice.installName)",
                    "\(choice) installs outside the app models directory")
        }
    }

    @Test func installURLMatchesCheckoutDefaultURLForEveryRow() {
        let root = URL(fileURLWithPath: "/repo")
        for choice in AppModelChoice.allCases {
            #expect(AppModelLocation.installURL(
                        for: choice,
                        packageRoot: root,
                        applicationSupportURL: URL(fileURLWithPath: "/support"))
                    == choice.defaultURL(packageRoot: root),
                    "\(choice) resolves differently in a checkout")
        }
    }

    @Test func installURLOutsideACheckoutUsesTheAppRoot() {
        #expect(AppModelLocation.installURL(
                    for: .qwen3_6_3bit,
                    packageRoot: nil,
                    applicationSupportURL: URL(fileURLWithPath: "/support")).path
                == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-3bit.finch")
    }

    /// The app-side equivalent of the checkout scan must find every row's
    /// install by its own name, so selecting a row and relaunching agree.
    @Test func standaloneAppScanFindsEveryQwenInstall() {
        for choice in AppModelChoice.allCases where choice != .gemma4 {
            let files: Set<String> = [
                "/support/FinchMoE/Models/\(choice.installName)/manifest.json",
            ]
            #expect(resolveStandalone(files).path
                        == "/support/FinchMoE/Models/\(choice.installName)")
        }
    }

    /// The raw-BF16-PLE 3.8 install is the last Qwen entry in the checkout
    /// scan and has no picker row, so it must be listed explicitly — not
    /// derived from `AppModelChoice`.
    @Test func standaloneAppScanFindsTheRawBf16Qwen38Install() {
        let files: Set<String> = [
            "/support/FinchMoE/Models/Qwen3.8-Flash-Next-125B.finch/manifest.json",
        ]
        #expect(resolveStandalone(files).path
                    == "/support/FinchMoE/Models/Qwen3.8-Flash-Next-125B.finch")
    }

    @Test func standaloneAppScanUsesCheckoutPreferenceOrder() {
        // Family: 3.6 before every 3.8.
        let family: Set<String> = [
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/support/FinchMoE/Models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch/manifest.json",
        ]
        #expect(resolveStandalone(family).path
                    == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-4bit.finch")

        // Abliterated before base within a family.
        let abliterated: Set<String> = [
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-abliterated-4bit.finch/manifest.json",
        ]
        #expect(resolveStandalone(abliterated).path
                    == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-abliterated-4bit.finch")

        // A narrower width never displaces the 4-bit sibling.
        let width: Set<String> = [
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-4bit.finch/manifest.json",
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-3bit.finch/manifest.json",
        ]
        #expect(resolveStandalone(width).path
                    == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-4bit.finch")
    }

    @Test func standaloneAppScanPrefersQwenOverGemma() {
        let files: Set<String> = [
            "/support/FinchMoE/gemma4.finch/manifest.json",
            "/support/FinchMoE/Models/Qwen3.8-Flash-Next-125B-ple2bit.finch/manifest.json",
        ]
        #expect(resolveStandalone(files).path
                    == "/support/FinchMoE/Models/Qwen3.8-Flash-Next-125B-ple2bit.finch")
    }

    @Test func standaloneAppScanIgnoresModelsDirectoryWithoutManifest() {
        let files: Set<String> = ["/support/FinchMoE/Models"]
        #expect(resolveStandalone(files).path
                    == "/support/FinchMoE/Models/Qwen3.6-35B-A3B-3bit.finch")
    }

    /// Qwen 3.6 3-bit is the empty-case default, but an install made before
    /// that change must keep launching; the legacy Gemma directory is checked
    /// after the Qwen scan and before the new destination is assumed.
    @Test func standaloneAppStillFindsALegacyGemmaInstall() {
        let files: Set<String> = ["/support/FinchMoE/gemma4.finch/manifest.json"]
        #expect(resolveStandalone(files).path == "/support/FinchMoE/gemma4.finch")
    }

    @Test func defaultDescriptorAppliesOnlyToTheAppOwnedDefaultDestination() {
        let support = URL(fileURLWithPath: "/support")
        let appDefault = URL(fileURLWithPath:
            "/support/FinchMoE/Models/Qwen3.6-35B-A3B-3bit.finch")

        #expect(AppModelLocation.defaultDescriptor(
                    for: appDefault,
                    packageRoot: nil,
                    applicationSupportURL: support) == .qwen3_6_3bit)

        // A checkout keeps its own Gemma fallback.
        #expect(AppModelLocation.defaultDescriptor(
                    for: appDefault,
                    packageRoot: URL(fileURLWithPath: "/repo"),
                    applicationSupportURL: support) == nil)

        // Any other app-owned directory is not the default row.
        #expect(AppModelLocation.defaultDescriptor(
                    for: URL(fileURLWithPath: "/support/FinchMoE/gemma4.finch"),
                    packageRoot: nil,
                    applicationSupportURL: support) == nil)
    }

    // MARK: - Documents storage (iOS / Files app)

    private func resolveDocuments(_ files: Set<String>) -> URL {
        AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/FinchMoEMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            documentsURL: URL(fileURLWithPath: "/documents"),
            storage: .documents,
            fileExists: files.contains)
    }

    @Test func documentsStoragePutsEveryRowUnderModels() {
        for choice in AppModelChoice.allCases {
            let url = AppModelLocation.appInstallURL(
                for: choice,
                storage: .documents,
                documentsURL: URL(fileURLWithPath: "/documents"),
                applicationSupportURL: URL(fileURLWithPath: "/support"))
            #expect(url.path == "/documents/Models/\(choice.installName)",
                    "\(choice) is not under the Files-visible Models directory")
        }
    }

    /// A vision install is a superset of its text sibling, so it wins when both
    /// directories exist; a text-only install still resolves to its own row.
    @Test func visionInstallWinsOverItsTextSibling() {
        let both: Set<String> = [
            "/documents/Models/Qwen3.6-35B-A3B-3bit.finch/manifest.json",
            "/documents/Models/Qwen3.6-35B-A3B-3bit-vision.finch/manifest.json",
        ]
        #expect(resolveDocuments(both).path
            == "/documents/Models/Qwen3.6-35B-A3B-3bit-vision.finch")

        let textOnly: Set<String> = [
            "/documents/Models/Qwen3.6-35B-A3B-3bit.finch/manifest.json",
        ]
        #expect(resolveDocuments(textOnly).path
            == "/documents/Models/Qwen3.6-35B-A3B-3bit.finch")
    }

    @Test func documentsStorageFindsInstallsAndFallsBackToTheDefaultRow() {
        let installed: Set<String> = [
            "/documents/Models/Qwen3.6-35B-A3B-abliterated-3bit.finch/manifest.json",
        ]
        #expect(resolveDocuments(installed).path
                    == "/documents/Models/Qwen3.6-35B-A3B-abliterated-3bit.finch")

        let gemmaOnly: Set<String> = [
            "/documents/Models/gemma4.finch/manifest.json",
        ]
        #expect(resolveDocuments(gemmaOnly).path
                    == "/documents/Models/gemma4.finch")

        #expect(resolveDocuments([]).path
                    == "/documents/Models/Qwen3.6-35B-A3B-3bit.finch")
    }

    /// The first iOS launch after this change moves installs out of
    /// Application Support (including partial downloads and their checkpoints)
    /// without overwriting anything, and a second launch is a no-op.
    @Test func migrationMovesLegacyModelsIntoDocumentsWithoutOverwriting() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-migration-\(UUID().uuidString)",
                                    isDirectory: true)
        let support = root.appendingPathComponent("support", isDirectory: true)
        let documents = root.appendingPathComponent("documents", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let legacyRoot = support.appendingPathComponent("FinchMoE", isDirectory: true)
        let legacyModels = legacyRoot.appendingPathComponent("Models", isDirectory: true)
        let legacyQwen = legacyModels.appendingPathComponent(
            "Qwen3.6-35B-A3B-3bit.finch", isDirectory: true)
        let legacyPartial = legacyModels.appendingPathComponent(
            "Qwen3.8-Flash-Next-125B-ple2bit.finch.partial", isDirectory: true)
        let legacyGemma = legacyRoot.appendingPathComponent("gemma4.finch",
                                                            isDirectory: true)
        for directory in [legacyQwen, legacyPartial, legacyGemma] {
            try FileManager.default.createDirectory(at: directory,
                                                    withIntermediateDirectories: true)
        }
        try Data("{}".utf8).write(
            to: legacyQwen.appendingPathComponent("manifest.json"))
        // A migrated install's receipt still names the Application Support
        // path; the move must rewrite it or the probe reports the model as
        // partial and the app never offers Load.
        try JSONSerialization.data(withJSONObject: [
            "schemaVersion": 1,
            "modelDirectoryPath": legacyQwen.standardizedFileURL.path,
        ], options: [.sortedKeys]).write(
            to: legacyQwen.appendingPathComponent("verified-install.json"))
        try Data("{}".utf8).write(
            to: legacyModels.appendingPathComponent(
                "Qwen3.6-35B-A3B-3bit.finch.resume.json"))
        // A legacy Gemma install's sidecars and settings live one level above
        // its model directory.
        let legacyGemmaPartial = legacyRoot.appendingPathComponent(
            "gemma4.finch.partial", isDirectory: true)
        try FileManager.default.createDirectory(at: legacyGemmaPartial,
                                                withIntermediateDirectories: true)
        try Data("{}".utf8).write(
            to: legacyRoot.appendingPathComponent("gemma4.finch.resume.json"))
        try Data("{}".utf8).write(
            to: legacyRoot.appendingPathComponent("mac-app-settings.json"))

        AppModelLocation.prepareAppModelStorage(
            storage: .documents,
            documentsURL: documents,
            applicationSupportURL: support)

        let models = documents.appendingPathComponent("Models", isDirectory: true)
        let fm = FileManager.default
        let migratedQwen = models.appendingPathComponent(
            "Qwen3.6-35B-A3B-3bit.finch", isDirectory: true)
        #expect(fm.fileExists(atPath: migratedQwen.appendingPathComponent(
            "manifest.json").path))
        let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf:
            migratedQwen.appendingPathComponent("verified-install.json")))
            as? [String: Any]
        #expect(receipt?["modelDirectoryPath"] as? String
            == migratedQwen.standardizedFileURL.path)
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "Qwen3.6-35B-A3B-3bit.finch.resume.json").path))
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "Qwen3.8-Flash-Next-125B-ple2bit.finch.partial").path))
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "gemma4.finch").path))
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "gemma4.finch.partial").path))
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "gemma4.finch.resume.json").path))
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "mac-app-settings.json").path))
        #expect(!fm.fileExists(atPath: legacyModels.path))
        #expect(!fm.fileExists(atPath: legacyGemma.path))
        #expect(!fm.fileExists(atPath: legacyGemmaPartial.path))

        AppModelLocation.prepareAppModelStorage(
            storage: .documents,
            documentsURL: documents,
            applicationSupportURL: support)
        #expect(fm.fileExists(atPath: models.appendingPathComponent(
            "Qwen3.6-35B-A3B-3bit.finch/manifest.json").path))
    }
}
