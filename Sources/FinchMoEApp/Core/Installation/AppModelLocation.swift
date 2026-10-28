import Foundation

/// Where an app bundle (as opposed to a package checkout) keeps its models.
enum AppModelStorage: Equatable {
    /// The historical app-owned location inside Application Support. Not
    /// visible in the iOS Files app.
    case applicationSupport
    /// The app's Documents directory, exposed to the Files app when
    /// `UIFileSharingEnabled` is set. Used on iOS so downloaded models can be
    /// found and selected by other tools.
    case documents

    static var `default`: AppModelStorage {
        #if canImport(UIKit)
        .documents
        #else
        .applicationSupport
        #endif
    }
}

enum AppModelLocation {
    static func packageRootURL() -> URL? {
        let fileManager = FileManager.default
        if let executableURL = Bundle.main.executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileManager.fileExists(atPath:)) {
            return root
        }
        let currentDirectoryURL = URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                      isDirectory: true)
        return packageRoot(startingAt: currentDirectoryURL,
                           fileExists: fileManager.fileExists(atPath:))
    }

    /// `<Application Support>` (not created), with the same home-directory
    /// fallback this code has always used.
    static func applicationSupportURL() -> URL {
        let fileManager = FileManager.default
        return (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.finchHomeDirectory
    }

    /// `<Documents>` (not created). On iOS this is the directory the Files app
    /// browses when `UIFileSharingEnabled` is on.
    static func documentsURL() -> URL {
        let fileManager = FileManager.default
        return (try? fileManager.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false))
            ?? fileManager.finchHomeDirectory.appendingPathComponent("Documents",
                                                                     isDirectory: true)
    }

    static func defaultURL() -> URL {
        let fileManager = FileManager.default
        return resolve(
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupportURL(),
            documentsURL: documentsURL(),
            fileExists: fileManager.fileExists(atPath:))
    }

    /// The app-owned models directory, used when no package checkout exists
    /// (an `.app` bundle — notably iOS). Nothing is created here; the
    /// installers create their destination's parents on demand.
    static func appModelsRootURL(
        storage: AppModelStorage = .default,
        documentsURL: URL = AppModelLocation.documentsURL(),
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        switch storage {
        case .documents:
            return documentsURL.appendingPathComponent("Models", isDirectory: true)
        case .applicationSupport:
            return applicationSupportURL
                .appendingPathComponent("FinchMoE", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
    }

    /// Where a row installs and starts from when there is no checkout.
    ///
    /// In Application Support, Gemma keeps its historical legacy path so
    /// downloads made before the app-owned rows existed are not moved. In
    /// Documents (iOS) every row lives under `Models/`, including Gemma, so the
    /// Files app exposes one clear folder.
    static func appInstallURL(
        for choice: AppModelChoice,
        storage: AppModelStorage = .default,
        documentsURL: URL = AppModelLocation.documentsURL(),
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        let base: URL
        switch storage {
        case .documents:
            base = appModelsRootURL(storage: storage,
                                    documentsURL: documentsURL,
                                    applicationSupportURL: applicationSupportURL)
        case .applicationSupport:
            base = choice == .gemma4
                ? applicationSupportURL.appendingPathComponent("FinchMoE",
                                                               isDirectory: true)
                : appModelsRootURL(storage: storage,
                                   documentsURL: documentsURL,
                                   applicationSupportURL: applicationSupportURL)
        }
        return base.appendingPathComponent(choice.installName, isDirectory: true)
            .standardizedFileURL
    }

    /// One resolver for both worlds: a checkout keeps the historical row paths,
    /// an app bundle gets the app-owned destination.
    static func installURL(
        for choice: AppModelChoice,
        packageRoot: URL?,
        storage: AppModelStorage = .default,
        documentsURL: URL = AppModelLocation.documentsURL(),
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        if let packageRoot {
            return choice.defaultURL(packageRoot: packageRoot)
        }
        return appInstallURL(for: choice,
                             storage: storage,
                             documentsURL: documentsURL,
                             applicationSupportURL: applicationSupportURL)
    }

    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        documentsURL: URL = AppModelLocation.documentsURL(),
                        storage: AppModelStorage = .default,
                        fileExists: (String) -> Bool) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return preferredInstallURL(inPackageRoot: root, fileExists: fileExists)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return preferredInstallURL(inPackageRoot: root, fileExists: fileExists)
        }
        return preferredAppInstallURL(storage: storage,
                                      documentsURL: documentsURL,
                                      applicationSupportURL: applicationSupportURL,
                                      fileExists: fileExists)
    }

    /// The app's default model inside a package checkout: a repack-made Qwen
    /// install under `models/` when present, else the Gemma target under
    /// `scratch/` (which is also the in-app download destination when nothing
    /// is installed yet). Outside a checkout the Application Support Gemma
    /// target above remains the fallback.
    ///
    /// Order matters, and the 3.6 *family* stays first: 3.8 is a 125B,
    /// 512-expert model that needs far more memory than 3.6 to run, so a
    /// checkout holding both must keep starting on a 3.6. 3.8 is the entry for
    /// a checkout that holds only it. That reasoning is about 3.6 against 3.8
    /// and is untouched by the abliterated entries below.
    ///
    /// Within a family, an abliterated install is tried before its base: the
    /// two are the same architecture and the same quality band (EvalPlus
    /// 2026-09-27 measured no regression), and a checkout that has deliberately
    /// repacked one means to run it. A checkout holding only the base keeps
    /// starting on the base, so no existing install changes behaviour.
    ///
    /// Newer widths sit directly behind the sibling that used to win, so no
    /// checkout that existed before them changes which model it starts on: the
    /// 3-bit 3.6 behind its 4-bit twin, and on 3.8 the 3-bit and 2-bit twins
    /// behind their own family's 4-bit install (abliterated widths behind the
    /// abliterated 4-bit). A checkout holding only a new width is found —
    /// rather than falling through to Gemma — and a narrower width never
    /// displaces a wider one that was already being started on.
    ///
    /// Within the 3.8 pair the quantized-PLE install comes first and the
    /// original is kept directly behind it as the fallback: they are the same
    /// weights with the n-gram table at int4 instead of BF16 (97 GiB against
    /// 162), and a checkout that still holds only the older tree must keep
    /// working rather than fall through to Gemma.
    private static func preferredInstallURL(inPackageRoot root: URL,
                                            fileExists: (String) -> Bool) -> URL {
        let models = root.appendingPathComponent("models", isDirectory: true)
        for name in preferredInstallNames {
            let candidate = models.appendingPathComponent(name, isDirectory: true)
            if fileExists(candidate.appendingPathComponent("manifest.json").path) {
                return candidate.standardizedFileURL
            }
        }
        return root.appendingPathComponent("scratch/gemma4.finch", isDirectory: true)
            .standardizedFileURL
    }

    /// The launch preference order — 3.6 family before 3.8, abliterated before
    /// base within a family, narrower widths behind their 4-bit sibling, the
    /// quantized-PLE installs before the raw-BF16 fallback. Shared by the
    /// checkout scan above and the app-owned scan below so the two cannot
    /// disagree about which installed model wins at launch.
    ///
    /// These are the twelve picker rows' install names, in that order. A vision
    /// row sits immediately before its text sibling: when both directories
    /// exist, the vision install can do everything the text one can, so it
    /// wins at launch; a text-only install resolves exactly as before.
    private static let preferredChoices: [AppModelChoice] = [
        .qwen3_6_abliterated, .qwen3_6_abliterated_3bit,
        .qwen3_6, .qwen3_6_3bit_vision, .qwen3_6_3bit,
        .qwen3_8_abliterated, .qwen3_8_abliterated_3bit,
        .qwen3_8_abliterated_2bit_vision, .qwen3_8_abliterated_2bit,
        .qwen3_8, .qwen3_8_3bit, .qwen3_8_2bit,
    ]

    /// Rowless raw-BF16-PLE 3.8 install (162 GiB): the last Qwen entry in the
    /// checkout scan, kept in the app scan for parity. It has no picker row.
    private static let rawQwen38InstallName = "Qwen3.8-Flash-Next-125B.finch"

    private static let preferredInstallNames: [String] =
        preferredChoices.map(\.installName) + [rawQwen38InstallName]

    /// The row a fresh app-owned app starts on. Gemma is no longer the default:
    /// Qwen 3.6 3-bit is the smallest Qwen build the engine ships (about 16 GB
    /// installed) and the one a first install should offer. An existing Gemma
    /// install keeps winning the scan below, so this changes only the empty case.
    static let defaultChoice: AppModelChoice = .qwen3_6_3bit

    /// The app-owned equivalent of `preferredInstallURL`: outside a checkout,
    /// launch preference is re-derived by scanning the app's Models directory
    /// in the same order. With nothing installed, a legacy Gemma install is
    /// still honoured; otherwise the destination is the default row, so a first
    /// launch offers Qwen 3.6 3-bit rather than Gemma.
    private static func preferredAppInstallURL(storage: AppModelStorage,
                                               documentsURL: URL,
                                               applicationSupportURL: URL,
                                               fileExists: (String) -> Bool) -> URL {
        let modelsRoot = appModelsRootURL(storage: storage,
                                          documentsURL: documentsURL,
                                          applicationSupportURL: applicationSupportURL)
        for name in preferredInstallNames {
            let candidate = modelsRoot.appendingPathComponent(name, isDirectory: true)
            if fileExists(candidate.appendingPathComponent("manifest.json").path) {
                return candidate.standardizedFileURL
            }
        }
        let gemma = appInstallURL(for: .gemma4,
                                  storage: storage,
                                  documentsURL: documentsURL,
                                  applicationSupportURL: applicationSupportURL)
        if fileExists(gemma.appendingPathComponent("manifest.json").path) {
            return gemma
        }
        return appInstallURL(for: defaultChoice,
                             storage: storage,
                             documentsURL: documentsURL,
                             applicationSupportURL: applicationSupportURL)
    }

    /// The descriptor a launch destination should start from when it holds no
    /// manifest yet. Only the app-owned default destination qualifies; a
    /// checkout keeps its Gemma fallback, and any other directory is left to
    /// the probe's own default.
    static func defaultDescriptor(
        for directory: URL,
        packageRoot: URL? = AppModelLocation.packageRootURL(),
        storage: AppModelStorage = .default,
        documentsURL: URL = AppModelLocation.documentsURL(),
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> AppModelInstallDescriptor? {
        guard packageRoot == nil else { return nil }
        let expected = appInstallURL(for: defaultChoice,
                                     storage: storage,
                                     documentsURL: documentsURL,
                                     applicationSupportURL: applicationSupportURL)
        // Compare paths, not URLs: an appending `isDirectory: true` URL and the
        // caller's URL differ by their trailing slash.
        guard expected.standardizedFileURL.path
            == directory.standardizedFileURL.path else {
            return nil
        }
        return defaultChoice.descriptor
    }

    /// Prepares the Files-visible model directory and moves app-owned installs
    /// that predate it out of Application Support.
    ///
    /// The move is a rename inside the app's container, so it is immediate even
    /// for a 100 GB install. It never overwrites a destination that already
    /// exists; a failed move leaves the old copy in place rather than losing
    /// bytes. Idempotent, so the app can call it on every launch.
    static func prepareAppModelStorage(
        storage: AppModelStorage = .default,
        documentsURL: URL = AppModelLocation.documentsURL(),
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL(),
        fileManager: FileManager = .default
    ) {
        guard storage == .documents else { return }
        let models = appModelsRootURL(storage: storage,
                                      documentsURL: documentsURL,
                                      applicationSupportURL: applicationSupportURL)
        try? fileManager.createDirectory(at: models, withIntermediateDirectories: true)
        excludeFromBackup(models)

        // Qwen installs: the whole Models directory, including any .partial
        // download and .resume.json checkpoints beside the model directories.
        let legacyRoot = applicationSupportURL
            .appendingPathComponent("FinchMoE", isDirectory: true)
        let legacyModels = legacyRoot.appendingPathComponent("Models", isDirectory: true)
        if fileManager.fileExists(atPath: legacyModels.path) {
            let children = (try? fileManager.contentsOfDirectory(
                at: legacyModels, includingPropertiesForKeys: nil)) ?? []
            for child in children {
                let destination = models.appendingPathComponent(child.lastPathComponent)
                if moveIfAbsent(from: child, to: destination, fileManager: fileManager) {
                    // The receipt records the physical directory it was made
                    // for, and this move changes that directory. Rewrite it
                    // in place: moving within the app's own container cannot
                    // change the bytes, and leaving the old path would make
                    // the probe report a perfectly good install as partial.
                    rewriteReceiptPath(in: destination)
                }
            }
            if let remaining = try? fileManager.contentsOfDirectory(
                atPath: legacyModels.path), remaining.isEmpty {
                try? fileManager.removeItem(at: legacyModels)
            }
        }

        // The legacy app-owned Gemma target lived one level above Models, with
        // its checkpoint, partial payload and settings beside it.
        let gemmaName = AppModelChoice.gemma4.installName
        let gemmaDestination = models.appendingPathComponent(gemmaName, isDirectory: true)
        if moveIfAbsent(from: legacyRoot.appendingPathComponent(gemmaName, isDirectory: true),
                        to: gemmaDestination,
                        fileManager: fileManager) {
            rewriteReceiptPath(in: gemmaDestination)
        }
        for suffix in [".partial", ".resume.json", ".install.lock"] {
            moveIfAbsent(
                from: legacyRoot.appendingPathComponent(gemmaName + suffix),
                to: models.appendingPathComponent(gemmaName + suffix),
                fileManager: fileManager)
        }
        moveIfAbsent(from: legacyRoot.appendingPathComponent(MacAppSettings.fileName),
                     to: models.appendingPathComponent(MacAppSettings.fileName),
                     fileManager: fileManager)
    }

    @discardableResult
    private static func moveIfAbsent(from source: URL,
                                     to destination: URL,
                                     fileManager: FileManager) -> Bool {
        guard fileManager.fileExists(atPath: source.path),
              !fileManager.fileExists(atPath: destination.path) else { return false }
        guard (try? fileManager.moveItem(at: source, to: destination)) != nil else {
            return false
        }
        return true
    }

    /// Points an existing receipt at the directory it now lives in. Only the
    /// physical path changes: the manifest hash and every file digest the
    /// receipt carries are the ones computed before the move. A malformed
    /// receipt is left alone so the probe's size-check fallback can decide.
    private static func rewriteReceiptPath(in directory: URL) {
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        guard let data = try? Data(contentsOf: receiptURL),
              var object = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any] else { return }
        let path = directory.standardizedFileURL.path
        guard object["modelDirectoryPath"] as? String != path else { return }
        object["modelDirectoryPath"] = path
        guard let rewritten = try? JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys]) else { return }
        try? rewritten.write(to: receiptURL, options: [.atomic])
    }

    private static func excludeFromBackup(_ url: URL) {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            // The app lives at `Apps/FinchMoEMac` in this monorepo; the older
            // engine-repo layout kept it under `Sources/FinchMoEApp/Mac`.
            let appMonorepo = candidate.appendingPathComponent(
                "Apps/FinchMoEMac", isDirectory: true).path
            let appEngineRepo = candidate.appendingPathComponent(
                "Sources/FinchMoEApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appMonorepo) || fileExists(appEngineRepo) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
