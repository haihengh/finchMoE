import Foundation

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

    static func defaultURL() -> URL {
        let fileManager = FileManager.default
        return resolve(
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupportURL(),
            fileExists: fileManager.fileExists(atPath:))
    }

    /// The app-owned models directory, used when no package checkout exists
    /// (an `.app` bundle — notably iOS). Nothing is created here; the
    /// installers create their destination's parents on demand.
    static func appModelsRootURL(
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        applicationSupportURL
            .appendingPathComponent("FinchMoE", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Where a row installs and starts from when there is no checkout. Gemma
    /// keeps its existing legacy path so downloads made before the app-owned
    /// rows existed are not moved; every Qwen row lives under `Models/`.
    static func appInstallURL(
        for choice: AppModelChoice,
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        let base = choice == .gemma4
            ? applicationSupportURL.appendingPathComponent("FinchMoE", isDirectory: true)
            : appModelsRootURL(applicationSupportURL: applicationSupportURL)
        return base.appendingPathComponent(choice.installName, isDirectory: true)
            .standardizedFileURL
    }

    /// One resolver for both worlds: a checkout keeps the historical row paths,
    /// an app bundle gets the app-owned destination.
    static func installURL(
        for choice: AppModelChoice,
        packageRoot: URL?,
        applicationSupportURL: URL = AppModelLocation.applicationSupportURL()
    ) -> URL {
        if let packageRoot {
            return choice.defaultURL(packageRoot: packageRoot)
        }
        return appInstallURL(for: choice, applicationSupportURL: applicationSupportURL)
    }

    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
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
        return preferredAppInstallURL(applicationSupportURL: applicationSupportURL,
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
    /// These are the ten picker rows' install names, in that order.
    private static let preferredChoices: [AppModelChoice] = [
        .qwen3_6_abliterated, .qwen3_6_abliterated_3bit,
        .qwen3_6, .qwen3_6_3bit,
        .qwen3_8_abliterated, .qwen3_8_abliterated_3bit, .qwen3_8_abliterated_2bit,
        .qwen3_8, .qwen3_8_3bit, .qwen3_8_2bit,
    ]

    /// Rowless raw-BF16-PLE 3.8 install (162 GiB): the last Qwen entry in the
    /// checkout scan, kept in the app scan for parity. It has no picker row.
    private static let rawQwen38InstallName = "Qwen3.8-Flash-Next-125B.finch"

    private static let preferredInstallNames: [String] =
        preferredChoices.map(\.installName) + [rawQwen38InstallName]

    /// The app-owned equivalent of `preferredInstallURL`: outside a checkout,
    /// launch preference is re-derived by scanning the app's Models directory
    /// in the same order. Nothing installed returns the Gemma fallback path
    /// (installed or not), exactly like the checkout scan.
    private static func preferredAppInstallURL(applicationSupportURL: URL,
                                               fileExists: (String) -> Bool) -> URL {
        let modelsRoot = appModelsRootURL(applicationSupportURL: applicationSupportURL)
        for name in preferredInstallNames {
            let candidate = modelsRoot.appendingPathComponent(name, isDirectory: true)
            if fileExists(candidate.appendingPathComponent("manifest.json").path) {
                return candidate.standardizedFileURL
            }
        }
        return appInstallURL(for: .gemma4, applicationSupportURL: applicationSupportURL)
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
