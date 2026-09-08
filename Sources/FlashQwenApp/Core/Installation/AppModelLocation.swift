import Foundation

enum AppModelLocation {
    static func defaultURL() -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
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
        return applicationSupportURL
            .appendingPathComponent("FlashQwen", isDirectory: true)
            .appendingPathComponent("gemma4.fqturbo", isDirectory: true)
            .standardizedFileURL
    }

    /// The app's default model inside a package checkout: the repack-made
    /// Qwen install under `models/` when present, else the Gemma target under
    /// `scratch/` (which is also the in-app download destination when nothing
    /// is installed yet). Outside a checkout the Application Support Gemma
    /// target above remains the fallback.
    private static func preferredInstallURL(inPackageRoot root: URL,
                                            fileExists: (String) -> Bool) -> URL {
        let qwen = root
            .appendingPathComponent("models/Qwen3.6-35B-A3B-4bit.fqturbo", isDirectory: true)
        let qwenManifest = qwen.appendingPathComponent("manifest.json").path
        if fileExists(qwenManifest) {
            return qwen.standardizedFileURL
        }
        return root.appendingPathComponent("scratch/gemma4.fqturbo", isDirectory: true)
            .standardizedFileURL
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
            let appSources = candidate.appendingPathComponent(
                "Sources/FlashQwenApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
