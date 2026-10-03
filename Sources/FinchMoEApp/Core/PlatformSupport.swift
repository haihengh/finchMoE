import Foundation

/// Platform spellings the app core needs on both macOS and iOS.
extension FileManager {
    /// The user's home directory.
    ///
    /// `homeDirectoryForCurrentUser` is marked unavailable on iOS; on iOS this
    /// resolves to the app container's home directory, which is the closest
    /// equivalent and never escaped — the surrounding code only uses it as a
    /// fallback when Application Support cannot be resolved at all.
    var finchHomeDirectory: URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
    }
}
