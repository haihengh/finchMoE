import FinchMoEMacPresentation
/// Putting text on the general pasteboard.
///
/// Every copy in the app is the same two calls: the pasteboard has to be
/// cleared before the string goes on, because setting a type onto a pasteboard
/// that still holds another leaves the old contents in place. The calls are
/// spelled per platform in `FinchPlatformPasteboard`, so a shared view can
/// call this one name on both macOS and iOS.
@MainActor
enum Clipboard {
    static func copy(_ text: String) {
        FinchPlatformPasteboard.copy(text)
    }
}
