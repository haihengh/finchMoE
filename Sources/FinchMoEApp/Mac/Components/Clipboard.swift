import AppKit

/// Putting text on the general pasteboard.
///
/// Every copy in the app is the same two calls: the pasteboard has to be
/// cleared before the string goes on, because setting a type onto a pasteboard
/// that still holds another leaves the old contents in place.
@MainActor
enum Clipboard {
    static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
