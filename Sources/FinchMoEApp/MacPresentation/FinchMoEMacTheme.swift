import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// The shared accent for both apps.
public enum FinchMoEMacTheme {
    #if canImport(AppKit)
    public static let accentNSColor = NSColor(
        srgbRed: 106.0 / 255.0,
        green: 186.0 / 255.0,
        blue: 113.0 / 255.0,
        alpha: 1)
    #endif

    public static var accentColor: Color {
        #if canImport(AppKit)
        Color(nsColor: accentNSColor)
        #else
        Color(red: 106.0 / 255.0, green: 186.0 / 255.0, blue: 113.0 / 255.0)
        #endif
    }
}
