import SwiftUI
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// Named system colors shared by the Mac and iOS apps.
///
/// `Color(nsColor:)` is the macOS spelling and `Color(uiColor:)` the iOS one,
/// so every shared view goes through this table instead of touching
/// AppKit/UIKit directly.
public enum FinchPlatformColors {
    #if canImport(AppKit)
    public static var controlBackground: Color { Color(nsColor: .controlBackgroundColor) }
    public static var windowBackground: Color { Color(nsColor: .windowBackgroundColor) }
    public static var textBackground: Color { Color(nsColor: .textBackgroundColor) }
    public static var underPageBackground: Color { Color(nsColor: .underPageBackgroundColor) }
    #elseif canImport(UIKit)
    // iOS has no separate text/control backgrounds; systemBackground is the
    // closest equivalent for both.
    public static var controlBackground: Color { Color(uiColor: .systemBackground) }
    public static var windowBackground: Color { Color(uiColor: .systemBackground) }
    public static var textBackground: Color { Color(uiColor: .systemBackground) }
    public static var underPageBackground: Color { Color(uiColor: .secondarySystemBackground) }
    #else
    public static var controlBackground: Color { Color.white }
    public static var windowBackground: Color { Color.white }
    public static var textBackground: Color { Color.white }
    public static var underPageBackground: Color { Color.gray.opacity(0.1) }
    #endif
}

/// Copying text to the system pasteboard, spelled once for both platforms.
public enum FinchPlatformPasteboard {
    public static func copy(_ string: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = string
        #endif
    }
}

/// Color attribute values for `AttributedString`s built by the shared
/// markdown renderer: `NSColor` on macOS, `UIColor` on iOS.
public enum FinchPlatformAttributeColors {
    #if canImport(AppKit)
    public static var secondaryLabel: Any { NSColor.secondaryLabelColor }
    public static var controlBackground: Any { NSColor.controlBackgroundColor }
    public static var link: Any { NSColor.linkColor }
    public static var label: Any { NSColor.labelColor }
    #elseif canImport(UIKit)
    public static var secondaryLabel: Any { UIColor.secondaryLabel }
    public static var controlBackground: Any { UIColor.systemBackground }
    public static var link: Any { UIColor.link }
    public static var label: Any { UIColor.label }
    #else
    public static var secondaryLabel: Any { Color.secondary }
    public static var controlBackground: Any { Color.white }
    public static var link: Any { Color.blue }
    public static var label: Any { Color.primary }
    #endif
}

/// System fonts spelled once for both platforms (`NSFont` on macOS,
/// `UIFont` on iOS); the returned value is stored in the `.font` attribute
/// of a cross-platform `NSAttributedString`.
///
/// `NSFont.Weight` and `UIFont.Weight` are distinct types, so the weight
/// parameter is spelled through this alias rather than either one directly.
#if canImport(AppKit)
public typealias FinchFontWeight = NSFont.Weight
#elseif canImport(UIKit)
public typealias FinchFontWeight = UIFont.Weight
#else
public typealias FinchFontWeight = Double
#endif

public enum FinchPlatformFont {
    public static var systemFontSize: CGFloat {
        #if canImport(AppKit)
        NSFont.systemFontSize
        #elseif canImport(UIKit)
        UIFont.systemFontSize
        #else
        17
        #endif
    }

    public static func systemFont(ofSize size: CGFloat, weight: FinchFontWeight) -> Any {
        #if canImport(AppKit)
        NSFont.systemFont(ofSize: size, weight: weight)
        #elseif canImport(UIKit)
        UIFont.systemFont(ofSize: size, weight: weight)
        #else
        size
        #endif
    }

    public static func monospacedSystemFont(ofSize size: CGFloat, weight: FinchFontWeight) -> Any {
        #if canImport(AppKit)
        NSFont.monospacedSystemFont(ofSize: size, weight: weight)
        #elseif canImport(UIKit)
        UIFont.monospacedSystemFont(ofSize: size, weight: weight)
        #else
        size
        #endif
    }

    /// The italic variant of a font produced above, at the same size.
    public static func italic(_ font: Any, size: CGFloat) -> Any {
        #if canImport(AppKit)
        let concrete = font as! NSFont
        let descriptor = concrete.fontDescriptor.withSymbolicTraits(.italic)
        return NSFont(descriptor: descriptor, size: size) ?? concrete
        #elseif canImport(UIKit)
        let concrete = font as! UIFont
        // iOS spells the trait `traitItalic` and returns nil when the
        // descriptor cannot take it; fall back to the upright font.
        if let descriptor = concrete.fontDescriptor.withSymbolicTraits(.traitItalic) {
            return UIFont(descriptor: descriptor, size: size) ?? concrete
        }
        return concrete
        #else
        font
        #endif
    }
}

/// Input-state helpers with no cross-platform spelling.
public enum FinchPlatformInput {
    /// True while an IME is composing marked text in the focused text field
    /// (return should commit the composition instead of submitting).
    ///
    /// On macOS this reads the key window's first responder; on iOS the
    /// composer's key-press path is not used, so it reports false.
    public static var isComposingMarkedText: Bool {
        #if canImport(AppKit)
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
        #elseif canImport(UIKit)
        false
        #else
        false
        #endif
    }
}
