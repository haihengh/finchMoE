import SwiftUI

/// The accent colors the iOS app offers.
///
/// The Mac app keeps `FinchMoEMacTheme.accentColor` (green) as its fixed
/// accent; this list exists so the phone can repaint the shared views without
/// forking them. Every case is a fixed sRGB triple rather than a system color,
/// so a chosen theme renders identically in light and dark mode and does not
/// drift with the device's accent setting.
public enum FinchAccentTheme: String, CaseIterable, Identifiable, Sendable {
    case darkBlue
    case green
    case teal
    case indigo
    case purple
    case pink
    case orange
    case graphite

    /// The `UserDefaults` key shared by the picker and the app shell.
    public static let storageKey = "finch.accent-theme"

    /// Dark blue is the iOS default; green remains available as the historical
    /// accent (and the Mac app's fixed color).
    public static let `default`: FinchAccentTheme = .darkBlue

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .darkBlue: return "Dark Blue"
        case .green: return "Green"
        case .teal: return "Teal"
        case .indigo: return "Indigo"
        case .purple: return "Purple"
        case .pink: return "Pink"
        case .orange: return "Orange"
        case .graphite: return "Graphite"
        }
    }

    public var color: Color {
        switch self {
        case .darkBlue:
            return Color(red: 30.0 / 255.0, green: 58.0 / 255.0, blue: 138.0 / 255.0)
        case .green:
            // The accent both apps shipped with.
            return Color(red: 106.0 / 255.0, green: 186.0 / 255.0, blue: 113.0 / 255.0)
        case .teal:
            return Color(red: 13.0 / 255.0, green: 148.0 / 255.0, blue: 136.0 / 255.0)
        case .indigo:
            return Color(red: 79.0 / 255.0, green: 70.0 / 255.0, blue: 229.0 / 255.0)
        case .purple:
            return Color(red: 126.0 / 255.0, green: 58.0 / 255.0, blue: 190.0 / 255.0)
        case .pink:
            return Color(red: 190.0 / 255.0, green: 24.0 / 255.0, blue: 93.0 / 255.0)
        case .orange:
            return Color(red: 194.0 / 255.0, green: 94.0 / 255.0, blue: 10.0 / 255.0)
        case .graphite:
            return Color(red: 71.0 / 255.0, green: 85.0 / 255.0, blue: 105.0 / 255.0)
        }
    }
}

/// The accent shared views read instead of calling the static theme directly.
///
/// The default is the Mac app's existing green, so any view built outside the
/// iOS shell (the Mac app, previews) is unchanged. The iOS root injects the
/// user's selection here, which also makes SwiftUI re-render every reader when
/// the picker changes.
private struct FinchAccentColorKey: EnvironmentKey {
    static let defaultValue: Color = FinchMoEMacTheme.accentColor
}

public extension EnvironmentValues {
    var finchAccentColor: Color {
        get { self[FinchAccentColorKey.self] }
        set { self[FinchAccentColorKey.self] = newValue }
    }
}
