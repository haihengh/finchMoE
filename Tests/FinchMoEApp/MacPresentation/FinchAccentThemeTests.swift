import Testing
@testable import FinchMoEMacPresentation

/// The picker persists a raw string, so the names it stores are part of the
/// app's on-disk contract: renaming a case must not silently reset a user's
/// choice to the default.
@Suite struct FinchAccentThemeTests {
    @Test func darkBlueIsTheDefault() {
        #expect(FinchAccentTheme.default == .darkBlue)
    }

    @Test func storedRawValuesRoundTrip() {
        for theme in FinchAccentTheme.allCases {
            #expect(FinchAccentTheme(rawValue: theme.rawValue) == theme)
        }
        #expect(FinchAccentTheme(rawValue: "dark-blue") == nil)
    }

    @Test func everyCaseHasAUniqueReadableName() {
        let names = FinchAccentTheme.allCases.map(\.displayName)
        let ids = FinchAccentTheme.allCases.map(\.id)
        #expect(Set(names).count == FinchAccentTheme.allCases.count)
        #expect(Set(ids).count == FinchAccentTheme.allCases.count)
        #expect(!names.contains { $0.isEmpty })
    }

    @Test func storageKeyIsStable() {
        #expect(FinchAccentTheme.storageKey == "finch.accent-theme")
    }
}
