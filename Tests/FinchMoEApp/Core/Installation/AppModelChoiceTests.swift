import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

/// The row ↔ descriptor ↔ directory contract, which had no coverage before the
/// per-width rows arrived. Three couplings make it load-bearing: the picker
/// binds `AppModelChoice.allCases`, the install path writes to the selected
/// row's `defaultURL`, and `AppModel.modelChoice(for:)` matches a directory to
/// a row by descriptor equality *and* `defaultURL` equality. Rows, descriptors
/// and paths therefore each have to be unique and mutually consistent — a
/// duplicate anywhere surfaces as a picker that flips rows or a Download that
/// writes to the wrong directory.
@Suite struct AppModelChoiceTests {
    @Test func labelsAreUniqueAndUseTheDescriptorNames() {
        let labels = AppModelChoice.allCases.map(\.label)
        #expect(Set(labels).count == labels.count, "two picker rows share a label")
        for choice in AppModelChoice.allCases {
            #expect(choice.label == choice.descriptor.displayName,
                    "\(choice) labels itself differently from its descriptor")
        }
    }

    @Test func shortNamesAreUnique() {
        let shortNames = AppModelChoice.allCases.map(\.descriptor.shortName)
        #expect(Set(shortNames).count == shortNames.count,
                "two rows share a short name; the status badge becomes ambiguous")
    }

    @Test func defaultURLsAreUnique() {
        let root = URL(fileURLWithPath: "/repo")
        let paths = AppModelChoice.allCases.map { $0.defaultURL(packageRoot: root).path }
        #expect(Set(paths).count == paths.count,
                "two rows share a download destination; an install would land in the wrong row")
    }

    /// `modelChoice(for:)` finds a row by descriptor equality as a value, so a
    /// duplicated descriptor makes one of its two rows unreachable — the picker
    /// would highlight the wrong entry for the directory on disk.
    @Test func descriptorsAreUniquePerRow() {
        let displayNames = AppModelChoice.allCases.map(\.descriptor.displayName)
        #expect(Set(displayNames).count == displayNames.count,
                "two rows carry equal descriptors")
    }

    @Test func everyInstallableDescriptorHasARowAndViceVersa() {
        let rowDescriptors = AppModelChoice.allCases.map(\.descriptor)
        for descriptor in AppModelInstallDescriptor.installable {
            #expect(rowDescriptors.contains(descriptor),
                    "\(descriptor.displayName) is installable but has no picker row")
        }
        for choice in AppModelChoice.allCases where choice != .gemma4 {
            #expect(AppModelInstallDescriptor.installable.contains(choice.descriptor),
                    "\(choice.label) has a picker row but is not installable")
        }
    }
}
