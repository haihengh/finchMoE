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

    /// `installName` is the single source of truth for the row's directory: the
    /// checkout path and the app-owned path both derive from it, so it must be
    /// exactly the basename those paths end in.
    @Test func installNamesAreDefaultURLBasenames() {
        let root = URL(fileURLWithPath: "/repo")
        for choice in AppModelChoice.allCases {
            #expect(choice.defaultURL(packageRoot: root).lastPathComponent
                        == choice.installName,
                    "\(choice) derives a different basename than its installName")
            #expect(!choice.installName.isEmpty)
        }
    }

    /// Pins the on-disk basenames so a rename cannot silently orphan an
    /// installed model — the app-owned destinations outside a checkout are
    /// derived from these names.
    @Test func installNamesMatchThePinnedLayout() {
        let expected: [(AppModelChoice, String)] = [
            (.gemma4, "gemma4.finch"),
            (.qwen3_6, "Qwen3.6-35B-A3B-4bit.finch"),
            (.qwen3_6_3bit, "Qwen3.6-35B-A3B-3bit.finch"),
            (.qwen3_6_3bit_vision, "Qwen3.6-35B-A3B-3bit-vision.finch"),
            (.qwen3_6_abliterated, "Qwen3.6-35B-A3B-abliterated-4bit.finch"),
            (.qwen3_6_abliterated_vision,
             "Qwen3.6-35B-A3B-abliterated-4bit-vision.finch"),
            (.qwen3_6_abliterated_3bit, "Qwen3.6-35B-A3B-abliterated-3bit.finch"),
            (.qwen3_6_abliterated_3bit_vision,
             "Qwen3.6-35B-A3B-abliterated-3bit-vision.finch"),
            (.qwen3_8, "Qwen3.8-Flash-Next-125B-ple4bit.finch"),
            (.qwen3_8_vision, "Qwen3.8-Flash-Next-125B-ple4bit-vision.finch"),
            (.qwen3_8_3bit, "Qwen3.8-Flash-Next-125B-ple3bit.finch"),
            (.qwen3_8_3bit_vision, "Qwen3.8-Flash-Next-125B-ple3bit-vision.finch"),
            (.qwen3_8_2bit, "Qwen3.8-Flash-Next-125B-ple2bit.finch"),
            (.qwen3_8_2bit_vision, "Qwen3.8-Flash-Next-125B-ple2bit-vision.finch"),
            (.qwen3_8_abliterated, "Qwen3.8-Flash-Next-abliterated-ple4bit.finch"),
            (.qwen3_8_abliterated_vision,
             "Qwen3.8-Flash-Next-abliterated-ple4bit-vision.finch"),
            (.qwen3_8_abliterated_3bit, "Qwen3.8-Flash-Next-abliterated-ple3bit.finch"),
            (.qwen3_8_abliterated_3bit_vision,
             "Qwen3.8-Flash-Next-abliterated-ple3bit-vision.finch"),
            (.qwen3_8_abliterated_2bit, "Qwen3.8-Flash-Next-abliterated-ple2bit.finch"),
            (.qwen3_8_abliterated_2bit_vision,
             "Qwen3.8-Flash-Next-abliterated-ple2bit-vision.finch"),
        ]
        #expect(expected.count == AppModelChoice.allCases.count)
        for (choice, name) in expected {
            #expect(choice.installName == name, "\(choice) install name drifted")
        }
        #expect(Set(expected.map { $0.1 }).count == expected.count,
                "two rows share an install name")
    }
}
