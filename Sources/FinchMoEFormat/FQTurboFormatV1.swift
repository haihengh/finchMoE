import Foundation

package enum FQTurboFormatV1 {
    package static let magic = "FQTURBO"
    // Canonical model-family strings written to manifest.arch.modelFamily.
    // Single home for the runtime (ArchConfig.qwen*Family) and the repacker
    // (ArchInfo) so a family can never drift between writer and reader.
    package static let gemma4Family = "gemma4"
    package static let qwen36Family = "qwen3_6"
    package static let qwen38Family = "qwen3_8"
    package static let versionMajor = 1
    /// Optional arch-key groups (Qwen3.6 GDN keys, then Qwen3.8-Flash-Next's
    /// `hyperConnection*`/`indexer*`/`ngram*`/`ple*`) are additive at minor 0
    /// by precedent: encode omits nil keys so old byte streams are unchanged,
    /// decode synthesizes missing keys to nil, and old readers ignore unknown
    /// JSON keys. versionMinor stays 0 — it is embedded in repacked layer
    /// files (`RangeCopyPlanner`), so bumping it would change every artifact
    /// for every family. A *breaking* wire change (required key, semantic
    /// reinterpretation) is what bumps minor, with fixture regeneration.
    package static let versionMinor = 0
    package static let alignmentBytes: UInt64 = 16_384
    package static let residentHeaderBytes = 24
    package static let residentEntryBytes = 72
    package static let residentIndexMaxBytes: UInt64 = 16 * 1024 * 1024

    package static let knownFlags: Set<String> = [
        "streamingPresent", "quantKV", "aneSharedExpert",
    ]

    package enum DType: UInt8, Sendable {
        case u32 = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
    }
}

package enum FQTurboFormatError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(field: String, reason: String)
    case overflow(field: String)
    case truncated(field: String)

    package var description: String {
        switch self {
        case let .invalid(field, reason): "\(field): \(reason)"
        case let .overflow(field): "\(field): arithmetic overflow"
        case let .truncated(field): "\(field): truncated"
        }
    }
}

@inline(__always)
package func fqturboCheckedAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else { throw FQTurboFormatError.overflow(field: field) }
    return value
}

@inline(__always)
package func fqturboCheckedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw FQTurboFormatError.overflow(field: field) }
    return value
}

package enum FQTurboPathValidator {
    package static func appleFilesystemKey(_ path: String) -> String {
        path.precomposedStringWithCanonicalMapping
            .lowercased(with: Locale(identifier: "en_US_POSIX"))
    }

    package static func validateRelativePath(_ path: String, field: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\0") else {
            throw FQTurboFormatError.invalid(field: field, reason: "unsafe relative path")
        }
        let components = path.components(separatedBy: "/")
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw FQTurboFormatError.invalid(field: field, reason: "non-canonical path")
        }
        let normalized = NSString.path(withComponents: components)
        guard normalized == path else {
            throw FQTurboFormatError.invalid(field: field, reason: "non-normalized path")
        }
    }

    package static func validateBasename(_ name: String, field: String) throws {
        try validateRelativePath(name, field: field)
        guard !name.contains("/") else {
            throw FQTurboFormatError.invalid(field: field, reason: "expected basename")
        }
    }
}
