import Foundation
import FinchMoE
import FinchMoERepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64
    /// The architecture this checkpoint loads as. Required rather than
    /// defaulted: the context-length menu sizes its KV figures off this, and a
    /// default here is what previously reported Gemma's numbers against a Qwen
    /// install.
    public let architecture: ArchConfig
    /// Short label for tight UI (status badge); falls back to `displayName`.
    public let shortDisplayName: String?

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64,
                architecture: ArchConfig,
                shortDisplayName: String? = nil) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
        self.architecture = architecture
        self.shortDisplayName = shortDisplayName
    }

    public var shortName: String { shortDisplayName ?? displayName }

    public var supportsRemoteInstall: Bool { approximateDownloadBytes > 0 }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    public static let `default` = AppModelInstallDescriptor(
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        architecture: .gemma4_26B_A4B,
        shortDisplayName: "Gemma 4 26B")

    /// Local Qwen 3.6 install made by `FinchMoERepack`. The app only ever
    /// *probes* this checkpoint — in-app remote install is not supported for
    /// it (the range-streaming installer targets the mlx 4-bit Gemma layout),
    /// so repo/revision are informational and download sizing is zero.
    public static let qwen3_6 = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B",
        repoID: "Qwen/Qwen3.6-35B-A3B",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        approximateDownloadBytes: 0,
        installedBytes: 20_014_114_816,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 35B-A3B")

    /// Local Qwen 3.8 Flash-Next 125B install made by `FinchMoERepack`. Like
    /// 3.6 it is probe-only — the range-streaming installer targets the mlx
    /// Gemma layout, so repo/revision are informational and there is nothing to
    /// download.
    ///
    /// `installedBytes` is measured from the install directory rather than
    /// estimated: 174 403 168 940 bytes, of which 102.4 GB is the PLE n-gram
    /// shards (128 × 800 003 840 B) and 68.0 GB the packed experts for 512
    /// experts over 48 layers. It
    /// feeds only `requiredFreeBytes`, which gates an install path this model
    /// does not use — but a wrong value there is the kind of number that gets
    /// trusted later. `revision` is the Hugging Face commit the snapshot was
    /// fetched at, taken from the local snapshot's download metadata.
    public static let qwen3_8 = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        approximateDownloadBytes: 0,
        installedBytes: 174_403_168_940,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 125B")

    /// Every descriptor the app can identify a local directory as, in probe
    /// order. `matchingDescriptor` scans this rather than testing each hash
    /// inline: the second family is what makes a list worth having, and a
    /// missed entry here reads as "unknown checkpoint" and silently resolves
    /// the directory to the Gemma default.
    public static let installable: [AppModelInstallDescriptor] = [.qwen3_6, .qwen3_8]
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
