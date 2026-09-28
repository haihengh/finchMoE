import Foundation
import FinchMoE
import FinchMoERepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Hex SHA-256 (no `sha256:` prefix) of the install's `model_weights.bin`,
    /// i.e. `manifest.files["model_weights.bin"]?.sha256`.
    ///
    /// This exists because `sourceIndexSHA256` cannot identify an *abliterated*
    /// install. That hash covers the source snapshot's tensor **index** — names,
    /// shapes, layout — so an abliterated checkpoint repacked from the same
    /// architecture produces the identical value (verified on disk 2026-09-27:
    /// base and abliterated 3.6 both `41b93561…`, both 3.8 `99e81524…`). The two
    /// fields therefore answer different questions and both are needed:
    /// `sourceIndexSHA256` means "same architecture and layout", this means "same
    /// bytes". `AppModelInstallationProbe.matchingDescriptor` tries this first.
    ///
    /// `nil` means the descriptor does not pin weights, which keeps the older
    /// snapshot-only matching for such an entry (`.default` relies on this).
    public let weightsSHA256: String?
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
                weightsSHA256: String? = nil,
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
        self.weightsSHA256 = weightsSHA256
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
        weightsSHA256: "9644b61a7369787c4397f2f8272e9e95f3eaaf11d6d7be40164a6ec288a8228d",
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
    /// estimated: 103 925 807 384 bytes for the int4-PLE install (`AppModelLocation`'s
    /// preferred `Qwen3.8-Flash-Next-125B-ple4bit.finch`, also the layout published
    /// at `haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit` on Hugging Face),
    /// down from 174 403 168 940 bytes for the raw-BF16-PLE install this repo keeps
    /// as a fallback. It feeds only `requiredFreeBytes`, which gates an install path
    /// this model does not use — but a wrong value there is the kind of number that
    /// gets trusted later. `repoID`/`revision` identify the upstream source
    /// checkpoint this `.finch` was repacked from, not the `.finch` distribution
    /// repo itself.
    public static let qwen3_8 = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "c522877f166d128e0c30ce58bf93322d48e3ebbda55ccf9b2c8dae86efa82a06",
        approximateDownloadBytes: 0,
        installedBytes: 103_925_807_384,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 125B")

    /// Abliterated (refusal-direction-removed) Qwen 3.6. Structurally identical
    /// to `.qwen3_6` — same architecture, same layout, **the same
    /// `sourceIndexSHA256`** — and separable only by `weightsSHA256`. It is
    /// probe-only like the base Qwen entries; there is nothing to download.
    ///
    /// `installedBytes` is the sum of every regular file in
    /// `models/Qwen3.6-35B-A3B-abliterated-4bit.finch`, measured 2026-09-27. It
    /// is a little above `.qwen3_6`'s pinned figure because that one was taken
    /// at repack time, before the receipt and tokenizer files landed.
    public static let qwen3_6_abliterated = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B (Abliterated)",
        repoID: "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        weightsSHA256: "f6862341c9688e234c682cef186af5a92445be1dbcdd36d59637338634d311bd",
        approximateDownloadBytes: 0,
        installedBytes: 20_059_538_761,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 Abl.")

    /// Abliterated Qwen 3.8 Flash-Next, the 3.8 counterpart of
    /// `.qwen3_6_abliterated`: same relationship, same reason it needs its own
    /// `weightsSHA256`, probe-only and not downloadable.
    ///
    /// `installedBytes` is the sum of every regular file in
    /// `models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch`, measured
    /// 2026-09-27.
    public static let qwen3_8_abliterated = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B (Abliterated)",
        repoID: "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
        revision: "",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "6af82b557d8207e470f50c1fba5dbb14e7ff4b48139a5b0b2e4f8ac56d188acb",
        approximateDownloadBytes: 0,
        installedBytes: 104_002_831_190,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 Abl.")

    /// Every descriptor the app can identify a local directory as, in probe
    /// order. `matchingDescriptor` scans this rather than testing each hash
    /// inline: the second family is what makes a list worth having, and a
    /// missed entry here reads as "unknown checkpoint" and silently resolves
    /// the directory to the Gemma default.
    ///
    /// A base install and its abliterated twin deliberately appear as **two
    /// entries sharing one `sourceIndexSHA256`** — that is a real property of
    /// the checkpoints, not an oversight, and it is why `weightsSHA256` exists
    /// and why `matchingDescriptor` consults it first. Order still matters for
    /// the fallback scan, which is first-match-wins.
    public static let installable: [AppModelInstallDescriptor] = [
        .qwen3_6, .qwen3_6_abliterated, .qwen3_8, .qwen3_8_abliterated,
    ]
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
