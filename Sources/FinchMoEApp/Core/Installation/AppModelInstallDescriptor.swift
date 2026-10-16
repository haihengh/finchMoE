import Foundation
import FinchMoE
import FinchMoERepackCore

/// How a descriptor's bytes get onto disk.
///
/// This used to be implicit — `supportsRemoteInstall` was literally
/// `approximateDownloadBytes > 0` — which worked while Gemma was the only
/// downloadable model. It stopped working the moment a second *kind* of remote
/// install appeared, because a published `.finch` distribution has nothing to
/// repack and a repack source has no finished layout to fetch: they need
/// different code, so the descriptor has to say which one it is.
public enum AppModelInstallRoute: Equatable, Sendable {
    /// Made out of band and only ever probed by the app.
    case none
    /// Stream an upstream safetensors checkpoint and repack it on the way down.
    case repackUpstream
    /// Fetch an already-repacked `.finch` directory, as published on Hugging Face.
    case finchDistribution(FinchDistribution)
}

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
    /// Routed-expert width this entry describes (2, 3 or 4), or `nil` to accept
    /// any width.
    ///
    /// `weightsSHA256` cannot separate the widths of one checkpoint: the
    /// resident weights are byte-identical across the 2-, 3- and 4-bit builds
    /// and only `packed_experts/` differs (verified on disk 2026-10-15 — all
    /// three 3.8 PLE widths share `c522877f…`, both 3.6 base widths share
    /// `9644b61a…`), so the width has to come from the manifest's
    /// `quant.routedExpert.weightBits`. Every width-bearing entry pins it —
    /// including the 4-bit ones — or the coarser entry would claim the finer
    /// directory by coming first in the scan.
    public let routedExpertBits: Int?
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

    /// The published `.finch` distribution this checkpoint can be fetched from,
    /// when one exists. `repoID`/`revision` above stay what they always were —
    /// the *upstream* checkpoint this `.finch` was repacked from, and the
    /// provenance the probe matches on — so this field is what actually answers
    /// "where do the bytes come from".
    public let finchDistribution: FinchDistribution?

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                weightsSHA256: String? = nil,
                routedExpertBits: Int? = nil,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64,
                architecture: ArchConfig,
                shortDisplayName: String? = nil,
                finchDistribution: FinchDistribution? = nil) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.weightsSHA256 = weightsSHA256
        self.routedExpertBits = routedExpertBits
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
        self.architecture = architecture
        self.shortDisplayName = shortDisplayName
        self.finchDistribution = finchDistribution
    }

    public var shortName: String { shortDisplayName ?? displayName }

    public var installRoute: AppModelInstallRoute {
        if let finchDistribution { return .finchDistribution(finchDistribution) }
        if approximateDownloadBytes > 0 { return .repackUpstream }
        return .none
    }

    public var supportsRemoteInstall: Bool { installRoute != .none }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    /// Whether this entry may describe an install whose manifest declares the
    /// given routed-expert width. Tolerant in one direction only: a descriptor
    /// that pins nothing accepts every width, while a pinned descriptor that
    /// disagrees rejects the match.
    func accepts(routedExpertBits declared: Int) -> Bool {
        routedExpertBits == nil || routedExpertBits == declared
    }

    /// Gemma has one build, so it pins no expert width (`nil` accepts any);
    /// it is a fallback, never scanned by the probe.
    public static let `default` = AppModelInstallDescriptor(
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        routedExpertBits: nil,
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        architecture: .gemma4_26B_A4B,
        shortDisplayName: "Gemma 4 26B")

    /// Qwen 3.6 35B-A3B, the 4-bit routed-expert build. Same checkpoint as
    /// `.qwen3_6_3bit` — resident weights byte-identical, only
    /// `packed_experts/` differs — so the two are separate entries pinning
    /// different `routedExpertBits` rather than one entry covering both: each
    /// picker row downloads to, and starts from, its own build. This entry has
    /// no distribution (the published base 3.6 is the 3-bit build), so it is
    /// probe-only; a local 4-bit directory resolves here.
    ///
    /// `installedBytes` is the sum of every regular file in
    /// `models/Qwen3.6-35B-A3B-4bit.finch`, measured 2026-09-27.
    public static let qwen3_6 = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B 4-bit",
        repoID: "Qwen/Qwen3.6-35B-A3B",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        weightsSHA256: "9644b61a7369787c4397f2f8272e9e95f3eaaf11d6d7be40164a6ec288a8228d",
        routedExpertBits: 4,
        approximateDownloadBytes: 0,
        installedBytes: 20_014_114_816,
        rangeStagingBytes: 0,
        reserveBytes: 0,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 4-bit")

    /// Qwen 3.6 35B-A3B at 3-bit routed experts — the base 3.6 the app
    /// installs. Carries the published distribution (`qwen36ThreeBitDistribution`);
    /// a fresh download lands in `models/Qwen3.6-35B-A3B-3bit.finch`.
    public static let qwen3_6_3bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B 3-bit",
        repoID: "Qwen/Qwen3.6-35B-A3B",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        weightsSHA256: "9644b61a7369787c4397f2f8272e9e95f3eaaf11d6d7be40164a6ec288a8228d",
        routedExpertBits: 3,
        approximateDownloadBytes: qwen36ThreeBitDistribution.approximateDownloadBytes,
        installedBytes: 16_032_975_496,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 3-bit",
        finchDistribution: qwen36ThreeBitDistribution)

    /// Qwen 3.8 Flash-Next 125B made by `FinchMoERepack`. Unlike 3.6 this one
    /// has a published distribution, so it is downloadable even though its own
    /// repack source is not — `repoID`/`revision` still identify the upstream
    /// checkpoint, and `finchDistribution` carries the published install.
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
        displayName: "Qwen 3.8 Flash-Next 125B 4-bit",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "c522877f166d128e0c30ce58bf93322d48e3ebbda55ccf9b2c8dae86efa82a06",
        routedExpertBits: 4,
        approximateDownloadBytes: qwen38Distribution.approximateDownloadBytes,
        installedBytes: 103_925_807_384,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 125B 4-bit",
        finchDistribution: qwen38Distribution)

    /// Qwen 3.8 Flash-Next 125B at 3-bit routed experts — the quality-neutral
    /// step down (HumanEval+ matches the 4-bit build exactly). `installedBytes`
    /// measured 2026-10-15 from `models/Qwen3.8-Flash-Next-125B-ple3bit.finch`.
    public static let qwen3_8_3bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B 3-bit",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "c522877f166d128e0c30ce58bf93322d48e3ebbda55ccf9b2c8dae86efa82a06",
        routedExpertBits: 3,
        approximateDownloadBytes: qwen38ThreeBitDistribution.approximateDownloadBytes,
        installedBytes: 89_104_658_029,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 125B 3-bit",
        finchDistribution: qwen38ThreeBitDistribution)

    /// Qwen 3.8 Flash-Next 125B at 2-bit routed experts — the fastest decode of
    /// the family, ~10 pp of quality below 3-bit. `installedBytes` measured
    /// 2026-10-15 from `models/Qwen3.8-Flash-Next-125B-ple2bit.finch`.
    public static let qwen3_8_2bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B 2-bit",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "c522877f166d128e0c30ce58bf93322d48e3ebbda55ccf9b2c8dae86efa82a06",
        routedExpertBits: 2,
        approximateDownloadBytes: qwen38TwoBitDistribution.approximateDownloadBytes,
        installedBytes: 73_803_784_477,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 125B 2-bit",
        finchDistribution: qwen38TwoBitDistribution)

    /// Abliterated (refusal-direction-removed) Qwen 3.6. Structurally identical
    /// to `.qwen3_6` — same architecture, same layout, **the same
    /// `sourceIndexSHA256`** — and separable only by `weightsSHA256`.
    ///
    /// `repoID` is the upstream abliteration this was repacked from;
    /// `finchDistribution` is where the finished install is published, and it is
    /// the one the Download button uses. `installedBytes` is the sum of every
    /// regular file in `models/Qwen3.6-35B-A3B-abliterated-4bit.finch`, measured
    /// 2026-09-27; the payload figure is within 18 KB of it, which is the
    /// manifest, receipt and tokenizer sidecars.
    public static let qwen3_6_abliterated = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B 4-bit (Abliterated)",
        repoID: "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        weightsSHA256: "f6862341c9688e234c682cef186af5a92445be1dbcdd36d59637338634d311bd",
        routedExpertBits: 4,
        approximateDownloadBytes: qwen36AbliteratedDistribution.approximateDownloadBytes,
        installedBytes: 20_059_538_761,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 Abl. 4-bit",
        finchDistribution: qwen36AbliteratedDistribution)

    /// Abliterated Qwen 3.6 at 3-bit routed experts, repacked 2026-10-16 from
    /// `models/Qwen3.6-35B-A3B-abliterated-bf16`. Payload is byte-identical to
    /// the base 3-bit build (the twin property); `installedBytes` carries the
    /// base's measurement until the abliterated install's own sum is taken.
    public static let qwen3_6_abliterated_3bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.6 35B-A3B 3-bit (Abliterated)",
        repoID: "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        revision: "",
        sourceIndexSHA256: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        weightsSHA256: "f6862341c9688e234c682cef186af5a92445be1dbcdd36d59637338634d311bd",
        routedExpertBits: 3,
        approximateDownloadBytes: qwen36ThreeBitAbliteratedDistribution.approximateDownloadBytes,
        installedBytes: 16_032_975_496,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_6_35B_A3B,
        shortDisplayName: "Qwen 3.6 Abl. 3-bit",
        finchDistribution: qwen36ThreeBitAbliteratedDistribution)

    /// Abliterated Qwen 3.8 Flash-Next, the 3.8 counterpart of
    /// `.qwen3_6_abliterated`: same relationship, same reason it needs its own
    /// `weightsSHA256`. As with 3.6, `repoID` is the upstream abliteration and
    /// `finchDistribution` is the published install.
    ///
    /// `installedBytes` is the sum of every regular file in
    /// `models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch`, measured
    /// 2026-09-27.
    public static let qwen3_8_abliterated = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B 4-bit (Abliterated)",
        repoID: "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
        revision: "",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "6af82b557d8207e470f50c1fba5dbb14e7ff4b48139a5b0b2e4f8ac56d188acb",
        routedExpertBits: 4,
        approximateDownloadBytes: qwen38AbliteratedDistribution.approximateDownloadBytes,
        installedBytes: 104_002_831_190,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 Abl. 4-bit",
        finchDistribution: qwen38AbliteratedDistribution)

    /// Abliterated Qwen 3.8 at 3-bit routed experts, repacked 2026-10-16 from
    /// `models/Qwen3.8-Flash-Next-abliterated-bf16`. Payload is byte-identical
    /// to the base 3-bit build (verified on disk 2026-10-16); `installedBytes`
    /// is this install's own measured sum.
    public static let qwen3_8_abliterated_3bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B 3-bit (Abliterated)",
        repoID: "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
        revision: "",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "6af82b557d8207e470f50c1fba5dbb14e7ff4b48139a5b0b2e4f8ac56d188acb",
        routedExpertBits: 3,
        approximateDownloadBytes: qwen38ThreeBitAbliteratedDistribution.approximateDownloadBytes,
        installedBytes: 89_104_658_006,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 Abl. 3-bit",
        finchDistribution: qwen38ThreeBitAbliteratedDistribution)

    /// Abliterated Qwen 3.8 at 2-bit routed experts, repacked 2026-10-16.
    /// Payload identical to the base 2-bit build (verified on disk
    /// 2026-10-16); `installedBytes` is this install's own measured sum.
    public static let qwen3_8_abliterated_2bit = AppModelInstallDescriptor(
        displayName: "Qwen 3.8 Flash-Next 125B 2-bit (Abliterated)",
        repoID: "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
        revision: "",
        sourceIndexSHA256: "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        weightsSHA256: "6af82b557d8207e470f50c1fba5dbb14e7ff4b48139a5b0b2e4f8ac56d188acb",
        routedExpertBits: 2,
        approximateDownloadBytes: qwen38TwoBitAbliteratedDistribution.approximateDownloadBytes,
        installedBytes: 73_803_784_454,
        rangeStagingBytes: 0,
        reserveBytes: distributionReserveBytes,
        architecture: .qwen3_8_flashNext_125B,
        shortDisplayName: "Qwen 3.8 Abl. 2-bit",
        finchDistribution: qwen38TwoBitAbliteratedDistribution)

    /// Every descriptor the app can identify a local directory as, in probe
    /// order. `matchingDescriptor` scans this rather than testing each hash
    /// inline: the second family is what makes a list worth having, and a
    /// missed entry here reads as "unknown checkpoint" and silently resolves
    /// the directory to the Gemma default.
    ///
    /// A base install and its abliterated twin deliberately appear as **two
    /// entries sharing one `sourceIndexSHA256`** — that is a real property of
    /// the checkpoints, not an oversight, and it is why `weightsSHA256` exists
    /// and why `matchingDescriptor` consults it first. The expert widths of one
    /// checkpoint do the same with `weightsSHA256` itself, which is what
    /// `routedExpertBits` separates.
    ///
    /// Every width-bearing entry pins its width, so order no longer decides
    /// which width claims a directory — it only orders the *fallback* scan for
    /// a directory whose weights digest no entry recognises, which stays
    /// first-match-wins. Each family therefore keeps its 4-bit entry first, so
    /// that fallback answers exactly as it did before widths existed.
    public static let installable: [AppModelInstallDescriptor] = [
        .qwen3_6, .qwen3_6_3bit, .qwen3_6_abliterated, .qwen3_6_abliterated_3bit,
        .qwen3_8, .qwen3_8_3bit, .qwen3_8_2bit,
        .qwen3_8_abliterated, .qwen3_8_abliterated_3bit, .qwen3_8_abliterated_2bit,
    ]
}

// MARK: - Published `.finch` distributions

// Each payload figure is the sum of `manifest.files[].size` in its repository —
// exactly what the download transfers, and what the Download button quotes.
// They are declared once, here, so the number the UI shows and the number the
// downloader is handed cannot drift apart.
//
// The commits are pinned rather than tracked at `main`: the install checkpoint
// and the receipt both record the resolved commit, and a distribution that
// moved under a resumed download would be refused rather than half-mixed.
//
// One distribution per published width, hung off the row that downloads it.
// Abliterated twins share their base sibling's payload figure byte-for-byte
// (verified for both 4-bit pairs), which is what lets their entries pin a
// download size before their uploads exist. Each `revision` starts as
// `PENDING-UPLOAD-REPLACE-ME` and is replaced with the resolved commit as the
// matching upload lands — until then the row is visible but its download will
// fail, which is deliberate (the alternative is a dead endpoint that looks
// live).

private let qwen36ThreeBitDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.6-35B-A3B-finch-3bit",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 16_032_957_550,
    installedBytes: 16_032_975_496)

private let qwen36ThreeBitAbliteratedDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.6-35B-A3B-finch-3bit-abliterated",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 16_032_957_550,
    installedBytes: 16_032_975_496)

private let qwen38ThreeBitDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple3bit",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 89_104_597_905,
    installedBytes: 89_104_658_029)

private let qwen38TwoBitDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple2bit",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 73_803_724_449,
    installedBytes: 73_803_784_477)

private let qwen38ThreeBitAbliteratedDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple3bit-abliterated",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 89_104_597_905,
    installedBytes: 89_104_658_029)

private let qwen38TwoBitAbliteratedDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple2bit-abliterated",
    revision: "PENDING-UPLOAD-REPLACE-ME",
    approximateDownloadBytes: 73_803_724_449,
    installedBytes: 73_803_784_477)

private let qwen36AbliteratedDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated",
    revision: "e1998dcbf984e2c3e8e38eab5eacd3b8ed0c93de",
    approximateDownloadBytes: 20_059_520_830,
    installedBytes: 20_059_538_761)

private let qwen38Distribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit",
    revision: "ad8151a34a5b5fd40c581e16fa526577315ac5df",
    approximateDownloadBytes: 104_002_771_089,
    installedBytes: 103_925_807_384)

private let qwen38AbliteratedDistribution = FinchDistribution(
    repoID: "haihengh/Qwen3.8-Flash-Next-125B-finchmoe-4bit-ple4bit-abliterated",
    revision: "d19beab26664be0e3becd66322d71631cec63ddb",
    approximateDownloadBytes: 104_002_771_089,
    installedBytes: 104_002_831_190)

/// Held back on top of the payload so a finished install is not immediately
/// under storage pressure. Mirrors
/// `FinchDistributionDownloader.defaultReserveBytes`.
private let distributionReserveBytes: UInt64 = 1 << 30

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
