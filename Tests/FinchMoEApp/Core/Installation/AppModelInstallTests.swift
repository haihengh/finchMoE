import Foundation
import Testing
import FinchMoE
import FinchMoERepackCore

@testable import FinchMoEAppCore

@Suite struct AppModelInstallTests {

  @MainActor
  @Test func missingModelCanInstall() {
    let installer = MockModelInstallerClient()
    let directory = temporaryInstallPath("missing")
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(!model.isModelInstalled)
    #expect(model.requiresModelInstallation)
    #expect(model.canInstallModel)
  }

  @MainActor
  @Test func installedModelShowsLoadNotInstall() throws {
    let directory = try makeCompleteModelInstall("installed")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(!model.canInstallModel)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainDetectsModelInstalledAfterLaunch() throws {
    let directory = try makeCompleteModelInstall("external-install")
    let stagedDirectory = directory.deletingLastPathComponent()
      .appendingPathComponent("staged-\(UUID().uuidString).finch")
    try FileManager.default.moveItem(at: directory, to: stagedDirectory)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.requiresModelInstallation)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.moveItem(at: stagedDirectory, to: directory)

    model.refreshInstallReadiness()

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainUsesCurrentModelLocation() throws {
    let initialDirectory = temporaryInstallPath("initial-location")
    let currentDirectory = try makeCompleteModelInstall("current-location")
    defer { try? FileManager.default.removeItem(at: currentDirectory) }
    let model = AppModel(
      modelDirectory: initialDirectory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    model.modelPathText = currentDirectory.path
    model.recheckModelAtCurrentLocation()

    #expect(model.modelPathText == currentDirectory.standardizedFileURL.path)
    #expect(model.isModelInstalled)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func defaultInstallDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.default
    #expect(descriptor.displayName == "Gemma 4 26B-A4B IT 4-bit")
    #expect(descriptor.shortDisplayName == "Gemma 4 26B")
    #expect(descriptor.shortName == "Gemma 4 26B")
    #expect(descriptor.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
    #expect(descriptor.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
    #expect(descriptor.sourceIndexSHA256 == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
    #expect(descriptor.approximateDownloadBytes == 14_620_479_420)
    #expect(descriptor.installedBytes == 14_291_921_884)
    #expect(descriptor.requiredFreeBytes == 15_432_772_572)
  }

  @MainActor
  @Test func qwenInstallProbesCompleteAndSelectsQwenDescriptor() throws {
    let directory = try makeCompleteModelInstall(
      "app-qwen",
      arch: ArchConfig.qwen3_6_35B_A3B,
      modelID: "local/Qwen3.6-35B-A3B",
      descriptor: .qwen3_6)
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient())

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(model.canLoadModel)
    #expect(model.installDescriptor == .qwen3_6)
    #expect(model.modelPathText == directory.standardizedFileURL.path)
  }

  /// The 3.6 4-bit row. It is probe-only — no base 4-bit `.finch` is
  /// published, so there is nothing to fetch and the Download button must stay
  /// disabled; the downloadable base 3.6 is the 3-bit row below.
  @MainActor
  @Test func qwenInstallDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.qwen3_6
    #expect(descriptor.displayName == "Qwen 3.6 35B-A3B 4-bit")
    #expect(descriptor.shortName == "Qwen 3.6 4-bit")
    #expect(descriptor.repoID == "Qwen/Qwen3.6-35B-A3B")
    #expect(descriptor.sourceIndexSHA256 == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83")
    #expect(descriptor.weightsSHA256 == "9644b61a7369787c4397f2f8272e9e95f3eaaf11d6d7be40164a6ec288a8228d")
    #expect(descriptor.routedExpertBits == 4)
    #expect(descriptor.approximateDownloadBytes == 0)
    #expect(descriptor.installedBytes == 20_014_114_816)
    #expect(descriptor.rangeStagingBytes == 0)
    #expect(descriptor.reserveBytes == 0)
    #expect(descriptor.requiredFreeBytes == 20_014_114_816)
    #expect(descriptor.installRoute == .none)
    #expect(!descriptor.supportsRemoteInstall)
  }

  /// The 3.6 3-bit row, which carries the published base distribution. Its
  /// distribution revision is a placeholder until the upload commits.
  @MainActor
  @Test func qwen36ThreeBitDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.qwen3_6_3bit
    #expect(descriptor.displayName == "Qwen 3.6 35B-A3B 3-bit")
    #expect(descriptor.shortName == "Qwen 3.6 3-bit")
    #expect(descriptor.repoID == "Qwen/Qwen3.6-35B-A3B")
    #expect(descriptor.sourceIndexSHA256
              == AppModelInstallDescriptor.qwen3_6.sourceIndexSHA256)
    #expect(descriptor.weightsSHA256
              == AppModelInstallDescriptor.qwen3_6.weightsSHA256)
    #expect(descriptor.routedExpertBits == 3)
    #expect(descriptor.installedBytes == 16_032_975_496)
    #expect(descriptor.rangeStagingBytes == 0)
    #expect(descriptor.reserveBytes == 1_073_741_824)
    #expect(descriptor.requiredFreeBytes == 17_106_717_320)
    #expect(descriptor.architecture == .qwen3_6_35B_A3B)

    guard case .finchDistribution(let distribution) = descriptor.installRoute else {
      Issue.record("Qwen 3.6 3-bit should install from a published distribution")
      return
    }
    #expect(distribution.repoID == "haihengh/Qwen3.6-35B-A3B-finch-3bit")
    #expect(distribution.revision == "8a6734f672742102dcb42ec8c106f10f3a83182d")
    #expect(distribution.approximateDownloadBytes == 16_032_957_550)
    #expect(descriptor.approximateDownloadBytes == 16_032_957_550)
  }

  /// The 3.8 counterpart. The hash is the one thing here that cannot be
  /// re-derived from the repo — it is the sha256 of the source snapshot's
  /// `model.safetensors.index.json`, which is what the manifest stores and
  /// what the probe matches on, so a typo silently makes the install
  /// unrecognisable rather than failing loudly.
  ///
  /// `repoID` and `finchDistribution.repoID` are deliberately different
  /// repositories and must both stay: the first is the upstream checkpoint this
  /// `.finch` was repacked from and is what the probe matches on, the second is
  /// where the finished install is published and is what the download fetches.
  @MainActor
  @Test func qwen38InstallDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.qwen3_8
    #expect(descriptor.displayName == "Qwen 3.8 Flash-Next 125B 4-bit")
    #expect(descriptor.shortName == "Qwen 3.8 125B 4-bit")
    #expect(descriptor.repoID == "Qwen/Qwen3.8-Flash-Next")
    #expect(descriptor.sourceIndexSHA256 == "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de")
    #expect(descriptor.weightsSHA256 == "c522877f166d128e0c30ce58bf93322d48e3ebbda55ccf9b2c8dae86efa82a06")
    #expect(descriptor.routedExpertBits == 4)
    #expect(descriptor.installedBytes == 103_925_807_384)
    #expect(descriptor.rangeStagingBytes == 0)
    #expect(descriptor.reserveBytes == 1_073_741_824)
    #expect(descriptor.requiredFreeBytes == 104_999_549_208)

    guard case .finchDistribution(let distribution) = descriptor.installRoute else {
      Issue.record("Qwen 3.8 base should install from a published distribution")
      return
    }
    #expect(distribution.repoID == "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit")
    #expect(distribution.revision == "ad8151a34a5b5fd40c581e16fa526577315ac5df")
    #expect(distribution.approximateDownloadBytes == 104_002_771_089)
    #expect(descriptor.approximateDownloadBytes == 104_002_771_089)
    #expect(descriptor.supportsRemoteInstall)
  }

  /// The abliterated pair. Each shares its base counterpart's
  /// `sourceIndexSHA256` deliberately — abliteration rewrites weight *values*
  /// and leaves the tensor index untouched — so `weightsSHA256` is the only
  /// field that separates them, and the assertion that it differs from the
  /// base's is the point of the test rather than a formality.
  @MainActor
  @Test func qwenAbliteratedInstallDescriptorsMatchPinnedAudit() {
    let qwen36 = AppModelInstallDescriptor.qwen3_6_abliterated
    #expect(qwen36.displayName == "Qwen 3.6 35B-A3B 4-bit (Abliterated)")
    #expect(qwen36.shortName == "Qwen 3.6 Abl. 4-bit")
    #expect(qwen36.repoID == "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated")
    #expect(qwen36.sourceIndexSHA256
              == AppModelInstallDescriptor.qwen3_6.sourceIndexSHA256)
    #expect(qwen36.weightsSHA256
              == "f6862341c9688e234c682cef186af5a92445be1dbcdd36d59637338634d311bd")
    #expect(qwen36.weightsSHA256 != AppModelInstallDescriptor.qwen3_6.weightsSHA256)
    #expect(qwen36.routedExpertBits == 4)
    #expect(qwen36.installedBytes == 20_059_538_761)
    #expect(qwen36.rangeStagingBytes == 0)
    #expect(qwen36.reserveBytes == 1_073_741_824)
    #expect(qwen36.requiredFreeBytes == 21_133_280_585)
    #expect(qwen36.architecture == .qwen3_6_35B_A3B)

    guard case .finchDistribution(let qwen36Distribution) = qwen36.installRoute else {
      Issue.record("Qwen 3.6 abliterated should install from a published distribution")
      return
    }
    #expect(qwen36Distribution.repoID
              == "haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated")
    #expect(qwen36Distribution.revision
              == "e1998dcbf984e2c3e8e38eab5eacd3b8ed0c93de")
    #expect(qwen36Distribution.approximateDownloadBytes == 20_059_520_830)

    // The abliterated 3-bit twin: same source index as its base counterpart
    // (asserted as this block's premise), same weights digest as its own 4-bit
    // sibling, separated from both by the width.
    let qwen36ThreeBit = AppModelInstallDescriptor.qwen3_6_abliterated_3bit
    #expect(qwen36ThreeBit.displayName == "Qwen 3.6 35B-A3B 3-bit (Abliterated)")
    #expect(qwen36ThreeBit.shortName == "Qwen 3.6 Abl. 3-bit")
    #expect(qwen36ThreeBit.sourceIndexSHA256
              == AppModelInstallDescriptor.qwen3_6_3bit.sourceIndexSHA256,
            "premise: base and abliterated 3.6 share the source index hash")
    #expect(qwen36ThreeBit.weightsSHA256 == qwen36.weightsSHA256)
    #expect(qwen36ThreeBit.routedExpertBits == 3)
    #expect(qwen36ThreeBit.architecture == .qwen3_6_35B_A3B)

    guard case .finchDistribution(let qwen36ThreeBitDistribution) = qwen36ThreeBit.installRoute else {
      Issue.record("Qwen 3.6 abliterated 3-bit should install from a published distribution")
      return
    }
    #expect(qwen36ThreeBitDistribution.repoID
              == "haihengh/Qwen3.6-35B-A3B-finch-3bit-abliterated")
    #expect(qwen36ThreeBitDistribution.revision == "PENDING-UPLOAD-REPLACE-ME")
    #expect(qwen36ThreeBitDistribution.approximateDownloadBytes == 16_032_957_550)

    let qwen38 = AppModelInstallDescriptor.qwen3_8_abliterated
    #expect(qwen38.displayName == "Qwen 3.8 Flash-Next 125B 4-bit (Abliterated)")
    #expect(qwen38.shortName == "Qwen 3.8 Abl. 4-bit")
    #expect(qwen38.repoID == "windowsxp811203/Qwen3.8-Flash-Next-Abliterated")
    #expect(qwen38.sourceIndexSHA256
              == AppModelInstallDescriptor.qwen3_8.sourceIndexSHA256)
    #expect(qwen38.weightsSHA256
              == "6af82b557d8207e470f50c1fba5dbb14e7ff4b48139a5b0b2e4f8ac56d188acb")
    #expect(qwen38.weightsSHA256 != AppModelInstallDescriptor.qwen3_8.weightsSHA256)
    #expect(qwen38.routedExpertBits == 4)
    #expect(qwen38.installedBytes == 104_002_831_190)
    #expect(qwen38.rangeStagingBytes == 0)
    #expect(qwen38.reserveBytes == 1_073_741_824)
    #expect(qwen38.requiredFreeBytes == 105_076_573_014)
    #expect(qwen38.architecture == .qwen3_8_flashNext_125B)

    guard case .finchDistribution(let qwen38Distribution) = qwen38.installRoute else {
      Issue.record("Qwen 3.8 abliterated should install from a published distribution")
      return
    }
    #expect(qwen38Distribution.repoID
              == "haihengh/Qwen3.8-Flash-Next-125B-finchmoe-4bit-ple4bit-abliterated")
    #expect(qwen38Distribution.revision
              == "d19beab26664be0e3becd66322d71631cec63ddb")
    #expect(qwen38Distribution.approximateDownloadBytes == 104_002_771_089)
  }

  /// The four new 3.8 width rows: 3-bit and 2-bit, base and abliterated. The
  /// payloads are the measured `manifest.files` sums (abliterated twins share
  /// their base sibling's payload byte-for-byte, which is why the abliterated
  /// rows can pin a size before their uploads exist); revisions are
  /// placeholders until each upload commits.
  @MainActor
  @Test func qwen38PerWidthDescriptorsMatchPinnedAudit() {
    let qwen38ThreeBit = AppModelInstallDescriptor.qwen3_8_3bit
    #expect(qwen38ThreeBit.displayName == "Qwen 3.8 Flash-Next 125B 3-bit")
    #expect(qwen38ThreeBit.shortName == "Qwen 3.8 125B 3-bit")
    #expect(qwen38ThreeBit.weightsSHA256
              == AppModelInstallDescriptor.qwen3_8.weightsSHA256)
    #expect(qwen38ThreeBit.routedExpertBits == 3)
    #expect(qwen38ThreeBit.installedBytes == 89_104_658_029)
    #expect(qwen38ThreeBit.requiredFreeBytes == 90_178_399_853)
    #expect(qwen38ThreeBit.architecture == .qwen3_8_flashNext_125B)
    guard case .finchDistribution(let threeBitDistribution) = qwen38ThreeBit.installRoute else {
      Issue.record("Qwen 3.8 3-bit should install from a published distribution")
      return
    }
    #expect(threeBitDistribution.repoID
              == "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple3bit")
    #expect(threeBitDistribution.revision == "80a63e6294655af8a9424ce1307478d294d9e463")
    #expect(threeBitDistribution.approximateDownloadBytes == 89_104_597_905)

    let qwen38TwoBit = AppModelInstallDescriptor.qwen3_8_2bit
    #expect(qwen38TwoBit.displayName == "Qwen 3.8 Flash-Next 125B 2-bit")
    #expect(qwen38TwoBit.shortName == "Qwen 3.8 125B 2-bit")
    #expect(qwen38TwoBit.weightsSHA256
              == AppModelInstallDescriptor.qwen3_8.weightsSHA256)
    #expect(qwen38TwoBit.routedExpertBits == 2)
    #expect(qwen38TwoBit.installedBytes == 73_803_784_477)
    #expect(qwen38TwoBit.requiredFreeBytes == 74_877_526_301)
    guard case .finchDistribution(let twoBitDistribution) = qwen38TwoBit.installRoute else {
      Issue.record("Qwen 3.8 2-bit should install from a published distribution")
      return
    }
    #expect(twoBitDistribution.repoID
              == "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple2bit")
    #expect(twoBitDistribution.approximateDownloadBytes == 73_803_724_449)

    let qwen38AbliteratedThreeBit = AppModelInstallDescriptor.qwen3_8_abliterated_3bit
    #expect(qwen38AbliteratedThreeBit.displayName
              == "Qwen 3.8 Flash-Next 125B 3-bit (Abliterated)")
    #expect(qwen38AbliteratedThreeBit.weightsSHA256
              == AppModelInstallDescriptor.qwen3_8_abliterated.weightsSHA256)
    #expect(qwen38AbliteratedThreeBit.routedExpertBits == 3)
    guard case .finchDistribution(let ablThreeBitDistribution) = qwen38AbliteratedThreeBit.installRoute else {
      Issue.record("Qwen 3.8 abliterated 3-bit should install from a published distribution")
      return
    }
    #expect(ablThreeBitDistribution.repoID
              == "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple3bit-abliterated")
    #expect(ablThreeBitDistribution.approximateDownloadBytes == 89_104_597_905)

    let qwen38AbliteratedTwoBit = AppModelInstallDescriptor.qwen3_8_abliterated_2bit
    #expect(qwen38AbliteratedTwoBit.displayName
              == "Qwen 3.8 Flash-Next 125B 2-bit (Abliterated)")
    #expect(qwen38AbliteratedTwoBit.weightsSHA256
              == AppModelInstallDescriptor.qwen3_8_abliterated.weightsSHA256)
    #expect(qwen38AbliteratedTwoBit.routedExpertBits == 2)
    guard case .finchDistribution(let ablTwoBitDistribution) = qwen38AbliteratedTwoBit.installRoute else {
      Issue.record("Qwen 3.8 abliterated 2-bit should install from a published distribution")
      return
    }
    #expect(ablTwoBitDistribution.repoID
              == "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple2bit-abliterated")
    #expect(ablTwoBitDistribution.approximateDownloadBytes == 73_803_724_449)
  }

  /// Cross-check across the whole list, so a copy-paste between the width rows
  /// cannot slip through the per-entry audits: a row that carries a
  /// distribution must quote the same payload the distribution does and must
  /// hold back the reserve; a row without one must do neither.
  @MainActor
  @Test func everyDescriptorReserveMatchesItsRoute() {
    for descriptor in AppModelInstallDescriptor.installable {
      if let distribution = descriptor.finchDistribution {
        #expect(descriptor.approximateDownloadBytes == distribution.approximateDownloadBytes,
                "\(descriptor.displayName) quotes a payload its distribution does not")
        #expect(descriptor.reserveBytes == 1_073_741_824,
                "\(descriptor.displayName) downloads but holds back no reserve")
      } else {
        #expect(descriptor.approximateDownloadBytes == 0,
                "\(descriptor.displayName) has no distribution but quotes a download size")
        #expect(descriptor.reserveBytes == 0,
                "\(descriptor.displayName) has no distribution but holds back a reserve")
      }
    }
  }

  @MainActor
  @Test func insufficientSpaceDisablesInstallAndExposesShortfall() {
    let requirement = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 100,
      availableBytes: 40)
    let installer = MockModelInstallerClient(requirement: requirement)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("space"),
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(model.installReadiness == .insufficientSpace(requirement))
    #expect(model.installRequirement?.shortfallBytes == 60)
    #expect(!model.canInstallModel)
  }

  @MainActor
  @Test func installProgressUpdatesStatusAndByteCounts() async throws {
    let directory = temporaryInstallPath("progress")
    let installer = MockModelInstallerClient(
      events: [
        .checking,
        .copyingPayload(
          reusedBytes: 1,
          downloadedThisRunBytes: 3,
          totalBytes: 10),
      ], holdOpen: true)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      model.installState == .copyingPayload(
        reusedBytes: 1,
        downloadedThisRunBytes: 3,
        totalBytes: 10)
    }
    #expect(model.installDownloadedBytes == 4)
    #expect(model.installTotalBytes == 10)
    #expect(model.installProgressFraction == 0.4)
    #expect(model.presentation.label == "Downloading model")
    model.cancelInstall()
    try await waitUntil { model.installState == .cancelled }
  }

  @MainActor
  @Test func installCompletionStopsUnloaded() async throws {
    let requestedDirectory = temporaryInstallPath("requested")
    let completedDirectory = try makeCompleteModelInstall("complete")
    defer { try? FileManager.default.removeItem(at: completedDirectory) }
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(events: [.installed(completedDirectory)])
    let model = AppModel(
      modelDirectory: requestedDirectory,
      client: client,
      installer: installer)

    model.installModel()
    try await waitUntil {
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL)
    }
    #expect(
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL))
    #expect(model.modelPathText == completedDirectory.standardizedFileURL.path)
    #expect(model.loadState == .notLoaded)
    #expect(model.canLoadModel)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func installFailureDoesNotAttemptLoad() async throws {
    struct SyntheticError: Error {}
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(failure: SyntheticError())
    let model = AppModel(
      modelDirectory: temporaryInstallPath("failure"),
      client: client,
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }
    #expect(model.loadState == .notLoaded)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func networkFailureWithSavedProgressLeavesResumeEnabled() async throws {
    struct NetworkFailure: Error {}
    let directory = temporaryInstallPath("network-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: NetworkFailure()))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func diskFailureCanResumeAfterSpaceRecheck() async throws {
    let directory = temporaryInstallPath("disk-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: error))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }
    #expect(!model.canInstallModel)

    model.recheckModelAtCurrentLocation()

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func invalidSavedDownloadsRemainDiscardOnly() throws {
    let descriptor = AppModelInstallDescriptor.default
    for incompatible in [false, true] {
      let directory = temporaryInstallPath(
        incompatible ? "incompatible-checkpoint" : "corrupt-checkpoint")
      let paths = try makeSavedDownload(at: directory)
      defer { cleanUpSavedDownload(paths) }
      if incompatible {
        try RemoteInstallCheckpoint(
          repoID: "other/model",
          requestedRevision: descriptor.revision,
          resolvedCommit: String(repeating: "a", count: 40),
          sourceIndexSHA256: String(repeating: "b", count: 64),
          planFingerprint: String(repeating: "c", count: 64),
          totalSourceBytes: 1
        ).write(
          to: paths.checkpointFile,
          parentDirectory: paths.parentDirectory)
      } else {
        try Data("{}".utf8).write(
          to: URL(fileURLWithPath: paths.checkpointFile))
      }
      let model = AppModel(
        modelDirectory: directory,
        client: MockLifecycleInferenceClient(),
        installer: RepackModelInstallerClient(descriptor: descriptor))

      #expect(!model.canInstallModel)
      #expect(model.canDiscardModelDownload)
      guard case .failed = model.installReadiness else {
        Issue.record("invalid checkpoint did not fail readiness")
        continue
      }
    }
  }

  @MainActor
  @Test func diskFailureKeepsExactRequirementAndShortfall() async throws {
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let installer = MockModelInstallerClient(failure: error)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("disk-failure"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }

    let expected = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 120,
      availableBytes: 45)
    #expect(model.installReadiness == .insufficientSpace(expected))
    #expect(model.installRequirement?.shortfallBytes == 75)
  }

  @MainActor
  @Test func cancelInstallWaitsForAcknowledgementAndAllowsRetry() async throws {
    let installer = MockModelInstallerClient(
      events: [.downloadingMetadata],
      holdOpen: true,
      delayCancellationAcknowledgement: true)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("cancel"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()
    try await waitUntil { model.installState == .downloadingMetadata }

    model.cancelInstall()
    #expect(installer.cancelCalled)
    try await waitUntil { installer.cancellationAcknowledgementPending }
    #expect(model.installState == .cancelling)
    #expect(!model.canInstallModel)

    await installer.releaseCancellationAcknowledgement()
    try await waitUntil { model.installState == .cancelled }

    #expect(model.loadState == .notLoaded)
    #expect(model.canInstallModel)
  }

  private func temporaryInstallPath(_ tag: String) -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("finchmoe-app-install-\(tag)-\(UUID().uuidString).finch")
  }

  private func makeSavedDownload(at directory: URL) throws -> RemoteInstallPaths {
    let paths = try RemoteInstallPaths(outputDirectory: directory.path)
    try FileManager.default.createDirectory(
      atPath: paths.partialDirectory,
      withIntermediateDirectories: true)
    return paths
  }

  private func cleanUpSavedDownload(_ paths: RemoteInstallPaths) {
    for path in [
      paths.finalDirectory,
      paths.partialDirectory,
      paths.checkpointFile,
      paths.lockFile,
    ] {
      try? FileManager.default.removeItem(atPath: path)
    }
  }

  @MainActor
  private func waitUntil(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for condition")
  }

}
