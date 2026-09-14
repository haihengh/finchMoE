import Foundation
import Darwin
import Metal
import Testing

@testable import FinchMoE

/// `ModelIntegrityPreference.automatic` is the "use the receipt if it is usable,
/// otherwise hash" rule. The dangerous failure mode is not "hashes too little on
/// a bad receipt" — falling back always hashes *more* — it is resolving to
/// `.fullSha256` while leaving `Model.integrityPolicy` at
/// `.sizeCheckTrustedReceipt`, which would skip the lazy layer/PLE hashes *and*
/// validate no receipt, verifying less than either mode alone. Most of these
/// tests exist to make that state impossible to reach unnoticed.
///
/// Every corruption here is a byte *flip*, never a truncation: a flip preserves
/// file size, so the size checks pass and only the SHA-256 gate can catch it.
/// Truncating would let a size check stand in for the hash and the test would
/// pass for the wrong reason.
extension ModelLoaderTests {
  private static func toyDevice() throws -> MTLDevice {
    try #require(MTLCreateSystemDefaultDevice())
  }

  private static func layerURL(in dir: URL) -> URL {
    dir.appendingPathComponent("packed_experts").appendingPathComponent("layer_00.bin")
  }

  /// Zero the receipt's manifest binding, leaving the file present and otherwise
  /// well-formed — the "present but invalid" case that must warn.
  private static func invalidateManifestBinding(directoryURL dir: URL) throws {
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["manifestSha256"] = String(repeating: "0", count: 64)
    }
  }

  // MARK: - The trap

  /// A receipt that is present but unusable must fall back to hashing, and the
  /// fallback must actually hash.
  ///
  /// Two independent detectors on purpose. `integrityPolicy` catches a resolver
  /// that returns the wrong policy; the `checksumMismatch` catches a resolver
  /// that returns `.fullSha256` but whose fallback never reaches the lazy gate.
  /// A half-fix satisfies only one of them.
  @Test func automaticFallsBackToHashingWhenReceiptIsPresentButInvalid() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.invalidateManifestBinding(directoryURL: dir)
    try Self.flipByte(in: Self.layerURL(in: dir), at: 64)
    let device = try Self.toyDevice()

    // The default. Under `.fullSha256` this leg is vacuous, which is why the
    // assertions below pin the outcome too.
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .automatic)

    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.integrityOutcome.isWarning)
    if case .automaticFellBackInvalid(let detail) = model.integrityOutcome {
      #expect(detail.contains("manifest SHA mismatch"))
    } else {
      Issue.record("expected .automaticFellBackInvalid, got \(model.integrityOutcome)")
    }

    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  // MARK: - Absent vs. invalid

  /// A fresh install has no receipt at all. That is normal, so it is silent —
  /// but the layer SHA must still fire.
  @Test func automaticFallsBackSilentlyWhenReceiptIsAbsent() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.flipByte(in: Self.layerURL(in: dir), at: 64)
    let device = try Self.toyDevice()

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .automatic)

    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.integrityOutcome == .automaticFellBackAbsent)
    #expect(!model.integrityOutcome.isWarning)

    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  // MARK: - The payoff

  /// A usable receipt is taken, and taking it really does skip the layer hash —
  /// the corrupt layer loads without complaint.
  @Test func automaticUsesAValidReceiptAndSkipsTheLayerSha() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.flipByte(in: Self.layerURL(in: dir), at: 64)
    let device = try Self.toyDevice()

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .automatic)

    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
    #expect(model.integrityOutcome == .automaticUsedReceipt)
    #expect(!model.integrityOutcome.isWarning)

    _ = try model.routedExpert(layer: 0, expert: 0)
  }

  // MARK: - Explicit requests stay strict

  /// Only `automatic` falls back. Asking for the receipt and not getting a
  /// usable one stays an error.
  @Test func explicitTrustedReceiptStillThrowsOnAnInvalidReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.invalidateManifestBinding(directoryURL: dir)
    let device = try Self.toyDevice()

    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid = error { return true }
      return false
    }
  }

  /// Same resolved policy as the absent-receipt fallback, different outcome —
  /// which is the whole reason the outcome is carried at all. Without it a
  /// caller cannot tell "you asked for hashing" from "I chose hashing for you".
  @Test func explicitFullSha256IsDistinguishableFromAFallback() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try Self.toyDevice()

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .fullSha256)

    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.integrityOutcome == .explicitFullSha256)
    #expect(!model.integrityOutcome.isWarning)
  }

  // MARK: - Presence probe

  /// `load` collapses every failure into `.trustedReceiptInvalid`, so deciding
  /// between "no receipt" (silent) and "broken receipt" (warn) needs a separate
  /// look at the filesystem. Only ENOENT is absent.
  ///
  /// The dangling symlink is the case that tells this apart from
  /// `FileManager.fileExists`: `fileExists` follows the link, finds nothing, and
  /// reports absent — silently swallowing a receipt somebody deliberately
  /// placed. `O_NOFOLLOW` makes it ELOOP, which is a presence signal.
  @Test func isPresentDistinguishesAbsentFromUnusable() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let receiptURL = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)

    #expect(!VerifiedInstallReceiptReader.isPresent(directoryURL: dir))

    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    #expect(VerifiedInstallReceiptReader.isPresent(directoryURL: dir))

    try FileManager.default.removeItem(at: receiptURL)
    try FileManager.default.createSymbolicLink(
      at: receiptURL,
      withDestinationURL: dir.appendingPathComponent("does-not-exist.json"))
    #expect(VerifiedInstallReceiptReader.isPresent(directoryURL: dir))
    // The probe is the stricter of the two, and deliberately so.
    #expect(!FileManager.default.fileExists(atPath: receiptURL.path))

    try FileManager.default.removeItem(at: receiptURL)
    #expect(mkfifo(receiptURL.path, 0o600) == 0)
    #expect(VerifiedInstallReceiptReader.isPresent(directoryURL: dir))

    try FileManager.default.removeItem(at: receiptURL)
    try FileManager.default.createDirectory(at: receiptURL, withIntermediateDirectories: false)
    #expect(VerifiedInstallReceiptReader.isPresent(directoryURL: dir))
  }
}
