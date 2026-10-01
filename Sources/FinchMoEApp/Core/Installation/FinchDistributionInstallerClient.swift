import Foundation
import FinchMoERepackCore
import Synchronization

/// Installs a published `.finch` distribution from Hugging Face.
///
/// The second of the two install routes. `RepackModelInstallerClient` streams
/// an upstream safetensors checkpoint and builds the `.finch` layout on the way
/// down; this one fetches a layout that already exists, verified against the
/// digests its own manifest declares. The lifecycle, event stream and
/// cancellation mirror the repack client deliberately — the two are the same
/// shape of operation from the UI's point of view, and the app picks between
/// them from `descriptor.installRoute`.
public final class FinchDistributionInstallerClient: AppModelInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        URL,
        @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL
    typealias DiscardRunner = @Sendable (URL) async throws -> Void

    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let descriptor: AppModelInstallDescriptor
    private let distribution: FinchDistribution
    private let runInstall: InstallRunner
    private let runDiscard: DiscardRunner
    private let taskState = InstallTaskState()

    /// Fails for a descriptor that has no published distribution — the caller
    /// is expected to have consulted `installRoute` first, so this is a
    /// programming error rather than a user-facing condition.
    public init?(descriptor: AppModelInstallDescriptor) {
        guard case .finchDistribution(let distribution) = descriptor.installRoute else {
            return nil
        }
        self.descriptor = descriptor
        self.distribution = distribution
        self.runInstall = { outputDirectory, progress in
            let result = try await FinchDistributionDownloader.run(
                source: distribution,
                outputDirectory: outputDirectory.path,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                reserveBytes: descriptor.reserveBytes,
                progress: progress)
            // The receipt names the directory the install was promoted to, so
            // this is the path the app should now probe.
            return URL(fileURLWithPath: result.receiptPath)
                .deletingLastPathComponent()
                .standardizedFileURL
        }
        self.runDiscard = { outputDirectory in
            try FinchDistributionDownloader.discardPartial(
                outputDirectory: outputDirectory.path)
        }
    }

    init(descriptor: AppModelInstallDescriptor,
         distribution: FinchDistribution,
         runInstall: @escaping InstallRunner,
         runDiscard: @escaping DiscardRunner = { _ in }) {
        self.descriptor = descriptor
        self.distribution = distribution
        self.runInstall = runInstall
        self.runDiscard = runDiscard
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        let remaining = descriptor.approximateDownloadBytes
            - min(descriptor.approximateDownloadBytes,
                  reusedBytes(outputDirectory: outputDirectory))
        let requirement = try DiskSpaceChecker.assess(
            path: outputDirectory.path,
            bytes: remaining,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: requirement.path,
                                          requiredBytes: requirement.requiredBytes,
                                          availableBytes: requirement.availableBytes)
    }

    /// Bytes a previous run already landed, for the disk estimate only.
    ///
    /// Size and presence are enough here because this number decides whether to
    /// *start* a download, and the downloader re-hashes every file it adopts
    /// before trusting it. A checkpoint that overstates reuse costs a disk check
    /// that passes too easily and then fails loudly mid-install; it can never
    /// produce a wrong install.
    private func reusedBytes(outputDirectory: URL) -> UInt64 {
        guard let checkpoint = try? FinchDistributionDownloader.inspectPersistentInstall(
            outputDirectory: outputDirectory.path) else { return 0 }
        let partial = outputDirectory.path + ".partial"
        var reused: UInt64 = 0
        for entry in checkpoint.entries {
            let path = (partial as NSString).appendingPathComponent(entry.relativePath)
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  let size = attributes[.size] as? UInt64,
                  size == entry.size else { continue }
            let sum = reused.addingReportingOverflow(entry.size)
            guard !sum.overflow else { return reused }
            reused = sum.partialValue
        }
        return reused
    }

    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    continuation.yield(.checking)
                    let completedDirectory = try await runInstall(outputDirectory) { progress in
                        continuation.yield(RepackModelInstallerClient.event(for: progress))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(completedDirectory))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()

            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    public func discardPartialInstall(outputDirectory: URL) async throws {
        let directory = outputDirectory.standardizedFileURL
        try await Task.detached(priority: .utility) { [runDiscard] in
            try await runDiscard(directory)
        }.value
    }
}
