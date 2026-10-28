import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

struct ModelInstallView: View {
    let model: AppModel
    #if !canImport(AppKit)
    @State private var showsFolderImporter = false
    #endif
    @State private var showingDiscardConfirmation = false
    @Environment(\.finchAccentColor) private var accentColor
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    init(model: AppModel) {
        self.model = model
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                storageCard
                progressArea
                actions
            }
            .frame(maxWidth: 560)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
        .confirmationDialog(
            "Discard the saved model download?",
            isPresented: $showingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Download", role: .destructive) {
                model.discardModelDownload()
            }
            Button("Keep Download", role: .cancel) {}
        } message: {
            Text("Downloaded ranges will be removed. The installed model, if any, is preserved.")
        }
        #if !canImport(AppKit)
        // iOS has no NSOpenPanel; the folder picker is SwiftUI's fileImporter.
        .fileImporter(
            isPresented: $showsFolderImporter,
            allowedContentTypes: [.folder]
        ) { result in
            switch result {
            case .success(let url):
                // A folder chosen from another provider (iCloud Drive, another
                // app's container) needs the security scope for the session.
                _ = url.startAccessingSecurityScopedResource()
                model.setModelURL(url)
            case .failure(let error):
                model.reportModelSelectionError(error.localizedDescription)
            }
        }
        #endif
    }

    private var identity: some View {
        VStack(spacing: 10) {
            Image(systemName: "externaldrive.badge.plus")
                .font(.system(.largeTitle, design: .rounded))
                .foregroundStyle(accentColor)
                .accessibilityHidden(true)
            Text("Model required")
                .font(.title.bold())
                .accessibilityHeading(.h1)
            Text("FinchMoE needs \(model.installDescriptor.displayName) before it can generate.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if !model.installDescriptor.supportsRemoteInstall {
                Text("This preset has no published download. Choose an existing .finch directory, or pick a downloadable preset.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
    }

    private var storageCard: some View {
        VStack(spacing: 12) {
            if let requirement = model.installRequirement {
                StorageRow(label: "Space required",
                           value: MetricFormat.storage(requirement.requiredBytes))
                StorageRow(label: "Available on this device",
                           value: MetricFormat.storage(requirement.availableBytes))
                capacityStatus(requirement)
            } else if case .failed(let message) = model.installReadiness {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(message).foregroundStyle(.secondary)
                    Spacer()
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(readinessLabel)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            Text(model.modelPathText)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("model-path")
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 20)
                .fill(FinchPlatformColors.controlBackground)
                .overlay {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    @ViewBuilder
    private func capacityStatus(_ requirement: AppModelInstallRequirement) -> some View {
        if requirement.canInstall {
            Label("Enough space to install", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Label("\(MetricFormat.storage(requirement.shortfallBytes)) more is required",
                  systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var progressArea: some View {
        if let message = model.modelSelectionError {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
        } else if !model.isInstallingModel,
                  case .partial(let message) = model.installationStatus {
            // The probe knows why a directory is not usable (a file missing,
            // a checkpoint mismatch); without this line the screen only shows
            // a disabled Download and no reason for it.
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
        }
        if model.isInstallingModel {
            VStack(alignment: .leading, spacing: 8) {
                if let fraction = model.installProgressFraction,
                   let downloaded = model.installDownloadedBytes,
                   let total = model.installTotalBytes {
                    ProgressView(value: fraction)
                        .accessibilityLabel("Model download")
                        .accessibilityValue(Text(accessibleProgressValue(fraction: fraction)))
                    HStack {
                        Text("Downloaded \(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))")
                        Spacer()
                        Text(MetricFormat.percent(fraction * 100))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 16) {
                        if let reused = model.installReusedBytes, reused > 0 {
                            Text("Reused \(MetricFormat.storage(reused)) from the saved download")
                                .font(.caption)
                        }
                        Spacer(minLength: 16)
                        if let eta = model.installETAText {
                            Text(eta)
                                .font(.caption.monospacedDigit())
                        }
                    }
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                } else {
                    Text(model.presentation.label)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
        } else if case .cancelled = model.installState {
            Label("Download paused", systemImage: "pause.circle")
                .foregroundStyle(.secondary)
        } else if case .failed(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
        } else if case .recoverable(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
        }
    }

    private var isLocalOnly: Bool {
        !model.installDescriptor.supportsRemoteInstall
    }

    private var actions: some View {
        Group {
            if model.isInstallingModel {
                Button("Cancel", action: model.cancelInstall)
                    .buttonStyle(.bordered)
                    .keyboardShortcut(.cancelAction)
                    .disabled(!model.canCancelInstall)
            } else if horizontalSizeClass == .compact {
                // A phone cannot fit Check Again + Choose Local + Download in
                // one row; the last button used to be clipped off-screen, which
                // left a not-installed model with no tappable Download.
                VStack(spacing: 10) {
                    if isLocalOnly {
                        chooseAction.frame(maxWidth: .infinity)
                    } else {
                        downloadAction.frame(maxWidth: .infinity)
                    }
                    if model.hasPartialModelDownload {
                        discardAction.frame(maxWidth: .infinity)
                    }
                    checkAction.frame(maxWidth: .infinity)
                    if !isLocalOnly {
                        chooseAction.frame(maxWidth: .infinity)
                    }
                }
            } else {
                HStack(spacing: 12) {
                    if model.hasPartialModelDownload {
                        discardAction
                    }
                    checkAction
                    chooseAction
                    if !isLocalOnly {
                        downloadAction
                    }
                }
            }
        }
        .controlSize(.large)
    }

    private var downloadAction: some View {
        Button(model.hasPartialModelDownload ? "Resume" : "Download",
               action: model.installModel)
            .buttonStyle(.borderedProminent)
            .disabled(!model.canInstallModel)
    }

    private var discardAction: some View {
        Button("Discard Download", role: .destructive) {
            showingDiscardConfirmation = true
        }
        .buttonStyle(.bordered)
        .disabled(!model.canDiscardModelDownload)
    }

    private var checkAction: some View {
        Button("Check Again", action: model.recheckModelAtCurrentLocation)
            .buttonStyle(.bordered)
            .disabled(model.isInstallingModel)
    }

    @ViewBuilder
    private var chooseAction: some View {
        if isLocalOnly {
            // A probe-only row has no published distribution; choosing an
            // existing directory is its only install path, so it gets the
            // primary weight instead of a disabled Download.
            chooseButton.buttonStyle(.borderedProminent)
        } else {
            chooseButton.buttonStyle(.bordered)
        }
    }

    private var chooseButton: some View {
        Button {
            chooseLocalModelDirectory()
        } label: {
            Label("Choose Local", systemImage: "folder")
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var readinessLabel: String {
        switch model.installReadiness {
        case .checking:
            return "Checking available space"
        case .failed(let message):
            return message
        case .ready, .insufficientSpace:
            return "Checking available space"
        }
    }

    private func accessibleProgressValue(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.installETAText else { return percent }
        return "\(percent), \(eta)"
    }

    private func chooseLocalModelDirectory() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            _ = url.startAccessingSecurityScopedResource()
            model.setModelURL(url)
        }
        #else
        showsFolderImporter = true
        #endif
    }
}

private struct StorageRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
    }
}
