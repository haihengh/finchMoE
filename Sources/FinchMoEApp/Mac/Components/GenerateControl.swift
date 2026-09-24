import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

struct GenerateControl: View {
    let model: AppModel

    var body: some View {
        if model.isRunning {
            stopButton
        } else {
            sendButton
        }
    }

    private var sendButton: some View {
        Button {
            model.run()
        } label: {
            Image(systemName: "arrow.up")
                .font(.system(size: 14, weight: .bold))
                .frame(width: 30, height: 30)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(FinchMoEMacTheme.accentColor, in: .circle)
        .overlay {
            Circle().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .opacity(model.canRun ? 1 : 0.45)
        .help("Send message (\u{2318}\u{21A9})")
        .accessibilityLabel("Send message")
    }

    private var stopButton: some View {
        Button {
            model.cancel()
        } label: {
            HStack(spacing: 7) {
                if !model.isCancellationPending, model.phase == .decode {
                    Text(MetricFormat.rate(model.liveTokensPerSecond))
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                }
                Image(systemName: "stop.fill")
                    .font(.system(size: 11, weight: .bold))
            }
            .padding(.horizontal, 10)
            .frame(minWidth: 30, minHeight: 30)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(FinchMoEMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.cancelAction)
        .disabled(!model.canCancel)
        .help("Stop generating")
        .accessibilityLabel("Stop generating")
        .animation(.smooth(duration: 0.2), value: model.presentation.label)
    }
}
