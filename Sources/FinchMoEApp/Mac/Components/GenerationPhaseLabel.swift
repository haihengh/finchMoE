import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

/// The status HUD's phase readout: a spinner while the app is busy, a pulsing
/// dot and live token counts while a prompt is being prefilled, and a steady
/// dot while the reply is decoding.
///
/// Shared with the phone shell, whose header otherwise only carries the model
/// badge — the prefill fraction ("Prefill (128/514, 64 cached)") is the status
/// the Mac window has always shown and the phone did not.
struct GenerationPhaseLabel: View {
    let model: AppModel
    @Environment(\.finchAccentColor) private var accentColor

    init(model: AppModel) {
        self.model = model
    }

    var body: some View {
        HStack(spacing: 6) {
            switch content {
            case .loading(let label):
                ProgressView().controlSize(.mini)
                Text(label)
            case .pulse(let label):
                PulsingDot()
                Text(label)
                    .monospacedDigit()
                    .contentTransition(.numericText())
            case .steady(let label):
                Circle().fill(accentColor).frame(width: 7, height: 7)
                Text(label).contentTransition(.opacity)
            case .quiet(let label):
                Text(label)
                    .foregroundStyle(.secondary)
                    .contentTransition(.opacity)
            }
        }
        .font(.caption.weight(.medium))
        .lineLimit(1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Model status")
        .accessibilityValue(model.presentation.label)
    }

    private enum Content {
        case loading(String)
        case pulse(String)
        case steady(String)
        case quiet(String)
    }

    private var content: Content {
        let presentation = model.presentation
        if presentation.showsActivity { return .loading(presentation.label) }
        if model.isRunning && model.phase == .prefill { return .pulse(presentation.label) }
        if model.isRunning && model.phase == .decode { return .steady(presentation.label) }
        return .quiet(presentation.label)
    }
}

private struct PulsingDot: View {
    @Environment(\.finchAccentColor) private var accentColor

    var body: some View {
        Circle()
            .fill(accentColor)
            .frame(width: 7, height: 7)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
