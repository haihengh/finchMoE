import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

struct StatusHUDView: View {
    let model: AppModel
    @Binding var showsSidebar: Bool
    /// False once the session list sits at the window's left edge under the
    /// traffic lights, so the strip does not leave room for them twice.
    var clearsTrafficLights: Bool
    @Environment(\.finchAccentColor) private var accentColor

    init(
        model: AppModel,
        showsSidebar: Binding<Bool>,
        clearsTrafficLights: Bool = true
    ) {
        self.model = model
        self._showsSidebar = showsSidebar
        self.clearsTrafficLights = clearsTrafficLights
    }

    var body: some View {
        HStack(spacing: 10) {
            sidebarToggle
            strip
        }
        .padding(.top, 10)
        .padding(.leading, clearsTrafficLights ? 84 : 12)
        .padding(.trailing, 20)
    }

    private var sidebarToggle: some View {
        Button {
            showsSidebar.toggle()
        } label: {
            Image(systemName: "sidebar.left")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(showsSidebar ? accentColor : .secondary)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
                .background(.regularMaterial, in: Circle())
                .overlay {
                    Circle().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .buttonStyle(.plain)
        .help(showsSidebar ? "Hide chats" : "Show chats")
        .accessibilityLabel(showsSidebar ? "Hide chat list" : "Show chat list")
    }

    private var strip: some View {
        HStack(spacing: 12) {
            ModelStatusBadge(model: model)
            Divider().frame(height: 16)
            GenerationPhaseLabel(model: model)
            Spacer(minLength: 12)
            if showsMetrics {
                HUDMetricView(value: rateText, label: "tok/s", animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: "tokens", animated: !model.isRunning)
                HUDMetricView(value: memoryText, label: "memory", animated: !model.isRunning)
            }
        }
        .frame(height: 30)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background {
            Capsule()
                .fill(FinchPlatformColors.controlBackground)
                .overlay {
                    Capsule().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        #if canImport(AppKit)
        // `WindowDragGesture` is macOS-only; the iOS shell does not use this
        // view, and there is no window frame to drag there.
        .gesture(WindowDragGesture())
        #endif
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let d = model.diagnostics { return "\(d.generatedTokens)" }
        return "\u{2014}"
    }

    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    private var showsMetrics: Bool {
        model.loadState.isReady || model.isRunning || model.diagnostics != nil
    }
}
