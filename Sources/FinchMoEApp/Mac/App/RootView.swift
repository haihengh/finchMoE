import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var showsSidebar = true

    var body: some View {
        HStack(spacing: 0) {
            if showsSidebar {
                ChatSessionSidebar(model: model)
                    .frame(maxHeight: .infinity)
                    .transition(.move(edge: .leading))
            }

            Divider()

            primaryContent
                .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            InspectorView(model: model)
                .frame(width: 320)
                .frame(maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
        }
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: FinchMoEMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(FinchMoEMacTheme.accentColor)
        .animation(.smooth(duration: 0.25), value: showsSidebar)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .animation(.smooth(duration: 0.2), value: model.presentation.conversationAction)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            StatusHUDView(
                model: model,
                showsSidebar: $showsSidebar,
                clearsTrafficLights: !showsSidebar)
        }
    }

    /// Chat transcript above, composer pinned below: the ordinary shape of a
    /// chat window, and the reason the old floating-chrome measurement could
    /// go away.
    private var conversationView: some View {
        VStack(spacing: 0) {
            ChatTranscriptView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            conversationChrome
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
    }
}
