import AppKit
import FinchMoEAppCore
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let iconURL = Bundle.module.url(
            forResource: "finchmoe-app-icon",
            withExtension: "png"
        ), let icon = NSImage(contentsOf: iconURL) {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct FinchMoEMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var model: AppModel

    init() {
        #if DEBUG
        SnapshotHarness.runIfRequested()
        #endif
        _model = State(initialValue: AppModel(
            client: DecodeServiceInferenceClient(),
            settingsPersistenceEnabled: true,
            chatStore: ChatSessionFileStore(
                fileURL: ChatSessionFileStore.defaultFileURL())))
    }

    var body: some Scene {
        Window("FinchMoE", id: "main") {
            RootView(model: model)
                .frame(minWidth: 1000, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1240, height: 800)
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("Chat") {
                Button("New Chat", action: model.newSession)
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(!model.canStartNewSession)
                Button("Regenerate Reply", action: model.regenerateLastReply)
                    .keyboardShortcut("r", modifiers: .command)
                    .disabled(!model.canRegenerateLastReply)
                Button("Clear Chat", action: model.clearOutput)
                    .disabled(model.isRunning || !model.activeSession.hasMessages)
            }
            CommandMenu("Generation") {
                Button("Cancel Generation") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)
                Button("Cancel Model Installation") { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }
            CommandMenu("Model") {
                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)
                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)
                Button("Unload Model", action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
            }
            CommandMenu("Settings") {
                Picker("Send Message With", selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("After Sending", selection: sentPromptBehaviorBinding) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
            }
        }
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding {
            model.newlineShortcut
        } set: { shortcut in
            model.setNewlineShortcut(shortcut)
        }
    }

    private var sentPromptBehaviorBinding: Binding<AppSentPromptBehavior> {
        Binding {
            model.sentPromptBehavior
        } set: { behavior in
            model.setSentPromptBehavior(behavior)
        }
    }
}
