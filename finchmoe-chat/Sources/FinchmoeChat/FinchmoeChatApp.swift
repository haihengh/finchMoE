import SwiftUI

@main
struct FinchmoeChatApp: App {
    @StateObject private var model = ChatModel()

    var body: some Scene {
        WindowGroup("FinchMoE Chat") {
            ChatView()
                .environmentObject(model)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1100, height: 700)
    }
}
