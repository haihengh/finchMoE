import FinchMoEMacPresentation
import SwiftUI

/// The little info control beside a settings row.
///
/// A plain `.help` tooltip is desktop-only, so this opens a real popover: the
/// row explains what the setting means and what changing it does, on both the
/// Mac inspector and the iOS Settings sheet. On a compact iPhone width the
/// popover is forced to stay a bubble rather than adapting to a full sheet.
struct SettingInfoButton: View {
    let title: String
    let detail: String
    @State private var isPresented = false

    init(title: String, detail: String) {
        self.title = title
        self.detail = detail
    }

    var body: some View {
        Button {
            isPresented = true
        } label: {
            Image(systemName: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 22, height: 22)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .accessibilityLabel("About \(title)")
        .popover(isPresented: $isPresented, arrowEdge: .top) {
            let content = VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
            .frame(width: 300)
            #if canImport(UIKit)
            content.presentationCompactAdaptation(.popover)
            #else
            content
            #endif
        }
    }
}
