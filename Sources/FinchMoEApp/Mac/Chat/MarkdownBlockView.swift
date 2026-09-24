import FinchMoEMacPresentation
import SwiftUI

/// Draws one parsed answer.
///
/// Block layout is SwiftUI's, inline styling comes from Foundation's markdown
/// parser, so bold spans, inline code and links keep the formatting the model
/// actually wrote while headings, lists and code panels get real chat-app
/// layout instead of one wall of text.
struct MarkdownBlockView: View {
    let document: MarkdownDocument

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(document.blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownDocument.Block) -> some View {
        switch block {
        case .paragraph(let text):
            Text(text)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)

        case .heading(let level, let text):
            Text(text)
                .font(.system(size: headingSize(level), weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)

        case .code(let code):
            Text(code)
                .font(.system(size: 12.5, design: .monospaced))
                .fixedSize(horizontal: false, vertical: true)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(nsColor: .textBackgroundColor))
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                        }
                }

        case .quote(let text):
            // The bar is an overlay, not an HStack sibling: a bare Shape is
            // greedy in both axes, so as a sibling it would stretch the row to
            // the full proposed height and leave a huge gap under the quote.
            Text(text)
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.leading, 14)
                .padding(.vertical, 2)
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(.quaternary)
                        .frame(width: 3)
                }

        case .listItem(let marker, let indent, let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(marker)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .frame(width: 22, alignment: .trailing)
                Text(text)
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(indent) * 18)

        case .thematicBreak:
            Divider().padding(.vertical, 2)
        }
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: 19
        case 2: 17
        case 3: 15.5
        default: 14.5
        }
    }
}
