import FinchMoEMacPresentation
import SwiftUI

/// Draws one parsed answer.
///
/// Block layout is SwiftUI's, inline styling comes from Foundation's markdown
/// parser, so bold spans, inline code and links keep the formatting the model
/// actually wrote while headings, lists, code panels and tables get real
/// chat-app layout instead of one wall of text.
struct MarkdownBlockView: View {
    let document: MarkdownDocument

    /// A table is drawn a little smaller than body text so that a four-column
    /// answer still fits the bubble; wider ones scroll.
    private static let tableFontSize: CGFloat = 12.5

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
                .padding(.top, 4)

        case .code(let code):
            codePanel(code)

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

        case .table(let header, let rows, let alignments):
            table(header: header, rows: rows, alignments: alignments)
        }
    }

    /// Code and the pictures drawn out of characters in it.
    ///
    /// Nothing here wraps: a chart is aligned by column, so folding one of its
    /// lines onto the next one destroys it. Text that fits fills the panel;
    /// text that does not scroll sideways.
    private func codePanel(_ code: String) -> some View {
        ScrollView(.horizontal) {
            Text(code)
                .font(.system(size: 12.5, design: .monospaced))
                .textSelection(.enabled)
                .fixedSize(horizontal: true, vertical: false)
                .padding(10)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .textBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private func table(
        header: [AttributedString],
        rows: [[AttributedString]],
        alignments: [MarkdownDocument.ColumnAlignment]
    ) -> some View {
        ScrollView(.horizontal) {
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(header.indices, id: \.self) { column in
                        tableCell(
                            header[column],
                            alignment: alignment(alignments, column),
                            isHeader: true,
                            drawsTrailingEdge: column < header.count - 1)
                            .gridColumnAlignment(
                                gridAlignment(alignment(alignments, column)))
                    }
                }
                ForEach(rows.indices, id: \.self) { row in
                    GridRow {
                        ForEach(rows[row].indices, id: \.self) { column in
                            tableCell(
                                rows[row][column],
                                alignment: alignment(alignments, column),
                                isHeader: false,
                                drawsTrailingEdge: column < rows[row].count - 1)
                        }
                    }
                }
            }
            .padding(10)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .textBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    /// The gridlines are drawn by the cells themselves.
    ///
    /// A `Divider()` between rows would be a row of its own that spans every
    /// column, and a view that spans the grid takes whatever width it is
    /// offered — which, inside the horizontal `ScrollView`, is none, so the
    /// table would stop hugging its content. Cell edges with `horizontalSpacing
    /// 0` abut into the same hairlines and cost nothing.
    private func tableCell(
        _ text: AttributedString,
        alignment: MarkdownDocument.ColumnAlignment,
        isHeader: Bool,
        drawsTrailingEdge: Bool
    ) -> some View {
        Text(text)
            .font(.system(
                size: Self.tableFontSize,
                weight: isHeader ? .semibold : .regular))
            .multilineTextAlignment(textAlignment(alignment))
            .fixedSize(horizontal: false, vertical: true)
            // The cell has to take its column's width, not its text's: the
            // gridlines are drawn on the cell's edges, and a cell that hugs
            // its text would draw them mid-column.
            .frame(maxWidth: .infinity, alignment: cellAlignment(alignment))
            .padding(.vertical, 5)
            .padding(.horizontal, 9)
            .background(isHeader ? Color.primary.opacity(0.06) : .clear)
            .overlay(alignment: .trailing) {
                if drawsTrailingEdge {
                    Rectangle()
                        .fill(.separator)
                        .frame(width: 1)
                }
            }
            .overlay(alignment: .bottom) {
                Rectangle()
                    .fill(.separator)
                    .frame(height: isHeader ? 1 : 0.5)
            }
    }

    private func cellAlignment(
        _ alignment: MarkdownDocument.ColumnAlignment
    ) -> Alignment {
        switch alignment {
        case .leading: .leading
        case .center: .center
        case .trailing: .trailing
        }
    }

    private func alignment(
        _ alignments: [MarkdownDocument.ColumnAlignment],
        _ column: Int
    ) -> MarkdownDocument.ColumnAlignment {
        column < alignments.count ? alignments[column] : .leading
    }

    private func gridAlignment(
        _ alignment: MarkdownDocument.ColumnAlignment
    ) -> HorizontalAlignment {
        switch alignment {
        case .leading: .leading
        case .center: .center
        case .trailing: .trailing
        }
    }

    private func textAlignment(
        _ alignment: MarkdownDocument.ColumnAlignment
    ) -> TextAlignment {
        switch alignment {
        case .leading: .leading
        case .center: .center
        case .trailing: .trailing
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
