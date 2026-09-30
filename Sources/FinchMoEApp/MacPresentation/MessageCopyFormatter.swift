import Foundation

/// Turns a rendered answer back into text for the clipboard.
///
/// A bubble offers two things, and they are not the same thing. Plain text is
/// the answer as the reader sees it — the markdown syntax consumed rather than
/// stripped off the source, so the `**` around a word and the `#` before a
/// title are gone while a table still lines up in columns. Markdown is the
/// answer as the model wrote it, which is what pastes back into a document
/// that renders markdown.
///
/// Reading plain text off the parsed document rather than off the source
/// string is what makes the first one honest: the parser is the only thing
/// that knows which `*` was emphasis and which was a bullet, and it is what
/// separates one table cell from the next.
public enum MessageCopyFormatter {
    /// The answer as the reader sees it: no markdown syntax, structure kept.
    ///
    /// Blocks are separated by a blank line, except for list items, which stay
    /// on consecutive lines because they are one list. A thematic break
    /// carries over as nothing at all — the blank line is already the break.
    public static func plainText(of document: MarkdownDocument) -> String {
        var output = ""
        var previous: MarkdownDocument.Block?
        for block in document.blocks {
            guard let text = text(of: block), !text.isEmpty else { continue }
            if !output.isEmpty {
                output += continues(previous, block) ? "\n" : "\n\n"
            }
            output += text
            previous = block
        }
        return output
    }

    /// Every code block in the answer, in order.
    ///
    /// A fence the model wrote and a table or chart it drew out of characters
    /// both land here: on screen they are the same monospace panel, so they
    /// are the same thing to copy.
    public static func codeBlocks(of document: MarkdownDocument) -> [String] {
        document.blocks.compactMap { block in
            guard case .code(let code) = block else { return nil }
            return code
        }
    }

    /// A menu label for one code block: what its first line says, clipped.
    ///
    /// A menu of code blocks has nothing else to tell them apart — the fence's
    /// language tag does not survive parsing — so the first line stands in for
    /// the whole block.
    public static func codeBlockLabel(for code: String, index: Int) -> String {
        let firstLine = code
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty }
        guard let firstLine else { return "Code Block \(index + 1)" }

        let collapsed = firstLine
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        guard collapsed.count > labelCharacterLimit else { return collapsed }
        let clipped = collapsed.prefix(labelCharacterLimit)
            .trimmingCharacters(in: .whitespaces)
        return clipped + "\u{2026}"
    }

    /// Long enough for a line of code to be recognizable, short enough that a
    /// menu of them stays a menu.
    private static let labelCharacterLimit = 34

    /// One block as plain text, or `nil` for a block that contributes none.
    private static func text(of block: MarkdownDocument.Block) -> String? {
        switch block {
        case .paragraph(let text), .heading(_, let text), .quote(let text):
            return String(text.characters)
        case .code(let code):
            return code
        case .listItem(let marker, let indent, let text):
            // The marker and the indent are the block's, not the text's, so
            // the sources below reattach them here.
            return String(repeating: "  ", count: indent)
                + marker + " " + String(text.characters)
        case .table(let header, let rows, _):
            return tableText(header: header, rows: rows)
        case .thematicBreak:
            return nil
        }
    }

    /// Cells joined by tabs, which is what a spreadsheet column paste wants
    /// and what a plain-text editor lines up.
    private static func tableText(
        header: [AttributedString],
        rows: [[AttributedString]]
    ) -> String {
        ([header] + rows)
            .map(rowText)
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func rowText(_ cells: [AttributedString]) -> String {
        cells
            .map { String($0.characters).trimmingCharacters(in: .whitespacesAndNewlines) }
            .joined(separator: "\t")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Whether two blocks are the same list, and so belong on consecutive
    /// lines rather than in a paragraph each.
    ///
    /// The marker is all a block carries of its list — the indent says how
    /// deep it sits, not which list it is in — so a bullet and a number are
    /// two lists even when the model wrote them one after the other. A nested
    /// item keeps the marker of its parent, which is what makes the indent
    /// read as nesting rather than as a list that starts again.
    private static func continues(
        _ previous: MarkdownDocument.Block?,
        _ block: MarkdownDocument.Block
    ) -> Bool {
        guard case .listItem(let previousMarker, _, _) = previous,
              case .listItem(let marker, _, _) = block else { return false }
        return isBullet(previousMarker) == isBullet(marker)
    }

    private static func isBullet(_ marker: String) -> Bool {
        marker == "\u{2022}"
    }
}
