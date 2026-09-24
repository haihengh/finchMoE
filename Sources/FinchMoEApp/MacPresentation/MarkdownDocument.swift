import Foundation

/// Whether a model answer can be trusted to the markdown parser at all.
///
/// Shared by the AppKit renderer and the SwiftUI block parser so both draw the
/// same answer the same way: an unfinished code fence, an HTML tag, a table, or
/// an image is shown verbatim rather than half-interpreted, because a partial
/// interpretation of those is worse than the raw text.
enum MarkdownSourcePolicy {
    static func requiresRawRendering(_ source: String) -> Bool {
        let fenceCount = source.components(separatedBy: "```").count - 1
        if !fenceCount.isMultiple(of: 2) { return true }
        if source.range(
            of: #"</?[A-Za-z][^>]*>"#,
            options: .regularExpression) != nil {
            return true
        }
        return source.range(
            of: #"!\[[^\]]*\]\([^\)]*\)"#,
            options: .regularExpression) != nil
    }

    static func containsTable(_ parsed: AttributedString) -> Bool {
        parsed.runs.contains { run in
            run.presentationIntent?.components.contains { component in
                switch component.kind {
                case .table, .tableHeaderRow, .tableRow, .tableCell:
                    return true
                default:
                    return false
                }
            } == true
        }
    }

    /// The model writes `**Label**` lines that mean "heading" to a reader but
    /// are one paragraph to the parser; a blank line after them restores the
    /// intended break.
    static func separatingBoldOnlyLines(_ source: String) -> String {
        source.replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)
    }

    static func parsed(_ source: String) throws -> AttributedString {
        try AttributedString(
            markdown: separatingBoldOnlyLines(source),
            options: .init(
                interpretedSyntax: .full,
                failurePolicy: .returnPartiallyParsedIfPossible))
    }
}

/// The block structure of one answer.
///
/// A chat bubble wants headings, list indent, code panels and quotes laid out
/// as separate views, which a single `AttributedString` cannot express in
/// SwiftUI (it drops paragraph styles). Parsing into blocks keeps the inline
/// styling from Foundation's markdown parser and hands the layout to SwiftUI.
public struct MarkdownDocument: Equatable, Sendable {
    public enum Block: Equatable, Sendable {
        case paragraph(AttributedString)
        case heading(level: Int, text: AttributedString)
        case code(String)
        case quote(AttributedString)
        case listItem(marker: String, indent: Int, text: AttributedString)
        case thematicBreak
    }

    public let blocks: [Block]
    /// True when the answer was shown verbatim, either because it is still
    /// missing a closing fence or because it uses markup SwiftUI should not
    /// try to interpret (tables, HTML, images).
    public let usedRawFallback: Bool

    public init(blocks: [Block], usedRawFallback: Bool) {
        self.blocks = blocks
        self.usedRawFallback = usedRawFallback
    }

    public static let empty = MarkdownDocument(blocks: [], usedRawFallback: false)

    public var isEmpty: Bool { blocks.isEmpty }

    public static func parse(_ source: String) -> MarkdownDocument {
        guard !source.isEmpty else { return .empty }
        guard !MarkdownSourcePolicy.requiresRawRendering(source) else {
            return raw(source)
        }
        guard let parsed = try? MarkdownSourcePolicy.parsed(source) else {
            return raw(source)
        }
        guard !MarkdownSourcePolicy.containsTable(parsed) else { return raw(source) }

        var builder = BlockBuilder()
        var currentIdentity: Int?
        var currentKind: BlockKind?
        var buffer = AttributedString()

        for run in parsed.runs {
            let info = BlockInfo(presentationIntent: run.presentationIntent)
            if info.identity != currentIdentity {
                builder.append(kind: currentKind, text: buffer)
                currentIdentity = info.identity
                currentKind = info.kind
                buffer = AttributedString()
            }
            guard info.kind != .thematicBreak else { continue }
            var slice = AttributedString(parsed[run.range])
            slice.presentationIntent = nil
            buffer.append(slice)
        }
        builder.append(kind: currentKind, text: buffer)

        guard !builder.blocks.isEmpty else { return raw(source) }
        return MarkdownDocument(blocks: builder.blocks, usedRawFallback: false)
    }

    private static func raw(_ source: String) -> MarkdownDocument {
        MarkdownDocument(
            blocks: [.paragraph(AttributedString(source))],
            usedRawFallback: true)
    }
}

private enum BlockKind: Equatable {
    case paragraph
    case heading(Int)
    case code
    case quote
    case unorderedList(indent: Int)
    case orderedList(ordinal: Int, indent: Int)
    case thematicBreak
}

private struct BlockInfo {
    let identity: Int
    let kind: BlockKind

    init(presentationIntent: PresentationIntent?) {
        guard let components = presentationIntent?.components,
              let leaf = components.first else {
            identity = 0
            kind = .paragraph
            return
        }

        var headingLevel: Int?
        var code = false
        var quote = false
        var thematicBreak = false
        var ordinal: Int?
        var ordered = false
        var unordered = false
        var listDepth = 0

        for component in components {
            switch component.kind {
            case .header(let level): headingLevel = level
            case .codeBlock: code = true
            case .blockQuote: quote = true
            case .thematicBreak: thematicBreak = true
            case .listItem(let itemOrdinal): ordinal = itemOrdinal
            case .orderedList:
                ordered = true
                listDepth += 1
            case .unorderedList:
                unordered = true
                listDepth += 1
            default: break
            }
        }

        identity = leaf.identity
        if thematicBreak {
            kind = .thematicBreak
        } else if let headingLevel {
            kind = .heading(headingLevel)
        } else if code {
            kind = .code
        } else if ordered, let ordinal {
            kind = .orderedList(ordinal: ordinal, indent: max(0, listDepth - 1))
        } else if unordered {
            kind = .unorderedList(indent: max(0, listDepth - 1))
        } else if quote {
            kind = .quote
        } else {
            kind = .paragraph
        }
    }
}

private struct BlockBuilder {
    private(set) var blocks: [MarkdownDocument.Block] = []

    mutating func append(kind: BlockKind?, text: AttributedString) {
        guard let kind else { return }
        switch kind {
        case .thematicBreak:
            blocks.append(.thematicBreak)
        case .paragraph:
            guard !isBlank(text) else { return }
            blocks.append(.paragraph(text))
        case .heading(let level):
            guard !isBlank(text) else { return }
            blocks.append(.heading(level: level, text: text))
        case .code:
            let code = String(text.characters)
                .trimmingCharacters(in: .newlines)
            guard !code.isEmpty else { return }
            blocks.append(.code(code))
        case .quote:
            guard !isBlank(text) else { return }
            blocks.append(.quote(text))
        case .unorderedList(let indent):
            guard !isBlank(text) else { return }
            blocks.append(.listItem(marker: "\u{2022}", indent: indent, text: text))
        case .orderedList(let ordinal, let indent):
            guard !isBlank(text) else { return }
            blocks.append(.listItem(
                marker: "\(ordinal).",
                indent: indent,
                text: text))
        }
    }

    private func isBlank(_ text: AttributedString) -> Bool {
        text.characters.allSatisfy { $0.isWhitespace || $0.isNewline }
    }
}
