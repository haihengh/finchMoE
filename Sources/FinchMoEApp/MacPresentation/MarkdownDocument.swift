import Foundation

/// The pass an answer makes before the markdown parser sees it, and the one
/// thing the parser must not be given at all.
///
/// Two properties of model answers shape this. They are line-oriented — a
/// title, a table row, one bar of a chart per line — while CommonMark joins
/// consecutive lines into a single paragraph, so the line structure has to be
/// restored first. And an answer is mostly prose even when part of it is
/// something SwiftUI has no layout for, so nothing here rejects a whole
/// answer: an image becomes its alt text, a fence the model never closed is
/// closed, and a hand-drawn table is lifted out verbatim.
enum MarkdownSourcePolicy {
    /// One run of lines that should be handed to the markdown parser, or kept
    /// away from it.
    enum Segment: Equatable {
        case markdown(String)
        case verbatim(String)
    }

    static func normalized(_ source: String) -> String {
        var lines = source.components(separatedBy: "\n")
        if let fence = unterminatedFence(in: lines) {
            lines.append(fence)
        }

        var output: [String] = []
        output.reserveCapacity(lines.count)
        var openFence: String?
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isCodeLine: Bool
            if let fence = openFence {
                isCodeLine = true
                if trimmed.hasPrefix(fence) { openFence = nil }
            } else if let fence = fenceMarker(of: trimmed) {
                isCodeLine = true
                openFence = fence
            } else {
                isCodeLine = false
            }

            // A source line break is a line break to the reader: the model
            // writes one line per paragraph, and a chart, a column of numbers
            // or a list written by hand only reads correctly if the breaks
            // survive. CommonMark would fold them into one paragraph, so they
            // are promoted to hard breaks.
            let next = index + 1 < lines.count ? lines[index + 1] : ""
            let keepsBreak = !isCodeLine
                && !trimmed.isEmpty
                && !next.trimmingCharacters(in: .whitespaces).isEmpty
            output.append(keepsBreak ? line + "  " : line)
        }

        return replacingImages(in: output.joined(separator: "\n"))
    }

    /// Splits an answer into what the markdown parser can lay out well and
    /// what it would flatten.
    ///
    /// A hand-drawn table or chart is the second kind. It is drawn with box
    /// characters and aligned spaces, and the parser strips the indentation
    /// that holds it together and reflows its lines like prose, so it is
    /// lifted out line for line and shown in a monospace panel instead. The
    /// lines are classified as the model wrote them; only the markdown
    /// segments are normalized afterwards.
    static func segments(of source: String) -> [Segment] {
        let lines = source.components(separatedBy: "\n")
        var kinds = Array(repeating: SegmentKind.markdown, count: lines.count)
        let fenced = fencedLines(in: lines)

        var index = 0
        while index < lines.count {
            guard !fenced[index], !isBlank(lines[index]) else {
                index += 1
                continue
            }
            var end = index
            while end < lines.count, !fenced[end], !isBlank(lines[end]) {
                end += 1
            }
            markDrawing(in: index..<end, lines: lines, kinds: &kinds)
            index = end
        }

        var segments: [Segment] = []
        var start = 0
        for index in 1...lines.count where index == lines.count
            || kinds[index] != kinds[start] {
            let text = lines[start..<index].joined(separator: "\n")
            segments.append(kinds[start] == .verbatim ? .verbatim(text) : .markdown(text))
            start = index
        }
        return segments
    }

    /// The model writes `**Label**` lines that mean "heading" to a reader but
    /// are one paragraph to the parser; a blank line after them restores the
    /// intended break. The two trailing spaces `normalized` adds are absorbed
    /// by the `[ \t]*` before the newline.
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

    // MARK: - Fences

    /// Returns the marker of a fence the answer opens and never closes, which
    /// is how an answer that ran out of tokens usually ends. CommonMark runs
    /// such a fence to the end of the document anyway; saying so keeps the
    /// code after it in a code panel instead of leaving the parser to treat
    /// the rest of the answer as prose.
    private static func unterminatedFence(in lines: [String]) -> String? {
        var openFence: String?
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let fence = openFence {
                if trimmed.hasPrefix(fence) { openFence = nil }
            } else if let fence = fenceMarker(of: trimmed) {
                openFence = fence
            }
        }
        return openFence
    }

    private static func fencedLines(in lines: [String]) -> [Bool] {
        var fenced = Array(repeating: false, count: lines.count)
        var openFence: String?
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let fence = openFence {
                fenced[index] = true
                if trimmed.hasPrefix(fence) { openFence = nil }
            } else if let fence = fenceMarker(of: trimmed) {
                fenced[index] = true
                openFence = fence
            }
        }
        return fenced
    }

    private static func fenceMarker(of trimmedLine: String) -> String? {
        if trimmedLine.hasPrefix("```") { return "```" }
        if trimmedLine.hasPrefix("~~~") { return "~~~" }
        return nil
    }

    // MARK: - Drawings

    private enum SegmentKind {
        case markdown
        case verbatim
    }

    /// Marks the lines of a run that the parser would destroy.
    ///
    /// A drawing is two or more lines carrying box-drawing or block
    /// characters, or an ASCII frame drawn with `+--`. The run is the whole
    /// group of lines those sit in — a chart's axis labels and title are part
    /// of it even though they carry no box characters — minus any leading or
    /// trailing line that starts a markdown block, so a `## Latency` sitting
    /// directly above a drawn table still renders as a heading.
    private static func markDrawing(
        in range: Range<Int>,
        lines: [String],
        kinds: inout [SegmentKind]
    ) {
        // A markdown table is a table; only what the parser cannot see is a
        // drawing.
        guard !range.contains(where: { isTableDelimiter(lines[$0]) }) else { return }

        let drawn = drawnLines(in: range, lines: lines)
        guard drawn.count >= 2, let first = drawn.first, let last = drawn.last else {
            return
        }

        var start = range.lowerBound
        while start < first, isBlockStart(lines[start]) || isAdjacentProse(lines[start]) {
            start += 1
        }
        var end = range.upperBound - 1
        while end > last, isBlockStart(lines[end]) || isAdjacentProse(lines[end]) {
            end -= 1
        }
        for index in start...end { kinds[index] = .verbatim }
    }

    /// A sentence that happens to sit against a drawing belongs to the prose
    /// around it. A sentence that is *indented* — an axis label, a legend —
    /// belongs to the drawing, and a fragment like `tok/s` is a label too.
    private static func isAdjacentProse(_ line: String) -> Bool {
        guard let first = line.first, first != " ", first != "\t" else { return false }
        guard let last = line.last, last == "." || last == "!" || last == "?" else {
            return false
        }
        return line.contains(" ")
    }

    /// The lines of a run that are actually drawn, not just decorated.
    ///
    /// A sentence can carry a couple of `─` characters where a writer would
    /// use a dash, so carrying drawing characters is not enough on its own: a
    /// drawn line also has to *line up* with another one — two bars of a chart
    /// start at the same column, a table's rows share their edges. Prose does
    /// not, and is left to the markdown parser.
    private static func drawnLines(in range: Range<Int>, lines: [String]) -> [Int] {
        let candidates = range
            .filter { isDrawingLine(lines[$0]) }
            .map { ($0, drawingColumns(lines[$0])) }
        guard candidates.count >= 2 else { return [] }

        return candidates.filter { candidate in
            candidates.filter {
                !$0.1.isDisjoint(with: candidate.1)
            }.count >= 2
        }.map(\.0)
    }

    /// Where a line's drawing characters sit, by character offset.
    private static func drawingColumns(_ line: String) -> Set<Int> {
        var columns: Set<Int> = []
        for (offset, character) in line.enumerated() {
            if character.unicodeScalars.count == 1,
               let scalar = character.unicodeScalars.first?.value,
               (0x2500...0x257F).contains(scalar)
                || (0x2580...0x259F).contains(scalar) {
                columns.insert(offset)
            } else if character == "+" || character == "-" || character == "|" {
                columns.insert(offset)
            }
        }
        return columns
    }

    private static func isDrawingLine(_ line: String) -> Bool {
        // A list bulleted with a glyph is a list, not a drawing, and it would
        // otherwise line up well enough to pass for one.
        guard !isBlockStart(line) else { return false }

        var glyphs = 0
        for scalar in line.unicodeScalars {
            switch scalar.value {
            case 0x2500...0x257F, 0x2580...0x259F: glyphs += 1
            default: break
            }
        }
        if glyphs >= 2 || line.contains("+--") || line.contains("--+") {
            return true
        }

        // A row of a pipe table the model drew without a delimiter row. Both
        // ends are piped, which is what separates a table from a shell
        // pipeline written into a sentence.
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix("|") && trimmed.hasSuffix("|")
            && trimmed.filter { $0 == "|" }.count >= 2
    }

    /// True for the `|---|---|` row that makes a pipe table a table. It has to
    /// contain a pipe: `---` on its own is a thematic break.
    private static func isTableDelimiter(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("|") else { return false }

        var cells = 0
        for cell in trimmed.split(separator: "|", omittingEmptySubsequences: false) {
            var body = Substring(cell.trimmingCharacters(in: .whitespaces))
            guard !body.isEmpty else { continue }
            if body.hasPrefix(":") { body = body.dropFirst() }
            if body.hasSuffix(":") { body = body.dropLast() }
            guard body.count >= 2, body.allSatisfy({ $0 == "-" }) else { return false }
            cells += 1
        }
        return cells > 0
    }

    private static func isBlockStart(_ line: String) -> Bool {
        let trimmed = line.drop { $0 == " " || $0 == "\t" }
        guard let first = trimmed.first else { return false }

        if first == ">" { return true }
        if first == "-" || first == "*" || first == "+" {
            return trimmed.dropFirst().first == " " || trimmed.allSatisfy { $0 == first }
        }
        if first == "#" {
            let marks = trimmed.prefix { $0 == "#" }
            let rest = trimmed.dropFirst(marks.count)
            return marks.count <= 6 && (rest.isEmpty || rest.first == " ")
        }
        if first.isNumber {
            let digits = trimmed.prefix { $0.isNumber }
            let rest = trimmed.dropFirst(digits.count)
            return rest.hasPrefix(". ") || rest.hasPrefix(") ")
        }
        return false
    }

    private static func isBlank(_ line: String) -> Bool {
        line.allSatisfy { $0 == " " || $0 == "\t" }
    }

    /// There is nothing to load, so the alt text is what the reader gets; an
    /// image with no alt text at least says what it pointed at.
    ///
    /// Which of the two it is has to be decided per match, and replacements
    /// are applied back to front so that each match's range is still the one
    /// the expression reported.
    private static func replacingImages(in source: String) -> String {
        guard source.contains("!["),
              let expression = try? NSRegularExpression(
                pattern: #"!\[([^\]]*)\]\(([^)\s]*)[^)]*\)"#) else {
            return source
        }
        let matches = expression.matches(
            in: source,
            range: NSRange(source.startIndex..., in: source))
        guard !matches.isEmpty else { return source }

        var output = source
        for match in matches.reversed() {
            guard let whole = Range(match.range, in: output),
                  let alt = Range(match.range(at: 1), in: output),
                  let url = Range(match.range(at: 2), in: output) else {
                continue
            }
            let text = String(output[alt])
            output.replaceSubrange(
                whole,
                with: text.isEmpty ? String(output[url]) : text)
        }
        return output
    }
}

/// The block structure of one answer.
///
/// A chat bubble wants headings, list indent, code panels, tables and quotes
/// laid out as separate views, which a single `AttributedString` cannot
/// express in SwiftUI (it drops paragraph styles). Parsing into blocks keeps
/// the inline styling from Foundation's markdown parser and hands the layout
/// to SwiftUI.
public struct MarkdownDocument: Equatable, Sendable {
    /// How a table column lines up, from the `:--` / `--:` markers the model
    /// wrote in the delimiter row.
    public enum ColumnAlignment: Equatable, Sendable {
        case leading
        case center
        case trailing
    }

    public enum Block: Equatable, Sendable {
        case paragraph(AttributedString)
        case heading(level: Int, text: AttributedString)
        case code(String)
        case quote(AttributedString)
        case listItem(marker: String, indent: Int, text: AttributedString)
        case thematicBreak
        case table(
            header: [AttributedString],
            rows: [[AttributedString]],
            alignments: [ColumnAlignment])
    }

    public let blocks: [Block]
    /// True when some of the answer could not be parsed and was kept as plain
    /// text. Parsing degrades one construct at a time, so this is rare: an
    /// unfinished fence, a table or an image no longer costs the whole answer.
    public let usedRawFallback: Bool

    public init(blocks: [Block], usedRawFallback: Bool) {
        self.blocks = blocks
        self.usedRawFallback = usedRawFallback
    }

    public static let empty = MarkdownDocument(blocks: [], usedRawFallback: false)

    public var isEmpty: Bool { blocks.isEmpty }

    public static func parse(_ source: String) -> MarkdownDocument {
        guard !source.isEmpty else { return .empty }

        var output: [Block] = []
        var usedRawFallback = false
        for segment in MarkdownSourcePolicy.segments(of: source) {
            switch segment {
            case .verbatim(let text):
                // Taken from the answer as written: a drawing is only itself
                // line for line, and normalizing it would rewrite its lines.
                guard !text.isEmpty else { continue }
                output.append(.code(text))
            case .markdown(let text):
                guard !text.isEmpty else { continue }
                let text = MarkdownSourcePolicy.normalized(text)
                guard let parsed = try? MarkdownSourcePolicy.parsed(text) else {
                    output.append(.paragraph(AttributedString(text)))
                    usedRawFallback = true
                    continue
                }
                output.append(contentsOf: blocks(from: parsed))
            }
        }

        guard !output.isEmpty else { return raw(source) }
        return MarkdownDocument(blocks: output, usedRawFallback: usedRawFallback)
    }

    private static func raw(_ source: String) -> MarkdownDocument {
        MarkdownDocument(
            blocks: [.paragraph(AttributedString(source))],
            usedRawFallback: true)
    }

    private static func blocks(from parsed: AttributedString) -> [Block] {
        var builder = BlockBuilder()
        var currentIdentity: Int?
        var currentKind: BlockKind?
        var buffer = AttributedString()
        var table: TableAccumulator?

        func flushText() {
            builder.append(kind: currentKind, text: buffer)
            buffer = AttributedString()
        }

        for run in parsed.runs {
            let info = BlockInfo(presentationIntent: run.presentationIntent)

            if let tableID = info.tableID {
                if currentIdentity != nil {
                    flushText()
                    currentIdentity = nil
                    currentKind = nil
                }
                if table?.tableID != tableID {
                    if let finished = table { builder.append(table: finished.build()) }
                    table = TableAccumulator(
                        tableID: tableID,
                        alignments: info.alignments)
                }
                table?.append(
                    column: info.columnIndex,
                    rowID: info.rowID,
                    isHeader: info.isHeaderRow,
                    text: AttributedString(parsed[run.range]))
                continue
            }

            if let finished = table {
                builder.append(table: finished.build())
                table = nil
            }

            if info.identity != currentIdentity {
                flushText()
                currentIdentity = info.identity
                currentKind = info.kind
            }
            guard info.kind != .thematicBreak else { continue }
            var slice = AttributedString(parsed[run.range])
            slice.presentationIntent = nil
            buffer.append(slice)
        }

        flushText()
        if let finished = table { builder.append(table: finished.build()) }
        return builder.blocks
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

/// The block one run belongs to, and — for a table — where in it.
///
/// Components arrive innermost-first, so a table cell is `tableCell` followed
/// by its row and then the table itself.
private struct BlockInfo {
    let identity: Int
    let kind: BlockKind
    let tableID: Int?
    let rowID: Int
    let columnIndex: Int
    let isHeaderRow: Bool
    let alignments: [MarkdownDocument.ColumnAlignment]

    init(presentationIntent: PresentationIntent?) {
        guard let components = presentationIntent?.components,
              let leaf = components.first else {
            identity = 0
            kind = .paragraph
            tableID = nil
            rowID = 0
            columnIndex = 0
            isHeaderRow = false
            alignments = []
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
        var innermostIsOrdered: Bool?
        var table: Int?
        var row: Int?
        var column: Int?
        var header = false
        var alignments: [MarkdownDocument.ColumnAlignment] = []

        for component in components {
            switch component.kind {
            case .header(let level): headingLevel = level
            case .codeBlock: code = true
            case .blockQuote: quote = true
            case .thematicBreak: thematicBreak = true
            case .listItem(let itemOrdinal):
                if ordinal == nil { ordinal = itemOrdinal }
            case .orderedList:
                ordered = true
                listDepth += 1
                // Components come innermost-first, so the first list that
                // turns up is the one this item actually belongs to: a bullet
                // nested under a numbered list is a bullet, not item 1 again.
                if innermostIsOrdered == nil { innermostIsOrdered = true }
            case .unorderedList:
                unordered = true
                listDepth += 1
                if innermostIsOrdered == nil { innermostIsOrdered = false }
            case .table(let columns):
                table = component.identity
                alignments = columns.map { column in
                    switch column.alignment {
                    case .left: return .leading
                    case .center: return .center
                    case .right: return .trailing
                    @unknown default: return .leading
                    }
                }
            case .tableHeaderRow:
                row = component.identity
                header = true
            case .tableRow:
                row = component.identity
            case .tableCell(let columnIndex):
                column = columnIndex
            default: break
            }
        }

        identity = leaf.identity
        tableID = table
        rowID = row ?? 0
        columnIndex = column ?? 0
        isHeaderRow = header

        if thematicBreak {
            kind = .thematicBreak
        } else if let headingLevel {
            kind = .heading(headingLevel)
        } else if code {
            kind = .code
        } else if innermostIsOrdered == true, let ordinal {
            kind = .orderedList(ordinal: ordinal, indent: max(0, listDepth - 1))
        } else if unordered {
            kind = .unorderedList(indent: max(0, listDepth - 1))
        } else if ordered {
            kind = .orderedList(ordinal: ordinal ?? 1, indent: max(0, listDepth - 1))
        } else if quote {
            kind = .quote
        } else {
            kind = .paragraph
        }

        self.alignments = alignments
    }
}

/// Collects a table's runs into rows and cells.
private struct TableAccumulator {
    let tableID: Int
    let alignments: [MarkdownDocument.ColumnAlignment]
    private var rows: [Row] = []

    private struct Row {
        let id: Int
        let isHeader: Bool
        var cells: [Int: AttributedString]
    }

    init(tableID: Int, alignments: [MarkdownDocument.ColumnAlignment]) {
        self.tableID = tableID
        self.alignments = alignments
    }

    mutating func append(
        column: Int,
        rowID: Int,
        isHeader: Bool,
        text: AttributedString
    ) {
        if rows.last?.id != rowID {
            rows.append(Row(id: rowID, isHeader: isHeader, cells: [:]))
        }
        rows[rows.count - 1].cells[column, default: AttributedString()].append(text)
    }

    func build() -> MarkdownDocument.Block {
        let columnCount = max(
            alignments.count,
            rows.map { ($0.cells.keys.max() ?? -1) + 1 }.max() ?? 0)

        func cells(of row: Row) -> [AttributedString] {
            (0..<columnCount).map { row.cells[$0] ?? AttributedString() }
        }

        let header = rows.first { $0.isHeader }.map(cells)
            ?? Array(repeating: AttributedString(), count: columnCount)
        let body = rows.filter { !$0.isHeader }.map(cells)
        return .table(header: header, rows: body, alignments: alignments)
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

    mutating func append(table: MarkdownDocument.Block) {
        guard case .table(let header, let rows, _) = table else { return }
        let isEmptyTable = header.allSatisfy(isBlank)
            && rows.allSatisfy { row in row.allSatisfy(isBlank) }
        guard !isEmptyTable else { return }
        blocks.append(table)
    }

    private func isBlank(_ text: AttributedString) -> Bool {
        text.characters.allSatisfy { $0.isWhitespace || $0.isNewline }
    }
}
