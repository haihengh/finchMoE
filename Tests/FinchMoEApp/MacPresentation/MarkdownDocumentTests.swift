import Foundation
import Testing
@testable import FinchMoEMacPresentation

@Suite struct MarkdownDocumentTests {
    @Test func parsesAWholeAnswerIntoBlocks() throws {
        let source = """
        # Heading

        A paragraph.

        - first
        - second

        1. one
        2. two

        ```swift
        let answer = 42
        ```

        > quoted text

        ---
        """

        let document = MarkdownDocument.parse(source)

        #expect(!document.usedRawFallback)
        #expect(document.blocks.map(describe) == [
            "heading1:Heading",
            "paragraph:A paragraph.",
            "item\u{2022}0:first",
            "item\u{2022}0:second",
            "item1.0:one",
            "item2.0:two",
            "code:let answer = 42",
            "quote:quoted text",
            "break",
        ])
    }

    @Test func nestedListItemsCarryTheirIndent() throws {
        let document = MarkdownDocument.parse("""
        - outer
          - inner
        """)

        #expect(document.blocks.map(describe) == [
            "item\u{2022}0:outer",
            "item\u{2022}1:inner",
        ])
    }

    /// A bullet under a numbered list is a bullet: components arrive
    /// innermost-first, and the outer list's ordinal used to win.
    @Test func aBulletUnderANumberedListStaysABullet() throws {
        let document = MarkdownDocument.parse("""
        1. outer
           - inner
        """)

        #expect(document.blocks.map(describe) == [
            "item1.0:outer",
            "item\u{2022}1:inner",
        ])
    }

    @Test func inlineEmphasisAndCodeStayAsInlineIntents() throws {
        let document = MarkdownDocument.parse("A **bold**, *italic* and `code` line.")

        let paragraph = try #require(document.blocks.first)
        guard case .paragraph(let text) = paragraph else {
            Issue.record("expected one paragraph, got \(paragraph)")
            return
        }
        let rendered = String(text.characters)
        #expect(!rendered.contains("**"))
        #expect(!rendered.contains("`"))
        let intents = text.runs.compactMap(\.inlinePresentationIntent)
        #expect(intents.contains { $0.contains(.stronglyEmphasized) })
        #expect(intents.contains { $0.contains(.emphasized) })
        #expect(intents.contains { $0.contains(.code) })
    }

    @Test func boldOnlyHeadingLinesKeepTheirBreak() {
        let document = MarkdownDocument.parse(
            "**Origins**\nFieldfares arrive from northern Europe.")

        #expect(document.blocks.map(describe) == [
            "paragraph:Origins",
            "paragraph:Fieldfares arrive from northern Europe.",
        ])
    }

    // MARK: - What used to take the whole answer down with it

    /// An answer that ran out of tokens still renders: the fence runs to the
    /// end of the answer, as CommonMark says it does, instead of turning
    /// everything before it into plain text.
    @Test func anUnfinishedFenceRunsToTheEndOfTheAnswer() {
        let document = MarkdownDocument.parse(
            "Before\n\n```python\nprint('unfinished')")

        #expect(!document.usedRawFallback)
        #expect(document.blocks.map(describe) == [
            "paragraph:Before",
            "code:print('unfinished')",
        ])
    }

    /// Inline and block HTML come through the parser as the literal text they
    /// are, so there is nothing to protect the rest of the answer from.
    @Test func htmlIsKeptAsTextWithoutFallingBack() {
        let samples = [
            "<div>Never interpret this</div>",
            "Set the flag to <true> before running.",
            "Use Array<Int> and Optional<String> in Swift.",
        ]

        for source in samples {
            let document = MarkdownDocument.parse(source)
            #expect(!document.usedRawFallback, "unexpected fallback for \(source)")
            let rendered = document.blocks.map(describe).joined(separator: "\n")
            #expect(rendered.contains(source), "lost text for \(source)")
        }
    }

    @Test func anImageBecomesItsAltText() {
        let document = MarkdownDocument.parse(
            "![remote](https://example.com/image.png)")

        #expect(!document.usedRawFallback)
        #expect(document.blocks.map(describe) == ["paragraph:remote"])
    }

    @Test func anImageWithNoAltTextFallsBackToItsURL() {
        let document = MarkdownDocument.parse(
            "![](https://example.com/image.png)")

        #expect(document.blocks.map(describe) == [
            "paragraph:https://example.com/image.png",
        ])
    }

    // MARK: - Tables

    @Test func aPipeTableParsesAsATable() throws {
        let document = MarkdownDocument.parse("""
        | Model | Params | Speed |
        |:------|-------:|------:|
        | A3B   | 35B    | 11.4  |
        | Dense | 7B     | 30.0  |
        """)

        #expect(!document.usedRawFallback)
        #expect(document.blocks.map(describe) == [
            "table(LTT) <Model|Params|Speed> [A3B|35B|11.4] [Dense|7B|30.0]",
        ])
    }

    /// A table used to cost the answer its headings, its lists and its code.
    @Test func anAnswerWithATableKeepsEverythingAroundIt() {
        let document = MarkdownDocument.parse("""
        ## Comparison

        | Model | Speed |
        |-------|------:|
        | A3B   | 11.4  |

        ### Notes

        - fast
        """)

        #expect(!document.usedRawFallback)
        #expect(document.blocks.map(describe) == [
            "heading2:Comparison",
            "table(LT) <Model|Speed> [A3B|11.4]",
            "heading3:Notes",
            "item\u{2022}0:fast",
        ])
    }

    /// A row shorter than the header still fills its columns, because the view
    /// draws the gridlines on the cells.
    @Test func aRaggedTableIsPaddedToTheHeaderWidth() {
        let document = MarkdownDocument.parse("""
        | A | B | C |
        |---|---|---|
        | 1 | 2 |
        """)

        #expect(document.blocks.map(describe) == [
            "table(LLL) <A|B|C> [1|2|]",
        ])
    }

    // MARK: - Drawings

    /// The model draws tables and charts by hand, and the parser would strip
    /// the indentation that holds them together and reflow the lines.
    @Test func aBoxDrawingIsKeptLineForLine() {
        let drawing = """
        ┌──────────┬────────┐
        │ Stage    │ ms     │
        ├──────────┼────────┤
        │ prefill  │ 1200   │
        └──────────┴────────┘
        """
        let document = MarkdownDocument.parse("## Latency\n\n\(drawing)")

        #expect(!document.usedRawFallback)
        #expect(document.blocks == [
            .heading(level: 2, text: AttributedString("Latency")),
            .code(drawing),
        ])
    }

    @Test func aChartKeepsItsColumnsAndLabels() {
        let chart = """
        tok/s
         40 |          ███
         20 |  ███     ███
          0 +--------------
              Q4_K    Q5_K
        """
        let document = MarkdownDocument.parse("Throughput:\n\n\(chart)")

        #expect(document.blocks == [
            .paragraph(AttributedString("Throughput:")),
            .code(chart),
        ])
    }

    /// A sentence can carry a couple of `─` characters where a writer would
    /// use a dash; it is still a sentence.
    @Test func aSentenceWithDashesIsNotADrawing() {
        let document = MarkdownDocument.parse("""
        ┌──────────┐
        │ Latency  │
        └──────────┘
        The p99 stays under 200ms ─ under load ─ it is fine.
        """)

        #expect(document.blocks.map(describe) == [
            "code:┌──────────┐\n│ Latency  │\n└──────────┘",
            "paragraph:The p99 stays under 200ms ─ under load ─ it is fine.",
        ])
    }

    @Test func aListBulletedWithGlyphsIsStillAList() {
        let document = MarkdownDocument.parse("""
        - ── note one
        - ── note two
        """)

        #expect(document.blocks.map(describe) == [
            "item\u{2022}0:── note one",
            "item\u{2022}0:── note two",
        ])
    }

    @Test func aPipeTableWithNoDelimiterRowIsKeptAsDrawn() {
        let document = MarkdownDocument.parse("""
        | Model | Speed |
        | A3B   | 11.4  |
        """)

        #expect(document.blocks.map(describe) == [
            "code:| Model | Speed |\n| A3B   | 11.4  |",
        ])
    }

    // MARK: - Line structure

    /// CommonMark folds consecutive lines into one paragraph, which is what
    /// used to turn a column of numbers into a sentence.
    @Test func linesKeepTheirBreaks() throws {
        let document = MarkdownDocument.parse("""
        Name      Qty    Price
        apple     3      1.20
        banana    12     0.35
        """)

        let block = try #require(document.blocks.first)
        guard case .paragraph(let text) = block else {
            Issue.record("expected one paragraph, got \(block)")
            return
        }
        #expect(String(text.characters) == """
        Name      Qty    Price
        apple     3      1.20
        banana    12     0.35
        """)
    }

    @Test func fencedCodeIsNeverReflowed() {
        let document = MarkdownDocument.parse("""
        ```text
        a       b
          c   d
        ```
        """)

        #expect(document.blocks == [.code("a       b\n  c   d")])
    }

    @Test func anEmptyAnswerHasNoBlocks() {
        #expect(MarkdownDocument.parse("").isEmpty)
        #expect(!MarkdownDocument.parse("").usedRawFallback)
    }

    private func describe(_ block: MarkdownDocument.Block) -> String {
        switch block {
        case .paragraph(let text):
            return "paragraph:\(String(text.characters))"
        case .heading(let level, let text):
            return "heading\(level):\(String(text.characters))"
        case .code(let code):
            return "code:\(code)"
        case .quote(let text):
            return "quote:\(String(text.characters))"
        case .listItem(let marker, let indent, let text):
            return "item\(marker)\(indent):\(String(text.characters))"
        case .thematicBreak:
            return "break"
        case .table(let header, let rows, let alignments):
            let columns = alignments.map { alignment in
                switch alignment {
                case .leading: "L"
                case .center: "C"
                case .trailing: "T"
                }
            }.joined()
            let header = header.map { String($0.characters) }.joined(separator: "|")
            let body = rows.map { row in
                "[" + row.map { String($0.characters) }.joined(separator: "|") + "]"
            }.joined(separator: " ")
            return "table(\(columns)) <\(header)> \(body)"
        }
    }
}
