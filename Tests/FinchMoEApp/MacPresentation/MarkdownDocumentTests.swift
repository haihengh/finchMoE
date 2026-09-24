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

    @Test func anUnfinishedFenceIsShownVerbatim() {
        let source = "Before\n\n```python\nprint('unfinished')"
        let document = MarkdownDocument.parse(source)

        #expect(document.usedRawFallback)
        #expect(document.blocks.map(describe) == ["paragraph:\(source)"])
    }

    @Test func tablesHTMLAndImagesAreShownVerbatim() {
        let samples = [
            "<div>Never interpret this</div>",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "![remote](https://example.com/image.png)",
        ]

        for source in samples {
            let document = MarkdownDocument.parse(source)
            #expect(document.usedRawFallback, "expected raw fallback for \(source)")
            #expect(document.blocks.map(describe) == ["paragraph:\(source)"])
        }
    }

    @Test func boldOnlyHeadingLinesKeepTheirBreak() {
        let document = MarkdownDocument.parse(
            "**Origins**\nFieldfares arrive from northern Europe.")

        #expect(document.blocks.map(describe) == [
            "paragraph:Origins",
            "paragraph:Fieldfares arrive from northern Europe.",
        ])
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
        }
    }
}
