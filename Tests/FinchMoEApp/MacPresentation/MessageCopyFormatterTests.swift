import Foundation
import Testing
@testable import FinchMoEMacPresentation

@Suite struct MessageCopyFormatterTests {
    @Test func plainTextDropsTheMarkdownButKeepsTheStructure() throws {
        let document = MarkdownDocument.parse("""
        ## Where it lands

        A **sparse** layer keeps `k` experts.

        1. The router picks experts.
        2. The rest stay on disk.

        - 4-bit experts stream in
        - the cache decides how often

        ```swift
        let selected = router.topK(token, k: 8)
        ```

        > Memory is set by the cache, not the checkpoint.

        ---

        That trade is why slots matter.
        """)

        #expect(MessageCopyFormatter.plainText(of: document) == """
        Where it lands

        A sparse layer keeps k experts.

        1. The router picks experts.
        2. The rest stay on disk.

        \u{2022} 4-bit experts stream in
        \u{2022} the cache decides how often

        let selected = router.topK(token, k: 8)

        Memory is set by the cache, not the checkpoint.

        That trade is why slots matter.
        """)
    }

    /// A numbered list followed by a bulleted one is two lists, and the blank
    /// line between them is what says so.
    @Test func aSecondListStartsANewBlock() throws {
        let document = MarkdownDocument.parse("""
        1. first
        2. second

        - alpha
        - beta
        """)

        #expect(MessageCopyFormatter.plainText(of: document) == """
        1. first
        2. second

        \u{2022} alpha
        \u{2022} beta
        """)
    }

    @Test func plainTextKeepsATableInColumns() throws {
        let document = MarkdownDocument.parse("""
        | Precision | Size | Speed |
        |:----------|-----:|------:|
        | 4-bit     | 1.95 GB | 11.4 tok/s |
        """)

        #expect(MessageCopyFormatter.plainText(of: document) == """
        Precision\tSize\tSpeed
        4-bit\t1.95 GB\t11.4 tok/s
        """)
    }

    /// The line structure is the model's: a line break in the source is a line
    /// break on screen, so it is one in the copied text too — without the two
    /// spaces the renderer adds to make it a hard break.
    @Test func plainTextKeepsTheLineBreaksOfAMultiLineAnswer() throws {
        let document = MarkdownDocument.parse("""
        Throughput
        14 rows per second
        """)

        #expect(MessageCopyFormatter.plainText(of: document)
            == "Throughput\n14 rows per second")
    }

    @Test func codeBlocksAreFencedCodeAndDrawingsAlike() throws {
        let drawing = """
        ┌──────────┐
        │ Latency  │
        └──────────┘
        """
        let document = MarkdownDocument.parse("""
        Before:

        ```swift
        let selected = router.topK(token, k: 8)
        ```

        \(drawing)
        """)

        #expect(MessageCopyFormatter.codeBlocks(of: document) == [
            "let selected = router.topK(token, k: 8)",
            drawing,
        ])
        // A drawing is only itself line for line, so plain text keeps it whole.
        #expect(MessageCopyFormatter.plainText(of: document) == """
        Before:

        let selected = router.topK(token, k: 8)

        \(drawing)
        """)
    }

    @Test func anAnswerWithNoCodeHasNoCodeBlocks() throws {
        let document = MarkdownDocument.parse("Just a sentence.")
        #expect(MessageCopyFormatter.codeBlocks(of: document).isEmpty)
    }

    @Test func aCodeBlockIsLabelledByItsFirstLine() throws {
        #expect(MessageCopyFormatter.codeBlockLabel(
            for: "print(\"late\")\nprint(\"again\")", index: 0)
            == "print(\"late\")")

        // A blank first line is not a label, so the next real one is used.
        #expect(MessageCopyFormatter.codeBlockLabel(
            for: "\n\n  let k = 8\n", index: 1) == "let k = 8")
    }

    @Test func aLongFirstLineIsClipped() throws {
        let label = MessageCopyFormatter.codeBlockLabel(
            for: "let selected = router.topK(token, k: 8, sorted: true)",
            index: 0)

        #expect(label.hasSuffix("\u{2026}"))
        #expect(label.count <= 35)
        #expect("let selected = router.topK(token, k: 8, sorted: true)"
            .hasPrefix(label.dropLast()))
    }

    /// Only reachable for a block that is whitespace throughout, which the
    /// parser drops — but a menu item with no label at all would be worse.
    @Test func aCodeBlockWithNothingInItFallsBackToItsPosition() throws {
        #expect(MessageCopyFormatter.codeBlockLabel(for: " \n\n", index: 1)
            == "Code Block 2")
    }

    @Test func anEmptyDocumentCopiesAsNothing() throws {
        #expect(MessageCopyFormatter.plainText(of: .empty).isEmpty)
    }
}
