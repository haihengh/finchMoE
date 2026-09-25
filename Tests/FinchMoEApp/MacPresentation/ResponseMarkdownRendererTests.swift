import AppKit
import Foundation
import Testing
@testable import FinchMoEMacPresentation

@MainActor
@Suite struct ResponseMarkdownRendererTests {
    @Test func rendersSupportedMarkdownWithNativeAttributes() throws {
        let source = """
        # Heading

        A **bold** and *italic* sentence with ~~obsolete~~ text, `inlineCode`, and a [link](https://example.com).

        - first
        - second

        > quoted text

        ```swift
        let answer = 42
        ```

        ---
        """

        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("Heading"))
        #expect(text.contains("bold"))
        #expect(text.contains("italic"))
        #expect(text.contains("•\tfirst\n•\tsecond"))
        #expect(text.contains("│\tquoted text"))
        #expect(text.contains("let answer = 42"))
        #expect(text.contains("────────────────"))
        #expect(!text.contains("**"))
        #expect(!text.contains("```"))

        let linkRange = (text as NSString).range(of: "link")
        #expect(result.attributedString.attribute(.link, at: linkRange.location,
                                                  effectiveRange: nil) == nil)
        let linkColor = result.attributedString.attribute(
            .foregroundColor, at: linkRange.location, effectiveRange: nil) as? NSColor
        #expect(linkColor?.isEqual(NSColor.linkColor) == true)
        #expect(result.attributedString.attribute(.underlineStyle,
                                                  at: linkRange.location,
                                                  effectiveRange: nil) as? Int
            == NSUnderlineStyle.single.rawValue)

        let codeRange = (text as NSString).range(of: "inlineCode")
        let codeFont = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(codeFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .backgroundColor, at: codeRange.location, effectiveRange: nil) != nil)

        let strikeRange = (text as NSString).range(of: "obsolete")
        #expect(result.attributedString.attribute(
            .strikethroughStyle, at: strikeRange.location, effectiveRange: nil) != nil)
    }

    /// The renderer closes the fence for the reader instead of handing back
    /// the answer as plain text.
    @Test func unfinishedFenceStillRendersWhatCameBeforeIt() {
        let result = ResponseMarkdownRenderer().render(
            "Before\n\n```python\nprint('unfinished')")

        #expect(!result.usedFallback)
        #expect(result.attributedString.string.contains("Before"))
        #expect(result.attributedString.string.contains("print('unfinished')"))
    }

    /// HTML and images are text the parser can keep; a table is the one thing
    /// this renderer has no layout for, so it stays as it was written.
    @Test func onlyATableFallsBackToRawText() {
        let renderer = ResponseMarkdownRenderer()

        for source in [
            "<div>Never execute this</div>",
            "![remote](https://example.com/image.png)",
        ] {
            let result = renderer.render(source)
            #expect(!result.usedFallback, "unexpected fallback for \(source)")
            #expect(result.attributedString.string.contains("Never")
                || result.attributedString.string.contains("remote"))
        }

        let table = "| A | B |\n|---|---|\n| 1 | 2 |"
        let result = renderer.render(table)
        #expect(result.usedFallback)
        #expect(result.attributedString.string.contains("| A | B |"))
    }

    @Test func latexRemainsReadableText() {
        let source = "Cosine is $\\frac{u \\cdot v}{||u|| ||v||}$."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string.contains("\\frac"))
        #expect(result.attributedString.string.contains("\\cdot"))
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }
}


@MainActor
@Suite struct FinchMoEMacThemeTests {
    @Test func appAccentMatchesProductRGB() {
        let color = FinchMoEMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 106.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 186.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 113.0 / 255.0) < 0.000_001)
    }
}
