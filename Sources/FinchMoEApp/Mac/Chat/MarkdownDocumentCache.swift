import FinchMoEMacPresentation
import Foundation

/// Parsed answers, keyed by their own text.
///
/// The transcript re-renders on every streaming update and on every resize, and
/// building an `AttributedString` per visible message each time is the
/// expensive part of that. A finished answer never changes, so parsing it once
/// is enough; the cap keeps a long session from growing without bound.
@MainActor
enum MarkdownDocumentCache {
    private static var documents: [String: MarkdownDocument] = [:]
    private static var insertionOrder: [String] = []
    private static let limit = 256

    static func document(for source: String) -> MarkdownDocument {
        if let cached = documents[source] { return cached }
        let document = MarkdownDocument.parse(source)
        documents[source] = document
        insertionOrder.append(source)
        if insertionOrder.count > limit {
            documents[insertionOrder.removeFirst()] = nil
        }
        return document
    }
}
