import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// The two tokenizer families get different treatment because they split
/// multi-byte characters differently:
///
/// - **Gemma (`<0xXX>` byte fallback).** BPE splits multi-byte codepoints
///   (e.g. emoji) across several byte-fallback tokens, and swift-transformers'
///   decoder silently drops byte-fallback tokens that sit at the **end** of the
///   decoded sequence (the bytes are committed only once a non-byte-fallback
///   token follows). Trailing byte IDs are therefore held back during the
///   stream and assembled manually at `flush()`.
/// - **Qwen (byte-level BPE).** There are no `<0xXX>` tokens, but an ordinary
///   token can still end mid-character: U+4E02, for example, is tokenized as
///   E4 B8 followed by 82. Decoding the prefix after the first token yields
///   U+FFFD, and the next token *repairs* that prefix, so a delta scheme would
///   emit a replacement character and then drop the completing token — the
///   diamond-with-question-mark seen in Chinese output. Here the raw token
///   bytes are assembled directly and only complete UTF-8 scalars are emitted;
///   at most three trailing bytes are held until the rest of the character
///   arrives.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline let family: GFTokenizerFamily
    /// IDs the library's `skipSpecialTokens: true` would drop. The Qwen
    /// byte-level path decodes tokens itself, so it needs the set explicitly.
    @usableFromInline let specialIDs: Set<Int32>

    // Gemma byte-fallback state.
    @usableFromInline var stableIDs: [Int] = []
    @usableFromInline var trailingByteIDs: [Int] = []
    @usableFromInline var emitted: String = ""

    /// Qwen byte-level state: the bytes of the character currently being
    /// assembled. Never more than three (the longest incomplete UTF-8 suffix).
    @usableFromInline var pendingByteLevelBytes: [UInt8] = []

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
        self.family = tokenizer.family
        var special = tokenizer.stopTokenIDs
        special.insert(tokenizer.bosID)
        special.insert(tokenizer.padID)
        if let imStart = tokenizer.tokenizer.convertTokenToId("<|im_start|>") {
            special.insert(Int32(imStart))
        }
        self.specialIDs = special
    }

    mutating func push(_ id: Int32) -> String {
        switch family {
        case .gemma4:
            return pushByteFallback(id)
        case .qwen3_6:
            return pushByteLevel(id)
        }
    }

    mutating func flush() -> String {
        switch family {
        case .gemma4:
            return flushByteFallback()
        case .qwen3_6:
            return flushByteLevel()
        }
    }

    // MARK: - Gemma byte fallback

    private mutating func pushByteFallback(_ id: Int32) -> String {
        let tokenID = Int(id)
        let token = tokenizer.convertIdToToken(tokenID) ?? ""
        if Self.isByteFallback(token) {
            trailingByteIDs.append(tokenID)
            return ""
        }

        if !trailingByteIDs.isEmpty {
            stableIDs.append(contentsOf: trailingByteIDs)
            trailingByteIDs.removeAll(keepingCapacity: true)
        }
        stableIDs.append(tokenID)

        let current = tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        return commitDelta(current)
    }

    private mutating func flushByteFallback() -> String {
        let stableText = stableIDs.isEmpty
            ? ""
            : tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)

        let trailingText = assembleByteFallback(trailingByteIDs)
        let fullText = stableText + trailingText
        return commitDelta(fullText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        // A combining mark can extend the last emitted grapheme, so compare the
        // append-only prefix without using Character boundaries.
        let currentUTF8 = current.utf8
        var boundary = currentUTF8.startIndex
        for byte in emitted.utf8 {
            guard boundary != currentUTF8.endIndex,
                  currentUTF8[boundary] == byte else {
                // Decoder altered the prefix — extremely rare in append-only streams.
                // Resync rather than emit garbage; the user-visible loss is bounded
                // to whatever was retokenized.
                emitted = current
                return ""
            }
            currentUTF8.formIndex(after: &boundary)
        }
        let delta = String(current[boundary...])
        emitted = current
        return delta
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id) else { continue }
            guard Self.isByteFallback(tok),
                  let byte = UInt8(tok.dropFirst(3).dropLast(), radix: 16)
            else { continue }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }

    // MARK: - Qwen byte-level BPE

    private mutating func pushByteLevel(_ id: Int32) -> String {
        guard !specialIDs.contains(id),
              let token = tokenizer.convertIdToToken(Int(id)),
              let bytes = Self.byteLevelBytes(token) else {
            return ""
        }
        pendingByteLevelBytes.append(contentsOf: bytes)
        let holdback = Self.incompleteUTF8SuffixCount(pendingByteLevelBytes)
        let completeCount = pendingByteLevelBytes.count - holdback
        guard completeCount > 0 else { return "" }
        let text = String(decoding: pendingByteLevelBytes[0..<completeCount],
                          as: UTF8.self)
        pendingByteLevelBytes.removeFirst(completeCount)
        return text
    }

    private mutating func flushByteLevel() -> String {
        guard !pendingByteLevelBytes.isEmpty else { return "" }
        // A truncated generation can end mid-character; replacement is the
        // honest rendering of bytes the model never finished.
        let text = String(decoding: pendingByteLevelBytes, as: UTF8.self)
        pendingByteLevelBytes.removeAll(keepingCapacity: true)
        return text
    }

    /// The number of trailing bytes that form an incomplete UTF-8 sequence.
    ///
    /// Qwen's bytes are valid UTF-8 in the long run, so the only undecodable
    /// suffix worth holding is an unfinished scalar. Anything else (an invalid
    /// lead or more than four continuation bytes) is left alone and decoded
    /// with replacement rather than silently swallowed.
    @usableFromInline
    static func incompleteUTF8SuffixCount(_ bytes: [UInt8]) -> Int {
        guard let last = bytes.last, last >= 0x80 else { return 0 }
        let lookback = min(4, bytes.count)
        for distance in 1...lookback {
            let byte = bytes[bytes.count - distance]
            if byte & 0xC0 == 0x80 { continue }   // continuation byte
            let needed: Int
            switch byte {
            case 0xC2...0xDF: needed = 2
            case 0xE0...0xEF: needed = 3
            case 0xF0...0xF4: needed = 4
            default: return 0                     // ASCII or an invalid lead
            }
            return distance < needed ? distance : 0
        }
        return 0
    }

    /// Reverse of the GPT-2 byte-to-unicode alphabet the ByteLevel decoder
    /// uses. Every scalar of a byte-level token maps back to one raw byte;
    /// `nil` means the token contains a scalar outside that alphabet.
    @usableFromInline
    static func byteLevelBytes(_ token: String) -> [UInt8]? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(token.utf8.count)
        for scalar in token.unicodeScalars {
            guard let byte = byteLevelByte(scalar) else { return nil }
            bytes.append(byte)
        }
        return bytes
    }

    @usableFromInline
    static func byteLevelByte(_ scalar: Unicode.Scalar) -> UInt8? {
        let value = scalar.value
        if Self.isPrintableByte(value) { return UInt8(value) }
        // The remaining bytes are mapped to U+0100, U+0101, … in ascending
        // byte order.
        let index = Int(value) - 256
        guard index >= 0, index < unprintableBytes.count else { return nil }
        return unprintableBytes[index]
    }

    private static func isPrintableByte(_ value: UInt32) -> Bool {
        (33...126).contains(value) || (161...172).contains(value)
            || (174...255).contains(value)
    }

    private static let unprintableBytes: [UInt8] = (0...255).compactMap { value -> UInt8? in
        let byte = UInt8(value)
        // 0...32, 127...160 and 173 map to U+0100+n.
        return isPrintableByte(UInt32(value)) ? nil : byte
    }
}
