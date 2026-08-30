import Foundation

/// Byte-level BPE tokenizer compatible with the HuggingFace `tokenizer.json`
/// used by Qwen3 (GPT-2 style byte-level pre-tokenization + BPE merges).
/// Loads vocab, merges, and added/special tokens; supports encode + decode.
public final class Tokenizer {

    public enum TokError: Error, CustomStringConvertible {
        case fileMissing(String)
        case parseFailed(String)
        public var description: String {
            switch self {
            case .fileMissing(let s): return "Tokenizer: file missing \(s)"
            case .parseFailed(let s): return "Tokenizer: parse failed \(s)"
            }
        }
    }

    private var encoder: [String: Int] = [:]     // token string → id
    private var decoder: [Int: String] = [:]     // id → token string
    private var bpeRanks: [String: Int] = [:]    // "a b" → rank
    private var specialTokens: [String: Int] = [:]

    // Byte ↔ unicode mapping (GPT-2 byte-level).
    private let byteToUnicode: [UInt8: Character]
    private let unicodeToByte: [Character: UInt8]

    public let bosTokenId: Int
    public let eosTokenId: Int
    public private(set) var imStartId: Int = -1
    public private(set) var imEndId: Int = -1

    private let pattern = try! NSRegularExpression(
        pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        options: [])

    public init(tokenizerJSONURL url: URL, bosTokenId: Int = 151643, eosTokenId: Int = 151645) throws {
        guard let data = try? Data(contentsOf: url) else { throw TokError.fileMissing(url.path) }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokError.parseFailed("root not object")
        }

        // Byte-level maps.
        var b2u: [UInt8: Character] = [:]
        var u2b: [Character: UInt8] = [:]
        Self.buildByteMaps(&b2u, &u2b)
        self.byteToUnicode = b2u
        self.unicodeToByte = u2b
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId

        // model.vocab and model.merges
        if let model = root["model"] as? [String: Any] {
            if let vocab = model["vocab"] as? [String: Any] {
                for (tok, idAny) in vocab {
                    if let id = (idAny as? NSNumber)?.intValue {
                        encoder[tok] = id
                        decoder[id] = tok
                    }
                }
            }
            if let merges = model["merges"] as? [Any] {
                for (i, m) in merges.enumerated() {
                    if let pair = m as? String {
                        bpeRanks[pair] = i
                    } else if let arr = m as? [String], arr.count == 2 {
                        bpeRanks["\(arr[0]) \(arr[1])"] = i
                    }
                }
            }
        }

        // added_tokens (special tokens like <|im_start|>).
        if let added = root["added_tokens"] as? [[String: Any]] {
            for t in added {
                if let content = t["content"] as? String, let id = (t["id"] as? NSNumber)?.intValue {
                    specialTokens[content] = id
                    encoder[content] = id
                    decoder[id] = content
                    if content == "<|im_start|>" { imStartId = id }
                    if content == "<|im_end|>" { imEndId = id }
                }
            }
        }
    }

    // MARK: - Encode

    /// Encodes text into token ids (no special tokens added automatically,
    /// except those literally present in the text).
    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for segment in splitOnSpecials(text) {
            switch segment {
            case .special(let id):
                ids.append(id)
            case .text(let s):
                ids.append(contentsOf: encodeOrdinary(s))
            }
        }
        return ids
    }

    private enum Segment { case text(String); case special(Int) }

    private func splitOnSpecials(_ text: String) -> [Segment] {
        guard !specialTokens.isEmpty else { return [.text(text)] }
        // Longest-first to avoid partial matches.
        let specials = specialTokens.keys.sorted { $0.count > $1.count }
        var segments: [Segment] = []
        var remainder = Substring(text)
        outer: while !remainder.isEmpty {
            for sp in specials {
                if let range = remainder.range(of: sp) {
                    let before = remainder[remainder.startIndex..<range.lowerBound]
                    if !before.isEmpty { segments.append(.text(String(before))) }
                    segments.append(.special(specialTokens[sp]!))
                    remainder = remainder[range.upperBound...]
                    continue outer
                }
            }
            segments.append(.text(String(remainder)))
            break
        }
        return segments
    }

    private func encodeOrdinary(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let piece = ns.substring(with: m.range)
            // Byte-level encode the piece.
            var mapped = ""
            for byte in Array(piece.utf8) {
                mapped.append(byteToUnicode[byte]!)
            }
            for tok in bpe(mapped) {
                if let id = encoder[tok] { ids.append(id) }
            }
        }
        return ids
    }

    // MARK: - BPE

    private func bpe(_ token: String) -> [String] {
        var word = token.map { String($0) }
        if word.count < 2 { return word }

        while true {
            var minRank = Int.max
            var minPairIndex = -1
            for i in 0..<(word.count - 1) {
                let pair = "\(word[i]) \(word[i + 1])"
                if let rank = bpeRanks[pair], rank < minRank {
                    minRank = rank
                    minPairIndex = i
                }
            }
            if minPairIndex == -1 { break }
            let merged = word[minPairIndex] + word[minPairIndex + 1]
            word.replaceSubrange(minPairIndex...(minPairIndex + 1), with: [merged])
            if word.count == 1 { break }
        }
        return word
    }

    // MARK: - Decode

    public func decode(_ ids: [Int], skipSpecial: Bool = true) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if skipSpecial, specialTokens.values.contains(id) { continue }
            guard let tok = decoder[id] else { continue }
            if specialTokens[tok] != nil && skipSpecial { continue }
            for ch in tok {
                if let b = unicodeToByte[ch] {
                    bytes.append(b)
                } else {
                    bytes.append(contentsOf: Array(String(ch).utf8))
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Decodes a single token id to its string fragment (for streaming).
    public func decodeToken(_ id: Int, skipSpecial: Bool = true) -> String {
        decode([id], skipSpecial: skipSpecial)
    }

    // MARK: - Byte maps

    private static func buildByteMaps(_ b2u: inout [UInt8: Character], _ u2b: inout [Character: UInt8]) {
        var bs: [Int] = []
        bs.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
        bs.append(contentsOf: 0xA1...0xAC)
        bs.append(contentsOf: 0xAE...0xFF)
        var cs = bs
        var n = 0
        for b in 0...255 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        for (b, c) in zip(bs, cs) {
            let scalar = Unicode.Scalar(c)!
            let ch = Character(scalar)
            b2u[UInt8(b)] = ch
            u2b[ch] = UInt8(b)
        }
    }
}
