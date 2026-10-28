import FinchMoE
import FinchMoEVision
import Foundation
import Metal

/// A prompt built for a vision turn: the ChatML ids, the mRoPE timeline, and
/// the image rows that replace the placeholder token embeddings.
struct AppVisionPrompt {
    let messages: [GFTokenizer.Message]
    let promptIDs: [Int32]
    let multimodal: MultimodalPrefillInput
    let imageCount: Int
}

enum AppVisionPromptBuilder {

    /// Markers as they appear inside one user turn. The tokenizer recognizes
    /// the special-token spellings and emits their ids, so no separate
    /// placeholder-expansion pass is needed.
    static func markerString(tokenCounts: [Int]) -> String {
        var output = ""
        for count in tokenCounts {
            output += "<|vision_start|>"
            output += String(repeating: "<|image_pad|>", count: count)
            output += "<|vision_end|>"
        }
        return output
    }

    static func messages(history: [AppChatTurn],
                         prompt: String,
                         images: [AppImageAttachment],
                         tokenCounts: [UUID: Int]) -> [GFTokenizer.Message] {
        var messages: [GFTokenizer.Message] = []
        messages.reserveCapacity(history.count + 1)
        for turn in history {
            let content: String
            if turn.role == .user, !turn.images.isEmpty {
                let counts = turn.images.compactMap { tokenCounts[$0.id] }
                content = markerString(tokenCounts: counts)
                    + turn.text.trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                content = turn.text
            }
            messages.append(GFTokenizer.Message(
                role: turn.role == .user ? .user : .assistant,
                content: content))
        }
        let currentCounts = images.compactMap { tokenCounts[$0.id] }
        let current = markerString(tokenCounts: currentCounts)
            + prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        messages.append(GFTokenizer.Message(role: .user, content: current))
        return messages
    }

    /// Build the final prompt and the multimodal inputs.
    ///
    /// `embeddings` must contain an entry for every image that survives
    /// `history`, plus every image in `images`. The image runs are matched to
    /// the embedding list in prompt order and their sizes are checked against
    /// the tower's token counts.
    static func build(history: [AppChatTurn],
                      prompt: String,
                      images: [AppImageAttachment],
                      tokenCounts: [UUID: Int],
                      embeddings: [UUID: VisionEngine.ImageEmbedding],
                      tokenizer: GFTokenizer,
                      imageTokenID: Int32,
                      device: MTLDevice,
                      queue: MTLCommandQueue) throws -> AppVisionPrompt {
        // The markers are spelled as special tokens in the rendered prompt, so
        // literal marker text from a user or a model would masquerade as an
        // image run.
        let reserved = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
        for text in [prompt] + history.map(\.text) {
            guard !reserved.contains(where: { text.contains($0) }) else {
                throw AppInferenceError.invalidRequest(
                    "Message text cannot contain vision marker tokens.")
            }
        }
        let messages = messages(history: history, prompt: prompt,
                                images: images, tokenCounts: tokenCounts)
        let rendered = try tokenizer.applyChatTemplate(messages)
        let promptIDs = tokenizer.encode(rendered, addBOS: false)

        let orderedImages = history.flatMap(\.images) + images
        let orderedEmbeddings = orderedImages.compactMap { embeddings[$0.id] }
        guard orderedEmbeddings.count == orderedImages.count else {
            throw AppInferenceError.visionUnsupported(
                "an image was not encoded before the prompt was assembled")
        }

        let runs = imageRuns(in: promptIDs, tokenID: imageTokenID)
        guard runs.count == orderedEmbeddings.count else {
            throw AppInferenceError.visionUnsupported(
                "prompt has \(runs.count) image runs but \(orderedEmbeddings.count) images were attached")
        }
        for (run, embedding) in zip(runs, orderedEmbeddings) {
            guard run.count == embedding.tokenCount else {
                throw AppInferenceError.visionUnsupported(
                    "image run has \(run.count) tokens, the tower produced \(embedding.tokenCount)")
            }
        }

        var positions: [MRoPEPosition] = []
        positions.reserveCapacity(promptIDs.count)
        var hiddenRowPositions: [Int32] = []
        hiddenRowPositions.reserveCapacity(runs.reduce(0) { $0 + $1.count })
        var current: Int32 = 0
        var runIndex = 0
        var index = 0
        while index < promptIDs.count {
            if runIndex < runs.count, runs[runIndex].lowerBound == index {
                let run = runs[runIndex]
                let embedding = orderedEmbeddings[runIndex]
                let mergedH = embedding.mergedGridH
                let mergedW = embedding.mergedGridW
                guard mergedH > 0, mergedW > 0, mergedH * mergedW == run.count else {
                    throw AppInferenceError.visionUnsupported(
                        "image grid \(mergedH)×\(mergedW) does not match its \(run.count) tokens")
                }
                let base = current
                for offset in 0..<run.count {
                    let row = Int32(offset / mergedW)
                    let column = Int32(offset % mergedW)
                    positions.append(MRoPEPosition(temporal: base,
                                                   height: base + row,
                                                   width: base + column))
                    hiddenRowPositions.append(Int32(index + offset))
                }
                current += Int32(max(mergedH, mergedW))
                index = run.upperBound
                runIndex += 1
            } else {
                positions.append(MRoPEPosition(current))
                current += 1
                index += 1
            }
        }
        guard positions.count == promptIDs.count else {
            throw AppInferenceError.visionUnsupported(
                "mRoPE positions cover \(positions.count) of \(promptIDs.count) tokens")
        }

        let combined = try combinedEmbeddings(
            orderedEmbeddings,
            hiddenSize: orderedEmbeddings.first?.textHiddenSize ?? 0,
            device: device, queue: queue)
        let multimodal = MultimodalPrefillInput(
            positions: positions,
            hiddenRows: combined,
            hiddenRowPositions: hiddenRowPositions,
            mropeSection: orderedEmbeddings.first?.mropeSection ?? [],
            decodePositionOffset: Int(current) - promptIDs.count)
        return AppVisionPrompt(messages: messages,
                               promptIDs: promptIDs,
                               multimodal: multimodal,
                               imageCount: orderedImages.count)
    }

    /// Contiguous runs of the image placeholder token.
    static func imageRuns(in ids: [Int32], tokenID: Int32) -> [Range<Int>] {
        var runs: [Range<Int>] = []
        var index = 0
        while index < ids.count {
            guard ids[index] == tokenID else {
                index += 1
                continue
            }
            let start = index
            while index < ids.count, ids[index] == tokenID {
                index += 1
            }
            runs.append(start..<index)
        }
        return runs
    }

    private static func combinedEmbeddings(_ embeddings: [VisionEngine.ImageEmbedding],
                                           hiddenSize: Int,
                                           device: MTLDevice,
                                           queue: MTLCommandQueue) throws -> MTLBuffer {
        let totalRows = embeddings.reduce(0) { $0 + $1.tokenCount }
        let rowBytes = hiddenSize * MemoryLayout<Float16>.stride
        guard totalRows > 0, rowBytes > 0,
              let destination = device.makeBuffer(length: totalRows * rowBytes,
                                                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw AppInferenceError.visionUnsupported(
                "could not allocate the image embedding buffer")
        }
        var offset = 0
        for embedding in embeddings {
            let bytes = embedding.tokenCount * rowBytes
            blit.copy(from: embedding.embeddings, sourceOffset: 0,
                      to: destination, destinationOffset: offset,
                      size: bytes)
            offset += bytes
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw AppInferenceError.visionUnsupported(
                "could not combine the image embeddings: \(error)")
        }
        return destination
    }
}
