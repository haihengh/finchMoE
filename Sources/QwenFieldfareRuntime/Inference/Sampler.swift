import Foundation
import Metal

/// Token sampler: greedy (temperature == 0), temperature scaling, and top-p
/// nucleus sampling. Reads fp16 logits directly from a shared `MTLBuffer`.
///
/// Sampling runs on the CPU by default (vocab ≈ 152K is trivial for the CPU and
/// avoids a GPU round-trip); the Metal `sample_argmax` / `sample_top_p` kernels
/// are available via `KernelPipelines` for a fully-on-GPU path if desired.
public struct Sampler {

    public struct Config: Sendable {
        public var temperature: Float
        public var topP: Float
        public var seed: UInt64?

        public init(temperature: Float = 0.7, topP: Float = 0.9, seed: UInt64? = nil) {
            self.temperature = temperature
            self.topP = topP
            self.seed = seed
        }
    }

    public var config: Config
    private var rng: SplitMix64

    public init(config: Config) {
        self.config = config
        self.rng = SplitMix64(seed: config.seed ?? UInt64.random(in: .min ... .max))
    }

    /// Samples the next token id from fp16 logits stored in `buffer`.
    public mutating func sample(logitsBuffer buffer: MTLBuffer, vocabSize: Int) -> Int {
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: vocabSize)
        return sample(logits: ptr, vocabSize: vocabSize)
    }

    /// Samples from a raw fp16 logits pointer.
    public mutating func sample(logits: UnsafePointer<UInt16>, vocabSize: Int) -> Int {
        if config.temperature <= 0 {
            return argmax(logits: logits, vocabSize: vocabSize)
        }
        return topPSample(logits: logits, vocabSize: vocabSize)
    }

    // MARK: - Greedy

    public func argmax(logits: UnsafePointer<UInt16>, vocabSize: Int) -> Int {
        var best = -Float.infinity
        var bestIdx = 0
        for i in 0..<vocabSize {
            let v = Self.halfToFloat(logits[i])
            if v > best { best = v; bestIdx = i }
        }
        return bestIdx
    }

    // MARK: - Top-p nucleus

    private mutating func topPSample(logits: UnsafePointer<UInt16>, vocabSize: Int) -> Int {
        let invT = 1.0 / config.temperature

        // 1. Max for numerical stability.
        var maxLogit = -Float.infinity
        for i in 0..<vocabSize {
            let v = Self.halfToFloat(logits[i]) * invT
            if v > maxLogit { maxLogit = v }
        }

        // 2. Softmax probabilities + index list.
        var probs = [Float](repeating: 0, count: vocabSize)
        var sum: Float = 0
        for i in 0..<vocabSize {
            let e = expf(Self.halfToFloat(logits[i]) * invT - maxLogit)
            probs[i] = e
            sum += e
        }
        let invSum = 1.0 / sum
        for i in 0..<vocabSize { probs[i] *= invSum }

        // 3. Sort indices by descending probability.
        var indices = Array(0..<vocabSize)
        indices.sort { probs[$0] > probs[$1] }

        // 4. Build nucleus until cumulative >= topP.
        var cumulative: Float = 0
        var nucleus: [Int] = []
        for idx in indices {
            nucleus.append(idx)
            cumulative += probs[idx]
            if cumulative >= config.topP { break }
        }

        // 5. Sample proportionally within the nucleus.
        let r = Float(rng.nextUnitDouble()) * cumulative
        var acc: Float = 0
        for idx in nucleus {
            acc += probs[idx]
            if acc >= r { return idx }
        }
        return nucleus.last ?? argmax(logits: logits, vocabSize: vocabSize)
    }

    // MARK: - fp16 → float

    @inline(__always)
    static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h & 0x7C00) >> 10
        let mant = UInt32(h & 0x03FF)
        var bits: UInt32
        if exp == 0 {
            if mant == 0 {
                bits = sign
            } else {
                // Subnormal — normalize.
                var e: Int32 = -1
                var m = mant
                repeat { e += 1; m <<= 1 } while (m & 0x0400) == 0
                m &= 0x03FF
                let newExp = UInt32(127 - 15 - e)
                bits = sign | (newExp << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            bits = sign | 0x7F800000 | (mant << 13) // Inf / NaN
        } else {
            let newExp = exp &- 15 &+ 127
            bits = sign | (newExp << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }
}

/// Small, fast, deterministic PRNG (SplitMix64) for reproducible sampling.
public struct SplitMix64 {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform double in [0, 1).
    public mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) * (1.0 / 9007199254740992.0)
    }
}
