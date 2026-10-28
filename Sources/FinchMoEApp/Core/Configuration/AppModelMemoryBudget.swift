#if canImport(UIKit)
import UIKit
#endif
import Foundation
import FinchMoE

/// Load-time caps a *device* (iPhone or iPad) applies to the largest vision
/// model, as opposed to a Mac.
///
/// Qwen 3.8 keeps ~3.9 GB of resident weights, and its vision payload adds a
/// ~0.9 GB BF16 tower. On an iPhone whose jetsam ceiling is around 6 GB, the
/// desktop defaults — 16 expert-cache slots and up to 64K context — leave no
/// room for the tower once the KV cache and prefill scratch are allocated.
/// The app therefore runs the 3.8 vision rows with a smaller budget while
/// leaving the user's persisted (shared) settings untouched.
public enum AppModelMemoryBudget {
    public struct Cap: Equatable, Sendable {
        public let expertCacheSlots: Int
        public let contextTokens: Int
        public let prefillChunkTokens: Int

        public init(expertCacheSlots: Int,
                    contextTokens: Int,
                    prefillChunkTokens: Int) {
            self.expertCacheSlots = expertCacheSlots
            self.contextTokens = contextTokens
            self.prefillChunkTokens = prefillChunkTokens
        }
    }

    /// Whether this process is running under a jetsam ceiling, which is to
    /// say on an iPhone or an iPad rather than on a Mac.
    ///
    /// This used to be `#if canImport(UIKit)`, which made the *build* decide a
    /// *runtime* fact. A name saying "phone" is what invited that proxy, and
    /// the proxy was wrong in both directions at once: it handed iPad the
    /// iPhone's numbers (an iPad imports UIKit too), and `isPhone` was the
    /// default argument to `cap(for:)`, so any non-UIKit host — a test
    /// harness, a CLI — silently skipped the cap and the OOM risk that comes
    /// with skipping it. The idiom is the fact, so read it at runtime.
    ///
    /// The cap itself stays the iPhone's, priced for a ~6 GB ceiling: the
    /// iPad's is higher but unmeasured here, and a cap that is too tight only
    /// costs prefill speed while one that is too loose gets the process killed
    /// mid-generation. `AppMemorySampler.memoryLimitBytes` reports the ceiling
    /// a given device actually grants, which is the number a pad tier would be
    /// priced from.
    public static var runsUnderAJetsamCeiling: Bool {
        #if canImport(UIKit)
        switch UIDevice.current.userInterfaceIdiom {
        case .phone, .pad: return true
        default:
            // `.tv`, `.watch`, `.vision`, `.unspecified`. None of them run
            // this app today, but every one of them is still a jetsam-managed
            // device, so returning true keeps an unexpected idiom from opting
            // into desktop-sized caches and getting killed mid-generation.
            return true
        }
        #else
        // A Mac has no jetsam ceiling. This is also what a non-GUI host sees,
        // so the name describes the risk rather than the hardware.
        return false
        #endif
    }

    /// Ten slots is the lowest the 3.8 decoder can use: it routes to ten
    /// experts per token, and a layer's cache must hold at least top-k.
    /// That still cuts the cache from ~1.2 GB at 16 to ~0.74 GB. A 4K context
    /// keeps KV near 0.1 GB, and a 128-token chunk halves the prefill scratch
    /// of the 256 setting. The remaining headroom is what lets the vision
    /// tower and its activations fit.
    public static func cap(for descriptor: AppModelInstallDescriptor,
                           underJetsam: Bool = Self.runsUnderAJetsamCeiling) -> Cap? {
        guard underJetsam,
              descriptor.requiresVision,
              descriptor.architecture.isQwen3_8 else { return nil }
        return Cap(expertCacheSlots: 10,
                   contextTokens: AppContextLengthOption.fourK.tokens,
                   prefillChunkTokens: 128)
    }
}
