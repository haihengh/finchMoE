import Testing
@testable import FinchMoEAppCore

@Suite struct AppMemorySamplerTests {
    @Test func samplingDoesNotThrowAndTracksPeak() {
        let sampler = AppMemorySampler()
        sampler.resetPeak()
        _ = sampler.sample()
        _ = sampler.sample()
        if let peak = sampler.peakBytes {
            #expect(peak > 0)
        }
    }

    /// The model's mmap'd weights are clean and absent from the footprint;
    /// the reported RAM has to include them, which is the resident figure.
    @Test func samplingReportsResidentWhenItExceedsFootprint() {
        let sample = AppMemorySample(footprintBytes: 1_700_000_000,
                                     residentBytes: 5_400_000_000)
        let sampler = AppMemorySampler(processMemory: { sample })

        #expect(sampler.sample() == 5_400_000_000)
        #expect(sampler.peakBytes == 5_400_000_000)
    }

    /// Compressed memory can push the footprint above the resident set; the
    /// readout must not go backwards when that happens either.
    @Test func samplingReportsFootprintWhenItExceedsResident() {
        let sample = AppMemorySample(footprintBytes: 900_000_000,
                                     residentBytes: 700_000_000)
        let sampler = AppMemorySampler(processMemory: { sample })

        #expect(sampler.sample() == 900_000_000)
    }

    @Test func failedSampleDoesNotSetPeak() {
        let sampler = AppMemorySampler(processMemory: { nil })

        #expect(sampler.sample() == nil)
        #expect(sampler.peakBytes == nil)
    }
}
