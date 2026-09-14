// A GPU ring-read load generator, the missing arm of the IO-20 open item.
//
// The hypothesis it exists to test: the engine's 3.8 read deficit is GPU
// contention on the expert slot ring -- the drive DMA-ing bytes into a page
// while the GPU reads expert weights back out of the same allocation. The
// engine cannot be asked this question directly (`FQ_EXPERT_TRACE` is
// capture-only, so nop'ing the readback changes the token stream and with it
// the reads), so the experiment is inverted: take the offline replay, which
// has the ring and no GPU, and ADD the GPU.
//
// It therefore has to match what the engine actually allocates, or the dose is
// not the engine's dose: same `posix_memalign` + `makeBuffer(bytesNoCopy:)`
// over pinned anonymous pages, same sizes (1.98 GiB for 3.8's ring, 1.05 for
// 3.6's). The rate is settable because the engine's is measurable -- 648.8
// MB/step over a 38.8 ms routed phase is 16.7 GB/s on 3.8, 227.5 over 19.1 is
// 11.9 on 3.6 -- and a generator that reads flat-out would test a dose no
// engine run produces.
//
// Deliberately memory-traffic only, no compute: the question is whether bytes
// crossing the fabric cost the drive, and a GEMV would confound that with
// ALU occupancy.
//
//   swiftc -O gpu_load.swift -o gpu_load
//   ./gpu_load --gib 1.98 --gbps 16.7 --seconds 30
//
// Prints one line per pass and a summary; `--quiet` prints only the summary.

import Foundation
import Metal

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void ring_read(device const uint4 *p      [[buffer(0)]],
                      device uint       *sink    [[buffer(1)]],
                      constant uint     &start   [[buffer(2)]],
                      constant uint     &count   [[buffer(3)]],
                      constant uint     &stride  [[buffer(4)]],
                      uint gid [[thread_position_in_grid]])
{
    uint acc = 0;
    for (uint i = gid; i < count; i += stride) {
        uint4 v = p[start + i];
        acc ^= (v.x ^ v.y) ^ (v.z ^ v.w);
    }
    // Data-dependent store. A plain accumulator would be dead code and the
    // loads with it; a per-thread atomic would put contention on the path
    // being measured. One never-taken branch per thread is neither.
    if (acc == 0x9E3779B9u) { sink[0] = gid; }
}
"""

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data("gpu_load: \(message)\n".utf8))
    exit(2)
}

func number(_ text: String, _ name: String) -> Double {
    guard let v = Double(text) else { die("\(name): not a number: \(text)") }
    return v
}

// MARK: - Arguments

var gib = 1.98
var gbps = 0.0          // 0 = flat out
var seconds = 30.0
var quiet = false
var burstMB = 0.0       // 0 = one burst reads the whole ring
var periodMS = 0.0      // 0 = back to back (period == burst duration)
var noDispatch = false  // allocate and idle; never launch the kernel

var args = Array(CommandLine.arguments.dropFirst())
while let flag = args.first {
    args.removeFirst()
    func value(_ name: String) -> String {
        guard let v = args.first else { die("\(name) needs a value") }
        args.removeFirst()
        return v
    }
    switch flag {
    case "--gib":     gib = number(value("--gib"), "--gib")
    case "--gbps":    gbps = number(value("--gbps"), "--gbps")
    case "--seconds": seconds = number(value("--seconds"), "--seconds")
    case "--burst-mb":  burstMB = number(value("--burst-mb"), "--burst-mb")
    case "--period-ms": periodMS = number(value("--period-ms"), "--period-ms")
    case "--no-dispatch": noDispatch = true
    case "--quiet":   quiet = true
    case "-h", "--help":
        print("""
        usage: gpu_load [--gib N] [--gbps R] [--seconds S]
                        [--burst-mb N] [--period-ms P] [--quiet]

          --gib       ring size, to match the engine's slot pool
          --burst-mb  bytes read per burst (default: the whole ring)
          --period-ms burst start to next burst start (default: back to back)
          --gbps      throttle the read itself (default: flat out)
        """)
        exit(0)
    default: die("unknown flag \(flag)")
    }
}

guard let device = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
guard let queue = device.makeCommandQueue() else { die("no command queue") }

let library: MTLLibrary
do { library = try device.makeLibrary(source: shaderSource, options: nil) }
catch { die("shader compile failed: \(error)") }
guard let function = library.makeFunction(name: "ring_read") else {
    die("ring_read not found")
}
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: function) }
catch { die("pipeline failed: \(error)") }

// MARK: - The ring, shaped like the engine's

let pageSize = Int(getpagesize())
let bytes = Int(gib * 1024 * 1024 * 1024)
let aligned = (bytes + pageSize - 1) / pageSize * pageSize

// `MakeBufferBytesNoCopy` on the engine's side wraps exactly this: page-aligned
// anonymous memory handed to Metal without a copy, so the GPU and the CPU (and
// therefore the NVMe DMA, in the engine's case) are looking at one allocation.
var raw: UnsafeMutableRawPointer?
guard posix_memalign(&raw, pageSize, aligned) == 0, let base = raw else {
    die("posix_memalign of \(aligned) bytes failed")
}
memset(base, 0xA5, aligned)   // touch every page: unfaulted pages read too fast

guard let ring = device.makeBuffer(bytesNoCopy: base,
                                   length: aligned,
                                   options: .storageModeShared,
                                   deallocator: nil) else {
    die("makeBuffer(bytesNoCopy:) returned nil for \(aligned) bytes")
}
guard let sink = device.makeBuffer(length: pageSize, options: .storageModeShared) else {
    die("sink allocation failed")
}

let nUint4 = aligned / 16
let perGroup = pipeline.maxTotalThreadsPerThreadgroup
let maxGroups = max(1, (1 << 22) / perGroup)

// Bytes read per burst -- the engine's own figure is the point of the flag.
// 648.8 MB/step over 3.8's 38.8 ms routed phase, 227.5 over 3.6's 19.1.
let burstUint4 = burstMB > 0
    ? min(nUint4, max(1, Int(burstMB * 1024 * 1024) / 16))
    : nUint4

// The stride the kernel walks with has to be the number of threads actually
// dispatched, or elements are read twice and others not at all.
var burstGroups = min(maxGroups, max(1, (burstUint4 + perGroup - 1) / perGroup))
var burstThreads = UInt32(truncatingIfNeeded: burstGroups * perGroup)

// One burst = `count` uint4s read from `start`, no wrapping. `start` advances
// by one burst per period and resets at the end, so the whole ring is swept
// rather than one hot range re-read -- otherwise the generator's own working
// set shrinks to the burst and it stops modelling a ring.
func runRange(start: Int, count: Int) -> Double {
    guard let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { die("encoder failed") }
    var start32 = UInt32(start)
    var count32 = UInt32(count)
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(ring, offset: 0, index: 0)
    enc.setBuffer(sink, offset: 0, index: 1)
    enc.setBytes(&start32, length: 4, index: 2)
    enc.setBytes(&count32, length: 4, index: 3)
    enc.setBytes(&burstThreads, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: burstGroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(
                                width: perGroup, height: 1, depth: 1))
    enc.endEncoding()
    let t0 = DispatchTime.now().uptimeNanoseconds
    cb.commit()
    cb.waitUntilCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - t0)
}

// `--gbps` throttles the read itself: the engine's 16.7 GB/s effective rate is
// compute-bound (it has only 648.8 MB to read and spends 38.8 ms doing it),
// not a bandwidth ceiling, so a generator reading the same bytes flat out
// would hit the memory system 5x harder over a 5x shorter window. Delivered by
// slicing the burst and idling between slices, since a single dispatch runs to
// completion and cannot be paced from outside.
let sliceBytes = gbps > 0
    ? max(16, Int(gbps * 1e9 * 0.005) / 16 * 16)   // ~5 ms of read per slice
    : Int.max                                      // unthrottled: one slice

func runBurst(start: Int, count: Int) -> Double {
    let burstStart = DispatchTime.now().uptimeNanoseconds
    var read = 0
    var off = start
    while read < count {
        let slice = min(count - read, sliceBytes / 16)
        // `--no-dispatch` keeps everything but the kernel: the same Metal
        // buffer, the same posix_memalign, the same pages touched, the same
        // burst/period shape. It separates "a Metal allocation of this size
        // competes with the replay's ring" from "the GPU reading it does",
        // which the CPU control cannot, because that one never touches Metal.
        if !noDispatch { _ = runRange(start: off, count: slice) }
        let used = Double(DispatchTime.now().uptimeNanoseconds - burstStart) / 1e9
        let want = Double((read + slice) * 16) / (gbps * 1e9)
        if want > used { Thread.sleep(forTimeInterval: want - used) }
        read += slice
        off += slice
    }
    return Double(DispatchTime.now().uptimeNanoseconds - burstStart)
}

// `--period-ms` idles between bursts to stand in for the rest of a step, so
// that one burst lands per replay pass rather than one per burst duration.
let targetPeriod = periodMS / 1000.0

let deadline = Date().addingTimeInterval(seconds)
// Calibrate with a plain range, not `runBurst`: that honours `--gbps` and
// would report the throttle as if it were the ring's ceiling.
let peak = Double(burstUint4 * 16) / runRange(start: 0, count: burstUint4) / 1e9

FileHandle.standardError.write(Data(String(
    format: "gpu_load: %.3f GiB ring, %.1f MB/burst, %.0f ms period,"
          + " %d threads, flat-out burst %.1f GB/s, %.0fs\n",
    Double(aligned) / 1073741824.0, Double(burstUint4 * 16) / 1e6,
    targetPeriod * 1000, burstGroups * perGroup, peak, seconds).utf8))

var totalNanos = 0.0
var bursts = 0
var bytesRead = 0
var offset = 0
let overallStart = DispatchTime.now().uptimeNanoseconds

while Date() < deadline {
    let burstStart = DispatchTime.now().uptimeNanoseconds
    let count = min(burstUint4, nUint4 - offset)
    let dt = runBurst(start: offset, count: count)
    bursts += 1
    totalNanos += dt
    bytesRead += count * 16
    offset = (offset + count >= nUint4) ? 0 : offset + count
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - overallStart) / 1e9

    if !quiet {
        FileHandle.standardError.write(Data(String(
            format: "  burst %4d  %.2f ms  %.1f MB  %.1f GB/s  elapsed %.1fs\n",
            bursts, dt / 1e6, Double(count * 16) / 1e6,
            Double(count * 16) / dt, elapsed).utf8))
    }

    // Idle out the rest of the period. Without this the generator reads at
    // the ring's own bandwidth, which is not a dose any engine run produces.
    if targetPeriod > 0 {
        let used = Double(DispatchTime.now().uptimeNanoseconds - burstStart) / 1e9
        let rest = targetPeriod - used
        if rest > 0 { Thread.sleep(forTimeInterval: rest) }
    }
}

let wall = Double(DispatchTime.now().uptimeNanoseconds - overallStart) / 1e9
let delivered = Double(bytesRead) / wall / 1e9
let duty = totalNanos / (wall * 1e9)

print(String(format: "gpu_load: %d bursts, %.1f MB read, %.1fs wall, "
                   + "%.2f GB/s delivered, %.1f%% duty, %.3f GiB ring",
             bursts, Double(bytesRead) / 1e6, wall, delivered, duty * 100,
             Double(aligned) / 1073741824.0))

// `deallocator: nil` means Metal never frees `base`; the process is about to
// exit, and the engine leaks its ring the same way for the same reason.
