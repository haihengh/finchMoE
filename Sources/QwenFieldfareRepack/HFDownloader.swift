import Foundation

/// Downloads model files from HuggingFace using a `URLSessionDownloadTask` with
/// a progress delegate, optional `HF_TOKEN` bearer auth, and size-based skipping
/// of files that are already present.
public final class HFDownloader: NSObject, @unchecked Sendable {

    public struct Progress: Sendable {
        public let filename: String
        public let bytesReceived: Int64
        public let totalBytes: Int64
        public var fraction: Double {
            totalBytes > 0 ? Double(bytesReceived) / Double(totalBytes) : 0
        }
    }

    public enum DownloadError: Error, CustomStringConvertible {
        case badURL(String)
        case httpError(Int, String)
        case ioError(String)

        public var description: String {
            switch self {
            case .badURL(let s): return "HFDownloader: bad URL \(s)"
            case .httpError(let code, let s): return "HFDownloader: HTTP \(code) for \(s)"
            case .ioError(let s): return "HFDownloader: IO error \(s)"
            }
        }
    }

    public let repo: String
    public let revision: String
    private let token: String?

    /// Default file list for `mlx-community/Qwen3-30B-A3B-4bit`.
    public static let defaultFiles: [String] = [
        "config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    // Progress + completion plumbing for the active download.
    private var progressHandler: (@Sendable (Progress) -> Void)?
    private var currentFilename: String = ""
    private var destination: URL?
    private var continuation: CheckedContinuation<URL, Error>?

    public init(repo: String = "mlx-community/Qwen3-30B-A3B-4bit",
                revision: String = "main",
                token: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]) {
        self.repo = repo
        self.revision = revision
        self.token = token
        super.init()
    }

    private func fileURL(_ filename: String) -> URL? {
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        comps.path = "/\(repo)/resolve/\(revision)/\(filename)"
        return comps.url
    }

    private func request(for url: URL) -> URLRequest {
        var req = URLRequest(url: url)
        if let token, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.setValue("QwenFieldfare/1.0", forHTTPHeaderField: "User-Agent")
        return req
    }

    /// Downloads a single file into `destDir`. Skips if a same-size file exists.
    public func download(filename: String,
                         into destDir: URL,
                         onProgress: (@Sendable (Progress) -> Void)? = nil) async throws -> URL {
        guard let url = fileURL(filename) else { throw DownloadError.badURL(filename) }
        let dest = destDir.appendingPathComponent(filename)
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let expectedSize = await contentLength(url: url)
        if let attrs = try? FileManager.default.attributesOfItem(atPath: dest.path),
           let sz = attrs[.size] as? Int64, expectedSize > 0, sz == expectedSize {
            onProgress?(Progress(filename: filename, bytesReceived: sz, totalBytes: sz))
            return dest
        }

        self.progressHandler = onProgress
        self.currentFilename = filename
        self.destination = dest
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        defer {
            self.progressHandler = nil
            self.destination = nil
            session.finishTasksAndInvalidate()
        }

        let result = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<URL, Error>) in
            self.continuation = cont
            let task = session.downloadTask(with: request(for: url))
            task.resume()
        }
        onProgress?(Progress(filename: filename, bytesReceived: expectedSize, totalBytes: expectedSize))
        return result
    }

    /// Downloads all files in the list into `destDir`, printing progress.
    public func downloadAll(_ files: [String] = HFDownloader.defaultFiles,
                            into destDir: URL) async throws -> [URL] {
        var urls: [URL] = []
        for f in files {
            FileHandle.standardError.write(Data("↓ \(f)\n".utf8))
            let u = try await download(filename: f, into: destDir) { p in
                let pct = Int(p.fraction * 100)
                let msg = String(format: "\r  %@  %3d%%  (%.2f/%.2f GB)",
                                 p.filename, pct,
                                 Double(p.bytesReceived) / 1e9,
                                 Double(p.totalBytes) / 1e9)
                FileHandle.standardError.write(Data(msg.utf8))
            }
            FileHandle.standardError.write(Data("\n".utf8))
            urls.append(u)
        }
        return urls
    }

    private func contentLength(url: URL) async -> Int64 {
        var req = request(for: url)
        req.httpMethod = "HEAD"
        if let (_, response) = try? await URLSession.shared.data(for: req),
           let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) {
            return response.expectedContentLength
        }
        return 0
    }
}

// MARK: - URLSessionDownloadDelegate (progress)

extension HFDownloader: URLSessionDownloadDelegate {
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64,
                           totalBytesWritten: Int64,
                           totalBytesExpectedToWrite: Int64) {
        progressHandler?(Progress(filename: currentFilename,
                                  bytesReceived: totalBytesWritten,
                                  totalBytes: totalBytesExpectedToWrite))
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didFinishDownloadingTo location: URL) {
        // Validate HTTP status.
        if let http = downloadTask.response as? HTTPURLResponse,
           !(200...299).contains(http.statusCode) {
            continuation?.resume(throwing: DownloadError.httpError(http.statusCode, currentFilename))
            continuation = nil
            return
        }
        guard let dest = destination else {
            continuation?.resume(throwing: DownloadError.ioError("no destination"))
            continuation = nil
            return
        }
        // Move the temp file into place synchronously inside this callback
        // (the framework deletes `location` once this method returns).
        do {
            if FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.moveItem(at: location, to: dest)
            continuation?.resume(returning: dest)
        } catch {
            continuation?.resume(throwing: DownloadError.ioError("moving \(currentFilename): \(error)"))
        }
        continuation = nil
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask,
                           didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
