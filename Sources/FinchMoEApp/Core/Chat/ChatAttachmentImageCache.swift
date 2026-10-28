import Foundation

/// Small in-memory cache for persisted chat images. Bubbles re-render while a
/// reply streams; without this every render would read the attachment files.
public enum ChatAttachmentImageCache {
    // NSCache is documented as thread-safe; the annotation only waives the
    // compiler's conservative global-state check.
    nonisolated(unsafe) private static let cache = NSCache<NSString, NSData>()

    public static func data(_ attachment: ChatImageAttachment,
                            store: ChatSessionFileStore?) -> Data? {
        let key = attachment.fileName as NSString
        if let cached = cache.object(forKey: key) {
            return cached as Data
        }
        guard let data = store?.attachmentData(relativePath: attachment.fileName) else {
            return nil
        }
        cache.setObject(data as NSData, forKey: key,
                        cost: data.count)
        return data
    }

    public static func remove(fileName: String) {
        cache.removeObject(forKey: fileName as NSString)
    }

    public static func removeAll() {
        cache.removeAllObjects()
    }
}
