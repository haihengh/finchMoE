import FinchMoEMacPresentation
#if canImport(UIKit)
import FinchMoEAppCore
import SwiftUI
import UIKit

/// The system camera sheet. iOS-only: the Mac shell has no camera source, and
/// the paperclip/file importer already covers photos there.
struct CameraPicker: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    /// Same model-aware import budget the photo/file path uses. A 3.8 image
    /// on iPhone is downscaled to 1024 px before it ever reaches the tower.
    let maxDimension: CGFloat
    let onCapture: (Data) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ picker: UIImagePickerController,
                                context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate,
        UINavigationControllerDelegate {
        private let parent: CameraPicker

        init(_ parent: CameraPicker) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage,
               let data = image.normalizedJPEGData(
                   maxDimension: parent.maxDimension) {
                parent.onCapture(data)
            }
            parent.isPresented = false
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.isPresented = false
        }
    }
}

private extension UIImage {
    /// Bake the camera orientation into the pixels and cap the longest side at
    /// the same budget `AppImagePreparation` uses, so a 48 MP capture never
    /// materializes a full-resolution RGBA context.
    func normalizedJPEGData(
        maxDimension: CGFloat = CGFloat(AppImagePreparation.maximumPixelDimension),
        quality: CGFloat = 0.9
    ) -> Data? {
        guard size.width > 0, size.height > 0 else { return nil }
        let longest = max(size.width, size.height)
        let factor = min(1, maxDimension / longest)
        let target = CGSize(width: max(1, (size.width * factor).rounded(.down)),
                            height: max(1, (size.height * factor).rounded(.down)))
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: target, format: format)
        let normalized = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
        return normalized.jpegData(compressionQuality: quality)
    }
}
#endif
