import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI
#if canImport(AppKit)
import AppKit
#else
import UIKit
#endif

/// A picked image rendered from its stored bytes. Shared by the composer's
/// pending strip and the user bubble so both show the same crop.
struct AttachmentThumbnail: View {
    let data: Data
    var size: CGFloat
    var cornerRadius: CGFloat

    init(data: Data, size: CGFloat = 56, cornerRadius: CGFloat = 10) {
        self.data = data
        self.size = size
        self.cornerRadius = cornerRadius
    }

    private var image: Image? {
        #if canImport(AppKit)
        NSImage(data: data).map { Image(nsImage: $0) }
        #else
        UIImage(data: data).map { Image(uiImage: $0) }
        #endif
    }

    var body: some View {
        Group {
            if let image {
                image
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(FinchPlatformColors.controlBackground)
                    Image(systemName: "photo")
                        .font(.system(size: size * 0.28))
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
        .overlay {
            RoundedRectangle(cornerRadius: cornerRadius)
                .stroke(.separator.opacity(0.5), lineWidth: 0.5)
        }
        .accessibilityLabel("Attached image")
    }
}
