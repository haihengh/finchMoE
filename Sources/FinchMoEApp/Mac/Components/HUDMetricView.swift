import FinchMoEMacPresentation
import SwiftUI

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated: Bool

    init(value: String, label: String, animated: Bool = true) {
        self.value = value
        self.label = label
        self.animated = animated
    }

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .contentTransition(animated ? .numericText() : .identity)
                .animation(animated ? .snappy(duration: 0.25) : nil, value: value)
            Text(label)
                .font(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
    }
}
