import SwiftUI

enum Brand {
    static let accent = Color(red: 0.11, green: 0.36, blue: 0.62)       // Bayside blue
    static let teal = Color(red: 0.05, green: 0.56, blue: 0.53)          // on device
    static let cloud = Color(red: 0.85, green: 0.47, blue: 0.02)         // cloud teacher
    static let person = Color(red: 0.42, green: 0.36, blue: 0.80)        // handed to a person
}

struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) { content }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

/// Text with redaction placeholders ([MEMBER_ID_1], [AMOUNT_2] …) highlighted.
struct PlaceholderText: View {
    var text: String
    var font: Font = .system(.footnote, design: .monospaced)

    var body: some View { Text(attributed).font(font).textSelection(.enabled) }

    private var attributed: AttributedString {
        var out = AttributedString(text)
        guard let re = try? NSRegularExpression(pattern: #"\[[A-Z_]+_\d+\]"#) else { return out }
        let ns = text as NSString
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed() {
            guard let r = Range(m.range, in: text), let ar = Range(r, in: out) else { continue }
            out[ar].foregroundColor = Brand.person
            out[ar].backgroundColor = Brand.person.opacity(0.12)
            out[ar].font = .system(.footnote, design: .monospaced).weight(.semibold)
        }
        return out
    }
}

struct ProvenanceTag: View {
    var served: String
    var body: some View {
        let (icon, label, color): (String, String, Color) = switch served {
        case "cloud": ("icloud", "Cloud", Brand.cloud)
        case "handoff": ("person.fill", "Handed to a person", Brand.person)
        default: ("iphone", "On device", Brand.teal)
        }
        return Label(label, systemImage: icon)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.12), in: Capsule())
    }
}

struct StatusChip: View {
    var text: String
    var color: Color
    var body: some View {
        Text(text).font(.caption.weight(.semibold)).foregroundStyle(color)
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.13), in: Capsule())
    }
}

func claimColor(_ status: String) -> Color {
    switch status {
    case "denied": return .red
    case "paid", "approved": return .green
    case "submitted": return Brand.accent
    default: return .orange
    }
}

struct ProgressRow: View {
    var title: String
    var value: Double
    var total: Double
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline)
                Spacer()
                Text("\(Fmt.money(value)) of \(Fmt.money(total))").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
            }
            ProgressView(value: min(value, total), total: max(total, 1)).tint(Brand.accent)
        }
    }
}

/// Banner shown when the runtime isn't reachable or the model is loading.
struct RuntimeBanner: View {
    @Environment(AppStore.self) private var store
    var body: some View {
        if let err = store.connectionError {
            Label(err, systemImage: "exclamationmark.triangle.fill")
                .font(.footnote).foregroundStyle(.white)
                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                .background(.red.gradient, in: RoundedRectangle(cornerRadius: 12))
        } else if let s = store.status, !s.ready {
            Label(s.error ?? "Loading the on-device model…", systemImage: s.error == nil ? "hourglass" : "exclamationmark.triangle")
                .font(.footnote)
                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
        }
    }
}
