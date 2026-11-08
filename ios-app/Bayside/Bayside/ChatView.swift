import SwiftUI

struct ChatView: View {
    @Environment(AppStore.self) private var store
    @State private var draft = ""
    @FocusState private var focused: Bool

    static let suggestions = [
        "Move my eye exam to Friday",
        "How much of my deductible is left?",
        "Is physical therapy covered?",
        "I was diagnosed with diabetes, is blood work covered?",
        "File a claim: I saw Dr. Percival Nilsson, my orthopedist in San Mateo, on November 2. The bill was $1,450.",
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if store.status?.settings.offline == true {
                    Label("Offline: answers come from the on-device model only", systemImage: "airplane")
                        .font(.caption.weight(.semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).padding(.vertical, 6)
                        .background(Color.orange.gradient)
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            RuntimeBanner()
                            if store.chat.isEmpty { intro }
                            ForEach(store.chat) { entry in
                                EntryView(entry: entry).id(entry.id)
                            }
                            if let who = store.thinking { ThinkingRow(who: who).id("thinking") }
                            Color.clear.frame(height: 4).id("bottom")
                        }
                        .padding(16)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: store.chat.count) { withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
                    .onChange(of: store.thinking) { withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
                }
                inputBar
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Ask Bayside")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { store.newConversation() } label: { Image(systemName: "square.and.pencil") }
                        .disabled(store.busy).accessibilityLabel("New conversation")
                }
            }
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 12) {
            Card {
                Label("Your assistant runs on this phone", systemImage: "iphone.gen3").font(.headline).foregroundStyle(Brand.teal)
                Text("Routine requests are answered on the device. If a check fails, the request goes to Bayside Cloud with your personal details replaced by placeholders, and it shows up in Privacy. Medical topics never leave the phone.")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            Text("Try").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(Self.suggestions, id: \.self) { s in
                Button { store.send(s) } label: {
                    Text(s).font(.subheadline).multilineTextAlignment(.leading)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .disabled(store.busy || store.status?.ready != true)
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message Bayside", text: $draft, axis: .vertical)
                .lineLimit(1...4)
                .focused($focused)
                .padding(.horizontal, 14).padding(.vertical, 9)
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20))
                .onSubmit(send)
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill").font(.system(size: 32))
            }
            .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty || store.busy)
            .accessibilityLabel("Send")
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.bar)
    }

    private func send() {
        let t = draft
        draft = ""
        store.send(t)
    }
}

private struct EntryView: View {
    @Environment(AppStore.self) private var store
    var entry: ChatEntry
    @State private var expanded = false

    var body: some View {
        switch entry.kind {
        case .user(let text):
            HStack {
                Spacer(minLength: 48)
                Text(text).padding(.horizontal, 14).padding(.vertical, 10)
                    .background(Brand.accent, in: RoundedRectangle(cornerRadius: 18))
                    .foregroundStyle(.white)
            }
        case .notice(let icon, let text, let tone):
            let color: Color = tone == .cloud ? Brand.cloud : tone == .local ? Brand.person : .orange
            let lines = text.components(separatedBy: "\n")
            VStack(alignment: .leading, spacing: 6) {
                Label(lines[0], systemImage: icon).font(.footnote.weight(.semibold)).foregroundStyle(color)
                if lines.count > 1 {
                    DisclosureGroup(isExpanded: $expanded) {
                        PlaceholderText(text: lines.dropFirst().joined(separator: "\n")).padding(.top, 4)
                    } label: { Text("Details").font(.caption) }
                }
            }
            .padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(color.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
        case .step(let label, let status, let who, let detail):
            VStack(alignment: .leading, spacing: 4) {
                Button { expanded.toggle() } label: {
                    HStack(spacing: 8) {
                        switch status {
                        case .running: ProgressView().controlSize(.small)
                        case .ok: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                        }
                        Text(label + (status == .running ? "…" : "")).font(.footnote.weight(.medium))
                        Image(systemName: who == "teacher" ? "icloud" : "iphone").font(.caption2)
                            .foregroundStyle(who == "teacher" ? Brand.cloud : Brand.teal)
                        if detail != nil { Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption2).foregroundStyle(.tertiary) }
                    }
                    .padding(.horizontal, 12).padding(.vertical, 7)
                    .background(Color(.secondarySystemGroupedBackground), in: Capsule())
                }
                .buttonStyle(.plain)
                if case .failed(let err) = status, let err { Text(err).font(.caption).foregroundStyle(.red).padding(.leading, 12) }
                if expanded, let detail {
                    PlaceholderText(text: detail).padding(10)
                        .background(Color(.tertiarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 10))
                }
            }
        case .confirm(let summary, let state):
            Card {
                Label("Confirm change", systemImage: "checkmark.shield").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(summary).font(.headline)
                switch state {
                case .pending:
                    HStack {
                        Button("Confirm") { store.confirm(true) }.buttonStyle(.borderedProminent)
                        Button("Change") { store.confirm(false) }.buttonStyle(.bordered)
                    }
                case .approved: Label("Confirmed", systemImage: "checkmark").font(.subheadline).foregroundStyle(.green)
                case .declined: Label("Not changed", systemImage: "xmark").font(.subheadline).foregroundStyle(.secondary)
                }
            }
        case .reply(let text, let served, let flagged, let latency, _):
            VStack(alignment: .leading, spacing: 6) {
                Text(text).padding(.horizontal, 14).padding(.vertical, 10)
                    .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18))
                HStack(spacing: 6) {
                    ProvenanceTag(served: served)
                    if flagged { StatusChip(text: "Flagged for review", color: .orange) }
                    if let latency { Text(String(format: "%.1f s", latency)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary) }
                }
            }
            .padding(.trailing, 32)
            .onTapGesture { if served == "cloud" { store.tab = .ledger } }
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.red)
        }
    }
}

private struct ThinkingRow: View {
    var who: String
    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(who == "teacher" ? "Bayside Cloud is working on it…" : "On-device model is thinking…")
                .font(.footnote).foregroundStyle(.secondary)
            Image(systemName: who == "teacher" ? "icloud" : "iphone").font(.caption).foregroundStyle(who == "teacher" ? Brand.cloud : Brand.teal)
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
    }
}
