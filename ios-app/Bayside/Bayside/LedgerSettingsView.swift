import SwiftUI

struct LedgerView: View {
    @Environment(AppStore.self) private var store
    @State private var selected: LedgerRow?

    var body: some View {
        NavigationStack {
            List {
                if let l = store.ledger {
                    Section {
                        HStack(spacing: 0) {
                            stat("\(l.stats.onDevice)", "on device", Brand.teal)
                            stat("\(l.stats.sent)", "sent", Brand.cloud)
                            stat("\(l.stats.blocked)", "blocked", .red)
                            stat("\(l.stats.queued)", "queued", .orange)
                        }
                        Label("\(l.stats.rawPiiFieldsSent) raw personal fields sent", systemImage: l.stats.rawPiiFieldsSent == 0 ? "checkmark.seal.fill" : "exclamationmark.octagon.fill")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(l.stats.rawPiiFieldsSent == 0 ? .green : .red)
                    } footer: {
                        Text("Every request that left this phone, with the exact text sent. Names, IDs and amounts are replaced by placeholders before anything leaves.")
                    }
                    Section("Cloud calls") {
                        if l.rows.isEmpty {
                            Text("Nothing has left this phone yet.").foregroundStyle(.secondary)
                        }
                        ForEach(l.rows) { r in
                            Button { selected = r } label: { row(r) }.buttonStyle(.plain)
                        }
                    }
                }
            }
            .navigationTitle("Privacy")
            .refreshable { await store.refreshLedger() }
            .task { await store.refreshLedger() }
            .sheet(item: $selected) { r in LedgerDetail(row: r) }
        }
    }

    private func stat(_ v: String, _ l: String, _ c: Color) -> some View {
        VStack(spacing: 2) {
            Text(v).font(.title2.weight(.bold).monospacedDigit()).foregroundStyle(c)
            Text(l).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func row(_ r: LedgerRow) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(reasonLabel(r.reason)).font(.subheadline.weight(.semibold))
                Spacer()
                StatusChip(text: r.status, color: statusColor(r.status))
            }
            if let req = r.request {
                PlaceholderText(text: "“\(req)”", font: .caption).foregroundStyle(.secondary).lineLimit(3)
            }
            if let n = r.note { Text(n).font(.caption).foregroundStyle(.secondary).lineLimit(2) }
            HStack(spacing: 6) {
                Text(timeLabel(r.at))
                if r.reason.hasPrefix("escalation:") { Text(String(r.reason.dropFirst("escalation:".count))).font(.caption2.monospaced()) }
            }
            .font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

func reasonLabel(_ r: String) -> String {
    if r == "shadow" { return "Shadow quality check" }
    if r == "offline_queue" { return "Saved while offline" }
    if r.hasPrefix("escalation:") {
        let why = String(r.dropFirst("escalation:".count))
        if why.contains("policy") { return "Escalated: claim over $1,000" }
        if why.contains("grounded") { return "Escalated: unverified number" }
        if why.contains("max_steps") { return "Escalated: needed more steps" }
        if why.contains("schema") || why.contains("json") || why.contains("unknown") { return "Escalated: invalid on-device step" }
        return "Escalated: check failed"
    }
    return r
}

func statusColor(_ s: String) -> Color {
    switch s { case "sent", "replayed": return Brand.cloud; case "blocked": return .red; default: return .orange }
}

func timeLabel(_ iso: String) -> String {
    let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
    guard let d = f.date(from: iso) else { return iso }
    return d.formatted(date: .omitted, time: .shortened)
}

struct LedgerDetail: View {
    var row: LedgerRow
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    LabeledContent("Why", value: reasonLabel(row.reason))
                    LabeledContent("Router reason", value: row.reason).font(.footnote.monospaced())
                    LabeledContent("Status", value: row.status)
                    LabeledContent("When", value: timeLabel(row.at))
                    if let ms = row.latencyMs { LabeledContent("Cloud latency", value: String(format: "%.1f s", ms / 1000)) }
                    if let c = row.costUsd { LabeledContent("Cost", value: String(format: "$%.4f", c)) }
                    if let m = row.model { LabeledContent("Model", value: m) }
                    if let f = row.findings, !f.isEmpty { LabeledContent("Blocked by scan", value: f.joined(separator: ", ")) }
                }
                if let q = row.request {
                    Section("Member's request, as sent") { PlaceholderText(text: q, font: .body) }
                }
                if let p = row.payload {
                    Section("Exact text sent") { PlaceholderText(text: p) }
                }
                if let r = row.response {
                    Section("What came back") { PlaceholderText(text: r) }
                }
                if let n = row.note { Section("Note") { Text(n) } }
            }
            .navigationTitle("Cloud call")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

struct SettingsView: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var url = ""
    @State private var backend = ""

    var body: some View {
        NavigationStack {
            Form {
                if let s = store.status {
                    Section("On-device model") {
                        LabeledContent("Model", value: s.model.name)
                        LabeledContent("Base", value: s.model.base)
                        LabeledContent("Round", value: s.model.round.map(String.init) ?? "?")
                        LabeledContent("Version", value: s.model.version)
                        LabeledContent("Size", value: String(format: "%.2f GB", s.model.sizeBytes / 1e9))
                        LabeledContent("Status", value: s.ready ? "Ready" : (s.error ?? "Loading…"))
                    }
                    Section {
                        Toggle("Airplane mode (simulated)", isOn: Binding(get: { s.settings.offline },
                                                                          set: { store.updateSettings(["offline": $0]) }))
                        Picker("Shadow quality checks", selection: Binding(get: { s.settings.shadowRate },
                                                                          set: { store.updateSettings(["shadow_rate": $0]) })) {
                            Text("Off").tag(0.0); Text("1%").tag(0.01); Text("15% (demo)").tag(0.15)
                        }
                        Toggle("Share redacted traces to improve the model", isOn: Binding(get: { s.settings.sync },
                                                                                       set: { store.updateSettings(["sync": $0]) }))
                    } header: { Text("Privacy and network") } footer: {
                        Text("Airplane mode blocks every cloud call from the assistant. Requests that would have gone to the cloud get a best-effort answer and are sent for review when you reconnect.")
                    }
                }
                Section {
                    #if targetEnvironment(simulator)
                    LabeledContent("Assistant runs on", value: RuntimeMode.mac.label)
                    #else
                    Picker("Assistant runs on", selection: Binding(get: { store.runtimeMode }, set: { store.runtimeMode = $0 })) {
                        ForEach(RuntimeMode.allCases) { Text($0.label).tag($0) }
                    }
                    #endif
                    if let s = store.status { LabeledContent("Status", value: s.runtime) }
                    if store.runtimeMode == .mac {
                        TextField("Mac runtime URL", text: $url).textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onSubmit { store.runtimeURL = url; store.start() }
                    } else {
                        TextField("Cloud backend URL", text: $backend).textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onSubmit { store.backendURL = backend }
                    }
                } header: { Text("Runtime") } footer: {
                    Text(store.runtimeMode == .device
                         ? "The fine-tuned model, router, redaction and tools run on this iPhone. Only escalations go to the cloud backend, and real airplane mode works."
                         : "MLX can't run in the iOS Simulator, so here the model runs on the Mac with the same weights, reached over the network.")
                }
                Section("Demo") {
                    Picker("Member", selection: Binding(get: { store.memberId }, set: { store.switchMember($0) })) {
                        ForEach(store.members) { m in Text("\(m.name) · \(m.plan)").tag(m.memberId) }
                    }
                    Button("Reset demo data", role: .destructive) { store.resetDemo() }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .onAppear { url = store.runtimeURL; backend = store.backendURL }
        }
    }
}
