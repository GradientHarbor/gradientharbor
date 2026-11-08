import SwiftUI

struct HomeView: View {
    @Environment(AppStore.self) private var store
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    RuntimeBanner()
                    header
                    if let s = store.state {
                        NavigationLink { PolicyView() } label: { planCard(s.policy) }.buttonStyle(.plain)
                        if let a = s.appointments.first { nextAppointment(a) }
                        if let c = s.claims.first { latestClaim(c) }
                        askButton
                        deviceCard
                    } else if store.connectionError == nil {
                        ProgressView().frame(maxWidth: .infinity).padding(40)
                    }
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Bayside Mutual")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                        .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .refreshable { await store.refreshAll() }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Good morning, \(store.state?.member.firstName ?? "there")").font(.title2.weight(.bold))
            Text(store.state.map { Fmt.longDay($0.today) } ?? "").font(.subheadline).foregroundStyle(.secondary)
        }
    }

    private func planCard(_ p: Policy) -> some View {
        Card {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(p.planName).font(.headline)
                    Text("Policy \(p.policyId) · \(Fmt.pretty(p.status))").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
            }
            ProgressRow(title: "Deductible", value: p.deductibleMet, total: p.deductible)
            ProgressRow(title: "Out-of-pocket", value: p.outOfPocketSpent, total: p.outOfPocketMax)
        }
    }

    private func nextAppointment(_ a: Appointment) -> some View {
        Card {
            Label("Next visit", systemImage: "calendar").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text("\(Fmt.specialty(a.specialty)) with \(a.providerName)").font(.headline)
            Text("\(Fmt.shortDay(a.datetime)) · \(Fmt.time(a.datetime))").font(.subheadline).foregroundStyle(Brand.accent)
        }
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Brand.teal, lineWidth: store.highlightedAppointment == a.id ? 2 : 0))
        .onTapGesture { store.tab = .appointments }
    }

    private func latestClaim(_ c: Claim) -> some View {
        Card {
            HStack {
                Label("Latest claim", systemImage: "doc.text").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                StatusChip(text: Fmt.pretty(c.status), color: claimColor(c.status))
            }
            Text("\(c.providerName) · \(Fmt.money(c.amountBilled))").font(.headline)
            Text("\(Fmt.pretty(c.service)) on \(Fmt.shortDay(c.serviceDate))").font(.subheadline).foregroundStyle(.secondary)
        }
        .onTapGesture { store.tab = .claims }
    }

    private var askButton: some View {
        Button { store.tab = .chat } label: {
            Label("Ask Bayside", systemImage: "bubble.left.and.text.bubble.right.fill")
                .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
    }

    private var deviceCard: some View {
        Card {
            HStack(spacing: 10) {
                Image(systemName: "iphone.gen3").font(.title2).foregroundStyle(Brand.teal)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Assistant runs on your phone").font(.subheadline.weight(.semibold))
                    if let s = store.status {
                        Text("\(s.model.base) · round \(s.model.round.map(String.init) ?? "?")\(s.settings.offline ? " · offline" : "")")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if let l = store.ledger {
                    VStack(alignment: .trailing, spacing: 0) {
                        Text("\(l.stats.onDevice)/\(l.stats.requests)").font(.headline.monospacedDigit())
                        Text("on device").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .onTapGesture { store.tab = .ledger }
    }
}

struct PolicyView: View {
    @Environment(AppStore.self) private var store
    var body: some View {
        List {
            if let s = store.state {
                Section("Plan") {
                    LabeledContent("Plan", value: s.policy.planName)
                    LabeledContent("Status", value: Fmt.pretty(s.policy.status))
                    LabeledContent("Effective", value: Fmt.shortDay(s.policy.effectiveDate))
                    LabeledContent("Renews", value: Fmt.shortDay(s.policy.renewalDate))
                    LabeledContent("Dependents", value: "\(s.policy.dependents)")
                }
                Section("This year") {
                    ProgressRow(title: "Deductible", value: s.policy.deductibleMet, total: s.policy.deductible)
                    ProgressRow(title: "Out-of-pocket", value: s.policy.outOfPocketSpent, total: s.policy.outOfPocketMax)
                }
                Section("Coverage") {
                    ForEach(s.coverage) { c in
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(c.label).font(.subheadline.weight(.semibold))
                                Spacer()
                                StatusChip(text: c.covered ? "Covered" : "Not covered", color: c.covered ? .green : .secondary)
                            }
                            if c.covered { Text(costLine(c)).font(.caption).foregroundStyle(.secondary) }
                        }
                    }
                }
            }
        }
        .navigationTitle("Policy")
    }

    private func costLine(_ c: Coverage) -> String {
        var parts: [String] = []
        if let p = c.copay, p > 0 { parts.append("\(Fmt.money(p)) copay") }
        if let ci = c.coinsurance, ci > 0 { parts.append("\(Int((ci * 100).rounded()))% coinsurance") }
        if c.requiresReferral { parts.append("referral") }
        if c.requiresPriorAuth { parts.append("prior auth") }
        if let v = c.visitLimitPerYear { parts.append("\(v) visits/yr") }
        return parts.isEmpty ? "No cost to you" : parts.joined(separator: " · ")
    }
}
