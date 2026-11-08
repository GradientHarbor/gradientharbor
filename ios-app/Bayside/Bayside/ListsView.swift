import SwiftUI

struct AppointmentsView: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        NavigationStack {
            List {
                let groups = Dictionary(grouping: store.state?.appointments ?? []) { String($0.datetime.prefix(10)) }
                if groups.isEmpty { ContentUnavailableView("No upcoming visits", systemImage: "calendar") }
                ForEach(groups.keys.sorted(), id: \.self) { day in
                    Section(Fmt.longDay(day)) {
                        ForEach(groups[day] ?? []) { a in
                            HStack(spacing: 12) {
                                Text(Fmt.time(a.datetime)).font(.subheadline.monospacedDigit().weight(.semibold))
                                    .frame(width: 76, alignment: .leading).foregroundStyle(Brand.accent)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(Fmt.specialty(a.specialty)).font(.subheadline.weight(.semibold))
                                    Text(a.providerName).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if store.highlightedAppointment == a.id {
                                    StatusChip(text: "Updated", color: Brand.teal)
                                }
                            }
                            .listRowBackground(store.highlightedAppointment == a.id ? Brand.teal.opacity(0.10) : nil)
                        }
                    }
                }
            }
            .animation(.spring(duration: 0.45), value: store.state?.appointments)
            .navigationTitle("Visits")
            .refreshable { await store.refreshState() }
        }
    }
}

struct ClaimsView: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        NavigationStack {
            List {
                ForEach(store.state?.claims ?? []) { c in
                    NavigationLink { ClaimDetail(claim: c) } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(c.providerName).font(.subheadline.weight(.semibold))
                                Spacer()
                                StatusChip(text: Fmt.pretty(c.status), color: claimColor(c.status))
                            }
                            HStack {
                                Text("\(Fmt.pretty(c.service)) · \(Fmt.shortDay(c.serviceDate))")
                                Spacer()
                                Text(Fmt.money(c.amountBilled)).monospacedDigit()
                            }
                            .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .listRowBackground(store.highlightedClaim == c.id ? Brand.teal.opacity(0.10) : nil)
                }
            }
            .animation(.spring(duration: 0.45), value: store.state?.claims)
            .navigationTitle("Claims")
            .refreshable { await store.refreshState() }
        }
    }
}

struct ClaimDetail: View {
    var claim: Claim
    var body: some View {
        List {
            Section {
                LabeledContent("Claim", value: claim.claimId)
                LabeledContent("Status") { StatusChip(text: Fmt.pretty(claim.status), color: claimColor(claim.status)) }
                LabeledContent("Provider", value: claim.providerName)
                LabeledContent("Service", value: Fmt.pretty(claim.service))
                LabeledContent("Date of service", value: Fmt.shortDay(claim.serviceDate))
                LabeledContent("Billed", value: Fmt.money(claim.amountBilled))
                if let v = claim.amountCovered { LabeledContent("Covered", value: Fmt.money(v)) }
                if let d = claim.expectedDecisionBy { LabeledContent("Decision expected", value: Fmt.shortDay(d)) }
            }
            if let r = claim.denialReason {
                Section("Why it was denied") { Text(r) }
            }
        }
        .navigationTitle(claim.claimId)
        .navigationBarTitleDisplayMode(.inline)
    }
}
