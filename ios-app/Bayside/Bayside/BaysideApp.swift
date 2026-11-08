import SwiftUI
import HarborMLX

@main
struct BaysideApp: App {
    @State private var store = AppStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(store)
                .tint(Brand.accent)
                .task { store.start() }
                .onOpenURL { store.open($0) }
                .onChange(of: scenePhase) { _, phase in
                    // No GPU work in the background (iOS terminates the app for it).
                    MLXStudent.isBackground = phase == .background
                    Diag.write("scene \(phase)")
                }
        }
    }
}

struct RootView: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        @Bindable var store = store
        TabView(selection: $store.tab) {
            HomeView().tabItem { Label("Home", systemImage: "house") }.tag(Tab.home)
            AppointmentsView().tabItem { Label("Visits", systemImage: "calendar") }.tag(Tab.appointments)
            ChatView().tabItem { Label("Ask Bayside", systemImage: "bubble.left.and.text.bubble.right") }.tag(Tab.chat)
            ClaimsView().tabItem { Label("Claims", systemImage: "doc.text") }.tag(Tab.claims)
            LedgerView().tabItem { Label("Privacy", systemImage: "lock.shield") }.tag(Tab.ledger)
        }
        .overlay(alignment: .top) {
            if let t = store.toast {
                Label(t, systemImage: "sparkles")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(.regularMaterial, in: Capsule())
                    .shadow(radius: 8, y: 4)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }
}
