import Foundation
import Observation
import SwiftUI
import HarborCore

/// One row in the support chat.
struct ChatEntry: Identifiable, Equatable {
    enum Kind: Equatable {
        case user(String)
        case notice(icon: String, text: String, tone: Tone)          // escalation, never-leave, handoff
        case step(label: String, status: StepStatus, who: String, detail: String?)
        case confirm(summary: String, state: ConfirmState)
        case reply(text: String, served: String, flagged: Bool, latency: Double?, reason: String?)
        case error(String)
    }
    enum Tone: Equatable { case cloud, local, warn }
    enum StepStatus: Equatable { case running, ok, failed(String?) }
    enum ConfirmState: Equatable { case pending, approved, declined }
    let id = UUID()
    var kind: Kind
}

enum Tab: Hashable { case home, appointments, claims, chat, ledger }

@MainActor
@Observable
final class AppStore {
    // settings
    // Simulator: the runtime is on this Mac at localhost. A real iPhone reaches the Mac over the network.
    static var defaultRuntimeURL: String {
        #if targetEnvironment(simulator)
        return "http://localhost:8100"
        #else
        return "http://Aydars-MacBook-Pro.local:8100"
        #endif
    }
    var runtimeURL: String = UserDefaults.standard.string(forKey: "runtimeURL") ?? AppStore.defaultRuntimeURL {
        didSet { UserDefaults.standard.set(runtimeURL, forKey: "runtimeURL") }
    }
    /// Where the assistant runs. On a real iPhone the default is the phone itself; the Simulator can't run MLX.
    var runtimeMode: RuntimeMode = {
        #if targetEnvironment(simulator)
        return .mac
        #else
        return RuntimeMode(rawValue: UserDefaults.standard.string(forKey: "runtimeMode") ?? "") ?? .device
        #endif
    }() {
        didSet {
            UserDefaults.standard.set(runtimeMode.rawValue, forKey: "runtimeMode")
            chat = []; conversationId = nil; state = nil; ledger = nil; status = nil
            start()
        }
    }
    /// The cloud side (teacher proxy) for on-device mode.
    var backendURL: String = UserDefaults.standard.string(forKey: "backendURL") ?? "http://Aydars-MacBook-Pro.local:8000" {
        didSet {
            UserDefaults.standard.set(backendURL, forKey: "backendURL")
            if let u = URL(string: backendURL) {
                LocalRuntime.shared.backendURL = u
                Task { await LocalRuntime.shared.agent?.setTeacher(TeacherClient(baseURL: u, deviceId: LocalRuntime.deviceId)) }
            }
        }
    }
    var memberId: String = UserDefaults.standard.string(forKey: "memberId") ?? "BM-10200" {
        didSet { UserDefaults.standard.set(memberId, forKey: "memberId") }
    }

    // data
    var state: DeviceState?
    var status: RuntimeStatus?
    var ledger: LedgerResponse?
    var members: [Member] = []
    var connectionError: String?

    // chat
    var chat: [ChatEntry] = []
    var conversationId: String?
    var busy = false
    var thinking: String?          // "student" | "teacher" while a model step runs
    var awaitingConfirm = false
    var lastNeverLeave = false

    // ui
    var tab: Tab = .home
    var toast: String?
    var highlightedAppointment: String?
    var highlightedClaim: String?

    private var client: any Runtime {
        runtimeMode == .device ? LocalRuntime.shared : DeviceClient(baseURL: URL(string: runtimeURL) ?? URL(string: "http://localhost:8100")!)
    }
    private var lastStepDetail: String?
    private var lastStepWho = "student"
    private var pollTask: Task<Void, Never>?

    // MARK: - loading

    func start() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshStatus()
                try? await Task.sleep(for: .seconds((self?.status?.ready ?? false) ? 10 : 2))
            }
        }
        if runtimeMode == .device {
            if let u = URL(string: backendURL) { LocalRuntime.shared.backendURL = u }
            let member = memberId
            Task {
                await refreshStatus()
                await LocalRuntime.shared.load(memberId: member)
                await refreshAll()
            }
        } else {
            Task { await refreshAll() }
        }
    }

    func refreshAll() async {
        await refreshStatus()
        await refreshState()
        await refreshLedger()
        if members.isEmpty { members = (try? await client.get("v1/device/members", as: [Member].self)) ?? [] }
        if runtimeMode == .device, status?.ready != true { return }
    }

    func refreshStatus() async {
        do {
            status = try await client.get("v1/device/status", as: RuntimeStatus.self)
            connectionError = nil
        } catch {
            connectionError = runtimeMode == .device ? "On-device runtime error: \(error.localizedDescription)"
                : "Can't reach the device runtime at \(runtimeURL). Start it on the Mac (see ios-app/README.md)."
        }
    }

    func refreshState() async {
        if let s = try? await client.get("v1/device/state", query: ["member_id": memberId], as: DeviceState.self) {
            withAnimation(.spring(duration: 0.45)) { state = s }
        }
    }

    func refreshLedger() async {
        if let l = try? await client.get("v1/device/ledger", as: LedgerResponse.self) { ledger = l }
    }

    // MARK: - chat

    func send(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !busy else { return }
        chat.append(.init(kind: .user(t)))
        var body: [String: Any] = ["text": t, "member_id": memberId]
        if let conversationId { body["conversation_id"] = conversationId }
        run(client.stream("v1/device/chat", body))
    }

    func confirm(_ approved: Bool) {
        guard awaitingConfirm, let conversationId else { return }
        if let i = chat.lastIndex(where: { if case .confirm(_, .pending) = $0.kind { return true }; return false }),
           case .confirm(let summary, _) = chat[i].kind {
            chat[i].kind = .confirm(summary: summary, state: approved ? .approved : .declined)
        }
        awaitingConfirm = false
        run(client.stream("v1/device/chat/continue", ["conversation_id": conversationId, "approved": approved]))
    }

    func newConversation() {
        guard !busy else { return }
        chat = []
        conversationId = nil
    }

    private func run(_ events: AsyncThrowingStream<ChatEvent, Error>) {
        busy = true
        Task {
            do {
                for try await ev in events { handle(ev) }
            } catch {
                chat.append(.init(kind: .error(error.localizedDescription)))
            }
            thinking = nil
            busy = awaitingConfirm
            await refreshLedger()
            await refreshStatus()
        }
    }

    private func handle(_ ev: ChatEvent) {
        if let c = ev.conversationId { conversationId = c }
        switch ev.type {
        case "start":
            lastNeverLeave = ev.neverLeave ?? false
            if lastNeverLeave {
                chat.append(.init(kind: .notice(icon: "lock.shield", text: "Medical topic: this request stays on your phone.", tone: .local)))
            }
        case "thinking":
            thinking = ev.who
        case "model_step":
            thinking = nil
            lastStepDetail = ev.action?.description ?? ev.raw
            lastStepWho = ev.who ?? "student"
        case "escalate":
            let shown = lastStepDetail.map { "\nDiscarded on-device step: \($0)" } ?? ""
            chat.append(.init(kind: .notice(icon: "icloud.and.arrow.up",
                                            text: "Sent to Bayside Cloud: \(ev.plain ?? "needs review")." + shown, tone: .cloud)))
        case "handoff":
            chat.append(.init(kind: .notice(icon: "person.fill.checkmark", text: ev.plain ?? "Handed to a person.", tone: .local)))
        case "tool":
            let label = ev.label ?? ev.tool ?? "Working"
            if ev.status == "running" {
                chat.append(.init(kind: .step(label: label, status: .running, who: lastStepWho, detail: lastStepDetail)))
            } else if let i = chat.lastIndex(where: { if case .step(label, .running, _, _) = $0.kind { return true }; return false }),
                      case .step(_, _, let who, let detail) = chat[i].kind {
                chat[i].kind = .step(label: label, status: ev.status == "ok" ? .ok : .failed(ev.error), who: who, detail: detail)
            }
        case "confirm":
            thinking = nil
            awaitingConfirm = true
            chat.append(.init(kind: .confirm(summary: ev.summary ?? "Confirm this change?", state: .pending)))
        case "state_changed":
            onStateChanged(tool: ev.tool ?? "")
        case "reply":
            thinking = nil
            chat.append(.init(kind: .reply(text: ev.text ?? "", served: ev.served ?? "device", flagged: ev.flagged ?? false,
                                           latency: ev.latencyS, reason: ev.reason)))
        case "error":
            chat.append(.init(kind: .error(ev.message ?? "Something went wrong.")))
        default:
            break
        }
    }

    private func onStateChanged(tool: String) {
        let before = state
        Task {
            await refreshState()
            guard let after = state else { return }
            switch tool {
            case "reschedule_appointment", "book_appointment":
                let moved = after.appointments.first { a in !(before?.appointments.contains(a) ?? false) }
                highlightedAppointment = moved?.id
                showToast(tool == "book_appointment" ? "Booked by Bayside assistant" : "Moved by Bayside assistant")
            case "file_claim":
                highlightedClaim = after.claims.first { c in !(before?.claims.contains(c) ?? false) }?.id
                showToast("Claim filed by Bayside assistant")
            case "escalate_to_human":
                showToast("Support ticket created")
            default: break
            }
        }
    }

    func showToast(_ text: String) {
        withAnimation { toast = text }
        Task {
            try? await Task.sleep(for: .seconds(2.5))
            withAnimation { if toast == text { toast = nil } }
        }
    }

    // MARK: - deep links (rehearsal and scripted demos)
    // bayside://ask?text=…   bayside://confirm?ok=1   bayside://tab/ledger   bayside://new

    func open(_ url: URL) {
        let q = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        let value = { (k: String) in q.first { $0.name == k }?.value }
        switch url.host() {
        case "ask": tab = .chat; send(value("text") ?? "")
        case "confirm": confirm(value("ok") != "0")
        case "new": newConversation(); tab = .chat
        case "tab":
            let t = url.pathComponents.last ?? ""
            tab = ["home": .home, "appointments": .appointments, "visits": .appointments, "claims": .claims,
                   "chat": .chat, "ledger": .ledger, "privacy": .ledger][t] ?? tab
        default: break
        }
    }

    // MARK: - settings

    func updateSettings(_ body: [String: Any]) {
        // Optimistic: flip the control now, then confirm with the runtime.
        if let v = body["offline"] as? Bool { status?.settings.offline = v }
        if let v = body["shadow_rate"] as? Double { status?.settings.shadowRate = v }
        if let v = body["sync"] as? Bool { status?.settings.sync = v }
        Task {
            _ = try? await client.post("v1/device/settings", body, as: RuntimeStatus.self)
            await refreshStatus()
            try? await Task.sleep(for: .seconds(1))
            await refreshLedger()
        }
    }

    func resetDemo() {
        Task {
            try? await client.postVoid("v1/device/reset", [:])
            chat = []; conversationId = nil
            await refreshAll()
        }
    }

    func switchMember(_ id: String) {
        memberId = id
        chat = []; conversationId = nil
        Task { await refreshState() }
    }
}
