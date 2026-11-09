import Foundation
import os
import HarborCore
import HarborMLX

/// What the app talks to: the Mac device runtime over HTTP (`DeviceClient`) or the stack running
/// on this iPhone (`LocalRuntime`). Both speak the same requests and chat events.
protocol Runtime {
    func get<T: Decodable>(_ path: String, query: [String: String], as: T.Type) async throws -> T
    func post<T: Decodable>(_ path: String, _ body: [String: Any], as: T.Type) async throws -> T
    func postVoid(_ path: String, _ body: [String: Any]) async throws
    func stream(_ path: String, _ body: [String: Any]) -> AsyncThrowingStream<ChatEvent, Error>
}

extension DeviceClient: Runtime {}

extension Runtime {
    func get<T: Decodable>(_ path: String, as t: T.Type) async throws -> T { try await get(path, query: [:], as: t) }
}

enum RuntimeMode: String, CaseIterable, Identifiable {
    case device, mac
    var id: String { rawValue }
    var label: String { self == .device ? "This iPhone (MLX)" : "Mac runtime (Wi-Fi)" }
}

/// The on-device stack: HarborKit's agent (tools, redactor, router, ledger) with the fine-tuned student
/// (Gemma 4 E4B, or LFM3-1.7B in the Bayside LFM3 copy) running in-process via MLX. Escalations go to the backend's teacher proxy.
final class LocalRuntime: Runtime, @unchecked Sendable {
    static let shared = LocalRuntime()

    enum Phase: Equatable { case idle, loading, ready, failed(String) }

    private let lock = NSLock()
    private(set) var phase: Phase = .idle
    private(set) var progress = 0.0
    private(set) var agent: Agent?
    private(set) var modelDir: URL?
    private var sink: AsyncThrowingStream<ChatEvent, Error>.Continuation?
    private var pending: CheckedContinuation<Bool, Never>?
    var backendURL = URL(string: "http://Aydars-MacBook-Pro.local:8000")!

    /// Separate ids per edition so the backend ledger tells the Gemma app and the LFM3 copy apart.
    static let deviceId = Bundle.main.bundleIdentifier?.hasSuffix(".lfm3") == true ? "iphone-lfm3" : "iphone-local"

    /// Model folders are side-loaded into Documents/Models/<name>/ (Finder file sharing or `devicectl device copy to`).
    static func findModel() -> URL? {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let root = docs.appendingPathComponent("Models")
        let names = ((try? fm.contentsOfDirectory(atPath: root.path)) ?? []).sorted().reversed()
        for n in names where fm.fileExists(atPath: root.appendingPathComponent(n).appendingPathComponent("config.json").path) {
            return root.appendingPathComponent(n)
        }
        return nil
    }

    static var modelsFolderHint: String {
        let app = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String ?? "Bayside"
        return "Files → On My iPhone → \(app) → Models/<model folder>"
    }

    func load(memberId: String) async {
        lock.lock()
        if phase == .loading || phase == .ready { lock.unlock(); return }
        phase = .loading; progress = 0
        lock.unlock()
        #if targetEnvironment(simulator)
        setPhase(.failed("MLX can't run in the iOS Simulator. Use the Mac runtime."))
        return
        #else
        guard let dir = Self.findModel() else {
            setPhase(.failed("No model on this iPhone yet. Copy the model folder to \(Self.modelsFolderHint)."))
            return
        }
        modelDir = dir
        Diag.write("---- launch · model \(dir.lastPathComponent)")
        MLXStudent.staticLog = { Diag.write($0) }
        do {
            let student = try await MLXStudent.load(directory: dir, name: dir.lastPathComponent) { [weak self] p in
                self?.progress = p
            }
            student.log = { Diag.write($0) }
            agent = Agent(student: student, teacher: TeacherClient(baseURL: backendURL, deviceId: Self.deviceId), memberId: memberId)
            setPhase(.ready)
        } catch {
            setPhase(.failed("Couldn't load the model: \(error.localizedDescription)"))
        }
        #endif
    }

    private func setPhase(_ p: Phase) { lock.lock(); phase = p; lock.unlock() }

    // MARK: Runtime

    private var decoder: JSONDecoder { let d = JSONDecoder(); d.keyDecodingStrategy = .convertFromSnakeCase; return d }

    private func decode<T: Decodable>(_ json: JSON, _ t: T.Type) throws -> T {
        try decoder.decode(T.self, from: Data(json.encode(sortKeys: false, separators: (",", ":"), ensureASCII: false).utf8))
    }

    func get<T: Decodable>(_ path: String, query: [String: String], as t: T.Type) async throws -> T {
        switch path {
        case "v1/device/status": return try decode(await statusJSON(), t)
        case "v1/device/members": return try decode(Agent.membersJSON(), t)
        case "v1/device/state":
            guard let agent else { throw DeviceClient.ClientError.http(503, "model loading") }
            if let m = query["member_id"], await agent.memberId != m { await agent.reset(memberId: m, clearLedger: false) }
            return try decode(await agent.stateJSON(), t)
        case "v1/device/ledger":
            guard let agent else { throw DeviceClient.ClientError.http(503, "model loading") }
            return try decode(await agent.ledgerJSON(), t)
        default: throw DeviceClient.ClientError.http(404, path)
        }
    }

    func post<T: Decodable>(_ path: String, _ body: [String: Any], as t: T.Type) async throws -> T {
        try await postVoid(path, body)
        return try decode(await statusJSON(), t)
    }

    func postVoid(_ path: String, _ body: [String: Any]) async throws {
        guard let agent else { return }
        switch path {
        case "v1/device/settings":
            _ = await agent.update(offline: body["offline"] as? Bool, shadowRate: body["shadow_rate"] as? Double, sync: body["sync"] as? Bool)
        case "v1/device/reset":
            await agent.reset()
        default: break
        }
    }

    func stream(_ path: String, _ body: [String: Any]) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { cont in
            self.lock.lock()
            self.sink = cont
            let pending = self.pending
            self.pending = nil
            self.lock.unlock()
            if path == "v1/device/chat/continue" {
                pending?.resume(returning: (body["approved"] as? Bool) ?? false)
                return
            }
            guard let agent = self.agent else {
                cont.yield(ChatEvent(type: "error", message: "The on-device model isn't loaded yet."))
                cont.finish()
                return
            }
            let text = body["text"] as? String ?? ""
            let fresh = body["conversation_id"] == nil
            Task.detached {
                if fresh { await agent.newConversation() }
                await agent.runTurn(text, emit: { [weak self] ev in self?.emit(ev) }, confirm: { [weak self] _, _ in
                    await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in
                        guard let self else { c.resume(returning: false); return }
                        self.lock.lock()
                        self.pending = c
                        // Like the server: the stream ends at a confirmation card; "continue" opens the next one.
                        self.sink?.finish()
                        self.sink = nil
                        self.lock.unlock()
                    }
                })
                self.lock.lock(); self.sink?.finish(); self.sink = nil; self.lock.unlock()
            }
        }
    }

    private func emit(_ ev: JObject) {
        guard let e = try? decode(.object(ev), ChatEvent.self) else { return }
        lock.lock()
        sink?.yield(e)
        lock.unlock()
    }

    private func statusJSON() async -> JSON {
        let fm = FileManager.default
        var size = 0.0, version = "-", name = "none"
        if let dir = modelDir ?? Self.findModel() {
            name = dir.lastPathComponent
            let files = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".safetensors") }.sorted()
            var sig = ""
            for f in files {
                let s = ((try? fm.attributesOfItem(atPath: dir.appendingPathComponent(f).path))?[.size] as? NSNumber)?.doubleValue ?? 0
                size += s
                sig += "\(f):\(Int(s))"
            }
            version = String(format: "%08x", sig.utf8.reduce(UInt32(2166136261)) { ($0 ^ UInt32($1)) &* 16777619 })
        }
        let round = name.split(separator: "-").first { $0.hasPrefix("r") && Int($0.dropFirst()) != nil }.flatMap { Int($0.dropFirst()) }
        var counters = JObject(), offline = false, shadow = 0.0, sync = false, queued = 0
        if let agent {
            for (k, v) in await agent.counters { counters[k] = .int(v) }
            let s = await agent.settings
            offline = s.offline; shadow = s.shadowRate; sync = s.sync
            queued = await agent.queuedCount
        }
        let error: JSON
        switch phase {
        case .failed(let m): error = .string(m)
        case .loading: error = .string(String(format: "Loading the on-device model… %d%%", Int(progress * 100)))
        default: error = .null
        }
        return .object(JObject([
            ("ready", .bool(phase == .ready)), ("error", phase == .ready ? .null : error),
            ("model", .object(JObject([("name", .string(name)), ("round", round.map { .int($0) } ?? .null), ("size_bytes", .double(size)),
                                       ("version", .string(version)), ("base", .string(Self.baseLabel(modelDir ?? Self.findModel())))]))),
            ("settings", .object(JObject([("offline", .bool(offline)), ("shadow_rate", .double(shadow)), ("sync", .bool(sync)),
                                          ("backend", .string(backendURL.absoluteString)), ("device_id", .string(Self.deviceId))]))),
            ("counters", .object(counters)), ("queued", .int(queued)),
            ("runtime", .string("On this iPhone (MLX) · \(Self.memoryLine)")),
        ]))
    }

    static func baseLabel(_ dir: URL?) -> String {
        guard let dir else { return "-" }
        return MLXStudent.modelType(dir).hasPrefix("lfm") ? "Liquid LFM3-1.7B, MLX 6-bit" : "Gemma 4 E4B, MLX 4-bit"
    }

    static var memoryLine: String {
        #if os(iOS) && !targetEnvironment(simulator)
        let avail = Double(os_proc_available_memory()) / 1e9
        return String(format: "%.1f GB free for the app", avail)
        #else
        return "Simulator"
        #endif
    }
}

extension ChatEvent {
    init(type: String, message: String) {
        self.type = type
        self.message = message
    }
}

/// Appends timestamped lines to Documents/bayside-diag.log (and stdout). Fetch it with
/// `xcrun devicectl device copy from --domain-type appDataContainer --domain-identifier com.gradientharbor.bayside --source Documents/bayside-diag.log`.
enum Diag {
    private static let queue = DispatchQueue(label: "bayside.diag")
    static let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("bayside-diag.log")

    static func write(_ line: String) {
        let stamp = ISO8601DateFormatter().string(from: Date())
        let text = "\(stamp) \(line)\n"
        print(text, terminator: "")
        queue.sync {
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(Data(text.utf8)); try? h.synchronize(); try? h.close()
            } else {
                try? Data(text.utf8).write(to: url)
            }
        }
    }
}
