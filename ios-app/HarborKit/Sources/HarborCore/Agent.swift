import Foundation

// MARK: - model + cloud interfaces

public struct StudentOutput {
    public var raw: String
    public var action: Action?
    public var latency: Double
    public var promptTokens: Int
    public var outputTokens: Int
    public var logprobs: [Double]
    public init(raw: String, action: Action?, latency: Double, promptTokens: Int, outputTokens: Int, logprobs: [Double]) {
        self.raw = raw; self.action = action; self.latency = latency
        self.promptTokens = promptTokens; self.outputTokens = outputTokens; self.logprobs = logprobs
    }
}

/// The fine-tuned student. HarborMLX provides the MLX implementation.
public protocol StudentModel: AnyObject, Sendable {
    var name: String { get }
    func step(_ messages: [Message]) async throws -> StudentOutput
}

public enum CloudError: Error { case offline, blocked([String]) }

/// The backend's teacher proxy (`POST /v1/teacher`). The phone never holds a provider key.
public struct TeacherClient: Sendable {
    public var baseURL: URL
    public var deviceId: String
    public init(baseURL: URL, deviceId: String) { self.baseURL = baseURL; self.deviceId = deviceId }

    public func step(_ messages: [Message], reason: String, traceId: String) async throws -> (action: Action?, native: JSON?, model: String?, latencyMs: Double, cost: Double) {
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/teacher"), timeoutInterval: 120)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = JSON.object(JObject([("device_id", .string(deviceId)), ("trace_id", .string(traceId)), ("reason", .string(reason)),
                                        ("messages", .array(messages.map(\.wire)))]))
        req.httpBody = Data(body.encode(sortKeys: false, separators: (",", ":"), ensureASCII: false).utf8)
        let data: Data, resp: URLResponse
        do { (data, resp) = try await URLSession.shared.data(for: req) } catch { throw CloudError.offline }
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code == 422 { throw CloudError.blocked(["server_scan"]) }
        if code >= 500 || code == 0 { throw CloudError.offline }
        guard code < 400, let out = try? JSON.parse(data) else { throw CloudError.offline }
        return (out["action"].flatMap(Action.init(json:)), out["native"], out["model"]?.string,
                out["latency_ms"]?.number ?? 0, out["cost_usd"]?.number ?? 0)
    }

    public func post(_ path: String, _ body: JSON) async {
        var req = URLRequest(url: baseURL.appendingPathComponent(path), timeoutInterval: 20)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data(body.encode(sortKeys: false, separators: (",", ":"), ensureASCII: false).utf8)
        _ = try? await URLSession.shared.data(for: req)
    }
}

// MARK: - the agent

/// The on-device agent: student + router + redactor + tools + privacy ledger, with escalation to the
/// teacher proxy. Events are the same JSON objects the Mac device runtime streams to the app.
public actor Agent {
    public struct Settings {
        public var offline = false
        public var shadowRate = 0.0
        public var sync = false
        public init() {}
    }

    public static let writeTools: Set<String> = ["book_appointment", "reschedule_appointment", "file_claim"]
    static let labels = [
        "get_policy": "Checking your plan", "check_coverage": "Checking coverage", "get_claim_status": "Looking up claims",
        "file_claim": "Filing the claim", "find_provider": "Finding providers", "book_appointment": "Booking the appointment",
        "reschedule_appointment": "Moving the appointment", "escalate_to_human": "Creating a support ticket",
    ]
    static let offlineReply = "I can't finish this one without a connection. I've saved your request and will complete it as soon as you're back online."
    static let declinedReply = "Okay, I haven't changed anything. What would you like instead?"

    public let student: StudentModel
    public var teacher: TeacherClient
    public var settings = Settings()
    public private(set) var memberId: String
    public private(set) var env: HarborEnv
    public private(set) var ledger: [JObject] = []
    public private(set) var counters: [String: Int] = [:]
    private var queue: [(convId: String, messages: [Message], reasons: [String], rowId: String)] = []
    private let router = Router()
    private let scanner = PIIScanner()
    private var red: Redactor
    private var messages: [Message] = []
    private var history: [Message] = []
    public private(set) var conversationId = UUID().uuidString.prefix(12).lowercased()

    public init(student: StudentModel, teacher: TeacherClient, memberId: String) {
        self.student = student
        self.teacher = teacher
        self.memberId = memberId
        let env = HarborEnv(memberId: memberId)
        self.env = env
        self.red = Redactor(known: env.knownPII())
    }

    public func setTeacher(_ t: TeacherClient) { teacher = t }

    public func update(offline: Bool? = nil, shadowRate: Double? = nil, sync: Bool? = nil) -> Int {
        let wasOffline = settings.offline
        if let offline { settings.offline = offline }
        if let shadowRate { settings.shadowRate = shadowRate }
        if let sync { settings.sync = sync }
        if wasOffline && !settings.offline && !queue.isEmpty { return replayQueue() }
        return 0
    }

    public func reset(memberId: String? = nil, clearLedger: Bool = true) {
        if let memberId { self.memberId = memberId }
        env = HarborEnv(memberId: self.memberId)
        newConversation()
        if clearLedger { ledger = []; queue = []; counters = [:] }
    }

    public func newConversation() {
        red = Redactor(known: env.knownPII())
        messages = []
        history = []
        conversationId = UUID().uuidString.prefix(12).lowercased()
    }

    // MARK: ledger + cloud

    @discardableResult
    private func addLedger(_ fields: [(String, JSON)]) -> String {
        let id = String(UUID().uuidString.prefix(10)).lowercased()
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime, .withDashSeparatorInDate]
        f.timeZone = .current
        let at = String(f.string(from: Date()).prefix(19))
        ledger.insert(JObject([("id", .string(id)), ("at", .string(at))] + fields), at: 0)
        return id
    }

    private func callTeacher(_ msgs: [Message], reason: String) async throws -> (action: Action?, native: JSON?) {
        if settings.offline { throw CloudError.offline }
        let payload = msgs.filter { $0.role == .user }.map(\.content).joined(separator: "\n")
        let findings = scanner.scan(payload)  // on-device scan, fail closed
        if !findings.isEmpty {
            let types = Array(Set(findings.map(\.type))).sorted()
            addLedger([("reason", .string(reason)), ("status", .string("blocked")), ("payload", .null),
                       ("conv_id", .string(conversationId)), ("findings", .array(types.map { .string($0) }))])
            throw CloudError.blocked(types)
        }
        do {
            let out = try await teacher.step(msgs, reason: reason, traceId: conversationId)
            let req = msgs.first { $0.kind == .request }.map { String($0.content.components(separatedBy: "\n\n").dropFirst().joined(separator: "\n\n")) }
            addLedger([("reason", .string(reason)), ("status", .string("sent")), ("payload", .string(payload)),
                       ("request", req.map { .string($0) } ?? .null), ("conv_id", .string(conversationId)),
                       ("response", .string(String((out.action?.text ?? "(no action)").prefix(600)))),
                       ("latency_ms", .double(out.latencyMs.rounded())), ("cost_usd", .double(out.cost)),
                       ("model", out.model.map { .string($0) } ?? .null)])
            return (out.action, out.native)
        } catch CloudError.blocked(let t) {
            addLedger([("reason", .string(reason)), ("status", .string("blocked")), ("payload", .null),
                       ("conv_id", .string(conversationId)), ("findings", .array(t.map { .string($0) }))])
            throw CloudError.blocked(t)
        }
    }

    private func replayQueue() -> Int {
        let items = queue
        queue = []
        Task {
            for it in items {
                do {
                    _ = try await self.callTeacher(it.messages, reason: "offline_queue")
                    self.markReplayed(it.rowId)
                } catch { self.requeue(it) }
            }
        }
        return items.count
    }

    private func markReplayed(_ rowId: String) {
        if let i = ledger.firstIndex(where: { $0["id"]?.string == rowId }) {
            ledger[i]["status"] = .string("replayed")
            ledger[i]["note"] = .string("sent for review after reconnecting")
        }
    }

    private func requeue(_ it: (convId: String, messages: [Message], reasons: [String], rowId: String)) { queue.append(it) }

    // MARK: the turn

    /// One member turn. `emit` receives UI events; `confirm` shows a confirmation card and returns the answer.
    public func runTurn(_ text: String, emit: @Sendable (JObject) -> Void, confirm: @Sendable (String, String) async -> Bool) async {
        let t0 = Date()
        var paused = 0.0
        // Keep the previous turn only when it ended with a question to the member (a clarifying question).
        if let last = messages.last(where: { $0.role == .assistant }), !(last.action?.replyText ?? "").trimmingCharacters(in: .whitespaces).hasSuffix("?") {
            history += messages
            messages = []
        }
        let neverLeave = router.isNeverLeave(text)
        let userRed = red.redactText(text)
        let start = messages.count
        messages.append(Message(role: .user, kind: .request, content: Prompts.request(session: red.redact(env.sessionContext()), text: userRed)))
        emit(ev("start", [("never_leave", .bool(neverLeave)), ("redacted", .string(userRed))]))

        var controller = "student", reasons: [String] = [], flagged = false, handoff = false
        var served: [String] = []
        var finalRed: String?
        var toolSteps = 0, modelSteps = 0
        let maxSteps = router.maxSteps

        func queueOffline(_ why: [String]) {
            let id = addLedger([("reason", .string("offline_queue")), ("status", .string("queued")), ("payload", .null),
                                ("request", .string(userRed)), ("conv_id", .string(conversationId)),
                                ("note", .string("waiting for a connection: " + why.joined(separator: ", ")))])
            queue.append((String(conversationId), Array(messages[start...]), why, id))
        }

        loop: for _ in 0..<(maxSteps + 3) {
            var action: Action?
            var native: JSON?
            var who = controller
            if controller == "student" {
                emit(ev("thinking", [("who", .string("student"))]))
                let out: StudentOutput
                do { out = try await student.step(messages) } catch {
                    emit(ev("error", [("message", .string("On-device model error: \(error)"))]))
                    return
                }
                modelSteps += 1
                action = out.action
                let realArgs = action?.args.map { red.refill($0) }
                var gate = router.check(action, realArgs: realArgs, env: env, logprobs: out.logprobs,
                                        context: messages.filter { $0.role == .user }.map(\.content))
                if action?.isTool == true, toolSteps >= maxSteps { gate.ok = false; gate.reasons.append("max_steps") }
                emit(ev("model_step", [("who", .string("student")), ("latency_s", .double((out.latency * 100).rounded() / 100)),
                                       ("prompt_tokens", .int(out.promptTokens)), ("action", action?.json ?? .null),
                                       ("raw", action == nil ? .string(String(out.raw.prefix(300))) : .null),
                                       ("gate", .array(gate.reasons.map { .string($0) }))]))
                if !gate.ok {
                    if neverLeave {
                        handoff = true; who = "local_handoff"
                        action = .tool(name: "escalate_to_human", args: red.redact(JObject([
                            ("member_id", .string(env.memberId)), ("topic", .string("medical")),
                            ("summary", .string("Member asked a medical question; needs a human."))])))
                        emit(ev("handoff", [("plain", .string("This stays on your phone. A Bayside team member will follow up."))]))
                    } else if settings.offline {
                        flagged = true
                        if case .reply(let r)? = action, !r.trimmingCharacters(in: .whitespaces).isEmpty {
                            messages.append(Message(role: .assistant, content: action!.text, action: action, servedBy: "student"))
                            finalRed = r
                        } else { finalRed = Self.offlineReply }
                        served.append("student")
                        queueOffline(gate.reasons)
                        break loop
                    } else {
                        controller = "teacher"; reasons = gate.reasons
                        emit(ev("escalate", [("reasons", .array(reasons.map { .string($0) })), ("plain", .string(Self.plainReason(reasons)))]))
                        continue
                    }
                } else if settings.shadowRate > 0, !neverLeave, !settings.offline, Double.random(in: 0..<1) < settings.shadowRate {
                    let snapshot = Array(messages[start...]), studentAction = action
                    Task { await self.shadow(snapshot, studentAction) }
                }
            } else {
                emit(ev("thinking", [("who", .string("teacher"))]))
                do {
                    // Only this turn goes to the cloud: earlier turns may include never-leave requests.
                    let out = try await callTeacher(Array(messages[start...]), reason: "escalation:" + (reasons.isEmpty ? ["teacher"] : reasons).joined(separator: ","))
                    action = out.action; native = out.native
                    emit(ev("model_step", [("who", .string("teacher")), ("action", action?.json ?? .null), ("gate", .array([]))]))
                    guard let a = action else {
                        finalRed = "Sorry, I can't help with that one. A Bayside team member can."
                        served.append("teacher"); break loop
                    }
                    if a.isTool, toolSteps >= maxSteps + 1 {
                        finalRed = "I've done what I can here. A Bayside team member can help with the rest."
                        served.append("teacher"); break loop
                    }
                } catch CloudError.blocked {
                    handoff = true; who = "blocked_by_pii_scan"
                    action = .tool(name: "escalate_to_human", args: red.redact(JObject([
                        ("member_id", .string(env.memberId)), ("topic", .string("other")),
                        ("summary", .string("Request could not be sent safely; needs a human."))])))
                    emit(ev("handoff", [("plain", .string("This couldn't be sent safely, so a person will follow up."))]))
                } catch {
                    flagged = true
                    finalRed = Self.offlineReply
                    queueOffline(reasons.isEmpty ? ["offline"] : reasons)
                    break loop
                }
            }

            guard let act = action else { break }
            served.append(who)
            let msg = Message(role: .assistant, content: act.text, action: act, servedBy: who, native: native)
            switch act {
            case .reply(let r):
                messages.append(msg)
                finalRed = r
                break loop
            case .tool(let tool, let args):
                let realArgs = red.refill(args)
                if Self.writeTools.contains(tool) {
                    let tw = Date()
                    let summary = summaryText(tool, realArgs)
                    emit(ev("confirm", [("tool", .string(tool)), ("summary", .string(summary))]))
                    let approved = await confirm(tool, summary)
                    paused += Date().timeIntervalSince(tw)
                    if !approved {
                        messages.append(msg)
                        messages.append(Message(role: .user, kind: .toolResult, tool: tool,
                                                content: Prompts.toolResult(tool, JObject([("error", .string("the member declined this change"))]))))
                        finalRed = Self.declinedReply
                        served.append("local_template")
                        break loop
                    }
                }
                let label = Self.labels[tool] ?? tool
                emit(ev("tool", [("tool", .string(tool)), ("label", .string(label)), ("status", .string("running"))]))
                let result = env.execute(tool, realArgs)
                toolSteps += 1
                messages.append(msg)
                messages.append(Message(role: .user, kind: .toolResult, tool: tool, content: Prompts.toolResult(tool, red.redact(result))))
                let ok = result["error"] == nil
                emit(ev("tool", [("tool", .string(tool)), ("label", .string(label)), ("status", .string(ok ? "ok" : "error")),
                                 ("error", result["error"] ?? .null)]))
                if ok, Self.writeTools.contains(tool) || tool == "escalate_to_human" { emit(ev("state_changed", [("tool", .string(tool))])) }
                if handoff {
                    finalRed = "I've passed this to a Bayside team member (ticket \(result["ticket_id"]?.pyStr ?? "")). They'll follow up \(result["expected_response"]?.pyStr ?? "soon")."
                    break loop
                }
            }
        }
        if finalRed == nil { finalRed = "I've done what I can here. A Bayside team member can help with the rest." }
        if !messages[start...].contains(where: { $0.action?.replyText == finalRed }) {
            messages.append(Message(role: .assistant, content: Action.reply(finalRed!).text, action: .reply(finalRed!), servedBy: "local_template"))
        }
        let escalated = served.contains("teacher")
        let kind = handoff ? "handoff" : escalated ? "cloud" : "device"
        let latency = ((Date().timeIntervalSince(t0) - paused) * 100).rounded() / 100
        counters["requests", default: 0] += 1
        counters[escalated ? "escalated" : "on_device", default: 0] += 1
        if flagged { counters["flagged", default: 0] += 1 }
        emit(ev("reply", [("text", .string(red.refillText(finalRed!))), ("served", .string(kind)), ("flagged", .bool(flagged)),
                          ("escalated", .bool(escalated)), ("reason", escalated ? .string(Self.plainReason(reasons)) : .null),
                          ("latency_s", .double(latency)), ("model_steps", .int(modelSteps)), ("tool_steps", .int(toolSteps)),
                          ("never_leave", .bool(neverLeave))]))
        emit(ev("done", []))
        if settings.sync { syncTrace(start: start, escalated: escalated, reasons: reasons, neverLeave: neverLeave, userRed: userRed,
                                     replyRed: finalRed!, latency: latency, steps: modelSteps) }
    }

    private func shadow(_ msgs: [Message], _ studentAction: Action?) async {
        guard let out = try? await callTeacher(msgs, reason: "shadow") else { return }
        let agree: Bool
        switch (studentAction, out.action) {
        case (.reply?, .reply?): agree = true
        case (.tool(let a, let x)?, .tool(let b, let y)?): agree = a == b && JSON.object(x).compact == JSON.object(y).compact
        default: agree = false
        }
        counters["shadow", default: 0] += 1
        if agree { counters["shadow_agree", default: 0] += 1 }
        await teacher.post("v1/shadow", .object(JObject([("trace_id", .string(conversationId)), ("run_id", .string(teacher.deviceId)),
                                                          ("agree", .bool(agree)), ("student", studentAction?.json ?? .null),
                                                          ("teacher", out.action?.json ?? .null)])))
    }

    private func syncTrace(start: Int, escalated: Bool, reasons: [String], neverLeave: Bool, userRed: String, replyRed: String, latency: Double, steps: Int) {
        let msgs = messages[start...].map { m -> JSON in
            var o = m.wire.object!
            o["native"] = nil
            if let s = m.servedBy { o["served_by"] = .string(s) }
            return .object(o)
        }
        let trace = JSON.array([.object(JObject([
            ("source", .string("phone")), ("run_id", .string(teacher.deviceId)), ("served_by", .string(escalated ? "teacher" : "student")),
            ("escalated", .bool(escalated)), ("escalation_reasons", .array(reasons.map { .string($0) })), ("never_leave", .bool(neverLeave)),
            ("request_redacted", .string(userRed)), ("messages", .array(msgs)), ("final_reply_redacted", .string(replyRed)),
            ("latency_ms", .double(latency * 1000)), ("steps", .int(steps)), ("model", .string(student.name)),
        ]))])
        let t = teacher
        Task { await t.post("v1/traces", trace) }
    }

    // MARK: helpers

    private func ev(_ type: String, _ fields: [(String, JSON)]) -> JObject {
        JObject([("type", .string(type))] + fields + [("conversation_id", .string(String(conversationId)))])
    }

    public static func plainReason(_ reasons: [String]) -> String {
        let r = reasons.joined(separator: " ")
        if r.contains("policy_claim_over_threshold") { return "claims over $1,000 are always reviewed" }
        if r.contains("reply_number_not_grounded") { return "the answer had a number I couldn't verify" }
        if r.contains("max_steps") { return "the request needed more steps" }
        if r.contains("low_confidence") { return "unusual request (low confidence)" }
        if r.contains("date") { return "a date didn't check out" }
        if r.contains("unknown_") || r.contains("mismatch") { return "it referenced something that doesn't exist" }
        if r.contains("schema:enum") { return "the on-device model picked an option that doesn't exist" }
        if r.contains("schema:required") { return "the on-device model left out a required detail" }
        if r.contains("invalid_json") { return "the on-device model's output wasn't a valid action" }
        return "the on-device step didn't pass a check"
    }

    static func niceDate(_ s: String) -> String {
        guard let d = Day(iso: s) else { return s }
        let (_, m, day) = d.ymd
        let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return "\(String(d.weekdayName.prefix(3))), \(months[m - 1]) \(day)"
    }

    static func niceTime(_ t: String) -> String {
        let p = t.prefix(5).split(separator: ":").compactMap { Int($0) }
        guard p.count == 2 else { return t }
        return "\((p[0] + 11) % 12 + 1):\(String(format: "%02d", p[1])) \(p[0] >= 12 ? "PM" : "AM")"
    }

    func summaryText(_ tool: String, _ a: JObject) -> String {
        switch tool {
        case "reschedule_appointment":
            let appt = env.appointments.first { $0["appointment_id"] == a["appointment_id"] }
            let kind = appt.map { ($0["reason"]?.string).flatMap { $0.isEmpty ? nil : $0.lowercased() }
                ?? $0["specialty"]!.string!.replacingOccurrences(of: "_", with: " ") + " appointment" }
            let what = appt.map { "your \(kind!) with \($0["provider_name"]!.string!)" } ?? "your appointment"
            let when = Self.niceDate(a["new_date"]?.pyStr ?? "") + ((a["new_time"]?.truthy ?? false) ? " at \(Self.niceTime(a["new_time"]!.pyStr))" : " (same time if it's free)")
            return "Move \(what) to \(when)?"
        case "book_appointment":
            let p = env.providers.first { $0["provider_id"] == a["provider_id"] }
            let when = Self.niceDate(a["date"]?.pyStr ?? "") + ((a["time"]?.truthy ?? false) ? " at \(Self.niceTime(a["time"]!.pyStr))" : " (first free time)")
            return "Book \(p?["name"]?.string ?? "an appointment")\(p.map { " (\($0["clinic"]!.string!))" } ?? "") on \(when)?"
        case "file_claim":
            let p = env.providers.first { $0["provider_id"] == a["provider_id"] }
            let amt = Redactor.formatAmount(a["amount"]?.number ?? 0)
            let svc = env.coverage[env.member["plan"]!.string!]?[a["service"]?.pyStr ?? ""]?["label"]?.string ?? a["service"]?.pyStr ?? ""
            return "File a \(amt) claim for a \(svc.lowercased()) with \(p?["name"]?.string ?? "the provider") on \(Self.niceDate(a["service_date"]?.pyStr ?? ""))?"
        default:
            return "Allow \((Self.labels[tool] ?? tool).lowercased())?"
        }
    }

    // MARK: data for the app screens (same shape as the device runtime's /v1/device/state)

    public func stateJSON() -> JSON {
        let m = env.member
        let pol = env.policies.first { $0["member_id"]?.string == memberId }!
        let cov = env.coverage[m["plan"]!.string!]!.object!
        let claims = stableSorted(env.claims.filter { $0["member_id"]?.string == memberId }, reverse: true) { $0["service_date"]?.string ?? "" }
        let appts = stableSorted(env.appointments.filter { $0["member_id"]?.string == memberId && $0["status"]?.string == "scheduled" }) { $0["datetime"]?.string ?? "" }
        return .object(JObject([
            ("today", .string(env.today.iso)),
            ("member", .object(m.pick(["member_id", "name", "first_name", "city", "plan", "policy_id", "dob", "phone", "email", "address"]))),
            ("policy", .object(pol.pick(["policy_id", "plan_name", "status", "effective_date", "renewal_date", "deductible", "deductible_met",
                                         "out_of_pocket_max", "out_of_pocket_spent", "dependents"]))),
            ("coverage", .array(cov.map { k, v in
                .object(JObject([("service", .string(k)), ("label", v["label"]!), ("covered", v["covered"]!), ("copay", v["copay"] ?? .null),
                                 ("coinsurance", v["coinsurance"] ?? .null), ("requires_referral", v["requires_referral"] ?? .bool(false)),
                                 ("requires_prior_auth", v["requires_prior_auth"] ?? .bool(false)),
                                 ("visit_limit_per_year", v["visit_limit_per_year"] ?? .null), ("out_of_network", v["out_of_network"] ?? .null)]))
            })),
            ("claims", .array(claims.map { .object($0.pick(["claim_id", "provider_name", "service", "service_date", "status", "amount_billed",
                                                            "amount_covered", "denial_reason", "expected_decision_by"])) })),
            ("appointments", .array(appts.map { .object($0.pick(["appointment_id", "provider_name", "specialty", "datetime", "reason"])) })),
            ("tickets", .array(env.tickets.map { .object($0) })),
        ]))
    }

    public func ledgerJSON() -> JSON {
        let sent = ledger.filter { $0["status"]?.string == "sent" }
        let raw = sent.reduce(0) { $0 + scanner.scan($1["payload"]?.string ?? "").count }
        return .object(JObject([
            ("rows", .array(ledger.map { .object($0) })),
            ("stats", .object(JObject([
                ("on_device", .int(counters["on_device"] ?? 0)), ("requests", .int(counters["requests"] ?? 0)),
                ("sent", .int(sent.count)), ("blocked", .int(ledger.filter { $0["status"]?.string == "blocked" }.count)),
                ("queued", .int(queue.count)), ("raw_pii_fields_sent", .int(raw)),
                ("cost_usd", .double(sent.reduce(0) { $0 + ($1["cost_usd"]?.number ?? 0) })),
            ]))),
        ]))
    }

    public var queuedCount: Int { queue.count }

    public static func membersJSON() -> JSON {
        .array(Shared.fixtures["members"]!.array!.map { m in
            .object(JObject([("member_id", m["member_id"]!), ("name", m["name"]!), ("plan", m["plan"]!), ("city", m["city"]!)]))
        })
    }
}
