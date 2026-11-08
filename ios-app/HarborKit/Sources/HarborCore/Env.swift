import Foundation

/// Bayside Mutual tool environment over the bundled fixtures. A line-by-line port of
/// `use-case-evals-poc/harbor_env/env.py`: same results, same key order, same error payloads.
public final class HarborEnv {
    public static let slotTimes: [String] = (9...16).flatMap { h in [0, 30].map { String(format: "%02d:%02d", h, $0) } }

    public let today: Day
    public let memberId: String
    public private(set) var member: JObject
    public var members: [JObject]
    public var policies: [JObject]
    public var providers: [JObject]
    public var claims: [JObject]
    public var appointments: [JObject]
    public var tickets: [JObject] = []
    public let coverage: JObject
    private var seq = 0
    public private(set) var calls: [(tool: String, args: JObject, result: JObject)] = []

    public init(memberId: String) {
        let fx = Shared.fixtures
        today = Day(iso: fx["meta"]!["sim_today"]!.string!)!
        members = fx["members"]!.array!.map { $0.object! }
        policies = fx["policies"]!.array!.map { $0.object! }
        providers = fx["providers"]!.array!.map { $0.object! }
        claims = fx["claims"]!.array!.map { $0.object! }
        appointments = fx["appointments"]!.array!.map { $0.object! }
        coverage = fx["coverage"]!.object!
        self.memberId = memberId
        member = members.first { $0["member_id"]?.string == memberId }!
    }

    // MARK: context

    public func sessionContext() -> JObject {
        let m = member
        let appts = stableSorted(appointments.filter { $0["member_id"]?.string == memberId && $0["status"]?.string == "scheduled" }) {
            $0["datetime"]?.string ?? ""
        }
        let recent = stableSorted(claims.filter { $0["member_id"]?.string == memberId }, reverse: true) {
            $0["service_date"]?.string ?? ""
        }.prefix(5)
        return JObject([
            ("today", .string(today.iso)),
            ("weekday", .string(today.weekdayName)),
            ("member_id", m["member_id"]!),
            ("first_name", m["first_name"]!),
            ("city", m["city"]!),
            ("plan", m["plan"]!),
            ("policy_id", m["policy_id"]!),
            ("upcoming_appointments", .array(appts.map { a in
                .object(JObject([("id", a["appointment_id"]!), ("provider", a["provider_name"]!),
                                 ("specialty", a["specialty"]!), ("datetime", a["datetime"]!)]))
            })),
            ("recent_claims", .array(recent.map { c in
                .object(JObject([("id", c["claim_id"]!), ("provider", c["provider_name"]!),
                                 ("service_date", c["service_date"]!), ("status", c["status"]!)]))
            })),
        ])
    }

    /// Python dict order: name, first_name, last_name, dob, phone, email, address.
    public func knownPII() -> [(String, String)] {
        let m = member
        let name = m["name"]!.string!
        return [("name", name), ("first_name", m["first_name"]!.string!), ("last_name", String(name.split(separator: " ").last ?? "")),
                ("dob", m["dob"]!.string!), ("phone", m["phone"]!.string!), ("email", m["email"]!.string!),
                ("address", m["address"]!.string!)]
    }

    // MARK: execution

    struct ToolError: Error { let payload: JObject }
    private func err(_ message: String, _ extra: [(String, JSON)] = []) -> ToolError {
        ToolError(payload: JObject([("error", .string(message))] + extra))
    }

    public func execute(_ tool: String, _ args: JObject) -> JObject {
        var result: JObject
        do {
            guard Self.signatures[tool] != nil else { throw err("unknown tool \(tool)") }
            if args["member_id"] != .string(memberId) { throw err("member_id does not match the signed-in member") }
            let kwargs = args.without(["member_id"])
            try checkSignature(tool, kwargs)
            result = try dispatch(tool, kwargs)
        } catch let e as ToolError {
            result = e.payload
        } catch {
            result = JObject([("error", .string("\(error)"))])
        }
        calls.append((tool, args, result))
        return result
    }

    /// Positional/keyword signature of each `_t_<tool>` in env.py: (required, optional).
    static let signatures: [String: ([String], [String])] = [
        "get_policy": ([], []),
        "check_coverage": (["service"], []),
        "get_claim_status": ([], ["claim_id"]),
        "file_claim": (["provider_id", "service_date", "service", "amount"], []),
        "find_provider": (["specialty"], ["city", "in_network_only"]),
        "book_appointment": (["provider_id", "date"], ["time", "reason"]),
        "reschedule_appointment": (["appointment_id", "new_date"], ["new_time"]),
        "escalate_to_human": (["topic", "summary"], []),
    ]

    /// Python raises TypeError for unexpected or missing keyword arguments.
    private func checkSignature(_ tool: String, _ kwargs: JObject) throws {
        let (req, opt) = Self.signatures[tool]!
        let fn = "HarborEnv._t_\(tool)()"
        for k in kwargs.keys where !req.contains(k) && !opt.contains(k) {
            throw err("bad arguments: \(fn) got an unexpected keyword argument '\(k)'")
        }
        let missing = req.filter { kwargs[$0] == nil }
        if !missing.isEmpty {
            let names = missing.map { "'\($0)'" }
            let list = names.count == 1 ? names[0] : names.dropLast().joined(separator: ", ") + " and " + names.last!
            throw err("bad arguments: \(fn) missing \(missing.count) required positional argument\(missing.count > 1 ? "s" : ""): \(list)")
        }
    }

    private func dispatch(_ tool: String, _ a: JObject) throws -> JObject {
        switch tool {
        case "get_policy": return try getPolicy()
        case "check_coverage": return try checkCoverage(a["service"]!)
        case "get_claim_status": return try getClaimStatus(a["claim_id"])
        case "file_claim": return try fileClaim(a["provider_id"]!, a["service_date"]!, a["service"]!, a["amount"]!)
        case "find_provider": return try findProvider(a["specialty"]!, a["city"], a["in_network_only"] ?? .bool(true))
        case "book_appointment": return try bookAppointment(a["provider_id"]!, a["date"]!, a["time"], a["reason"])
        case "reschedule_appointment": return try reschedule(a["appointment_id"]!, a["new_date"]!, a["new_time"])
        case "escalate_to_human": return try escalate(a["topic"]!, a["summary"]!)
        default: throw err("unknown tool \(tool)")
        }
    }

    private func nextId(_ prefix: String, _ base: Int) -> String {
        seq += 1
        return "\(prefix)-\(base + seq)"
    }

    private func provider(_ id: JSON) throws -> JObject {
        guard let p = providers.first(where: { $0["provider_id"] == id }) else { throw err("provider \(id.pyStr) not found") }
        return p
    }

    private func policy() -> JObject { policies.first { $0["member_id"]?.string == memberId }! }

    private func parseDate(_ s: JSON, _ field: String) throws -> Day {
        guard let d = Day(iso: s.pyStr) else { throw err("\(field) must be YYYY-MM-DD") }
        return d
    }

    private func workingDays(_ p: JObject) -> [Int] { p["working_days"]!.array!.compactMap(\.int) }

    private func freeTimes(_ p: JObject, _ d: Day, ignore: String? = nil) -> [String] {
        guard workingDays(p).contains(d.weekday) else { return [] }
        let taken = Set(appointments.filter {
            $0["provider_id"] == p["provider_id"] && $0["status"]?.string == "scheduled"
                && String($0["datetime"]!.string!.prefix(10)) == d.iso && $0["appointment_id"]?.string != ignore
        }.map { String($0["datetime"]!.string!.dropFirst(11).prefix(5)) })
        return Self.slotTimes.filter { !taken.contains($0) }
    }

    private func checkBookableDay(_ p: JObject, _ d: Day) throws {
        if d <= today { throw err("date must be after today", [("today", .string(today.iso))]) }
        if d > today.adding(60) { throw err("bookings open up to 60 days ahead") }
        if !workingDays(p).contains(d.weekday) {
            let days = workingDays(p).map { JSON.string(Day.weekdayNames[$0]) }
            throw err("\(p["name"]!.string!) does not work on \(d.weekdayName)s", [("working_days", .array(days))])
        }
    }

    private func getPolicy() throws -> JObject {
        policy().pick(["policy_id", "plan_name", "status", "effective_date", "renewal_date", "deductible",
                       "deductible_met", "out_of_pocket_max", "out_of_pocket_spent", "dependents"])
    }

    private func checkCoverage(_ service: JSON) throws -> JObject {
        let plan = member["plan"]!.string!
        guard let cov = coverage[plan]?.object?[service.pyStr]?.object else {
            let services = (coverage[plan]?.object?.keys ?? []).sorted(by: JSON.pyLess).map { JSON.string($0) }
            throw err("unknown service \(service.pyStr)", [("services", .array(services))])
        }
        let p = policy()
        var out = JObject([("service", service), ("plan_name", p["plan_name"]!)])
        for (k, v) in cov { out[k] = v }
        if case .int(let a) = p["deductible"]!, case .int(let b) = p["deductible_met"]! {
            out["deductible_remaining"] = .int(a - b)  // Python: int - int stays an int
        } else {
            out["deductible_remaining"] = .double(pyRound2(p["deductible"]!.number! - p["deductible_met"]!.number!))
        }
        return out
    }

    private func getClaimStatus(_ claimId: JSON?) throws -> JObject {
        let mine = claims.filter { $0["member_id"]?.string == memberId }
        if let claimId, claimId.truthy {
            guard let c = mine.first(where: { $0["claim_id"] == claimId }) else {
                throw err("claim \(claimId.pyStr) not found for this member")
            }
            return c.without(["member_id"])
        }
        let sorted = stableSorted(mine, reverse: true) { $0["service_date"]?.string ?? "" }
        return JObject([("claims", .array(sorted.prefix(5).map {
            .object($0.pick(["claim_id", "provider_name", "service", "service_date", "status", "amount_billed"]))
        }))])
    }

    private func fileClaim(_ providerId: JSON, _ serviceDate: JSON, _ service: JSON, _ amount: JSON) throws -> JObject {
        let prov = try provider(providerId)
        let d = try parseDate(serviceDate, "service_date")
        if d > today { throw err("service_date cannot be in the future") }
        if d < today.adding(-365) { throw err("claims must be filed within 365 days of service") }
        if coverage[member["plan"]!.string!]?.object?[service.pyStr] == nil { throw err("unknown service \(service.pyStr)") }
        guard let amt = amount.number else { throw err("amount must be a number") }
        if amt <= 0 { throw err("amount must be positive") }
        let claim = JObject([
            ("claim_id", .string(nextId("CLM", 90000))), ("member_id", .string(memberId)), ("provider_id", providerId),
            ("provider_name", prov["name"]!), ("service", service), ("service_date", .string(d.iso)),
            ("submitted_date", .string(today.iso)), ("amount_billed", .double(pyRound2(amt))), ("amount_covered", .null),
            ("status", .string("submitted")), ("denial_reason", .null), ("last_update", .string(today.iso)),
            ("expected_decision_by", .string(today.adding(10).iso)),
        ])
        claims.append(claim)
        return claim.without(["member_id"])
    }

    private func findProvider(_ specialty: JSON, _ city: JSON?, _ inNetworkOnly: JSON) throws -> JObject {
        let plan = member["plan"]!.string!
        var rows = providers.filter { $0["specialty"] == specialty }
        if rows.isEmpty { throw err("unknown specialty \(specialty.pyStr)") }
        if let city, city.truthy {
            let c = city.pyStr.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            rows = rows.filter { $0["city"]!.string!.lowercased() == c }
        }
        let inNet: (JObject) -> Bool = { p in p["in_network_plans"]!.array!.contains(.string(plan)) }
        if inNetworkOnly.truthy { rows = rows.filter(inNet) }
        rows = stableSorted(rows) { p in
            "\((p["accepting_new_patients"]?.bool ?? false) ? 0 : 1)|\(p["name"]!.string!)"
        }
        let short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        let out: [JSON] = rows.prefix(5).map { p in
            .object(JObject([
                ("provider_id", p["provider_id"]!), ("name", p["name"]!), ("clinic", p["clinic"]!), ("city", p["city"]!),
                ("in_network", .bool(inNet(p))), ("accepting_new_patients", p["accepting_new_patients"]!),
                ("working_days", .array(workingDays(p).map { .string(short[$0]) })),
            ]))
        }
        return JObject([("providers", .array(out)), ("count", .int(rows.count))])
    }

    private func bookAppointment(_ providerId: JSON, _ date: JSON, _ time: JSON?, _ reason: JSON?) throws -> JObject {
        let prov = try provider(providerId)
        let d = try parseDate(date, "date")
        try checkBookableDay(prov, d)
        if prov["accepting_new_patients"]?.bool != true { throw err("\(prov["name"]!.string!) is not accepting new patients") }
        let free = freeTimes(prov, d)
        let t: String? = (time?.truthy ?? false) ? normTime(time!.pyStr) : free.first
        guard let t, free.contains(t) else { throw err("that time is not available", [("available_times", .array(free.prefix(8).map { .string($0) }))]) }
        let appt = JObject([
            ("appointment_id", .string(nextId("APT", 8000))), ("member_id", .string(memberId)), ("provider_id", providerId),
            ("provider_name", prov["name"]!), ("specialty", prov["specialty"]!), ("datetime", .string("\(d.iso)T\(t)")),
            ("status", .string("scheduled")), ("reason", (reason?.truthy ?? false) ? reason! : .string("")),
        ])
        appointments.append(appt)
        return appt.without(["member_id"])
    }

    private func reschedule(_ appointmentId: JSON, _ newDate: JSON, _ newTime: JSON?) throws -> JObject {
        guard let idx = appointments.firstIndex(where: {
            $0["appointment_id"] == appointmentId && $0["member_id"]?.string == memberId && $0["status"]?.string == "scheduled"
        }) else { throw err("appointment \(appointmentId.pyStr) not found for this member") }
        let appt = appointments[idx]
        let prov = try provider(appt["provider_id"]!)
        let d = try parseDate(newDate, "new_date")
        try checkBookableDay(prov, d)
        let free = freeTimes(prov, d, ignore: appointmentId.string)
        let t: String?
        if let newTime, newTime.truthy {
            t = normTime(newTime.pyStr)
        } else {
            let orig = String(appt["datetime"]!.string!.dropFirst(11).prefix(5))
            t = free.contains(orig) ? orig : free.first
        }
        guard let t, free.contains(t) else { throw err("that time is not available", [("available_times", .array(free.prefix(8).map { .string($0) }))]) }
        let old = appt["datetime"]!
        appointments[idx]["datetime"] = .string("\(d.iso)T\(t)")
        return JObject([("appointment_id", appointmentId), ("provider_name", appt["provider_name"]!), ("old_datetime", old),
                        ("new_datetime", appointments[idx]["datetime"]!), ("status", .string("scheduled"))])
    }

    private func escalate(_ topic: JSON, _ summary: JSON) throws -> JObject {
        guard let t = topic.string, ["claim_appeal", "complaint", "billing", "medical", "other"].contains(t) else {
            throw err("unknown topic \(topic.pyStr)")
        }
        let ticket = JObject([("ticket_id", .string(nextId("HT", 4000))), ("topic", topic), ("summary", summary),
                              ("expected_response", .string("within 1 business day"))])
        tickets.append(ticket)
        return ticket
    }
}

/// env.py `_norm_time`.
func normTime(_ raw: String) -> String {
    var s = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().replacingOccurrences(of: ".", with: ":")
    let pm = s.hasSuffix("pm")
    if s.hasSuffix("pm") { s.removeLast(2) }
    if s.hasSuffix("am") { s.removeLast(2) }
    s = s.trimmingCharacters(in: .whitespacesAndNewlines)
    let hh: String, mm: String
    if s.contains(":") {
        let parts = s.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        hh = parts[0]; mm = parts.count > 1 ? parts[1] : ""
    } else { hh = s; mm = "00" }
    guard var h = Int(hh), let m = Int(mm) else { return s }
    if pm && h < 12 { h += 12 }
    return String(format: "%02d:%02d", h, m)
}
