import Foundation

/// One model action: `{"tool": name, "args": {...}}` or `{"reply": text}` (shared/action_format.md).
public enum Action: Equatable {
    case tool(name: String, args: JObject)
    case reply(String)

    public var json: JSON {
        switch self {
        case .tool(let n, let a): return .object(JObject([("tool", .string(n)), ("args", .object(a))]))
        case .reply(let r): return .object(JObject([("reply", .string(r))]))
        }
    }

    /// prompts.py `action_json`: `json.dumps(action, ensure_ascii=False, separators=(", ", ": "))`.
    public var text: String { json.encode(sortKeys: false, separators: (", ", ": "), ensureASCII: false) }

    public var isTool: Bool { if case .tool = self { return true }; return false }
    public var toolName: String? { if case .tool(let n, _) = self { return n }; return nil }
    public var args: JObject? { if case .tool(_, let a) = self { return a }; return nil }
    public var replyText: String? { if case .reply(let r) = self { return r }; return nil }

    public init?(json: JSON) {
        guard let o = json.object else { return nil }
        if let t = o["tool"]?.string {
            let a = o["args"] ?? .object(JObject())
            guard let args = a.object else { return nil }
            self = .tool(name: t, args: args)
        } else if let r = o["reply"]?.string {
            self = .reply(r)
        } else {
            return nil
        }
    }

    /// prompts.py `parse_action`: first JSON object in the output (code fences allowed).
    public static func parse(_ text: String) -> Action? {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let re = try? NSRegularExpression(pattern: "```(?:json)?\\s*(.*?)```", options: [.dotMatchesLineSeparators]),
           let m = re.firstMatch(in: s, range: NSRange(location: 0, length: (s as NSString).length)) {
            s = (s as NSString).substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let start = s.firstIndex(of: "{"), let v = JSON.rawDecode(String(s[start...])) else { return nil }
        return Action(json: v)
    }
}

public struct Gate {
    public var ok: Bool
    public var reasons: [String]
    public var minLogprob: Double?
    public var meanLogprob: Double?
}

/// Port of harbor_env/router.py + grounding.py, configured by shared/router.json.
public final class Router {
    public let maxSteps: Int
    public let minLP: Double
    public let meanLP: Double
    public let claimThreshold: Double
    public var useConfidence = true
    private let neverLeave: NSRegularExpression

    public init(config: JSON = Shared.routerConfig) {
        maxSteps = config["max_steps"]?.int ?? 4
        minLP = config["confidence"]?["min_token_logprob"]?.number ?? -6
        meanLP = config["confidence"]?["mean_token_logprob"]?.number ?? -0.35
        claimThreshold = config["policy"]?["claim_review_threshold"]?.number ?? 1000
        neverLeave = try! NSRegularExpression(pattern: config["never_leave"]!["regex"]!.string!, options: [.caseInsensitive])
    }

    public func isNeverLeave(_ text: String) -> Bool {
        neverLeave.firstMatch(in: text, range: NSRange(location: 0, length: (text as NSString).length)) != nil
    }

    public func validate(_ action: Action?, realArgs: JObject?, env: HarborEnv, context: [String]?) -> [String] {
        guard let action else { return ["invalid_json"] }
        switch action {
        case .reply(let r):
            if r.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return ["empty_reply"] }
            if let context, !Grounding.ungrounded(r, context).isEmpty { return ["reply_number_not_grounded"] }
            return []
        case .tool(let tool, _):
            guard let spec = Shared.toolsByName[tool] else { return ["unknown_tool"] }
            let args = realArgs ?? JObject()
            var reasons: [String] = []
            if let e = Schema.firstError(.object(args), spec["input_schema"]!) { reasons.append(e) }
            if args["member_id"] != .string(env.memberId) { reasons.append("member_id_mismatch") }
            if let p = args["provider_id"], !env.providers.contains(where: { $0["provider_id"] == p }) { reasons.append("unknown_provider_id") }
            if let a = args["appointment_id"], !env.appointments.contains(where: { $0["appointment_id"] == a && $0["member_id"]?.string == env.memberId }) {
                reasons.append("unknown_appointment_id")
            }
            if let c = args["claim_id"], c.truthy, !env.claims.contains(where: { $0["claim_id"] == c && $0["member_id"]?.string == env.memberId }) {
                reasons.append("unknown_claim_id")
            }
            for key in ["date", "new_date"] {
                if let v = args[key] {
                    if let d = Day(iso: v.pyStr) { if d <= env.today { reasons.append("booking_date_not_after_today") } }
                    else { reasons.append("bad_\(key)") }
                }
            }
            if let v = args["service_date"] {
                if let d = Day(iso: v.pyStr) { if d > env.today { reasons.append("service_date_in_future") } }
                else { reasons.append("bad_service_date") }
            }
            return reasons
        }
    }

    public func policy(_ action: Action?, realArgs: JObject?) -> [String] {
        if action?.toolName == "file_claim", let a = (realArgs ?? JObject())["amount"]?.number, a > claimThreshold {
            return ["policy_claim_over_threshold"]
        }
        return []
    }

    public func check(_ action: Action?, realArgs: JObject?, env: HarborEnv, logprobs: [Double], context: [String]?) -> Gate {
        var reasons = validate(action, realArgs: realArgs, env: env, context: context) + policy(action, realArgs: realArgs)
        var mn: Double?, mean: Double?
        if !logprobs.isEmpty {
            mn = logprobs.min()
            mean = logprobs.reduce(0, +) / Double(logprobs.count)
            // Confidence gates tool calls only (replies are checked by validators).
            if useConfidence, action?.isTool == true, mn! < minLP || mean! < meanLP { reasons.append("low_confidence") }
        }
        return Gate(ok: reasons.isEmpty, reasons: reasons, minLogprob: mn, meanLogprob: mean)
    }
}

/// Minimal JSON Schema (the keywords tools.json uses: type, properties, required, additionalProperties, enum).
enum Schema {
    static func firstError(_ v: JSON, _ s: JSON, path: [String] = []) -> String? {
        let where_ = path.isEmpty ? "" : path.joined(separator: ".")
        if let t = s["type"]?.string, !typeOK(v, t) { return "schema:type:\(where_.isEmpty ? "\(v.pyStr) is not of type '\(t)'".prefix(40) : Substring(where_))" }
        if let e = s["enum"]?.array, !e.contains(v) { return "schema:enum:\(where_.isEmpty ? "not in enum" : where_)" }
        guard let o = v.object else { return nil }
        let props = s["properties"]?.object ?? JObject()
        for r in s["required"]?.array?.compactMap(\.string) ?? [] where o[r] == nil {
            return "schema:required:\(where_.isEmpty ? "'\(r)' is a required property".prefix(40) : Substring(where_))"
        }
        if s["additionalProperties"] == .bool(false) {
            for k in o.keys where props[k] == nil {
                return "schema:additionalProperties:\(where_.isEmpty ? "Additional properties are not allowed".prefix(40) : Substring(where_))"
            }
        }
        for (k, sub) in props {
            if let val = o[k], let e = firstError(val, sub, path: path + [k]) { return e }
        }
        return nil
    }

    static func typeOK(_ v: JSON, _ t: String) -> Bool {
        switch (t, v) {
        case ("string", .string), ("object", .object), ("array", .array), ("boolean", .bool), ("null", .null): return true
        case ("integer", .int): return true
        case ("integer", .double(let d)): return d == d.rounded()
        case ("number", .int), ("number", .double): return true
        default: return false
        }
    }
}

/// grounding.py: every number in a reply must appear in (or be derived from) the conversation.
public enum Grounding {
    static let time = try! NSRegularExpression(pattern: "\\b(\\d{1,2})(?::(\\d{2}))?\\s*([ap])\\.?m\\.?\\b", options: [.caseInsensitive])
    static let num = try! NSRegularExpression(pattern: "(?<![\\w\\[])\\$?(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?(%|st|nd|rd|th)?(?![\\w\\]])")
    static let placeholder = try! NSRegularExpression(pattern: "\\[[A-Z_]+_\\d+\\]")
    static let time24 = try! NSRegularExpression(pattern: "\\b(\\d{1,2}):(\\d{2})\\b")
    static let isoTime = try! NSRegularExpression(pattern: "T(\\d{2}):(\\d{2})")
    static let isoDate = try! NSRegularExpression(pattern: "\\b(\\d{4})-(\\d{2})-(\\d{2})(?=\\b|T)")
    static let plain = try! NSRegularExpression(pattern: "^[\\d.]+$")
    static let numeric = try! NSRegularExpression(pattern: "^\\d+(\\.\\d+)?$")
    static let safe: Set<String> = ["1", "2", "3", "0"]

    static func sub(_ re: NSRegularExpression, _ s: String, _ with: String) -> String {
        re.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: (s as NSString).length), withTemplate: with)
    }

    static func groups(_ re: NSRegularExpression, _ s: String) -> [[String?]] {
        let ns = s as NSString
        return re.matches(in: s, range: NSRange(location: 0, length: ns.length)).map { m in
            (0..<m.numberOfRanges).map { i in m.range(at: i).location == NSNotFound ? nil : ns.substring(with: m.range(at: i)) }
        }
    }

    static func full(_ re: NSRegularExpression, _ s: String) -> Bool {
        re.firstMatch(in: s, range: NSRange(location: 0, length: (s as NSString).length)) != nil
    }

    /// Python `f"{v:g}"`.
    static func g(_ v: Double) -> String { String(format: "%g", v) }

    static func canon(_ intpart: String, _ frac: String?) -> String {
        g(Double(intpart.replacingOccurrences(of: ",", with: "") + (frac.map { "." + $0 } ?? "")) ?? 0)
    }

    static func numbersIn(_ text: String, asReply: Bool = false) -> Set<String> {
        var t = sub(placeholder, text, " ")
        var out = Set<String>()
        for m in groups(time, t) {
            let h = (Int(m[1]!)! % 12) + (m[3]!.lowercased() == "p" ? 12 : 0)
            out.insert(String(format: "%02d:%@", h, m[2] ?? "00"))
        }
        t = sub(time, t, " ")
        for m in groups(time24, t) { out.insert(String(format: "%02d:%@", Int(m[1]!)!, m[2]!)) }
        t = sub(time24, t, " ")
        for m in groups(num, t) {
            let v = canon(m[1]!, m[2])
            if m[3] == "%" {
                out.insert(g((Double(v) ?? 0) / 100))
                if !asReply { out.insert(v) }
            } else {
                out.insert(v)
            }
        }
        return out
    }

    static func contextNumbers(_ contents: [String]) -> Set<String> {
        var nums = Set<String>()
        for c in contents {
            nums.formUnion(numbersIn(c))
            for m in groups(isoTime, c) { nums.insert("\(m[1]!):\(m[2]!)") }
            for m in groups(isoDate, c) { nums.formUnion([g(Double(m[2]!)!), g(Double(m[3]!)!), m[1]!]) }
        }
        return nums
    }

    public static func ungrounded(_ reply: String, _ contents: [String]) -> [String] {
        let ctx = contextNumbers(contents)
        let pct = Set(ctx.filter { full(plain, $0) && (Double($0) ?? 9) < 1 }.map { g((Double($0) ?? 0) * 100) })
        let vals = Set(ctx.filter { full(numeric, $0) }.compactMap(Double.init).filter { $0 >= 10 }).sorted()
        var derived = Set<String>()
        for a in vals { for b in vals { if a > b { derived.insert(g(a - b)) } else if a < b { derived.insert(g(a + b)) } } }
        return numbersIn(reply, asReply: true).filter { !(ctx.contains($0) || safe.contains($0) || pct.contains($0) || derived.contains($0)) }.sorted()
    }
}
