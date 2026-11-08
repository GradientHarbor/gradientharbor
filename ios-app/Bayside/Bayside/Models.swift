import Foundation

// Shapes returned by the device runtime (use-case-evals-poc/device_runtime/server.py).
// Decoded with .convertFromSnakeCase.

struct DeviceState: Decodable {
    var today: String
    var member: Member
    var policy: Policy
    var coverage: [Coverage]
    var claims: [Claim]
    var appointments: [Appointment]
    var tickets: [Ticket]
}

struct Member: Decodable, Identifiable, Hashable {
    var memberId: String
    var name: String
    var firstName: String?
    var city: String
    var plan: String
    var policyId: String?
    var id: String { memberId }
}

struct Policy: Decodable {
    var policyId: String
    var planName: String
    var status: String
    var effectiveDate: String
    var renewalDate: String
    var deductible: Double
    var deductibleMet: Double
    var outOfPocketMax: Double
    var outOfPocketSpent: Double
    var dependents: Int
}

struct Coverage: Decodable, Identifiable {
    var service: String
    var label: String
    var covered: Bool
    var copay: Double?
    var coinsurance: Double?
    var requiresReferral: Bool
    var requiresPriorAuth: Bool
    var visitLimitPerYear: Int?
    var outOfNetwork: String?
    var id: String { service }
}

struct Claim: Decodable, Identifiable, Equatable {
    var claimId: String
    var providerName: String
    var service: String
    var serviceDate: String
    var status: String
    var amountBilled: Double
    var amountCovered: Double?
    var denialReason: String?
    var expectedDecisionBy: String?
    var id: String { claimId }
}

struct Appointment: Decodable, Identifiable, Equatable {
    var appointmentId: String
    var providerName: String
    var specialty: String
    var datetime: String
    var reason: String?
    var id: String { appointmentId }
}

struct Ticket: Decodable, Identifiable {
    var ticketId: String
    var topic: String
    var summary: String
    var expectedResponse: String?
    var id: String { ticketId }
}

struct RuntimeStatus: Decodable {
    struct ModelInfo: Decodable {
        var name: String
        var round: Int?
        var sizeBytes: Double
        var version: String
        var base: String
    }
    struct Settings: Decodable {
        var offline: Bool
        var shadowRate: Double
        var sync: Bool
        var backend: String
        var deviceId: String
    }
    var ready: Bool
    var error: String?
    var model: ModelInfo
    var settings: Settings
    var counters: [String: Int]
    var queued: Int
    var runtime: String
}

struct LedgerResponse: Decodable {
    struct Stats: Decodable {
        var onDevice: Int
        var requests: Int
        var sent: Int
        var blocked: Int
        var queued: Int
        var rawPiiFieldsSent: Int
        var costUsd: Double
    }
    var rows: [LedgerRow]
    var stats: Stats
}

struct LedgerRow: Decodable, Identifiable {
    var id: String
    var at: String
    var reason: String
    var status: String
    var payload: String?
    var request: String?
    var response: String?
    var latencyMs: Double?
    var costUsd: Double?
    var note: String?
    var findings: [String]?
    var model: String?
}

/// Minimal JSON value, for tool actions in chat events.
enum JSONValue: Decodable, CustomStringConvertible {
    case string(String), number(Double), bool(Bool), object([String: JSONValue]), array([JSONValue]), null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    var string: String? { if case .string(let s) = self { return s }; return nil }

    var description: String {
        switch self {
        case .string(let s): return "\"\(s)\""
        case .number(let n): return n == n.rounded() && abs(n) < 1e15 ? String(Int(n)) : String(n)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .array(let a): return "[" + a.map(\.description).joined(separator: ", ") + "]"
        case .object(let o):
            // tool first, then args, then the rest: reads like the model's own output
            let order = ["tool", "args", "reply", "member_id"]
            let keys = o.keys.sorted { (order.firstIndex(of: $0) ?? 99, $0) < (order.firstIndex(of: $1) ?? 99, $1) }
            return "{" + keys.map { "\"\($0)\": \(o[$0]!.description)" }.joined(separator: ", ") + "}"
        }
    }
}

/// One newline-delimited event from POST /v1/device/chat.
struct ChatEvent: Decodable {
    var type: String
    var conversationId: String?
    var who: String?
    var latencyS: Double?
    var promptTokens: Int?
    var action: JSONValue?
    var raw: String?
    var gate: [String]?
    var reasons: [String]?
    var plain: String?
    var tool: String?
    var label: String?
    var status: String?
    var error: String?
    var summary: String?
    var text: String?
    var served: String?
    var flagged: Bool?
    var escalated: Bool?
    var reason: String?
    var neverLeave: Bool?
    var redacted: String?
    var message: String?
}

// MARK: - formatting

enum Fmt {
    private static let iso: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd'T'HH:mm"; return f
    }()
    private static let isoDay: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    private static func out(_ format: String) -> DateFormatter {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US"); f.dateFormat = format; return f
    }
    static let dayF = out("EEE, MMM d"), longDayF = out("EEEE, MMMM d"), timeF = out("h:mm a")

    static func dateTime(_ s: String) -> Date? { iso.date(from: String(s.prefix(16))) }
    static func day(_ s: String) -> Date? { isoDay.date(from: String(s.prefix(10))) }
    static func shortDay(_ s: String) -> String { day(s).map { dayF.string(from: $0) } ?? s }
    static func longDay(_ s: String) -> String { day(s).map { longDayF.string(from: $0) } ?? s }
    static func time(_ s: String) -> String { dateTime(s).map { timeF.string(from: $0) } ?? "" }
    static func money(_ v: Double) -> String {
        let f = NumberFormatter(); f.numberStyle = .currency; f.locale = Locale(identifier: "en_US")
        f.maximumFractionDigits = v == v.rounded() ? 0 : 2
        return f.string(from: NSNumber(value: v)) ?? "$\(v)"
    }
    static func pretty(_ s: String) -> String {
        s.replacingOccurrences(of: "_", with: " ").capitalized(with: Locale(identifier: "en_US"))
    }
    static func specialty(_ s: String) -> String {
        [
            "optometrist": "Eye exam", "dentist": "Dentist", "dermatologist": "Dermatology",
            "physical_therapist": "Physical therapy", "primary_care": "Primary care", "orthopedist": "Orthopedics",
            "therapist": "Therapy", "chiropractor": "Chiropractic", "cardiologist": "Cardiology",
        ][s] ?? pretty(s)
    }
}
