import Foundation

/// Port of `backend/sdk/gradientharbor/privacy.py`: redaction with placeholder re-fill, and the
/// fail-closed PII scanner. Must reproduce `shared/redaction_vectors.json` exactly.
public final class Redactor {
    private enum Original { case text(String), amount(Double) }

    private let placeholderFormat: String
    private let patterns: [(String, NSRegularExpression)]
    private let amountKeys: Set<String>
    private let known: [(type: String, value: String, re: NSRegularExpression)]
    private var forward: [String: String] = [:]      // "TYPE|normalized value" -> placeholder
    private var reverse: [String: Original] = [:]    // placeholder -> original value
    private var counts: [String: Int] = [:]

    public init(rules: JSON = Shared.redactionRules, known: [(String, String)] = []) {
        placeholderFormat = rules["placeholder"]?.string ?? "[{type}_{n}]"
        patterns = rules["patterns"]!.array!.map { ($0["type"]!.string!, try! NSRegularExpression(pattern: $0["regex"]!.string!)) }
        amountKeys = Set(rules["amount_keys"]?.array?.compactMap(\.string) ?? [])
        let fields = rules["known_fields"]?.object ?? JObject()
        // Longest values first (stable), so "owen.novak@example.com" wins over "Owen".
        let pairs = known.compactMap { k, v -> (String, String)? in
            guard let t = fields[k]?.string, !v.isEmpty else { return nil }
            return (t, v)
        }
        let ordered = pairs.enumerated().sorted { a, b in
            a.element.1.count != b.element.1.count ? a.element.1.count > b.element.1.count : a.offset < b.offset
        }.map(\.element)
        self.known = ordered.map { t, v in
            (t, v, try! NSRegularExpression(pattern: "(?<![\\w@.])" + NSRegularExpression.escapedPattern(for: v) + "(?![\\w@])",
                                            options: [.caseInsensitive]))
        }
    }

    private func placeholder(_ type: String, text: String? = nil, amount: Double? = nil) -> String {
        let norm = amount.map { String(format: "%.2f", $0) } ?? (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let key = "\(type)|\(norm)"
        if let ph = forward[key] { return ph }
        counts[type, default: 0] += 1
        let ph = placeholderFormat.replacingOccurrences(of: "{type}", with: type).replacingOccurrences(of: "{n}", with: String(counts[type]!))
        forward[key] = ph
        reverse[ph] = amount.map { .amount($0) } ?? .text(text ?? "")
        return ph
    }

    private static func amountValue(_ s: String) -> Double {
        Double(s.filter { $0.isNumber || $0 == "." }) ?? 0
    }

    private static func replace(_ text: String, _ re: NSRegularExpression, _ with: (String) -> String) -> String {
        let ns = text as NSString
        var out = ""
        var last = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            out += with(ns.substring(with: m.range))
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        return out
    }

    public func redactText(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        var t = text
        for k in known {
            t = Self.replace(t, k.re) { _ in self.placeholder(k.type, text: k.value) }
        }
        for (type, re) in patterns {
            if type == "AMOUNT" {
                t = Self.replace(t, re) { self.placeholder("AMOUNT", amount: Self.amountValue($0)) }
            } else {
                t = Self.replace(t, re) { self.placeholder(type, text: $0) }
            }
        }
        return t
    }

    public func redact(_ obj: JSON, key: String? = nil) -> JSON {
        switch obj {
        case .object(let o):
            return .object(JObject(o.map { ($0.key, redact($0.value, key: $0.key)) }))
        case .array(let a):
            return .array(a.map { redact($0, key: key) })
        case .int(let i) where key.map(amountKeys.contains) ?? false:
            return .string(placeholder("AMOUNT", amount: Double(i)))
        case .double(let d) where key.map(amountKeys.contains) ?? false:
            return .string(placeholder("AMOUNT", amount: d))
        case .string(let s):
            return .string(redactText(s))
        default:
            return obj
        }
    }

    public func redact(_ o: JObject) -> JObject { redact(.object(o)).object! }

    public func refillText(_ text: String) -> String {
        guard !text.isEmpty, let re = try? NSRegularExpression(pattern: "\\$?\\[[A-Z_]+_\\d+\\]") else { return text }
        return Self.replace(text, re) { m in
            let ph = m.hasPrefix("$") ? String(m.dropFirst()) : m
            switch self.reverse[ph] {
            case .amount(let v): return Self.formatAmount(v)
            case .text(let s): return s
            case nil: return m
            }
        }
    }

    public func refill(_ obj: JSON) -> JSON {
        switch obj {
        case .object(let o): return .object(JObject(o.map { ($0.key, refill($0.value)) }))
        case .array(let a): return .array(a.map(refill))
        case .string(let s):
            var bare = s.trimmingCharacters(in: .whitespacesAndNewlines)
            while bare.hasPrefix("$") { bare.removeFirst() }
            if case .amount(let v) = reverse[bare] { return .double(v) }
            return .string(refillText(s))
        default: return obj
        }
    }

    public func refill(_ o: JObject) -> JObject { refill(.object(o)).object! }

    /// privacy.py `format_amount`: "$1,450" / "$1,450.50".
    public static func formatAmount(_ v: Double) -> String {
        let f = NumberFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.groupingSeparator = ","
        f.minimumFractionDigits = 2
        f.maximumFractionDigits = 2
        f.roundingMode = .halfEven
        let s = "$" + (f.string(from: NSNumber(value: v)) ?? String(v))
        return s.hasSuffix(".00") ? String(s.dropLast(3)) : s
    }
}

/// Finds raw PII left in an outbound payload. Non-empty result: do not send.
public final class PIIScanner {
    private let checks: [(String, NSRegularExpression)]

    public init(rules: JSON = Shared.redactionRules, members: [JSON] = Shared.fixtures["members"]!.array!) {
        let scan = rules["scan"]!
        var byType: [String: NSRegularExpression] = [:]
        for p in rules["patterns"]!.array! { byType[p["type"]!.string!] = try! NSRegularExpression(pattern: p["regex"]!.string!) }
        var checks: [(String, NSRegularExpression)] = scan["patterns"]!.array!.compactMap { t in byType[t.string!].map { (t.string!, $0) } }
        let fields = scan["known_fields"]!.array!.compactMap(\.string)
        for m in members {
            guard var o = m.object else { continue }
            if let name = o["name"]?.string, o["last_name"] == nil { o["last_name"] = .string(String(name.split(separator: " ").last ?? "")) }
            for f in fields {
                guard let v = o[f]?.pyStr, o[f] != nil, !(o[f]!.isNull), !v.isEmpty else { continue }
                let re = try! NSRegularExpression(pattern: "(?<![\\w@.])" + NSRegularExpression.escapedPattern(for: v) + "(?![\\w@])",
                                                  options: [.caseInsensitive])
                checks.append((f.uppercased(), re))
            }
        }
        self.checks = checks
    }

    public func scan(_ text: String) -> [(type: String, match: String)] {
        let ns = text as NSString
        var out: [(String, String)] = []
        for (t, re) in checks {
            for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
                out.append((t, ns.substring(with: m.range)))
            }
        }
        return out
    }
}
