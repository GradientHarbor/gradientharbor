import Foundation

/// Order-preserving JSON with Python-compatible encoding.
///
/// The student was trained on prompts rendered by Python's `json.dumps`, so the phone has to produce the
/// same bytes: key order, int vs float (`600.0` stays `600.0`), separators and escaping all matter.
public indirect enum JSON: Equatable, CustomStringConvertible {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSON])
    case object(JObject)

    public var description: String { encode(sortKeys: false, separators: (", ", ": "), ensureASCII: false) }
}

/// An insertion-ordered JSON object (Python dict semantics).
public struct JObject: Equatable, Sequence {
    public private(set) var keys: [String] = []
    private var storage: [String: JSON] = [:]

    public init() {}
    public init(_ pairs: [(String, JSON)]) { for (k, v) in pairs { self[k] = v } }

    public subscript(key: String) -> JSON? {
        get { storage[key] }
        set {
            if let newValue {
                if storage[key] == nil { keys.append(key) }
                storage[key] = newValue
            } else if storage.removeValue(forKey: key) != nil {
                keys.removeAll { $0 == key }
            }
        }
    }

    public var count: Int { keys.count }
    public func makeIterator() -> AnyIterator<(key: String, value: JSON)> {
        var i = 0
        return AnyIterator {
            guard i < keys.count else { return nil }
            defer { i += 1 }
            return (keys[i], storage[keys[i]]!)
        }
    }

    /// A copy without the given keys (Python: `{k: v for k, v in d.items() if k not in ...}`).
    public func without(_ drop: Set<String>) -> JObject {
        JObject(self.filter { !drop.contains($0.key) }.map { ($0.key, $0.value) })
    }

    /// A copy with only the given keys, in the given order (Python: `{k: d[k] for k in [...]}`).
    public func pick(_ order: [String]) -> JObject {
        JObject(order.map { ($0, self[$0] ?? .null) })
    }
}

// MARK: - accessors

extension JSON {
    public subscript(key: String) -> JSON? {
        if case .object(let o) = self { return o[key] }
        return nil
    }
    public var object: JObject? { if case .object(let o) = self { return o }; return nil }
    public var array: [JSON]? { if case .array(let a) = self { return a }; return nil }
    public var string: String? { if case .string(let s) = self { return s }; return nil }
    public var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var int: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d) where d == d.rounded(): return Int(d)
        default: return nil
        }
    }
    /// Python `float(x)` for numbers and numeric strings.
    public var number: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        case .string(let s): return Double(s.trimmingCharacters(in: .whitespaces))
        case .bool(let b): return b ? 1 : 0
        default: return nil
        }
    }
    public var isNull: Bool { if case .null = self { return true }; return false }

    /// Python truthiness.
    public var truthy: Bool {
        switch self {
        case .null: return false
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .double(let d): return d != 0
        case .string(let s): return !s.isEmpty
        case .array(let a): return !a.isEmpty
        case .object(let o): return o.count > 0
        }
    }

    /// Python `str(x)`.
    public var pyStr: String {
        switch self {
        case .string(let s): return s
        case .null: return "None"
        case .bool(let b): return b ? "True" : "False"
        case .int(let i): return String(i)
        case .double(let d): return JSON.pyFloat(d)
        default: return encode(sortKeys: false, separators: (", ", ": "), ensureASCII: false)
        }
    }
}

// MARK: - encoding (Python json.dumps)

extension JSON {
    /// `json.dumps(obj, separators=(",", ":"), sort_keys=True)`: the compact form used in prompts.
    public var compact: String { encode(sortKeys: true, separators: (",", ":"), ensureASCII: true) }

    public func encode(sortKeys: Bool, separators: (String, String), ensureASCII: Bool) -> String {
        var out = ""
        write(into: &out, sortKeys: sortKeys, sep: separators, ascii: ensureASCII)
        return out
    }

    private func write(into out: inout String, sortKeys: Bool, sep: (String, String), ascii: Bool) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .int(let i): out += String(i)
        case .double(let d): out += JSON.pyFloat(d)
        case .string(let s): JSON.writeString(s, into: &out, ascii: ascii)
        case .array(let a):
            out += "["
            for (i, v) in a.enumerated() {
                if i > 0 { out += sep.0 }
                v.write(into: &out, sortKeys: sortKeys, sep: sep, ascii: ascii)
            }
            out += "]"
        case .object(let o):
            out += "{"
            let keys = sortKeys ? o.keys.sorted(by: JSON.pyLess) : o.keys
            for (i, k) in keys.enumerated() {
                if i > 0 { out += sep.0 }
                JSON.writeString(k, into: &out, ascii: ascii)
                out += sep.1
                o[k]!.write(into: &out, sortKeys: sortKeys, sep: sep, ascii: ascii)
            }
            out += "}"
        }
    }

    /// Python sorts str keys by code point.
    static func pyLess(_ a: String, _ b: String) -> Bool {
        a.unicodeScalars.map(\.value).lexicographicallyPrecedes(b.unicodeScalars.map(\.value))
    }

    /// Python `repr(float)`: shortest round-trip digits, `.0` for integral values, exponent outside 1e-4..1e16.
    public static func pyFloat(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d > 0 ? "Infinity" : "-Infinity" }
        if d == d.rounded() && abs(d) < 1e16 { return String(format: "%.1f", d) }
        var s = "\(d)"  // Swift also prints the shortest round-trip representation
        if s.contains("e") {
            // Swift: 1e-05 / 1e+16 forms match Python already; normalise "e-5" to "e-05".
            let parts = s.split(separator: "e", maxSplits: 1)
            var exp = String(parts[1])
            let sign = exp.hasPrefix("-") ? "-" : "+"
            exp = exp.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
            if exp.count < 2 { exp = "0" + exp }
            s = "\(parts[0])e\(sign)\(exp)"
        }
        return s
    }

    static func writeString(_ s: String, into out: inout String, ascii: Bool) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if u.value < 0x20 {
                    out += String(format: "\\u%04x", u.value)
                } else if ascii && u.value > 0x7E {
                    if u.value > 0xFFFF {
                        let v = u.value - 0x10000
                        out += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
                    } else {
                        out += String(format: "\\u%04x", u.value)
                    }
                } else {
                    out.unicodeScalars.append(u)
                }
            }
        }
        out += "\""
    }
}

// MARK: - parsing (order-preserving; ints stay ints, "2400.0" stays a float)

public struct JSONParseError: Error { public let message: String; public let offset: Int }

extension JSON {
    public static func parse(_ text: String) throws -> JSON {
        var p = Parser(Array(text.utf8))
        p.skipWS()
        let v = try p.value()
        p.skipWS()
        guard p.i == p.b.count else { throw JSONParseError(message: "trailing data", offset: p.i) }
        return v
    }

    public static func parse(_ data: Data) throws -> JSON {
        try parse(String(decoding: data, as: UTF8.self))
    }

    /// Python `json.JSONDecoder().raw_decode(s)`: parse one value at the start, ignore what follows.
    public static func rawDecode(_ text: String) -> JSON? {
        var p = Parser(Array(text.utf8))
        return try? p.value()
    }

    struct Parser {
        let b: [UInt8]
        var i = 0
        init(_ b: [UInt8]) { self.b = b }

        mutating func skipWS() { while i < b.count, [0x20, 0x09, 0x0A, 0x0D].contains(b[i]) { i += 1 } }

        mutating func value() throws -> JSON {
            skipWS()
            guard i < b.count else { throw JSONParseError(message: "unexpected end", offset: i) }
            switch b[i] {
            case UInt8(ascii: "{"): return try object()
            case UInt8(ascii: "["): return try array()
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try lit("true"); return .bool(true)
            case UInt8(ascii: "f"): try lit("false"); return .bool(false)
            case UInt8(ascii: "n"): try lit("null"); return .null
            default: return try number()
            }
        }

        mutating func lit(_ s: String) throws {
            let u = Array(s.utf8)
            guard i + u.count <= b.count, Array(b[i..<i + u.count]) == u else {
                throw JSONParseError(message: "bad literal", offset: i)
            }
            i += u.count
        }

        mutating func object() throws -> JSON {
            i += 1
            var o = JObject()
            skipWS()
            if i < b.count, b[i] == UInt8(ascii: "}") { i += 1; return .object(o) }
            while true {
                skipWS()
                guard i < b.count, b[i] == UInt8(ascii: "\"") else { throw JSONParseError(message: "expected key", offset: i) }
                let k = try string()
                skipWS()
                guard i < b.count, b[i] == UInt8(ascii: ":") else { throw JSONParseError(message: "expected :", offset: i) }
                i += 1
                o[k] = try value()
                skipWS()
                guard i < b.count else { throw JSONParseError(message: "unterminated object", offset: i) }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "}") { i += 1; return .object(o) }
                throw JSONParseError(message: "expected , or }", offset: i)
            }
        }

        mutating func array() throws -> JSON {
            i += 1
            var a: [JSON] = []
            skipWS()
            if i < b.count, b[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
            while true {
                a.append(try value())
                skipWS()
                guard i < b.count else { throw JSONParseError(message: "unterminated array", offset: i) }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
                throw JSONParseError(message: "expected , or ]", offset: i)
            }
        }

        mutating func string() throws -> String {
            i += 1
            var bytes: [UInt8] = []
            while i < b.count {
                let c = b[i]
                if c == UInt8(ascii: "\"") { i += 1; return String(decoding: bytes, as: UTF8.self) }
                if c == UInt8(ascii: "\\") {
                    i += 1
                    guard i < b.count else { break }
                    let e = b[i]
                    i += 1
                    switch e {
                    case UInt8(ascii: "n"): bytes.append(0x0A)
                    case UInt8(ascii: "t"): bytes.append(0x09)
                    case UInt8(ascii: "r"): bytes.append(0x0D)
                    case UInt8(ascii: "b"): bytes.append(0x08)
                    case UInt8(ascii: "f"): bytes.append(0x0C)
                    case UInt8(ascii: "u"):
                        var code = try hex4()
                        if (0xD800...0xDBFF).contains(code), i + 1 < b.count, b[i] == UInt8(ascii: "\\"), b[i + 1] == UInt8(ascii: "u") {
                            i += 2
                            let low = try hex4()
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        }
                        bytes.append(contentsOf: Array(String(Character(Unicode.Scalar(code) ?? "\u{FFFD}")).utf8))
                    default: bytes.append(e)
                    }
                } else {
                    bytes.append(c)
                    i += 1
                }
            }
            throw JSONParseError(message: "unterminated string", offset: i)
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= b.count, let v = UInt32(String(decoding: b[i..<i + 4], as: UTF8.self), radix: 16) else {
                throw JSONParseError(message: "bad \\u escape", offset: i)
            }
            i += 4
            return v
        }

        mutating func number() throws -> JSON {
            let start = i
            var isFloat = false
            if i < b.count, b[i] == UInt8(ascii: "-") { i += 1 }
            while i < b.count {
                let c = b[i]
                if (0x30...0x39).contains(c) { i += 1 }
                else if c == UInt8(ascii: ".") || c == UInt8(ascii: "e") || c == UInt8(ascii: "E") || c == UInt8(ascii: "+")
                            || (c == UInt8(ascii: "-") && i > start) { isFloat = true; i += 1 }
                else { break }
            }
            let s = String(decoding: b[start..<i], as: UTF8.self)
            guard !s.isEmpty, s != "-" else { throw JSONParseError(message: "bad value", offset: start) }
            if !isFloat, let v = Int(s) { return .int(v) }
            guard let d = Double(s) else { throw JSONParseError(message: "bad number", offset: start) }
            return .double(d)
        }
    }
}
