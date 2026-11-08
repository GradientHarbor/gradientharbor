import Foundation

/// The files in `shared/`, bundled with the app: the same tools, fixtures, prompts, router and redaction
/// rules the evals used.
public enum Shared {
    public static let bundleDir: URL = Bundle.module.url(forResource: "shared", withExtension: nil)!

    public static func url(_ path: String) -> URL { bundleDir.appendingPathComponent(path) }

    public static func json(_ path: String) -> JSON {
        // Resources are part of the build; a parse failure is a packaging bug.
        try! JSON.parse(Data(contentsOf: url(path)))
    }

    public static func text(_ path: String) -> String {
        (try? String(contentsOf: url(path), encoding: .utf8)) ?? ""
    }

    public static let tools: [JSON] = json("tools.json")["tools"]!.array!
    public static let toolsByName: [String: JSON] = Dictionary(uniqueKeysWithValues: tools.map { ($0["name"]!.string!, $0) })
    public static let routerConfig: JSON = json("router.json")
    public static let redactionRules: JSON = json("redaction.json")

    public static let fixtures: [String: JSON] = {
        var out: [String: JSON] = [:]
        for name in ["meta", "members", "policies", "coverage", "providers", "claims", "appointments"] {
            out[name] = json("fixtures/\(name).json")
        }
        return out
    }()
}

/// Calendar-free ISO date helpers (UTC days), mirroring Python's `datetime.date`.
public struct Day: Comparable, Hashable {
    public let ordinal: Int  // days since 1970-01-01

    public init(ordinal: Int) { self.ordinal = ordinal }

    /// Python `date.fromisoformat(s[:10])`.
    public init?(iso: String) {
        let s = String(iso.prefix(10))
        let parts = s.split(separator: "-")
        guard s.count == 10, parts.count == 3, let y = Int(parts[0]), let m = Int(parts[1]), let d = Int(parts[2]),
              (1...12).contains(m), d >= 1, d <= Day.daysIn(y, m) else { return nil }
        ordinal = Day.toOrdinal(y, m, d)
    }

    public static func < (a: Day, b: Day) -> Bool { a.ordinal < b.ordinal }
    public func adding(_ days: Int) -> Day { Day(ordinal: ordinal + days) }

    public var ymd: (Int, Int, Int) {
        // civil-from-days (Howard Hinnant)
        let z = ordinal + 719468
        let era = (z >= 0 ? z : z - 146096) / 146097
        let doe = z - era * 146097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp < 10 ? mp + 3 : mp - 9
        return (yoe + era * 400 + (m <= 2 ? 1 : 0), m, d)
    }

    public var iso: String {
        let (y, m, d) = ymd
        return String(format: "%04d-%02d-%02d", y, m, d)
    }

    /// Python `date.weekday()`: Monday = 0.
    public var weekday: Int { ((ordinal % 7) + 7 + 3) % 7 }  // 1970-01-01 was a Thursday (3)

    public static let weekdayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    public var weekdayName: String { Day.weekdayNames[weekday] }

    static func toOrdinal(_ y0: Int, _ m: Int, _ d: Int) -> Int {
        let y = m <= 2 ? y0 - 1 : y0
        let era = (y >= 0 ? y : y - 399) / 400
        let yoe = y - era * 400
        let mp = m > 2 ? m - 3 : m + 9
        let doy = (153 * mp + 2) / 5 + d - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return era * 146097 + doe - 719468
    }

    static func daysIn(_ y: Int, _ m: Int) -> Int {
        switch m {
        case 2: return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 ? 29 : 28
        case 4, 6, 9, 11: return 30
        default: return 31
        }
    }
}

/// Python `sorted(..., key=..., reverse=...)`: stable in both directions.
func stableSorted<T, K: Comparable>(_ xs: [T], reverse: Bool = false, by key: (T) -> K) -> [T] {
    let indexed = xs.enumerated().map { ($0.offset, $0.element, key($0.element)) }
    return indexed.sorted { a, b in
        if a.2 != b.2 { return reverse ? a.2 > b.2 : a.2 < b.2 }
        return a.0 < b.0
    }.map { $0.1 }
}

/// Python `round(x, 2)` for the values in the fixtures (banker's rounding on the decimal digits).
func pyRound2(_ x: Double) -> Double { (x * 100).rounded(.toNearestOrEven) / 100 }
