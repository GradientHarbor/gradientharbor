import Foundation
import HarborCore

// HarborCore parity without MLX: redaction vectors, replay of recorded eval episodes, prompt rendering.
//   harbor-check <parity dir>
//
//   harbor-parity core <parity dir>                  redaction vectors + replay of recorded eval episodes
//   harbor-parity prompts <parity dir> <model dir>   prompt rendering + token ids vs Python
//   harbor-parity generate <parity dir> <model dir> [n]   greedy generation vs the Mac's raw outputs

func lines(_ url: URL) -> [JSON] {
    (try? String(contentsOf: url, encoding: .utf8))?.split(separator: "\n").compactMap { try? JSON.parse(String($0)) } ?? []
}

var failures = 0
func check(_ ok: Bool, _ what: @autoclosure () -> String) {
    if !ok { failures += 1; if failures <= 25 { print("  ✕ \(what())") } }
}

func vectors() {
    let v = Shared.json("redaction_vectors.json")
    var n = 0
    for g in v["groups"]!.array! {
        let known = g["known"]!.object!.map { ($0.key, $0.value.string ?? "") }
        let r = Redactor(known: known)
        for c in g["cases"]!.array! {
            n += 1
            let got = r.redactText(c["input"]!.string!)
            check(got == c["expected"]!.string!, "vector: \(got) != \(c["expected"]!.string!)")
        }
        for o in g["objects"]?.array ?? [] {
            n += 1
            let got = r.redact(o["input"]!)
            check(got.compact == o["expected"]!.compact, "object vector: \(got.compact) != \(o["expected"]!.compact)")
        }
    }
    print("redaction vectors: \(n) checked")
}

/// Replays the messages of recorded eval episodes through the Swift env + redactor and compares every
/// model-visible string; also recomputes router gates for the student's steps.
func replay(_ file: URL, scenarios: [String: JSON]) {
    let router = Router()
    var episodes = 0, strings = 0, gates = 0
    for row in lines(file) {
        let ep = row["episode"]!
        guard let sc = scenarios[row["scenario_id"]!.string!] else { continue }
        episodes += 1
        let env = HarborEnv(memberId: sc["member_id"]!.string!)
        let red = Redactor(known: env.knownPII())
        let msgs = ep["messages"]!.array!
        let first = Prompts.request(session: red.redact(env.sessionContext()), text: red.redactText(sc["message"]!.string!))
        strings += 1
        check(first == msgs[0]["content"]!.string!, "\(row["scenario_id"]!.string!) request:\n    swift:  \(first.prefix(300))\n    python: \(msgs[0]["content"]!.string!.prefix(300))")
        var context = [first]
        // Steps drive the replay so gates see the env exactly as the student did.
        var mi = 1
        for st in ep["steps"]!.array! {
            let actJSON = st["action"].flatMap { $0.isNull ? nil : $0 } ?? st["discarded_action"]
            // A never-leave handoff replaces the student's step; the gate was computed on the raw output.
            let action = st["served_by"]?.string == "local_handoff" ? st["raw"]?.string.flatMap(Action.parse) : actJSON.flatMap(Action.init(json:))
            if file.lastPathComponent.hasPrefix("hybrid"), let g = st["gate"], st["served_by"]?.string != "teacher", let pyReasons = g["reasons"]?.array?.compactMap(\.string) {
                gates += 1
                let real = action?.args.map { red.refill($0) }
                let swiftReasons = router.validate(action, realArgs: real, env: env, context: context) + router.policy(action, realArgs: real)
                let norm = { (r: [String]) in r.filter { $0 != "low_confidence" && $0 != "max_steps" }.map { $0.hasPrefix("schema:") ? "schema" : $0 } }
                check(norm(swiftReasons) == norm(pyReasons), "\(row["scenario_id"]!.string!) gate: swift \(swiftReasons) python \(pyReasons)")
            }
            guard st["tool_result"] != nil, let action = st["action"].flatMap(Action.init(json:)), case .tool(let tool, let args) = action else { continue }
            // assistant message
            while mi < msgs.count, msgs[mi]["role"]?.string != "assistant" { mi += 1 }
            if mi < msgs.count {
                strings += 1
                check(action.text == msgs[mi]["content"]!.string!, "\(row["scenario_id"]!.string!) action text: \(action.text) != \(msgs[mi]["content"]!.string!)")
            }
            let result = env.execute(tool, red.refill(args))
            let content = Prompts.toolResult(tool, red.redact(result))
            context.append(content)
            if mi + 1 < msgs.count {
                strings += 1
                check(content == msgs[mi + 1]["content"]!.string!, "\(row["scenario_id"]!.string!) \(tool) result:\n    swift:  \(content.prefix(400))\n    python: \(msgs[mi + 1]["content"]!.string!.prefix(400))")
            }
            mi += 2
        }
    }
    print("\(file.lastPathComponent): \(episodes) episodes, \(strings) model-visible strings, \(gates) router gates compared")
}

func messages(_ row: JSON) -> [Message] {
    row["messages"]!.array!.map { m in
        Message(role: m["role"]!.string! == "assistant" ? .assistant : .user, content: m["content"]!.string!)
    }
}


let dir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "parity")
vectors()
var scen: [String: JSON] = [:]
for s in lines(dir.appendingPathComponent("scenarios.jsonl")) { scen[s["id"]!.string!] = s }
for f in ["ft-e4b-r2-eval", "hybrid-e4b-r2-eval", "teacher-eval"] {
    replay(dir.appendingPathComponent("episodes/\(f).jsonl"), scenarios: scen)
}
var n = 0
for row in lines(dir.appendingPathComponent("prompts.jsonl")) {
    n += 1
    let rendered = Prompts.render(system: Prompts.studentSystem, messages: messages(row))
    if rendered != row["prompt"]!.string! {
        let a = Array(rendered), b = Array(row["prompt"]!.string!)
        let i = zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(a.count, b.count)
        check(false, "\(row["scenario_id"]!.string!) step \(row["step"]!.pyStr): prompt differs at \(i): swift …\(String(a[max(0, i - 40)..<min(a.count, i + 40)]))… python …\(String(b[max(0, i - 40)..<min(b.count, i + 40)]))…")
    }
}
print("prompts: \(n) rendered")
print(failures == 0 ? "PASS" : "FAIL (\(failures) mismatches)")
exit(failures == 0 ? 0 : 1)
