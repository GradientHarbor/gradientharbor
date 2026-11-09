import Foundation
import HarborCore
import HarborMLX
import Tokenizers

// Parity between HarborKit (Swift, on the phone) and harbor_env (Python, the evals).
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
            if let g = st["gate"], st["served_by"]?.string != "teacher", let pyReasons = g["reasons"]?.array?.compactMap(\.string) {
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

/// PARITY_PROMPTS=prompts-lfm3.jsonl picks another export (default prompts.jsonl, the Gemma run).
let promptsFile = ProcessInfo.processInfo.environment["PARITY_PROMPTS"] ?? "prompts.jsonl"

func messages(_ row: JSON) -> [HarborCore.Message] {
    row["messages"]!.array!.map { m in
        HarborCore.Message(role: m["role"]!.string! == "assistant" ? .assistant : .user, content: m["content"]!.string!)
    }
}

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: harbor-parity core|prompts|generate <parity dir> [model dir] [n]"); exit(2) }
let dir = URL(fileURLWithPath: args[2])

switch args[1] {
case "core":
    vectors()
    var scen: [String: JSON] = [:]
    for s in lines(dir.appendingPathComponent("scenarios.jsonl")) { scen[s["id"]!.string!] = s }
    for f in ["ft-e4b-r2-eval", "hybrid-e4b-r2-eval", "teacher-eval"] {
        replay(dir.appendingPathComponent("episodes/\(f).jsonl"), scenarios: scen)
    }
    // system prompt
    print("student system prompt: \(Prompts.studentSystem.count) chars")

case "prompts":
    let modelDir = URL(fileURLWithPath: args[3])
    let tok = try await AutoTokenizer.from(modelFolder: modelDir)
    let template = Prompts.Template.forModel(type: MLXStudent.modelType(modelDir))
    print("template: \(template.rawValue) · \(promptsFile)")
    var n = 0
    for row in lines(dir.appendingPathComponent(promptsFile)) {
        n += 1
        let rendered = Prompts.render(system: Prompts.studentSystem, messages: messages(row), template: template)
        check(rendered == row["prompt"]!.string!, "\(row["scenario_id"]!.string!) step \(row["step"]!.pyStr): rendered prompt differs")
        let ids = tok.encode(text: rendered, addSpecialTokens: false)
        let py = row["ids"]!.array!.compactMap(\.int)
        if ids != py {
            let firstDiff = zip(ids, py).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(ids.count, py.count)
            check(false, "\(row["scenario_id"]!.string!) token ids differ at \(firstDiff): swift \(ids.count) python \(py.count)")
        }
    }
    print("prompts: \(n) rendered + tokenized")

case "generate":
    let modelDir = URL(fileURLWithPath: args[3])
    let limit = args.count > 4 ? Int(args[4]) ?? 10 : 10
    let t0 = Date()
    let student = try await MLXStudent.load(directory: modelDir)
    print(String(format: "model loaded in %.1f s", Date().timeIntervalSince(t0)))
    var n = 0, same = 0, sameAction = 0
    var lat: [Double] = []
    for row in lines(dir.appendingPathComponent(promptsFile)).prefix(limit) {
        n += 1
        let out = try await student.step(messages(row))
        lat.append(out.latency)
        let py = row["raw"]!.string!
        if out.raw == py { same += 1 }
        if Action.parse(out.raw) == Action.parse(py) { sameAction += 1 } else {
            check(false, "\(row["scenario_id"]!.string!) step \(row["step"]!.pyStr):\n    swift:  \(out.raw.prefix(240))\n    python: \(py.prefix(240))")
        }
    }
    lat.sort()
    print("generate: \(n) steps, identical raw \(same), same action \(sameAction); p50 \(String(format: "%.2f", lat.isEmpty ? 0 : lat[lat.count / 2])) s/step")

default:
    print("unknown command \(args[1])")
    exit(2)
}

print(failures == 0 ? "PASS" : "FAIL (\(failures) mismatches)")
exit(failures == 0 ? 0 : 1)
