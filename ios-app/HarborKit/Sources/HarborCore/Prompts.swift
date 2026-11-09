import Foundation

/// harbor_env/prompts.py, plus the students' chat templates (Gemma 4, LFM3) rendered by hand (no Jinja engine on the
/// phone: a template mismatch would silently corrupt the fine-tuned student's input).
public enum Prompts {
    public static let core = Shared.text("prompts/system_core.txt").trimmingCharacters(in: .whitespacesAndNewlines)
    public static let actionFormat = Shared.text("prompts/action_format.txt").trimmingCharacters(in: .whitespacesAndNewlines)

    /// `tool_block("short")`: one line per tool, optional args marked with `?`.
    public static var shortToolBlock: String {
        let lines = Shared.tools.map { t -> String in
            let schema = t["input_schema"]!
            let req = Set(schema["required"]?.array?.compactMap(\.string) ?? [])
            let args = (schema["properties"]?.object?.keys ?? []).map { req.contains($0) ? $0 : $0 + "?" }.joined(separator: ", ")
            return "- \(t["name"]!.string!)(\(args)): \(t["short"]!.string!)"
        }
        return "Tools:\n" + lines.joined(separator: "\n")
    }

    /// `system_prompt("short")` for the fine-tuned student.
    public static let studentSystem = [core, actionFormat, shortToolBlock].joined(separator: "\n\n")

    public static func request(session: JObject, text: String) -> String { "[session] \(JSON.object(session).compact)\n\n\(text)" }

    public static func toolResult(_ name: String, _ result: JObject) -> String {
        "<tool_result name=\"\(name)\">\(JSON.object(result).compact)</tool_result>"
    }

    /// The chat templates the student models use, rendered by hand.
    public enum Template: String, Sendable {
        case gemma4, lfm3

        /// From a model folder's config.json `model_type`.
        public static func forModel(type: String) -> Template { type.hasPrefix("lfm") ? .lfm3 : .gemma4 }
    }

    /// The model's chat template with `add_generation_prompt=True, enable_thinking=False`.
    public static func render(system: String, messages: [Message], template: Template = .gemma4) -> String {
        switch template {
        case .gemma4:
            var s = "<bos><|turn>system\n\(system)<turn|>\n"
            for m in messages {
                s += "<|turn>\(m.role == .assistant ? "model" : "user")\n\(m.content)<turn|>\n"
            }
            return s + "<|turn>model\n"
        case .lfm3:
            // Liquid LFM3 (ChatML with an empty think block on every assistant turn).
            var s = "<|startoftext|><|im_start|>system\n\(system)<|im_end|>\n"
            for m in messages {
                s += m.role == .assistant ? "<|im_start|>assistant\n<think></think>\(m.content)<|im_end|>\n"
                                          : "<|im_start|>user\n\(m.content)<|im_end|>\n"
            }
            return s + "<|im_start|>assistant\n<think></think>"
        }
    }
}

/// A canonical (redacted) conversation message, as in harbor_env/models.py.
public struct Message {
    public enum Role: String { case user, assistant }
    public enum Kind: String { case request, toolResult = "tool_result" }
    public var role: Role
    public var kind: Kind?
    public var tool: String?
    public var content: String
    public var action: Action?
    public var servedBy: String?
    public var native: JSON?  // the teacher's raw content blocks, replayed unchanged to the cloud

    public init(role: Role, kind: Kind? = nil, tool: String? = nil, content: String, action: Action? = nil,
                servedBy: String? = nil, native: JSON? = nil) {
        self.role = role; self.kind = kind; self.tool = tool; self.content = content
        self.action = action; self.servedBy = servedBy; self.native = native
    }

    /// The shape the backend's /v1/teacher expects.
    public var wire: JSON {
        var o = JObject([("role", .string(role.rawValue))])
        if let kind { o["kind"] = .string(kind.rawValue) }
        if let tool { o["tool"] = .string(tool) }
        o["content"] = .string(content)
        if let action { o["action"] = action.json }
        if let native { o["native"] = native }
        return .object(o)
    }
}
