import Foundation
import os
import HarborCore
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// The fine-tuned Gemma 4 student running in-process with MLX: the same fused 4-bit weights the evals used.
///
/// Mirrors harbor_env/models.py `MLXBackend.step`: the Gemma chat template rendered by hand, greedy
/// decoding, logprobs of non-whitespace tokens, stop at an end-of-turn token or once the JSON action closes.
public final class MLXStudent: StudentModel, @unchecked Sendable {
    public let name: String
    public let directory: URL
    private let context: ModelContext
    private let stopIds: Set<Int>
    private let maxTokens: Int
    private let lock = NSLock()
    /// Diagnostics sink (memory at each phase); the app writes it to Documents/bayside-diag.log.
    public var log: (@Sendable (String) -> Void)?
    /// Prompt tokens per prefill forward. Gemma 4's 262k vocabulary makes each forward's logits
    /// ~1 MB per position, so small chunks keep the prefill spike well below the phone's limit.
    public var prefillStepSize = 64

    private init(name: String, directory: URL, context: ModelContext, stopIds: Set<Int>, maxTokens: Int) {
        self.name = name; self.directory = directory; self.context = context; self.stopIds = stopIds; self.maxTokens = maxTokens
    }

    /// Load a fused MLX model folder (config.json, *.safetensors, tokenizer.json).
    public static func load(directory: URL, name: String? = nil, maxTokens: Int = 320,
                            progress: (@Sendable (Double) -> Void)? = nil) async throws -> MLXStudent {
        // The model is ~5.9 GB in memory and iOS gives the app ~6.4 GB: keep MLX's buffer cache tiny.
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)  // the phone has ~0.4 GB headroom after loading
        progress?(0.05)
        MLXStudent.staticLog?("load start · \(MLXStudent.memoryLine())")
        let context = try await LLMModelFactory.shared.load(from: directory, using: #huggingFaceTokenizerLoader())
        progress?(1.0)
        MLXStudent.staticLog?("load done · \(MLXStudent.memoryLine())")
        var stops: Set<Int> = [1, 106, 50]  // generation_config.json eos_token_id (<eos>, <turn|>, …)
        if let data = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
           let cfg = try? JSON.parse(data) {
            if let arr = cfg["eos_token_id"]?.array { stops = Set(arr.compactMap(\.int)) }
            else if let one = cfg["eos_token_id"]?.int { stops = [one] }
        }
        return MLXStudent(name: name ?? directory.lastPathComponent, directory: directory, context: context, stopIds: stops, maxTokens: maxTokens)
    }

    public func promptTokens(_ messages: [HarborCore.Message]) -> [Int] {
        context.tokenizer.encode(text: Prompts.render(system: Prompts.studentSystem, messages: messages), addSpecialTokens: false)
    }

    public func step(_ messages: [HarborCore.Message]) async throws -> StudentOutput {
        lock.lock()
        defer { lock.unlock() }
        let tokens = promptTokens(messages)
        let t0 = Date()
        let recorder = LogprobRecorder()
        log?("step start · \(tokens.count) prompt tokens · \(Self.memoryLine())")
        var iterator = try TokenIterator(input: LMInput(tokens: MLXArray(tokens.map { Int32($0) })), model: context.model,
                                         processor: recorder, sampler: ArgMaxSampler(),
                                         prefill: PrefillParameters(stepSize: prefillStepSize), maxTokens: maxTokens)
        log?(String(format: "prefill done %.2f s · ", Date().timeIntervalSince(t0)) + Self.memoryLine())
        var out: [Int] = []
        var lps: [Double] = []
        var text = ""
        while let t = iterator.next() {
            // iOS kills apps that submit GPU work from the background: wait until we're visible again.
            while MLXStudent.isBackground { try await Task.sleep(for: .milliseconds(200)) }
            if stopIds.contains(t) { break }
            out.append(t)
            if out.count == 1 { log?(String(format: "first token %.2f s · ", Date().timeIntervalSince(t0)) + Self.memoryLine()) }
            let piece = context.tokenizer.decode(tokenIds: [t], skipSpecialTokens: false)
            if !piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, out.count - 1 < recorder.values.count {
                lps.append(Double(recorder.values[out.count - 1].item(Float.self)))
            }
            text = context.tokenizer.decode(tokenIds: out, skipSpecialTokens: false)
            // One action per step: stop once a complete JSON object has been produced.
            let opens = text.filter { $0 == "{" }.count
            if opens > 0, opens == text.filter({ $0 == "}" }).count,
               text.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("}") { break }
        }
        MLX.GPU.clearCache()
        log?(String(format: "step done %.2f s · %d tokens · ", Date().timeIntervalSince(t0), out.count) + Self.memoryLine())
        return StudentOutput(raw: text, action: Action.parse(text), latency: Date().timeIntervalSince(t0),
                             promptTokens: tokens.count, outputTokens: out.count, logprobs: lps)
    }
}

extension MLXStudent {
    /// Set by the app from its scene phase; decoding pauses while true.
    nonisolated(unsafe) public static var isBackground = false

    /// Set before `load` to capture load-time diagnostics.
    nonisolated(unsafe) public static var staticLog: (@Sendable (String) -> Void)?

    /// "free 6.1 GB · mlx active 5.4 GB peak 5.9 GB".
    public static func memoryLine() -> String {
        var free = ""
        #if os(iOS)
        free = String(format: "free %.2f GB · ", Double(os_proc_available_memory()) / 1e9)
        #endif
        return free + String(format: "mlx active %.2f GB peak %.2f GB cache %.2f GB",
                             Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9, Double(Memory.cacheMemory) / 1e9)
    }
}

/// Records the chosen token's log probability at each step. With greedy (argmax) sampling that's the
/// maximum of the log-softmax, matching Python's `logprobs[token]`.
final class LogprobRecorder: LogitProcessor {
    var values: [MLXArray] = []
    func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        let l = logits.asType(.float32)
        values.append(l.max(axis: -1) - logSumExp(l, axis: -1))  // log-softmax of the argmax token
        return logits
    }
    func didSample(token: MLXArray) {}
    func copy() -> Self { self }
}
