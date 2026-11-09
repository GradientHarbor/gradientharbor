// Liquid AI LFM3 for mlx-swift-lm: a port of mlx_vlm/models/lfm3/language.py (mlx-vlm 0.7.12), which is
// what the Mac evals ran. mlx-swift-lm 3.32 has LFM2 but not LFM3; HarborMLX registers "lfm3" itself.
//
// 24 layers: gated short convolutions (as in LFM2) interleaved with full and sliding-window (256)
// attention. Attention adds per-head sigmoid output gates and a partial rotary embedding that rotates
// the first `angles` coordinate pairs (i, i + D/2) and leaves the rest unrotated. Sandwich norms around
// every operator and feed-forward. Weights keep mlx-vlm's `language_model.model.…` paths so the
// per-layer quantization table in config.json applies unchanged.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct LFM3Configuration: Codable, Sendable {
    public struct Rope: Codable, Sendable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?
        var ropeType: String?
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta", partialRotaryFactor = "partial_rotary_factor", ropeType = "rope_type"
        }
    }

    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int?
    var vocabularySize: Int
    var normEps: Float
    var convLCache: Int
    var convBias: Bool
    var layerTypes: [String]
    var ropeParameters: [String: Rope]?
    var slidingWindow: Int?
    var useSandwichNorm: Bool?
    var tieWordEmbeddings: Bool?

    var resolvedHeadDim: Int { headDim ?? hiddenSize / attentionHeads }
    var sandwich: Bool { useSandwichNorm ?? true }
    var window: Int { slidingWindow ?? 256 }
    func rope(_ layerType: String) -> Rope {
        ropeParameters?[layerType]
            ?? (layerType == "full_attention" ? Rope(ropeTheta: 1e6, partialRotaryFactor: 0.5) : Rope(ropeTheta: 1e4))
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case vocabularySize = "vocab_size", normEps = "norm_eps", convLCache = "conv_L_cache", convBias = "conv_bias"
        case layerTypes = "layer_types", ropeParameters = "rope_parameters", slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm", tieWordEmbeddings = "tie_word_embeddings"
    }
}

/// mlx-vlm's `RotaryEmbedding`: rotate-half over the full head, with zero frequency past `angles`.
final class LFM3Rope {
    let invFreq: MLXArray  // [D/2]

    init(dim: Int, _ p: LFM3Configuration.Rope) {
        let base = p.ropeTheta ?? 1e6
        let angles = Int(Float(dim) * (p.partialRotaryFactor ?? 1)) / 2
        var f = (0 ..< angles).map { pow(base, -Float(2 * $0) / Float(dim)) }
        f += Array(repeating: 0, count: dim / 2 - angles)
        invFreq = MLXArray(f)
    }

    /// x: [B, H, L, D]
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let positions = MLXArray(Int32(offset) ..< Int32(offset + x.dim(2))).asType(.float32)
        let a = positions[0..., .newAxis] * invFreq[.newAxis, 0...]
        let angles = concatenated([a, a], axis: -1)
        let cos = MLX.cos(angles).asType(x.dtype), sin = MLX.sin(angles).asType(x.dtype)
        let halves = x.split(parts: 2, axis: -1)
        return x * cos + concatenated([-halves[1], halves[0]], axis: -1) * sin
    }
}

final class LFM3Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    let rope: LFM3Rope

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(_ c: LFM3Configuration, layerType: String) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.resolvedHeadDim
        scale = pow(Float(headDim), -0.5)
        rope = LFM3Rope(dim: headDim, c.rope(layerType))
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        _gateProj.wrappedValue = Linear(c.hiddenSize, heads, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        let k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        var out = attentionWithCacheUpdate(queries: rope(q, offset: offset), keys: rope(k, offset: offset), values: v,
                                           cache: cache, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3)  // [B, L, H, D]
        out = out * sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj(out.reshaped(B, L, -1))
    }
}

/// LFM2's gated short convolution (mlx-swift-lm `LFM2ShortConv`, which is internal to MLXLLM).
final class LFM3ShortConv: Module {
    let lCache: Int, hidden: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: LFM3Configuration) {
        lCache = c.convLCache; hidden = c.hiddenSize
        _conv.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden, kernelSize: lCache,
                                    groups: hidden, bias: c.convBias)
        _inProj.wrappedValue = Linear(hidden, 3 * hidden, bias: c.convBias)
        _outProj.wrappedValue = Linear(hidden, hidden, bias: c.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let bcx = inProj(x).split(parts: 3, axis: -1)
        var bx = bcx[0] * bcx[2]
        let state = cache?[0] ?? MLXArray.zeros([bx.dim(0), lCache - 1, hidden], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            cache[0] = contiguous(bx[0..., (bx.dim(1) - (lCache - 1))..., 0...])
            cache.advance(x.dim(1))
        }
        return outProj(bcx[1] * conv(bx))
    }
}

final class LFM3MLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ c: LFM3Configuration) {
        _w1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _w2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class LFM3DecoderLayer: Module {
    let layerType: String
    var isAttention: Bool { layerType != "conv" }
    let sandwich: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ c: LFM3Configuration, index: Int) {
        layerType = c.layerTypes[index]
        sandwich = c.sandwich
        if layerType != "conv" {
            _attention.wrappedValue = LFM3Attention(c, layerType: layerType)
        } else {
            _conv.wrappedValue = LFM3ShortConv(c)
        }
        _feedForward.wrappedValue = LFM3MLP(c)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        if sandwich {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        var h = operatorNorm(x)
        h = isAttention ? attention!(h, mask: mask, cache: cache) : conv!(h, cache: cache as? MambaCache)
        let x = x + (postOperatorNorm.map { $0(h) } ?? h)
        let f = feedForward(ffnNorm(x))
        return x + (postFFNNorm.map { $0(f) } ?? f)
    }
}

public final class LFM3Inner: Module {
    let config: LFM3Configuration
    let layers: [LFM3DecoderLayer]
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ c: LFM3Configuration) {
        config = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0 ..< c.hiddenLayers).map { LFM3DecoderLayer(c, index: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let c = cache?[i]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layer.layerType {
            case "full_attention": mask = createAttentionMask(h: h, cache: c, windowSize: nil)
            case "sliding_attention": mask = createAttentionMask(h: h, cache: c, windowSize: config.window)
            default: mask = .none
            }
            h = layer(h, mask: mask, cache: c)
        }
        return embeddingNorm(h)
    }
}

public final class LFM3LanguageModel: Module {
    @ModuleInfo(key: "model") var model: LFM3Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ c: LFM3Configuration) {
        _model.wrappedValue = LFM3Inner(c)
        if c.tieWordEmbeddings == false {
            _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }
}

public final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let config: LFM3Configuration
    @ModuleInfo(key: "language_model") var languageModel: LFM3LanguageModel

    public init(_ c: LFM3Configuration) {
        config = c
        vocabularySize = c.vocabularySize
        kvHeads = c.layerTypes.map { $0 == "conv" ? 0 : c.kvHeads }
        _languageModel.wrappedValue = LFM3LanguageModel(c)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = languageModel.model(inputs, cache: cache)
        if let head = languageModel.lmHead { return head(h) }
        return languageModel.model.embedTokens.asLinear(h)
    }

    /// mlx-vlm `Model.sanitize`: prefix bare HF names, drop a tied lm_head, put conv weights in MLX layout.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("model.") || name.hasPrefix("lm_head.") { name = "language_model." + name }
            if config.tieWordEmbeddings != false, name == "language_model.lm_head.weight" { continue }
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1, value.dim(2) > 1 {
                out[name] = value.transposed(0, 2, 1)
            } else {
                out[name] = value
            }
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try config.layerTypes.map { t -> KVCache in
            switch t {
            case "conv": return MambaCache()
            case "sliding_attention": return RotatingKVCache(maxSize: config.window)
            default: return try makeAttentionKVCache(parameters: parameters)
            }
        }
    }

    public var loraLayers: [Module] { languageModel.model.layers }
}

public enum LFM3 {
    /// Adds "lfm3" to mlx-swift-lm's model-type registry (idempotent).
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType("lfm3") { data in
            LFM3Model(try JSONDecoder().decode(LFM3Configuration.self, from: data))
        }
    }
}
