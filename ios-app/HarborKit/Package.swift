// swift-tools-version: 6.0
// HarborKit: the on-device GradientHarbor stack for the Bayside app.
// HarborCore mirrors use-case-evals-poc/harbor_env (tools, redactor, router, prompts, agent loop) and must
// produce byte-identical model inputs; HarborMLX runs the fine-tuned student with mlx-swift-lm.
// Build the parity tool with xcodebuild (MLX needs its Metal shaders):
//   xcodebuild -scheme harbor-parity -destination 'platform=macOS' -derivedDataPath .build/xc build
import PackageDescription

let package = Package(
    name: "HarborKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "HarborCore", targets: ["HarborCore"]),
        .library(name: "HarborMLX", targets: ["HarborMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(name: "HarborCore", resources: [.copy("Resources/shared")]),
        .target(
            name: "HarborMLX",
            dependencies: [
                "HarborCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]),
        .executableTarget(name: "harbor-parity", dependencies: ["HarborCore", "HarborMLX"]),
        .executableTarget(name: "harbor-check", dependencies: ["HarborCore"]),
    ],
    swiftLanguageModes: [.v5]
)
