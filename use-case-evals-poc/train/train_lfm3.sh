#!/usr/bin/env bash
# Fine-tune Liquid AI LFM3-1.7B on a round's SFT data and build the on-device model.
#   train/train_lfm3.sh <round>            # LoRA (default, what we ship) -> models/ft-lfm3-r<round> (6-bit merged)
#   MODE=full train/train_lfm3.sh <round>  # full fine-tune of the bf16 model -> models/ft-lfm3-r<round>-full-4bit
#
# mlx-lm doesn't support the lfm3 architecture yet, so this uses mlx-vlm (>= 0.7.11, Liquid's recommended
# path) as a uv overlay; the project's own environment is unchanged.
#
# Settings used for the Nov 8 runs (round-2 data, 1,010 train / 100 valid examples), on an M5 with 24 GB:
#   LoRA: rank 16, alpha 32, lr 1e-4, batch 4 x grad-accum 2, 316 iterations (~2.5 epochs), train on completions,
#         4-bit base. ~40 min. Merged at 6 bits: merging into the 4-bit weights rounds away much of the
#         adapter (15/15 -> 10/15 on a check); 6-bit keeps it (15/15) at ~1.0 s/step and 1.36 GB.
#   Full: bf16 base, lr 1e-5, batch 2 x grad-accum 4, 316 iterations, ~14-17 GB RAM, ~55 min, then 4-bit.
#         It overfit (86.2% vs 91.4% with the router), so LoRA is the default.
set -euo pipefail
cd "$(dirname "$0")/.."
ROUND=${1:?round}
MODE=${MODE:-lora}
DATA=train/data/r$ROUND
OV=(uv run --with "mlx-vlm>=0.7.11")
[ -f "$DATA/train.jsonl" ] || { echo "missing $DATA/train.jsonl (run train/build_sft.py first)"; exit 1; }
N=$(wc -l < "$DATA/train.jsonl" | tr -d ' ')
ITERS=${ITERS:-$(( (N * 5 / 2 + 7) / 8 ))}   # ~2.5 epochs at an effective batch of 8
mkdir -p models

if [ "$MODE" = "full" ]; then
  BF16=$("${OV[@]}" python -c "from huggingface_hub import snapshot_download as s; print(s('LiquidAI/LFM3-1.7B-MLX-bf16'))" 2>/dev/null | tail -1)
  OUT=models/ft-lfm3-r$ROUND-full
  echo "full fine-tune of $BF16 on $N examples, $ITERS iters -> $OUT"
  "${OV[@]}" python -m mlx_vlm.train --model "$BF16" --data "$DATA" --train-type full --learning-rate 1e-5 \
    --batch-size 2 --gradient-accumulation-steps 4 --iters "$ITERS" --max-seq-length 2048 --train-on-completions \
    --grad-checkpoint --steps-per-report 20 --steps-per-eval 80 --val-batches 8 --output-path "$OUT" \
    2>&1 | tee "$OUT.train.log"
  # The trainer saves all weights as adapters.safetensors: assemble a model folder, then quantize to 4-bit.
  rm -rf "$OUT-bf16" && mkdir -p "$OUT-bf16"
  for f in "$BF16"/*; do [ "$(basename "$f")" = model.safetensors ] || cp -L "$f" "$OUT-bf16/"; done
  cp "$OUT/adapters.safetensors" "$OUT-bf16/model.safetensors"
  "${OV[@]}" python -m mlx_vlm.convert --hf-path "$OUT-bf16" --mlx-path "$OUT-4bit" -q --q-bits 4 --q-group-size 64 --q-mode affine
  echo "model: $OUT-4bit"
  exit 0
fi

ADAPTER=models/ft-lfm3-r$ROUND-adapter
echo "LoRA on LiquidAI/LFM3-1.7B-MLX-4bit with $N examples, $ITERS iters -> $ADAPTER"
"${OV[@]}" python -m mlx_vlm.train --model LiquidAI/LFM3-1.7B-MLX-4bit --data "$DATA" --train-type lora \
  --lora-rank 16 --lora-alpha 32 --learning-rate 1e-4 --batch-size 4 --gradient-accumulation-steps 2 \
  --iters "$ITERS" --max-seq-length 2048 --train-on-completions --grad-checkpoint --steps-per-report 20 \
  --steps-per-eval 80 --val-batches 8 --output-path "$ADAPTER" 2>&1 | tee "models/ft-lfm3-r$ROUND.train.log"
"${OV[@]}" python train/fuse_vlm.py --base LiquidAI/LFM3-1.7B-MLX-4bit --adapter "$ADAPTER" \
  --out "models/ft-lfm3-r$ROUND" --bits 6
echo "model: models/ft-lfm3-r$ROUND (LoRA merged at 6-bit)"
echo "eval:  uv run --with 'mlx-vlm>=0.7.11' python runs/run_eval.py --run hybrid-lfm3 --split heldout,novel,never_leave,policy --model models/ft-lfm3-r$ROUND"
