#!/usr/bin/env bash
# LoRA-train the student with mlx-lm, fuse it, and register it with the collector.
#   train/train_mlx.sh <round> [e4b|e2b] [extra mlx_lm.lora args...]
# Output: models/ft-<size>-r<round>/ (fused MLX 4-bit, the same artifact the iPhone loads)
set -euo pipefail
cd "$(dirname "$0")/.."
ROUND=${1:?round}; SIZE=${2:-e4b}; shift $(( $# >= 2 ? 2 : 1 )) || true
case "$SIZE" in
  e4b) BASE=${GH_BASE_MODEL:-lmstudio-community/gemma-4-E4B-it-MLX-4bit} ;;
  e2b) BASE=${GH_BASE_MODEL:-lmstudio-community/gemma-4-E2B-it-MLX-4bit} ;;
  *) echo "size must be e4b or e2b"; exit 1 ;;
esac
DATA=train/data/r$ROUND
ADAPTER=models/ft-$SIZE-r$ROUND-adapter
FUSED=models/ft-$SIZE-r$ROUND
LOG=models/ft-$SIZE-r$ROUND.train.log
COLLECTOR=${GH_COLLECTOR_URL:-http://localhost:8000}
mkdir -p models
[ -f "$DATA/train.jsonl" ] || { echo "missing $DATA/train.jsonl (run train/build_sft.py first)"; exit 1; }
N=$(wc -l < $DATA/train.jsonl | tr -d ' ')
# ~2.5 epochs at an effective batch of 8 (batch 4 x grad accumulation 2), unless ITERS is set.
ITERS=${ITERS:-$(( (N * 5 / 2 + 7) / 8 ))}
echo "train $SIZE round $ROUND on $N examples, $ITERS iters -> $FUSED"
curl -s -X POST "$COLLECTOR/v1/rounds" -H 'content-type: application/json' \
  -d "{\"round\": $ROUND, \"status\": \"training\"}" >/dev/null || true
if [ "${DRY_RUN:-0}" = "1" ]; then echo "dry run: skipping training"; exit 0; fi
uv run mlx_lm.lora -c train/lora.yaml --model "$BASE" --train --data "$DATA" --adapter-path "$ADAPTER" --iters "$ITERS" "$@" 2>&1 | tee "$LOG"
uv run mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTER" --save-path "$FUSED"
uv run python train/register.py --round "$ROUND" --size "$SIZE" --base "$BASE" --fused "$FUSED" --adapter "$ADAPTER" --log "$LOG"
