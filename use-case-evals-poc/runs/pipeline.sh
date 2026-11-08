#!/usr/bin/env bash
# End-to-end flywheel after round-0 data and the baseline evals exist.
#   runs/pipeline.sh [size]   (size: e4b default)
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=${1:-e4b}
S=heldout,novel,never_leave,policy
step() { echo "=== $(date +%H:%M:%S) $*"; }
step "train round 1"
train/train_mlx.sh 1 "$SIZE"
step "tune router on train[400:500]"
uv run python runs/tune_router.py --model models/ft-$SIZE-r1 --from 400 --to 500 --write
step "eval ft-$SIZE r1"
uv run python runs/run_eval.py --run ft-$SIZE --split $S --model models/ft-$SIZE-r1 --round 1 --name ft-$SIZE-r1-eval
step "eval hybrid-$SIZE r1"
uv run python runs/run_eval.py --run hybrid-$SIZE --split $S --model models/ft-$SIZE-r1 --round 1 --shadow-rate 0.15 --name hybrid-$SIZE-r1-eval
step "live traffic through hybrid r1"
uv run python runs/run_eval.py --run hybrid-$SIZE --split traffic --model models/ft-$SIZE-r1 --round 1 --no-judge --name hybrid-$SIZE-r1-traffic
step "dataset round 2"
uv run python train/build_sft.py --round 2 --run-ids datagen-r0-train,hybrid-$SIZE-r1-traffic
step "train round 2"
train/train_mlx.sh 2 "$SIZE"
step "eval hybrid-$SIZE r2"
uv run python runs/run_eval.py --run hybrid-$SIZE --split $S --model models/ft-$SIZE-r2 --round 2 --shadow-rate 0.15 --name hybrid-$SIZE-r2-eval
step "eval ft-$SIZE r2"
uv run python runs/run_eval.py --run ft-$SIZE --split $S --model models/ft-$SIZE-r2 --round 2 --name ft-$SIZE-r2-eval
step "PIPELINE DONE"
