#!/usr/bin/env bash
# Package a self-contained training bundle for another Mac (e.g. the 64 GB M1 Max).
#   train/bundle.sh <round>   ->  train/gh-train-bundle-r<round>.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
ROUND=${1:?round}
B=bundle/gh-train-bundle-r$ROUND
rm -rf "$B" && mkdir -p "$B/data"
cp -R data/r$ROUND/. "$B/data/"
cp lora.yaml "$B/"
cat > "$B/run.sh" <<'RUN'
#!/usr/bin/env bash
# GradientHarbor student training on any Apple-silicon Mac.
#   ./run.sh                 QLoRA on the MLX 4-bit base (default, ~same result as the M5)
#   ./run.sh --bf16          LoRA on bf16 weights, then quantize to 4-bit (needs ~40 GB RAM)
#   ./run.sh --e2b           train the E2B student instead
#   ./run.sh --dry-run       check the setup and data, train nothing
set -euo pipefail
cd "$(dirname "$0")"
SIZE=E4B; BF16=0; DRY=0
for a in "$@"; do case $a in --bf16) BF16=1;; --e2b) SIZE=E2B;; --dry-run) DRY=1;; esac; done
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
UVX="uv run --python 3.12 --with mlx-lm==0.32.2"
if [ $BF16 = 1 ]; then BASE=google/gemma-4-$SIZE-it; else BASE=lmstudio-community/gemma-4-$SIZE-it-MLX-4bit; fi
N=$(wc -l < data/train.jsonl); V=$(wc -l < data/valid.jsonl)
echo "base=$BASE train=$N valid=$V"
python3 - <<'PY'
import json
for f in ("data/train.jsonl", "data/valid.jsonl"):
    for i, l in enumerate(open(f)):
        m = json.loads(l)["messages"]
        assert m[0]["role"] == "system" and m[-1]["role"] == "assistant", (f, i)
print("data ok")
PY
[ $DRY = 1 ] && { $UVX python -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)"; echo "dry run ok"; exit 0; }
$UVX mlx_lm.lora -c lora.yaml --model "$BASE" --train --data data --adapter-path out/adapter 2>&1 | tee out-train.log
if [ $BF16 = 1 ]; then
  $UVX mlx_lm.fuse --model "$BASE" --adapter-path out/adapter --save-path out/fused-bf16
  $UVX mlx_lm.convert --hf-path out/fused-bf16 --mlx-path out/fused -q --q-bits 4
else
  $UVX mlx_lm.fuse --model "$BASE" --adapter-path out/adapter --save-path out/fused
fi
echo "done: copy out/fused back to use-case-evals-poc/models/ft-$(echo $SIZE | tr A-Z a-z)-r<round>/"
RUN
chmod +x "$B/run.sh"
cat > "$B/README.md" <<EOF2
# GradientHarbor training bundle, round $ROUND

\`\`\`bash
tar xzf gh-train-bundle-r$ROUND.tar.gz && cd gh-train-bundle-r$ROUND
./run.sh --dry-run   # check
./run.sh             # train + fuse -> out/fused
\`\`\`

Copy \`out/fused\` back to the main laptop as \`use-case-evals-poc/models/ft-e4b-r$ROUND/\`, then register it:
\`uv run python train/register.py --round $ROUND --size e4b --base <base> --fused models/ft-e4b-r$ROUND --adapter <adapter dir> --log <out-train.log>\`
EOF2
tar czf gh-train-bundle-r$ROUND.tar.gz -C bundle gh-train-bundle-r$ROUND
echo "-> train/gh-train-bundle-r$ROUND.tar.gz ($(du -h gh-train-bundle-r$ROUND.tar.gz | cut -f1))"
