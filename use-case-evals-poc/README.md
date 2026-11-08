# use-case-evals-poc

The simulated customer: the ML team at **Bayside Mutual**, a fictional insurer whose member app has a support agent with 8 tools. This folder is their environment. It holds the tool sandbox, generated member requests with ground truth, the eval harness, and student training. The eval compares:

| Run | What runs | Prompt |
| --- | --- | --- |
| `teacher` | Claude Sonnet 5.5 (cloud baseline, today's setup) | full, native tool use |
| `base-e4b` | Gemma 4 E4B, MLX 4-bit, no fine-tuning | full tool schemas + 3 examples |
| `ft-e4b` | Gemma 4 E4B + LoRA distilled from Sonnet traces | short |
| `hybrid-e4b` | `ft-e4b` + router, escalating to Sonnet; the product | short |
| `ft-e2b` (optional) | Gemma 4 E2B + LoRA | short |

Every model sees **redacted** conversations (placeholders such as `[MEMBER_ID_1]`). Tool arguments are re-filled on the "device" before execution, and replies are re-filled before the member sees them. So the hybrid path can send the same conversation to the cloud, and stored traces are train-ready. This is what the iPhone app does too (`../ios-app/REQUIREMENTS.md`).

## Setup

```bash
uv sync                      # Python 3.12, mlx-lm, anthropic, openai
cp ~/GitHub/Loopr/.env .env  # ANTHROPIC_API / OPEN_AI_API (already copied, chmod 600)
uv run pytest -q
```

Start the collector first (`../backend`, port 8000); runs post their traces and results there and also save locally in `runs/output/<run>/` (mode 600).

## Pipeline

```bash
# 1. scenarios: templates x personas x fixtures, ground truth from executing the gold calls, paraphrased by Sonnet
uv run python scenarios/generate.py               # --no-llm for templates only

# 2. round 0: Sonnet serves the training requests (traces -> collector)
uv run python runs/run_eval.py --run teacher-datagen --split train --round 0 --name datagen-r0-train

# 3. dataset + LoRA (M5, ~mlx-lm QLoRA) -> models/ft-e4b-r1 (fused MLX 4-bit, what the phone loads)
uv run python train/build_sft.py --round 1 --run-ids datagen-r0-train --max-traces 250
train/train_mlx.sh 1 e4b

# 4. router thresholds on a calibration slice of train
uv run python runs/tune_router.py --model models/ft-e4b-r1 --from 400 --to 500 --write

# 5. the eval runs (same 152 scenarios: heldout 98, novel 29, never_leave 15, policy 10)
S=heldout,novel,never_leave,policy
uv run python runs/run_eval.py --run teacher    --split $S --name teacher-eval
uv run python runs/run_eval.py --run base-e4b   --split $S --name base-e4b-eval
uv run python runs/run_eval.py --run ft-e4b     --split $S --model models/ft-e4b-r1 --round 1 --name ft-e4b-r1-eval
uv run python runs/run_eval.py --run hybrid-e4b --split $S --model models/ft-e4b-r1 --round 1 --shadow-rate 0.15 --name hybrid-e4b-r1-eval

# 6. round 2: live traffic through the hybrid, escalations become training data
uv run python runs/run_eval.py --run hybrid-e4b --split traffic --model models/ft-e4b-r1 --round 1 --no-judge --name hybrid-r1-traffic
uv run python train/build_sft.py --round 2 --run-ids datagen-r0-train,hybrid-r1-traffic
train/train_mlx.sh 2 e4b
uv run python runs/run_eval.py --run hybrid-e4b --split $S --model models/ft-e4b-r2 --round 2 --name hybrid-e4b-r2-eval

# 7. round 3 and later: all three collection paths. Shadow-sample the traffic, have the teacher relabel the
#    out-of-distribution traces, then build with both. Use traffic runs only; eval runs are held out.
uv run python runs/run_eval.py --run hybrid-e4b --split traffic --model models/ft-e4b-r2 --round 2 --shadow-rate 0.15 --no-judge --name hybrid-r2-traffic
uv run python train/label_ood.py --run-ids hybrid-r2-traffic --dry-run      # score only, no cost
uv run python train/label_ood.py --run-ids hybrid-r2-traffic --limit 20     # ~0.5¢ per student step
uv run python train/build_sft.py --round 3 --run-ids datagen-r0-train,hybrid-r1-traffic,hybrid-r2-traffic --with-shadow --with-ood
```

Training data comes from three paths. **Escalations:** the teacher's steps after a router gate fails, already in the trace. **Shadow disagreements** (`--with-shadow`): a random sample of student steps that the teacher also answered, where the two differed. **OOD relabels** (`--with-ood`): synced traces far from every training request, whose student steps the teacher relabels. Each teacher call is PII-scanned (fail closed) and ledgered. Dataset stats break examples down `by_source`.

## Metrics

- **Task success** (primary): every required tool call made with the right arguments, the final state diff equals the expected one (no extra writes), and a reply. Ground truth comes from the scenario template, not from the teacher.
- **First call match**: the first action is the expected tool with normalized arguments. A provider lookup counts as the right first step for booking or filing.
- **Judge** (`judge/judge.py`): an OpenAI GPT judge (`EVAL_JUDGE_MODEL`, default `gpt-6.1-sol`), on a different provider from the teacher, grades the reply: correct, grounded, 1–5. It is calibrated on 8 labeled cases before every run, and a run aborts if more than one miss. The judge sees the fictional fixture data unredacted.
- **Product metrics** (hybrid): escalation rate, task success on locally served requests, policy compliance (claims over $1,000 go to the teacher), never-leave violations (target 0), raw PII fields in anything sent to the cloud (target 0), shadow agreement.
- **Latency/tokens**: p50/p95 per step. The Mac's MLX latency is a proxy; the phone's comes from the iOS eval mode.

## Training backups

| Where | How |
| --- | --- |
| M5 (primary) | `train/train_mlx.sh <round> [e4b|e2b]`: mlx-lm QLoRA, `train/lora.yaml` |
| M1 Max 64 GB | `train/bundle.sh <round>` → copy `gh-train-bundle-r<round>.tar.gz` → `./run.sh` (`--bf16` for full-precision LoRA) |
| Colab | `train/colab_unsloth.ipynb` + `colab_unsloth.py` → `mlx_lm.convert -q` on the Mac |
| AWS g6.xlarge | `train/aws/train_g6.sh <round>`, once the G-instance quota request (us-west-2, 8 vCPU) is approved; self-terminating |

## Layout

```
harbor_env/   env.py (tools over fixtures, state diff) · agent.py (loop, hybrid routing, redaction) · router.py
              models.py (MLX + Sonnet backends) · prompts.py · scoring.py
scenarios/    generate.py · splits/*.jsonl
judge/        judge.py (GPT judge + calibration)
runs/         run_eval.py · tune_router.py · output/ (ignored)
train/        build_sft.py · train_mlx.sh · lora.yaml · register.py · bundle.sh · colab_unsloth.* · aws/
```

## Eval report (HTML, for the demo)

```bash
uv run python runs/build_report.py   # -> reports/eval-report.html, also served at http://localhost:8000/report
```

A self-contained page that works offline: headline numbers, every run side by side, results by split, six demo picks, and a browser for all 152 requests. You can compare any two configurations step by step, including what the model saw (redacted), router reasons, tool results, the judge's verdict and what left the device.

## Device runtime (the "phone" for the iOS Simulator)

```bash
uv run uvicorn device_runtime.server:app --host 0.0.0.0 --port 8100
```

The round-2 student, router, redactor and tools served over HTTP for the Simulator app (MLX doesn't run in the Simulator). Escalations go to the backend teacher proxy. See `ios-app/README.md`.
