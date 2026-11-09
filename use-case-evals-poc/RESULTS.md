# Results — Nov 8, 2026 (overnight run)

Same 152 scenarios for every configuration: 98 held-out (members never seen in training), 29 novel (OOD: appeals, multi-step, cancel, compare), 15 never-leave (medical), 10 policy (claims over $1,000). Primary metric: **task success** against template ground truth: required tool calls with the right arguments, the exact final state, and a reply. Judge: GPT (`gpt-6.1-sol`), calibrated 8/8 before each run. All hybrid runs use the same final router (validators + reply grounding + policy + never-leave).

| Configuration | Task success | Held-out | Novel | Judge: reply correct | Escalated | Success when kept on device | p50 / step | Prompt tokens / step | Cloud cost |
|---|---|---|---|---|---|---|---|---|---|
| Sonnet 5.5 (cloud baseline) | **97.4%** | 100% | 100% | 92.8% | 100% | – | 1.31 s (cloud) | 3,416 | $1.91 |
| Gemma 4 E4B, no fine-tuning | 64.5% | 73.5% | 51.7% | 71.7% | 0% | 64.5% | 2.96 s | 2,293 | $0 |
| E4B fine-tuned, round 1 | 77.0% | 84.7% | 48.3% | 63.8% | 0% | 77.0% | 2.35 s | 965 | $0 |
| E4B fine-tuned, round 2 | 86.8% | 94.9% | 75.9% | 76.3% | 0% | 86.8% | 2.35 s | 975 | $0 |
| E4B fine-tuned + router, round 1 | 86.8% | 96.9% | 58.6% | 75.0% | 20.4% | 86.8% | 2.38 s | 1,338 | $0.73 |
| **E4B fine-tuned + router, round 2** | **94.1%** | **98.0%** | **89.7%** | **81.6%** | **16.4%** | **95.3%** | 2.35 s | 1,240 | **$0.71** |

Every run: **0 raw PII fields sent** to the cloud (re-scanned in the backend ledger: 1,768 calls, 0 findings).

Latency is measured on the M5 Mac with MLX as a proxy. The iPhone Air numbers come from the app's eval mode (`ios-app/REQUIREMENTS.md` §7).

## Liquid AI LFM3-1.7B (added Nov 8, afternoon)

Same 152 scenarios, same round-2 training data (1,010 examples) and the same scoring. LFM3-1.7B is Liquid AI's text model from the LFM3 release: 0.97 GB at 4-bit (official `LiquidAI/LFM3-1.7B-MLX-4bit`), against 5.5 GB for Gemma 4 E4B. mlx-lm doesn't support the `lfm3` architecture yet, so these runs use mlx-vlm 0.7.12, Liquid's recommended path (`VLMBackend` in `harbor_env/models.py`). The prompts, chat-template settings (`enable_thinking=False`), greedy decoding and stop rule are the same as for Gemma.

| Configuration | Task success | Held-out | Novel | Judge: reply correct | Escalated | Success when kept on device | p50 / step | Prompt tokens / step | Cloud cost |
|---|---|---|---|---|---|---|---|---|---|
| LFM3-1.7B, no fine-tuning | 37.5% | 50.0% | 24.1% | 30.9% | 0% | 37.5% | 3.50 s* | 2,179 | $0 |
| LFM3-1.7B LoRA (rank 16) | 86.8% | 98.0% | 58.6% | 77.6% | 0% | 86.8% | 1.42 s* | 906 | $0 |
| LFM3-1.7B full fine-tune | 78.9% | 96.9% | 34.5% | 63.8% | 0% | 78.9% | 0.81 s | 905 | $0 |
| **LFM3-1.7B LoRA + router** | **91.4%** | **98.0%** | 75.9% | **82.2%** | **10.5%** | 91.9% | 1.49 s | — | **$0.18** |
| LFM3-1.7B full fine-tune + router | 86.2% | 99.0% | 48.3% | 75.0% | 13.2% | 85.6% | 0.74 s | — | $0.24 |
| LFM3-1.7B LoRA merged at 6-bit (deployed) | 86.8% | 98.0% | 58.6% | 78.3% | 0% | 86.8% | 1.18 s | 907 | $0 |
| **LFM3-1.7B LoRA merged at 6-bit + router (main student)** | **91.4%** | **98.0%** | 75.9% | **82.2%** | **11.8%** | 91.8% | **0.94 s** | — | **$0.21** |
| *Gemma 4 E4B + router, round 2 (for comparison)* | *94.1%* | *98.0%* | *89.7%* | *81.6%* | *16.4%* | *95.3%* | *2.35 s* | *1,240* | *$0.71* |

\* These runs shared the GPU with a training job. Measured again with the GPU free (15 held-out requests), LFM3 LoRA runs at **1.34 s** p50 per step with the adapter applied on the fly. The full-fine-tune rows had the GPU to themselves (0.74–0.81 s).

**The deployed student is the LoRA merged into the weights at 6-bit** (`train/fuse_vlm.py --bits 6` → `models/ft-lfm3-r2`, 1.36 GB). Merging at the base's 4 bits is faster (0.78 s/step) but rounds away much of the small adapter delta: 10/15 vs 15/15 on a held-out check. 8-bit kept accuracy (15/15) but was barely faster (1.21 s, 1.72 GB); 6-bit keeps it (15/15, then the full runs above) at 1.04 s on that check and 1.36 GB. On the full 152 requests the merged model scores exactly what the unmerged adapter did (86.8% alone, 91.4% with the router), at 0.94 s per step. It is the main student since Nov 8 (Mac device runtime and the Bayside LFM3 iOS build).

- **Fine-tuning fixes the format.** 77 of the untrained model's 95 failures were prose answers instead of the JSON action. After LoRA there are none, and the model alone matches fine-tuned Gemma E4B (86.8%) at a sixth of the size.
- **With the router:** 91.4%, with only 10.5% sent to the cloud, at a quarter of Gemma's cloud cost. 10 of the 16 escalations are the mandatory $1,000+ claim reviews. LFM3 trails Gemma on novel requests (75.9% vs 89.7%), where it is more often confidently wrong without failing a check.
- **LoRA beats full fine-tuning here.** Updating all 1.7B weights on about 1,000 examples overfits to the training shapes: novel requests dropped to 34.5% alone and 48.3% with the router. Its validation loss was also worse (0.0095 vs 0.0065). LoRA stays the training method, and each round ships as a small adapter.
- **Two policy misses are scenario ambiguity, not model error.** In policy-0000 and policy-0004, Sonnet asks whether a dentist visit was a cleaning or a filling instead of filing. Both Gemma and LFM3 hybrid runs fail these two the same way.
- **Training cost on the M5:** LoRA took about 40 minutes for 316 iterations (16.5M trainable parameters), and full fine-tuning about 55 minutes, both while sharing the GPU.

## The flywheel

| Round | Training data | Escalated (hybrid) | Hybrid task success | Shadow agreement |
|---|---|---|---|---|
| 0 | none (Sonnet serves everything) | 100% | 97.4% | – |
| 1 | 250 Sonnet traces → 492 steps | 20.4% | 86.8% | 80% |
| 2 | 500 Sonnet traces + escalations from 150 live-traffic requests → 1,010 steps | **16.4%** | **94.1%** | 88% |

10 of the 25 round-2 escalations are the mandatory $1,000+ claim reviews (by policy, not model failures). Excluding them, 10.6% of requests left the device. Validation loss: round 1 0.330, round 2 0.267.

## What we learned

1. **Fine-tuning plus routing gets close to the cloud model.** Round 2's hybrid is within 3.3 points of Sonnet overall (2 points on held-out) while serving 84% of requests on device, at 37% of Sonnet's cloud cost for the same scenarios.
2. **Distillation alone isn't enough; the router carries the quality.** The fine-tuned student by itself reached 86.8%; with validators and escalation, 94.1%. Escalation reasons in round 2: schema errors 10, $1,000+ policy 10, invalid JSON 2, bad dates 3, invented numbers 3, and one each for max steps and an unknown tool.
3. **Token-logprob confidence didn't separate right from wrong** for this student. The tuner's best utility was validators only. The confidence gate is kept loose and only on tool-call steps (`shared/router.json`).
4. **Reply grounding catches the student's typical error.** The student would sometimes add a number that's not in the tool results (a total deductible when only the remainder was returned). A cheap on-device check (every number in the reply must appear in, or be derived from, the conversation) flags those with 0 false positives on the student's good replies. It caught 10 replies in the live-traffic run.
5. **Short prompts cut prefill by 58%** (2,293 → ~970 tokens per step), because the fine-tuned student no longer needs the tool schemas. That's the main lever for the phone.
6. **Round 2 fixed novel requests** (58.6% → 89.7% in the hybrid). The extra data covered more multi-step shapes. Novel requests are still where the student is most often confidently wrong, which is why the OOD check exists.

## Gaps found (and fixed or open)

- **Never-leave keyword rules are brittle.** Two paraphrases ("doctor say I have anxiety", "…a herniated disc") weren't recognized. Neither was escalated, but in round 1 one went out as a redacted *shadow* sample. Fixed by widening the pattern (now 31/31 never-leave requests detected, 0 false positives on 771 others), but it was tuned on these same scenarios. **Open:** a small on-device classifier for medical intent.
- **The judge is stricter than the programmatic metric** (for example, it marks "expected by Nov 12" phrased as a promise as wrong). Treat judge numbers as reply quality and task success as action correctness.
- **The data is synthetic:** fictional insurer, generated requests, paraphrased by Sonnet. The SDK and collector are built for real traffic.
- **Escalation at the step limit (fixed after the run).** When the router escalated a student step at the 4-step limit, the loop ended before the teacher was called. 1 of the 25 round-2 escalations (heldout-0073) never reached the cloud, so 24 actually left the device. `harbor_env/agent.py` now gives the teacher one last step; the published numbers are from before the fix.
- **Router thresholds were tuned once** (round 1, train[400:500]). That slice is in round 2's training data, so re-tune on held-back traffic before relying on the confidence gate.

## Reproduce

See `README.md` → Pipeline. Run outputs (episodes + summaries) are in `runs/output/<run>/`, and the dashboard is at http://localhost:8000/ with the backend running.
