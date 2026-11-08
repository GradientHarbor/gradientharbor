"""Backup trainer for a CUDA GPU (Colab or AWS g6.xlarge): Unsloth LoRA on Gemma 4, merged 16-bit output.

    python colab_unsloth.py --data data --out merged [--model unsloth/gemma-4-E4B-it]

Then convert on the Mac: uv run mlx_lm.convert --hf-path merged --mlx-path models/ft-e4b-rN -q
Hyperparameters mirror train/lora.yaml. Written against Unsloth's FastModel API (2026.11); if a
name moved, follow Unsloth's Gemma 4 fine-tuning notebook, the data format doesn't change.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="merged")
ap.add_argument("--model", default="unsloth/gemma-4-E4B-it")
ap.add_argument("--steps", type=int, default=450)
a = ap.parse_args()

from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

model, tokenizer = FastModel.from_pretrained(a.model, max_seq_length=2048, load_in_4bit=True)
model = FastModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, finetune_vision_layers=False,
                                 finetune_language_layers=True, finetune_attention_modules=True,
                                 finetune_mlp_modules=True, random_state=7)


def load(name):
    rows = [json.loads(l) for l in (Path(a.data) / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    return Dataset.from_list([{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)} for r in rows])


trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=load("train"), eval_dataset=load("valid"),
                     args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=4,
                                    gradient_accumulation_steps=2, max_steps=a.steps, learning_rate=1e-4,
                                    warmup_steps=10, logging_steps=10, eval_strategy="steps", eval_steps=50,
                                    lr_scheduler_type="linear", seed=7, report_to="none", output_dir="ckpt"))
# Loss only on the model's turns (same as --mask-prompt in mlx-lm).
trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
trainer.train()
model.save_pretrained_merged(a.out, tokenizer, save_method="merged_16bit")
print("saved", a.out)
