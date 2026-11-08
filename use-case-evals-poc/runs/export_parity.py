"""Export what the Mac evals fed the student, for the Swift parity tool (ios-app/HarborKit).

    uv run python runs/export_parity.py --limit 40   # -> ../ios-app/HarborKit/parity/prompts.jsonl

For each student step of the round-2 fine-tuned eval run: the canonical messages the student saw, the
rendered Gemma prompt, its token ids, and the raw output the Mac produced. The Swift side must render the
same string, tokenize to the same ids and (greedy) generate the same output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor_env.agent import read_jsonl  # noqa: E402
from harbor_env.paths import ROOT  # noqa: E402
from harbor_env.prompts import system_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ft-e4b-r2-eval")
    ap.add_argument("--model", default=str(ROOT / "models" / "ft-e4b-r2"))
    ap.add_argument("--limit", type=int, default=40, help="episodes")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    system = system_prompt("short")
    out = ROOT.parent / "ios-app" / "HarborKit" / "parity" / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in read_jsonl(ROOT / "runs" / "output" / args.run / "episodes.jsonl")[: args.limit]:
        ep = e["episode"]
        msgs = ep["messages"]
        users_seen = 0
        for i, m in enumerate(msgs):
            if m["role"] != "user":
                continue
            users_seen += 1
            step = users_seen - 1
            if step >= len(ep["steps"]) or not ep["steps"][step].get("raw"):
                continue
            prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[: i + 1]]
            chat = [{"role": "system", "content": system}] + prefix
            prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            rows.append({"scenario_id": e["scenario_id"], "step": step, "messages": prefix, "prompt": prompt,
                         "ids": tok.encode(prompt, add_special_tokens=False), "raw": ep["steps"][step]["raw"]})
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} student steps to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
