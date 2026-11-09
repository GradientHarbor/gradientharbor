"""Re-run the first N exported LFM3 parity prompts on the merged 6-bit model (what the phone runs) and keep
those outputs as the reference for the Swift port (harbor-parity generate). The adapter run's outputs stay as raw_adapter.

    uv run python runs/export_parity.py --run ft-lfm3-r2-eval --model models/ft-lfm3-r2 --out prompts-lfm3.jsonl
    uv run --with "mlx-vlm>=0.7.11" python runs/lfm3_parity_ref.py 16
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor_env.models import VLMBackend
n = int(sys.argv[1])
p = Path(__file__).resolve().parents[2] / "ios-app" / "HarborKit" / "parity" / "prompts-lfm3.jsonl"
rows = [json.loads(l) for l in p.read_text().splitlines()]
b = VLMBackend("models/ft-lfm3-r2", prompt="short")
for r in rows[:n]:
    assert b.render(r["messages"]) == r["prompt"]
    out = b.step(r["messages"])
    r["raw_adapter"] = r["raw"]; r["raw"] = out.raw; r["model"] = "ft-lfm3-r2"
    r["latency_s"] = round(out.latency_s, 3)
    print(r["scenario_id"], r["step"], f"{out.latency_s:.2f}s", "same-as-adapter" if out.raw == r["raw_adapter"] else "differs", flush=True)
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:n]))
