from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # use-case-evals-poc/
SHARED = Path(os.environ.get("GH_SHARED_DIR", ROOT.parent / "shared")).resolve()
DATA = ROOT / "data"
RUNS = ROOT / "runs" / "output"
MODELS = ROOT / "models"


def load_env() -> None:
    """Load ./.env and map the non-standard key names to the SDK defaults. Never prints values."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if "ANTHROPIC_API_KEY" not in os.environ and os.environ.get("ANTHROPIC_API"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API"]
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPEN_AI_API"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API"]
