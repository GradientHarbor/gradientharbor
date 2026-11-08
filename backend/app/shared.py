"""Access to the shared spec (tools, prompts, redaction rules, fixtures)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from gradientharbor.privacy import PIIScanner, load_rules

SHARED = Path(os.environ.get("GH_SHARED_DIR", Path(__file__).resolve().parents[2] / "shared")).resolve()


@lru_cache(maxsize=1)
def tools() -> list[dict]:
    return json.loads((SHARED / "tools.json").read_text())["tools"]


@lru_cache(maxsize=1)
def rules() -> dict:
    return load_rules(SHARED)


@lru_cache(maxsize=1)
def scanner() -> PIIScanner:
    return PIIScanner(rules(), json.loads((SHARED / "fixtures" / "members.json").read_text()))


@lru_cache(maxsize=1)
def router_config() -> dict:
    return json.loads((SHARED / "router.json").read_text())


def prompt(name: str) -> str:
    return (SHARED / "prompts" / name).read_text().strip()


def short_system_prompt() -> str:
    """System prompt for the fine-tuned student (must match harbor_env.prompts.system_prompt('short'))."""
    lines = []
    for t in tools():
        props = t["input_schema"]["properties"]
        req = set(t["input_schema"].get("required", []))
        args = ", ".join(p if p in req else p + "?" for p in props)
        lines.append(f"- {t['name']}({args}): {t['short']}")
    return "\n\n".join([prompt("system_core.txt"), prompt("action_format.txt"), "Tools:\n" + "\n".join(lines)])
