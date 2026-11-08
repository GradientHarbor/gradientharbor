"""Model backends. Every backend takes the canonical (redacted) conversation and returns one action.

Canonical messages:
    {"role": "user", "kind": "request" | "tool_result", "content": str, "tool": name?}
    {"role": "assistant", "content": action_json, "action": {...}, "native": [anthropic blocks]?}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .env import load_tools
from .prompts import action_json, parse_action, system_prompt

TEACHER_MODEL = os.environ.get("GH_TEACHER_MODEL", "claude-sonnet-5-5")


@dataclass
class StepOut:
    action: dict[str, Any] | None
    raw: str
    latency_s: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    logprobs: list[float] = field(default_factory=list)
    native: list[dict[str, Any]] | None = None
    cost_usd: float = 0.0
    error: str | None = None


class MLXBackend:
    """Gemma 4 (or any mlx-lm model) on Apple silicon. Same weights and chat template the iPhone runs."""

    def __init__(self, model: str, adapter: str | None = None, prompt: str = "full", few_shot: bool = False,
                 max_tokens: int = 320, name: str | None = None):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.sampler = make_sampler(temp=0.0)
        self.system = system_prompt(prompt, few_shot=few_shot)
        self.max_tokens = max_tokens
        self.name = name or model.split("/")[-1]
        self.kind = "local"

    def render(self, messages: list[dict[str, Any]]) -> str:
        chat = [{"role": "system", "content": self.system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages]
        return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)

    def step(self, messages: list[dict[str, Any]]) -> StepOut:
        from mlx_lm import stream_generate

        prompt = self.render(messages)
        t0 = time.perf_counter()
        text, lps, last = "", [], None
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=self.max_tokens, sampler=self.sampler):
            text += r.text
            if r.text.strip():  # skip whitespace/control tokens; score the action tokens
                lps.append(float(r.logprobs[r.token]))
            last = r
            # One action per step: stop once a complete JSON object has been produced.
            if text.count("{") and text.count("{") == text.count("}") and text.strip().endswith("}"):
                break
        dt = time.perf_counter() - t0
        return StepOut(action=parse_action(text), raw=text, latency_s=dt,
                       prompt_tokens=last.prompt_tokens if last else 0,
                       output_tokens=last.generation_tokens if last else 0, logprobs=lps)


class VLMBackend(MLXBackend):
    """Models mlx-lm doesn't support yet but mlx-vlm does (Liquid AI LFM3). Same prompt rendering,
    greedy decoding, logprobs of non-whitespace tokens and closed-JSON stop as MLXBackend.
    Run with the mlx-vlm overlay: `uv run --with "mlx-vlm>=0.7.11" python runs/run_eval.py ...`."""

    def __init__(self, model: str, adapter: str | None = None, prompt: str = "full", few_shot: bool = False,
                 max_tokens: int = 320, name: str | None = None):
        from mlx_vlm import load

        self.model, self.processor = load(model, adapter_path=adapter) if adapter else load(model)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.system = system_prompt(prompt, few_shot=few_shot)
        self.max_tokens = max_tokens
        self.name = name or model.rstrip("/").split("/")[-1]
        self.kind = "local"

    def step(self, messages: list[dict[str, Any]]) -> StepOut:
        from mlx_vlm import stream_generate

        prompt = self.render(messages)
        t0 = time.perf_counter()
        text, lps, last = "", [], None
        for r in stream_generate(self.model, self.processor, prompt, max_tokens=self.max_tokens, temperature=0.0):
            text += r.text
            if r.text.strip() and r.logprobs is not None:
                lps.append(float(r.logprobs.reshape(-1)[r.token]))
            last = r
            if text.count("{") and text.count("{") == text.count("}") and text.strip().endswith("}"):
                break
        dt = time.perf_counter() - t0
        return StepOut(action=parse_action(text), raw=text, latency_s=dt,
                       prompt_tokens=last.prompt_tokens if last else 0,
                       output_tokens=last.generation_tokens if last else 0, logprobs=lps)


def local_backend(model: str, **kw: Any) -> MLXBackend:
    """Pick the MLX engine for a model: mlx-vlm for LFM3, mlx-lm otherwise."""
    return VLMBackend(model, **kw) if "lfm3" in model.lower() else MLXBackend(model, **kw)


# Sonnet 5.5 pricing per 1M tokens (claude-api skill table, cached 2026-10-06).
PRICES = {"claude-sonnet-5-5": (2.00, 10.00, 0.20), "claude-haiku-5-5": (0.10, 0.50, 0.01),
          "claude-opus-5-5": (4.00, 20.00, 0.20)}


def native_tools() -> list[dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in load_tools()]


def to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical -> Anthropic messages. Teacher turns are replayed unchanged (thinking blocks included)."""
    out: list[dict[str, Any]] = []
    last_tool_id = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if m.get("native"):
                content = m["native"]
            elif "tool" in m["action"]:
                content = [{"type": "tool_use", "id": f"toolu_gh{i:03d}", "name": m["action"]["tool"],
                            "input": m["action"]["args"]}]
            else:
                content = [{"type": "text", "text": m["action"]["reply"]}]
            last_tool_id = next((b["id"] for b in content if b.get("type") == "tool_use"), None)
            out.append({"role": "assistant", "content": content})
        elif m.get("kind") == "tool_result" and last_tool_id:
            body = m["content"].split(">", 1)[1].rsplit("</tool_result>", 1)[0]
            out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": last_tool_id,
                                                     "content": body}]})
            last_tool_id = None
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


class TeacherBackend:
    """Claude Sonnet 5.5 with native tool use, mapped to the shared action format."""

    def __init__(self, model: str = TEACHER_MODEL, effort: str = "low", name: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.effort = effort
        self.system = system_prompt("teacher")
        self.tools = native_tools()
        self.name = name or model
        self.kind = "cloud"
        self.spent_usd = 0.0

    def step(self, messages: list[dict[str, Any]]) -> StepOut:
        import anthropic

        t0 = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system,
                tools=self.tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=to_native(messages),
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIStatusError as e:
            return StepOut(action=None, raw="", latency_s=time.perf_counter() - t0, error=f"api {e.status_code}")
        except anthropic.APIConnectionError:
            return StepOut(action=None, raw="", latency_s=time.perf_counter() - t0, error="connection")
        dt = time.perf_counter() - t0
        u = resp.usage
        pin, pout, pcache = PRICES.get(self.model, (2.0, 10.0, 0.2))
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        cost = (u.input_tokens * pin + cache_write * pin * 1.25 + cache_read * pcache + u.output_tokens * pout) / 1e6
        self.spent_usd += cost
        native = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        if resp.stop_reason == "refusal":
            return StepOut(action=None, raw="", latency_s=dt, native=native, cost_usd=cost, error="refusal")
        tool = next((b for b in resp.content if b.type == "tool_use"), None)
        if tool is not None:
            action = {"tool": tool.name, "args": dict(tool.input)}
        else:
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            action = {"reply": text}
        return StepOut(action=action, raw=action_json(action), latency_s=dt,
                       prompt_tokens=u.input_tokens + cache_read + cache_write, output_tokens=u.output_tokens,
                       native=native, cost_usd=cost)
