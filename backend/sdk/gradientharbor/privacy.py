"""Redaction with placeholder re-fill, plus a fail-closed PII scanner.

Rules come from shared/redaction.json. The iOS HarborKit Redactor implements the same
rules and must pass the same golden vectors (shared/redaction_vectors.json).

    r = Redactor(rules, known={"name": "Owen Novak", ...})
    outbound = r.redact_text("I'm Owen Novak, member BM-10200")   # "I'm [NAME_1], member [MEMBER_ID_1]"
    reply = r.refill_text(teacher_reply)                          # placeholders -> real values, on device
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)_(\d+)\]")


def load_rules(shared_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(shared_dir) / "redaction.json").read_text())


def _amount_value(s: str) -> float:
    return float(re.sub(r"[^\d.]", "", s) or 0)


def format_amount(v: float) -> str:
    s = f"${v:,.2f}"
    return s[:-3] if s.endswith(".00") else s


class Redactor:
    def __init__(self, rules: dict[str, Any], known: dict[str, str] | None = None):
        self.rules = rules
        self.patterns = [(p["type"], re.compile(p["regex"])) for p in rules["patterns"]]
        self.amount_keys = set(rules.get("amount_keys", []))
        known = known or {}
        fields = rules.get("known_fields", {})
        # Longest values first so "owen.novak@example.com" wins over "Owen".
        self.known = sorted(((fields[k], v) for k, v in known.items() if k in fields and v), key=lambda kv: -len(kv[1]))
        self.forward: dict[tuple[str, str], str] = {}   # (type, normalized value) -> placeholder
        self.reverse: dict[str, Any] = {}               # placeholder -> original value
        self.counts: dict[str, int] = {}

    def _placeholder(self, typ: str, value: Any) -> str:
        key = (typ, f"{float(value):.2f}" if typ == "AMOUNT" else str(value).strip().lower())
        if key not in self.forward:
            self.counts[typ] = self.counts.get(typ, 0) + 1
            ph = self.rules["placeholder"].format(type=typ, n=self.counts[typ])
            self.forward[key] = ph
            self.reverse[ph] = float(value) if typ == "AMOUNT" else value
        return self.forward[key]

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        for typ, value in self.known:
            pat = re.compile(r"(?<![\w@.])" + re.escape(value) + r"(?![\w@])", re.IGNORECASE)
            text = pat.sub(lambda m, t=typ, v=value: self._placeholder(t, v), text)
        for typ, pat in self.patterns:
            if typ == "AMOUNT":
                text = pat.sub(lambda m: self._placeholder("AMOUNT", _amount_value(m.group(0))), text)
            else:
                text = pat.sub(lambda m, t=typ: self._placeholder(t, m.group(0)), text)
        return text

    def redact_obj(self, obj: Any, key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {k: self.redact_obj(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(v, key) for v in obj]
        if isinstance(obj, (int, float)) and not isinstance(obj, bool) and key in self.amount_keys:
            return self._placeholder("AMOUNT", obj)
        if isinstance(obj, str):
            return self.redact_text(obj)
        return obj

    def refill_text(self, text: str) -> str:
        if not text:
            return text

        def sub(m: re.Match[str]) -> str:
            ph = m.group(0).lstrip("$")
            if ph not in self.reverse:
                return m.group(0)
            v = self.reverse[ph]
            return format_amount(v) if isinstance(v, float) else str(v)

        return re.sub(r"\$?\[[A-Z_]+_\d+\]", sub, text)

    def refill_obj(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self.refill_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.refill_obj(v) for v in obj]
        if isinstance(obj, str):
            bare = obj.strip().lstrip("$")
            if bare in self.reverse and isinstance(self.reverse[bare], float):
                return self.reverse[bare]
            return self.refill_text(obj)
        return obj


class PIIScanner:
    """Finds raw PII left in an outbound payload. A non-empty result means: do not send."""

    def __init__(self, rules: dict[str, Any], members: Iterable[dict[str, Any]] = ()):
        scan = rules["scan"]
        by_type = {p["type"]: re.compile(p["regex"]) for p in rules["patterns"]}
        self.patterns = [(t, by_type[t]) for t in scan["patterns"]]
        values: list[tuple[str, str]] = []
        for m in members:
            fields = dict(m)
            if "name" in fields and "last_name" not in fields:
                fields["last_name"] = str(fields["name"]).split()[-1]
            for f in scan["known_fields"]:
                if fields.get(f):
                    values.append((f.upper(), str(fields[f])))
        self.known = [(t, re.compile(r"(?<![\w@.])" + re.escape(v) + r"(?![\w@])", re.IGNORECASE)) for t, v in values]

    def scan(self, text: str) -> list[dict[str, str]]:
        found = []
        for typ, pat in self.patterns + self.known:
            for m in pat.finditer(text or ""):
                found.append({"type": typ, "match": m.group(0)})
        return found

    def scan_obj(self, obj: Any) -> list[dict[str, str]]:
        return self.scan(json.dumps(obj, ensure_ascii=False))
