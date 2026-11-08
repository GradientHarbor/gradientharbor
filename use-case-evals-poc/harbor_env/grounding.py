"""Reply grounding check: every number in a reply must appear in the conversation.

Cheap, on-device, and catches the student's typical failure (an invented total, limit or
amount). Runs on the redacted conversation, so placeholders like [AMOUNT_1] are ignored.
"""

from __future__ import annotations

import re

_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", re.I)
_NUM = re.compile(r"(?<![\w\[])\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%|st|nd|rd|th)?(?![\w\]])")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


def _canon(intpart: str, frac: str | None) -> str:
    v = float(intpart.replace(",", "") + ("." + frac if frac else ""))
    return f"{v:g}"


def numbers_in(text: str, *, as_reply: bool = False) -> set[str]:
    text = _PLACEHOLDER.sub(" ", text or "")
    out: set[str] = set()
    for m in _TIME.finditer(text):
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        out.add(f"{h:02d}:{m.group(2) or '00'}")
    text = _TIME.sub(" ", text)
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text):  # 24h times
        out.add(f"{int(m.group(1)):02d}:{m.group(2)}")
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    for m in _NUM.finditer(text):
        v = _canon(m.group(1), m.group(2))
        if m.group(3) == "%":
            out.add(f"{float(v) / 100:g}")  # "30%" matches 0.3 in a tool result
            if not as_reply:
                out.add(v)
        else:
            out.add(v)
    return out


def context_numbers(contents: list[str]) -> set[str]:
    nums: set[str] = set()
    for c in contents:
        nums |= numbers_in(c)
        for m in re.finditer(r"T(\d{2}):(\d{2})", c):  # ISO datetimes
            nums.add(f"{m.group(1)}:{m.group(2)}")
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})(?=\b|T)", c):  # ISO dates: month and day as plain numbers
            nums |= {f"{int(m.group(2)):g}", f"{int(m.group(3)):g}", m.group(1)}
    return nums


SAFE = {"1", "2", "3", "0"}  # "one appointment", "2 options" etc.


def ungrounded(reply: str, contents: list[str]) -> list[str]:
    ctx = context_numbers(contents)
    pct_ctx = {f"{float(x) * 100:g}" for x in ctx if re.fullmatch(r"[\d.]+", x) and float(x) < 1}
    vals = sorted({float(x) for x in ctx if re.fullmatch(r"\d+(\.\d+)?", x) and float(x) >= 10})
    derived = {f"{a - b:g}" for a in vals for b in vals if a > b} | {f"{a + b:g}" for a in vals for b in vals if a < b}
    missing = []
    for n in numbers_in(reply, as_reply=True):
        if n in ctx or n in SAFE or n in pct_ctx or n in derived:  # derived: "$4,000 max, $580 spent -> $3,420 left"
            continue
        missing.append(n)
    return sorted(missing)
