"""Async OOD scoring on synced, redacted requests: distance to the nearest training request.

POC uses hashed character-trigram TF vectors with cosine similarity (no model download).
The production plan in shared/router.json is embeddinggemma-300m; swap `embed()` for it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

DIM = 2048


def embed(text: str) -> dict[int, float]:
    s = re.sub(r"\[[A-Z_]+_\d+\]", " ", text.lower())
    s = re.sub(r"\s+", " ", s)
    grams = Counter(hash(s[i:i + 3]) % DIM for i in range(max(len(s) - 2, 0)))
    norm = math.sqrt(sum(v * v for v in grams.values())) or 1.0
    return {k: v / norm for k, v in grams.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class OODIndex:
    def __init__(self, texts: list[str]):
        self.vecs = [embed(t) for t in texts]

    def distance(self, text: str) -> float:
        if not self.vecs:
            return 1.0
        v = embed(text)
        return 1.0 - max(cosine(v, u) for u in self.vecs)
