"""URL-first embeddings for Kimchi (no ONNX)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Iterable

DEFAULT_EMBED_URL = "https://dev-bge-m3-embedding.sionic.im/embed"
_HTTP_DISABLED = False


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def _hash_embed(text: str, dims: int = 128) -> list[float]:
    """Deterministic fallback embedding when HTTP fails."""
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    vec = [0.0] * dims
    for i in range(dims):
        b = digest[i % len(digest)]
        vec[i] = (b / 255.0) * 2.0 - 1.0
    return _l2_normalize(vec)


def _extract_vectors(payload: object) -> list[list[float]] | None:
    """Accept common payload shapes: {results:[...]}, {embeddings:[...]}, list[...]"""
    candidates: object = payload

    if isinstance(payload, dict):
        for key in ("results", "embeddings", "vectors", "data"):
            if key in payload:
                candidates = payload[key]
                break

    if not isinstance(candidates, list):
        return None

    # [[...], [...]]
    if candidates and isinstance(candidates[0], list):
        return [[float(x) for x in row] for row in candidates]

    # [{"embedding": [...]}, {"vector": [...]}]
    if candidates and isinstance(candidates[0], dict):
        out: list[list[float]] = []
        for row in candidates:
            if not isinstance(row, dict):
                return None
            vec = row.get("embedding") or row.get("vector") or row.get("dense")
            if not isinstance(vec, list):
                return None
            out.append([float(x) for x in vec])
        return out

    # [...] => single vector
    if candidates and isinstance(candidates[0], (int, float)):
        return [[float(x) for x in candidates]]

    return None


class HttpEmbedder:
    """HTTP embedder for endpoints expecting JSON: {inputs, truncate}."""

    def __init__(self, url: str, timeout: float = 8.0, batch_size: int = 64):
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size

    def _post(self, inputs: list[str], truncate: bool = True) -> list[list[float]]:
        body = json.dumps({"inputs": inputs, "truncate": truncate}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")

        parsed = json.loads(raw)
        vectors = _extract_vectors(parsed)
        if not vectors:
            raise RuntimeError("Invalid embedding response format")
        if len(vectors) != len(inputs):
            raise RuntimeError(f"Embedding mismatch: sent={len(inputs)} got={len(vectors)}")
        return [_l2_normalize(vec) for vec in vectors]

    def encode(self, texts: list[str], prefix: str = "", truncate: bool = True) -> list[list[float]]:
        if prefix:
            texts = [prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._post(batch, truncate=truncate))
        return out


def embed_texts(
    texts: Iterable[str],
    prefix: str = "",
    *,
    url: str | None = None,
    truncate: bool = True,
    strict_http: bool | None = None,
) -> list[list[float]]:
    """Embed text with HTTP first, deterministic fallback second (unless strict mode)."""
    global _HTTP_DISABLED
    rows = list(texts)
    if not rows:
        return []

    endpoint = (url or os.environ.get("KIMCHI_EMBED_URL") or DEFAULT_EMBED_URL).strip()
    strict = strict_http if strict_http is not None else os.environ.get("KIMCHI_EMBED_STRICT", "0") == "1"

    if _HTTP_DISABLED and not strict:
        if prefix:
            rows = [prefix + t for t in rows]
        return [_hash_embed(t) for t in rows]

    try:
        return HttpEmbedder(endpoint).encode(rows, prefix=prefix, truncate=truncate)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, ValueError):
        if strict:
            raise
        _HTTP_DISABLED = True
        if prefix:
            rows = [prefix + t for t in rows]
        return [_hash_embed(t) for t in rows]
