"""Model access behind two small interfaces.

The workflow only sees ``LLM.generate`` and ``Embedder.embed``. The default
implementation talks to a local Ollama server. Swapping the model or the
server is a config change in ``config.yaml``; no code changes.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

import httpx
import numpy as np


class ModelUnavailable(Exception):
    """A model server cannot be reached or returned an error."""


class LLM(Protocol):
    def generate(self, prompt: str, *, json_mode: bool = False) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaLLM:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = cfg["model"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.temperature = cfg.get("temperature", 0.0)
        self.num_ctx = cfg.get("num_ctx", 4096)
        self.timeout = cfg.get("timeout_s", 90)

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise ModelUnavailable(f"LLM {self.model} unavailable: {exc}") from exc
        return resp.json().get("response", "")


class OllamaEmbedder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = cfg["model"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.timeout = cfg.get("timeout_s", 60)

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise ModelUnavailable(f"Embedder {self.model} unavailable: {exc}") from exc
        vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)


def make_llm(cfg: dict[str, Any]) -> LLM:
    if cfg["provider"] != "ollama":
        raise ValueError(f"unsupported LLM provider {cfg['provider']!r}")
    return OllamaLLM(cfg)


def make_embedder(cfg: dict[str, Any]) -> Embedder:
    if cfg["provider"] != "ollama":
        raise ValueError(f"unsupported embedder provider {cfg['provider']!r}")
    return OllamaEmbedder(cfg)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a model reply. Returns None if none parses."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
