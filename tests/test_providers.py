"""Ollama error handling: an aborted reply is empty output; anything else that fails is ModelUnavailable."""
import httpx
import pytest

from ragdocs.providers import ModelUnavailable, OllamaLLM

CFG = {"model": "m", "base_url": "http://ollama.test", "timeout_s": 1}


def _respond(monkeypatch, status, body):
    request = httpx.Request("POST", "http://ollama.test/api/generate")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(status, text=body, request=request))


def test_aborted_prediction_is_empty_reply(monkeypatch):
    _respond(monkeypatch, 500, '{"error":"prediction aborted, token repeat limit reached"}')
    assert OllamaLLM(CFG).generate("p") == ""


def test_other_server_error_is_model_unavailable(monkeypatch):
    _respond(monkeypatch, 500, '{"error":"model failed to load"}')
    with pytest.raises(ModelUnavailable):
        OllamaLLM(CFG).generate("p")


def test_unreachable_server_is_model_unavailable(monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ModelUnavailable):
        OllamaLLM(CFG).generate("p")
