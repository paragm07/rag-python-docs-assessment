from pathlib import Path

import pytest
from fastapi.testclient import TestClient

INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "index" / "ragdocs.sqlite"
pytestmark = pytest.mark.skipif(not INDEX_PATH.exists(), reason="run ingest.py first")


@pytest.fixture(scope="module")
def client():
    from ragdocs.api import app
    with TestClient(app) as c:
        yield c


def test_empty_question_is_422(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 422 and r.json()["error"] == "empty_question"


def test_oversized_question_is_422(client):
    r = client.post("/query", json={"question": "x" * 2000})
    assert r.status_code == 422
    assert r.json()["error"] in ("question_too_long",) or r.status_code == 422


def test_wrong_content_type_is_415(client):
    r = client.post("/query", content="question=hi", headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_missing_field_is_422(client):
    r = client.post("/query", json={"q": "hi"})
    assert r.status_code == 422


@pytest.fixture
def swap_state():
    """Temporarily replace entries in the service state; restore them afterwards."""
    from ragdocs import api
    saved = dict(api.STATE)
    yield api.STATE
    api.STATE.clear()
    api.STATE.update(saved)


def test_model_down_is_503_not_500(client, swap_state):
    # Point both models at a port nothing listens on, whether or not Ollama is running.
    from ragdocs.graph import build_graph
    from ragdocs.providers import make_embedder, make_llm
    from ragdocs.retrieval import Index
    cfg = swap_state["cfg"]
    dead = "http://127.0.0.1:9"
    index = Index(INDEX_PATH, make_embedder({**cfg["models"]["embedder"], "base_url": dead}),
                  cfg["retrieval"]["use_dense"])
    swap_state["graph"] = build_graph(cfg, index, make_llm({**cfg["models"]["llm"], "base_url": dead}))
    r = client.post("/query", json={"question": "How do I set a timeout in requests?"})
    assert r.status_code == 503
    assert r.json()["error"] == "model_unavailable"
    assert "Traceback" not in r.text


def test_slow_workflow_is_504_with_a_body(client, swap_state):
    import time

    class Slow:
        def stream(self, *args, **kwargs):
            time.sleep(3)
            return iter(())

    swap_state["graph"] = Slow()
    swap_state["cfg"] = {**swap_state["cfg"], "api": {**swap_state["cfg"]["api"], "request_timeout_s": 0.5}}
    r = client.post("/query", json={"question": "How do I set a timeout in requests?"})
    assert r.status_code == 504
    assert r.json()["termination"] == "timeout" and r.json()["refused"] is True


def test_documents_lists_both_sources(client):
    r = client.get("/documents")
    assert r.status_code == 200
    names = {d["name"] for d in r.json()["documents"]}
    assert names == {"python-3.12.7", "requests-2.32.3"}
    assert all(d["chunk_count"] > 0 and d["ingested_at"] for d in r.json()["documents"])
