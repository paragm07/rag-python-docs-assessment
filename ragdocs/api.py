"""FastAPI service in front of the workflow.

    uvicorn ragdocs.api:app --host 127.0.0.1 --port 8000

POST /query      a question in, an answer with citations out (refusals are 200)
GET  /documents  the indexed documents with version, source, chunk count, time
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import load_config, resolve
from .graph import build_graph, render_mermaid, run_question
from .providers import ModelUnavailable, make_embedder, make_llm
from .retrieval import Index

MAX_CHARS = 500  # overridden from config at startup


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_CHARS)


class Citation(BaseModel):
    doc: str
    location: str
    chunk_id: str


class QueryOut(BaseModel):
    request_id: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[str]
    retained: list[str]
    verified: bool | None
    retries: int
    termination: str
    latency_s: float


STATE: dict[str, Any] = {}


def _citation(index: Index, cid: str) -> Citation:
    row = index.get(cid)
    line = row.meta.get("start_line"), row.meta.get("end_line")
    location = f"{row.file}:{line[0]}-{line[1]} ({row.section})"
    return Citation(doc=row.doc, location=location, chunk_id=cid)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MAX_CHARS
    cfg = load_config()
    MAX_CHARS = cfg["api"]["max_question_chars"]
    STATE["cfg"] = cfg
    db_path = resolve(cfg, cfg["index"]["path"])
    STATE["graph"] = None
    STATE["index"] = None
    STATE["error"] = None
    try:
        embedder = make_embedder(cfg["models"]["embedder"])
        index = Index(db_path, embedder, cfg["retrieval"]["use_dense"])
        STATE["index"] = index
        STATE["graph"] = build_graph(cfg, index, make_llm(cfg["models"]["llm"]))
    except FileNotFoundError as exc:
        STATE["error"] = f"index not built: {exc}"
    except Exception as exc:  # startup must not crash the service; /query reports it
        STATE["error"] = f"startup failed: {exc}"
    yield


app = FastAPI(title="Python docs RAG assistant", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    # No stack traces to clients. The detail goes to the server log.
    import logging
    logging.getLogger("ragdocs").exception("unhandled error on %s", request.url.path)
    return _error(500, "internal_error", "Unexpected server error.")


@app.post("/query", response_model=QueryOut, responses={
    415: {"description": "Content type is not application/json"},
    422: {"description": "Question empty or longer than the limit"},
    503: {"description": "Model server or index unavailable"},
    504: {"description": "Request exceeded the time limit"},
})
async def query(request: Request):
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _error(415, "unsupported_media_type", "Send the question as application/json.")
    try:
        body = await request.json()
    except ValueError:
        return _error(422, "invalid_json", "Request body is not valid JSON.")
    if not isinstance(body, dict) or "question" not in body:
        return _error(422, "missing_question", "Body must be an object with a 'question' string.")
    if not isinstance(body["question"], str):
        return _error(422, "invalid_question", "'question' must be a string.")
    question = body["question"].strip()
    if not question:
        return _error(422, "empty_question", "'question' must not be empty.")
    if len(question) > MAX_CHARS:
        return _error(422, "question_too_long", f"'question' must be at most {MAX_CHARS} characters.")
    if STATE.get("graph") is None:
        return _error(503, "service_unavailable", STATE.get("error") or "workflow not ready")

    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    timeout = STATE["cfg"]["api"]["request_timeout_s"]
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(run_question, STATE["graph"], question), timeout=timeout)
    except asyncio.TimeoutError:
        # The worker thread cannot be killed. It finishes in the background and its result is dropped.
        return JSONResponse(status_code=504, content={
            "request_id": request_id, "answer": "The request timed out. Try again.",
            "refused": True, "citations": [], "retrieved": [], "retained": [],
            "verified": None, "retries": 0, "termination": "timeout",
            "latency_s": round(time.perf_counter() - started, 3)})
    except ModelUnavailable as exc:
        return _error(503, "model_unavailable", str(exc))

    index: Index = STATE["index"]
    return QueryOut(
        request_id=request_id,
        answer=result["answer"],
        refused=result["refused"],
        citations=[_citation(index, c) for c in result["citations"]],
        retrieved=result["retrieved"],
        retained=result["retained"],
        verified=result["verified"],
        retries=result["retries"],
        termination=result["termination"],
        latency_s=round(time.perf_counter() - started, 3),
    )


@app.get("/documents")
async def documents():
    if STATE.get("index") is None:
        return _error(503, "service_unavailable", STATE.get("error") or "index not ready")
    return {"documents": [
        {"name": m["name"], "version": m["version"], "source": m["source"],
         "chunk_count": m["chunk_count"], "ingested_at": m["ingested_at"]}
        for m in STATE["index"].manifest
    ]}


@app.get("/health")
async def health():
    return {"ready": STATE.get("graph") is not None, "error": STATE.get("error")}
