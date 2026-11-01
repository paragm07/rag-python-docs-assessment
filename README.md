# Python documentation RAG assistant

A question-answering service over pinned Python documentation. A LangGraph workflow
retrieves passages, decides whether they are enough, recovers or refuses when they are
not, generates an answer with citations, and verifies the answer against the cited
passages. A FastAPI service sits in front of it. Everything runs locally and offline.

## Corpus (pinned, local copies in `corpus/`, no live fetching)

| Document | Version | Source | Files |
|---|---|---|---|
| CPython library docs | 3.12.7 (commit `0b05ead877f909b7efe712db758012d9dbece7ce`) | https://github.com/python/cpython/tree/v3.12.7/Doc/library | `json.rst`, `string.rst`, `datetime.rst`, `urllib.request.rst` |
| requests docs | 2.32.3 (commit `0e322af87745eff34caffe4df68456ebc20d9068`) | https://github.com/psf/requests/tree/v2.32.3/docs | `api.rst`, `quickstart.rst`, `advanced.rst` |

The licences of both projects permit local copies. The files are reStructuredText, which
the chunker parses directly.

## Setup (offline run)

Tested on Python 3.13 (sandbox). Pinned versions are in `requirements.txt`.

1. Install Ollama and pull the two local models named in `config.yaml`:

       ollama pull qwen2.5:3b-instruct
       ollama pull nomic-embed-text

2. Install the Python packages:

       pip install -r requirements.txt

## The three commands

    python ingest.py                 # build data/index/ragdocs.sqlite (chunks + embeddings)
    uvicorn ragdocs.api:app          # serve POST /query and GET /documents on port 8000
    python evaluate.py               # run the 20 questions, write data/results.jsonl

Tests (no model server needed for most; the 503 test checks the model-down path):

    python -m pytest -q

Render the compiled graph: `python scripts/render_graph.py` writes `docs/graph.mmd`.
Check the question set's gold quotes against the corpus: `python scripts/check_questions.py`.

## Example request

    curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
         -d '{"question": "What range of values does the %j directive produce in datetime strftime?"}'

Response shape (abridged; the real values come from the run on the target machine):

    {"request_id": "...", "answer": "...", "refused": false,
     "citations": [{"doc": "python-3.12.7", "location": "python-3.12.7/datetime.rst:2359-2457 (...)",
                    "chunk_id": "python-3.12.7/datetime#0177"}],
     "retrieved": [...], "retained": [...], "verified": true, "retries": 0,
     "termination": "answered", "latency_s": 7.4}

Status codes: 200 answer or refusal, 415 wrong content type, 422 empty or oversized question
or bad body, 503 model server or index unavailable, 504 request timeout.

## Layout

| Path | What it is |
|---|---|
| `config.yaml` | every tunable: corpus, chunking, retrieval, models, retry limits, timeouts |
| `ragdocs/chunking.py` | structure-aware chunker for reStructuredText |
| `ragdocs/retrieval.py` | SQLite index, BM25, dense search, reciprocal rank fusion |
| `ragdocs/providers.py` | LLM and embedder interfaces; Ollama implementation |
| `ragdocs/graph.py` | the LangGraph StateGraph, its state and routing |
| `ragdocs/prompts.py` | prompt templates |
| `ragdocs/api.py` | FastAPI service |
| `ingest.py` | ingestion script (run once) |
| `evaluate.py` | runs the question set and writes `results.jsonl` and the summary |
| `data/questions.jsonl` | the 20 questions with categories, gold answers and gold passages |
| `data/results.jsonl` | the run's results in the required schema |
| `data/eval_summary.json` | metrics, including per-question records |
| `data/groundedness_review.md` | answered questions listed for the manual groundedness pass |
| `docs/graph.mmd` | the compiled workflow as Mermaid |
| `tests/` | chunking, workflow termination paths, API validation |

## Results

Filled in from the run on the target machine. See the report for the discussion.

RESULTS_PLACEHOLDER

## AI assistance

I used Claude (Anthropic, Claude Haiku 5.5, through Claude Code) as a coding assistant. It
helped with code scaffolding, debugging test failures, and drafting parts of the README and
report. I wrote the design decisions myself, reviewed and ran every part, and can explain each
line. The question set, the gold passages and the choice of corpus were written and checked
by me against the documentation text.

## Confidentiality

The assessment brief and the question set are not for publication. Do not publish this
repository outside the review.
