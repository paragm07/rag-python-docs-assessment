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

Requires Python 3.12 or newer (the pinned numpy 2.5 needs it). Tested on Python 3.13 in the build
sandbox. Pinned versions are in `requirements.txt`.

1. Install Ollama and pull the two local models named in `config.yaml`:

       ollama pull qwen2.5:3b-instruct
       ollama pull nomic-embed-text

2. Install the Python packages:

       pip install -r requirements.txt

`config.yaml` sets no hardware options, so Ollama uses a GPU when one is present and the CPU
otherwise; on the CPU-only target server it runs unchanged. `config.cpu.yaml` is identical
except that it pins both models to the CPU with 4 threads (`num_gpu: 0`, `num_thread: 4`), to
measure latency as on the target. Select it with `--config config.cpu.yaml` for the scripts or
`RAGDOCS_CONFIG=config.cpu.yaml` for the service. Which configuration produced which number is
stated in the Results section.

## The three commands

    python ingest.py                 # build data/index/ragdocs.sqlite (chunks + embeddings)
    uvicorn ragdocs.api:app          # serve POST /query and GET /documents on port 8000
    python evaluate.py               # run the 20 questions, write data/results.jsonl

Tests (no model server needed; the workflow tests drive a scripted model through every
termination path, and the 503 test points the service at a port nothing listens on):

    python -m pytest -q

Render the compiled graph: `python scripts/render_graph.py` writes `docs/graph.mmd`.
Check the question set's gold quotes against the corpus: `python scripts/check_questions.py`.

## Example request

    curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
         -d '{"question": "What range of values does the %j directive produce in datetime strftime?"}'

Real response from this machine (`config.yaml`, GTX 1650; the section path is shortened here):

    {"request_id": "1722169a99b7",
     "answer": "The %j directive produces day of the year values as a zero-padded decimal number, ranging from 001 to 366.",
     "refused": false,
     "citations": [{"doc": "python-3.12.7",
                    "location": "python-3.12.7/datetime.rst:2425-2427 (... > strftime() and strptime() Format Codes)",
                    "chunk_id": "python-3.12.7/datetime#0191"}],
     "retrieved": ["python-3.12.7/datetime#0191", "python-3.12.7/datetime#0196", "...6 more"],
     "retained": ["python-3.12.7/datetime#0191", "python-3.12.7/datetime#0198", "python-3.12.7/datetime#0197"],
     "verified": true, "retries": 0, "termination": "answered", "latency_s": 30.476}

The first request after start-up includes loading the model. Each request's node-by-node trace
is appended to `data/api_traces.jsonl`.

Status codes: 200 answer or refusal, 415 wrong content type, 422 empty or oversized question
or bad body, 503 model server or index unavailable, 504 request timeout.

## Layout

| Path | What it is |
|---|---|
| `config.yaml` | every tunable: corpus, chunking, retrieval, models, retry limits, timeouts |
| `config.cpu.yaml` | the same, pinned to the target server (CPU only, 4 threads) |
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
| `data/traces.jsonl` | per question: every node visit in order, its seconds, what it decided |
| `data/latency/` | CPU-only (4 threads) latency measurements |
| `data/groundedness_review.md` | answered questions listed for the manual groundedness pass |
| `docs/graph.mmd` | the compiled workflow as Mermaid |
| `docs/report.pdf` | the report (source: `docs/report.html`) |
| `tests/` | chunking, workflow termination paths, API validation and timeouts, Ollama error handling |

## Results

Filled in from the run on the target machine. See the report for the discussion.

All runs offline on one laptop: i5-9300H (4 cores), 8 GB RAM, GTX 1650 (4 GB). Models tested:
`qwen2.5:3b-instruct` (Q4_K_M) and `nomic-embed-text`, through Ollama.

| | Result |
|---|---|
| Gold passage in the first 8 retrieved (hit@8), answerable | 16/16 |
| Gold passage retained by the grader, answerable | 14/16 |
| Unanswerable questions refused | 4/4 |
| Answerable questions wrongly refused | 2/16 (q06, q08) |
| Answered, fully grounded in their citations (manual pass) | 8/14 |
| Answered, right but a citation or completeness defect | 4/14 |
| Answered with a claim the citations do not support | 2/14 (q15 partly, q16 wrong) |
| Latency, `config.yaml` (Ollama used the GTX 1650): median / p95 | 20.9 s / 35.9 s |
| Latency, `config.cpu.yaml` (CPU only, 4 threads, as on the target): median / p95 | 85.1 s / 176.3 s |

The target is a median under 20 s on 4 CPU cores; on CPU this system does not meet it. The
report (`docs/report.pdf`, sections 4 and 6) says where the time goes, what was cut and what a
GPU would change. Quality results are in `data/results.jsonl` and `data/eval_summary.json`; the
CPU run's own results and traces are in `data/latency/cpu_4thread/`.

## AI assistance

I used Claude (Anthropic) through Claude Code as a coding assistant, in two stages.

- Claude Haiku 5.5: code scaffolding for the first version, debugging test failures, and
  drafting parts of the README.
- Claude Opus 5.5: a debugging and measurement pass after the first end-to-end run, which
  refused 11 of 16 answerable questions. It traced the failures, proposed and implemented the
  fixes I accepted (row-wise table splitting and linearisation, weighted fusion, the
  evidence-copying grader, schema-constrained model output, regeneration feedback, the
  127.0.0.1 latency fix, node traces), wrote the tests for them, and drafted the report.

I chose the corpus and wrote the question set and gold passages myself, before the graph was
built, and checked them against the documentation text. I reviewed every change, ran every
measurement reported here, and can explain each line.

## Confidentiality

The assessment brief and the question set are not for publication. Do not publish this
repository outside the review.
