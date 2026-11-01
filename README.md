# Python documentation RAG assistant (assessment)

Work in progress. Corpus and question set are committed first, before the graph.

## Corpus (pinned, local copies in `corpus/`)

| Doc | Source | Version (commit) | File |
|---|---|---|---|
| CPython library docs | https://github.com/python/cpython, `Doc/library/` | v3.12.7 (0b05ead877f909b7efe712db758012d9dbece7ce) | `corpus/python-3.12.7/{json,string,datetime,urllib.request}.rst` |
| requests docs | https://github.com/psf/requests, `docs/` | v2.32.3 (0e322af87745eff34caffe4df68456ebc20d9068) | `corpus/requests-2.32.3/{api,quickstart,advanced}.rst` |

No live fetching at runtime.

## Question set

`data/questions.jsonl`: 20 questions (9 single_passage, 4 table_lookup, 2 multi_document,
1 version_dependent, 4 unanswerable). Check the gold quotes against the corpus with:

    python3 scripts/check_questions.py
