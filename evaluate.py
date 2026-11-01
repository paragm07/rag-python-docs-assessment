"""Run the workflow over data/questions.jsonl and write results.jsonl.

    python evaluate.py [--config config.yaml] [--out-dir data]

Writes:
  data/results.jsonl      one line per question, the schema in the brief
  data/eval_summary.json  retrieval hit rates, refusal counts, latency, automatic checks
  data/traces.jsonl       per question, every node visit in order with its seconds and decisions
  data/groundedness_review.md  answered questions listed for the manual groundedness pass

Gold matching: a chunk matches a gold passage when it comes from the same file and
its text contains the gold quote after whitespace normalisation. Chunk boundaries are
not used for matching. scripts/check_questions.py already confirmed each quote is
in the corpus, and tests/test_chunking.py confirms each quote sits inside one chunk.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from ragdocs.config import load_config, resolve
from ragdocs.graph import build_graph, run_question, unsupported_identifiers
from ragdocs.providers import ModelUnavailable, make_embedder, make_llm
from ragdocs.retrieval import Index

ANSWERABLE = ("single_passage", "table_lookup", "multi_document", "version_dependent")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def gold_chunk_ids(index: Index, passage: dict) -> set[str]:
    needle = norm(passage["quote"])
    return {cid for cid in index.order
            if index.get(cid).file == passage["doc"] and needle in norm(index.get(cid).text)}


def automatic_checks(index: Index, result: dict) -> dict:
    """Mechanical groundedness checks. The manual pass is separate."""
    cited = result["citations"]
    bad_citations = [c for c in cited if c not in result["retained"]]
    missing_ids = unsupported_identifiers(result["answer"], " ".join(index.get(c).text for c in cited))
    return {"citations_in_retained": not bad_citations,
            "backticked_identifiers_in_cited_text": not missing_ids,
            "unsupported_identifiers": missing_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="data", help="where the outputs go (questions are always read from data/)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = resolve(cfg, args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    questions = [json.loads(line) for line in resolve(cfg, "data/questions.jsonl").read_text().splitlines()
                 if line.strip()]
    index = Index(resolve(cfg, cfg["index"]["path"]), make_embedder(cfg["models"]["embedder"]),
                  cfg["retrieval"]["use_dense"])
    graph = build_graph(cfg, index, make_llm(cfg["models"]["llm"]))
    top_k = cfg["retrieval"]["top_k"]

    rows, per_q, traces = [], [], []
    for q in questions:
        started = time.perf_counter()
        try:
            res = run_question(graph, q["question"])
        except ModelUnavailable as exc:
            sys.exit(f"Model server not reachable: {exc}\nStart Ollama and pull the models in config.yaml, then rerun.")
        latency = round(time.perf_counter() - started, 3)
        citations = [{"doc": index.get(c).doc,
                      "location": f"{index.get(c).file}:{index.get(c).meta['start_line']}-"
                                  f"{index.get(c).meta['end_line']} ({index.get(c).section})",
                      "chunk_id": c} for c in res["citations"]]
        row = {"id": q["id"], "answer": res["answer"], "refused": res["refused"],
               "citations": citations, "retrieved": res["retrieved"], "retained": res["retained"],
               "verified": res["verified"], "retries": res["retries"],
               "termination": res["termination"], "latency_s": latency}
        rows.append(row)
        traces.append({"id": q["id"], "question": q["question"], "latency_s": latency, "trace": res["trace"]})

        gold = [p for p in q["gold_passages"]]
        gold_ids = [gold_chunk_ids(index, p) for p in gold]
        first_attempt = res["retrieved"][:top_k]
        hit_retrieved = any(ids & set(first_attempt) for ids in gold_ids) if gold else None
        hit_retained = any(ids & set(res["retained"]) for ids in gold_ids) if gold else None
        per_q.append({"id": q["id"], "category": q["category"], "refused": res["refused"],
                      "hit_retrieved": hit_retrieved, "hit_retained": hit_retained,
                      "latency_s": latency, "termination": res["termination"],
                      "retries": res["retries"], "verified": res["verified"],
                      "checks": automatic_checks(index, res) if res["citations"] else None})
        print(f"{q['id']} {q['category']:<18} refused={res['refused']!s:<5} "
              f"term={res['termination']:<18} retries={res['retries']} {latency}s", flush=True)

    with (data_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (data_dir / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = summarise(per_q, top_k)
    (data_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_groundedness_review(data_dir / "groundedness_review.md", questions, rows)
    print(json.dumps(summary["headline"], indent=2))


def summarise(per_q: list[dict], top_k: int) -> dict:
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_hit_retrieved": 0, "n_hit_retained": 0})
    for r in per_q:
        if r["category"] not in ANSWERABLE:
            continue
        c = by_cat[r["category"]]
        c["n"] += 1
        c["n_hit_retrieved"] += int(bool(r["hit_retrieved"]))
        c["n_hit_retained"] += int(bool(r["hit_retained"]))
    answerable = [r for r in per_q if r["category"] in ANSWERABLE]
    unanswerable = [r for r in per_q if r["category"] == "unanswerable"]
    latencies = sorted(r["latency_s"] for r in per_q)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    answered = [r for r in per_q if not r["refused"]]
    return {
        "top_k_for_hit_at_k": top_k,
        "retrieval_by_category": {k: {**v,
                                      "hit_at_k_retrieved": round(v["n_hit_retrieved"] / v["n"], 3),
                                      "hit_retained": round(v["n_hit_retained"] / v["n"], 3)}
                                  for k, v in by_cat.items()},
        "refusals": {
            "unanswerable_refused": sum(r["refused"] for r in unanswerable),
            "unanswerable_total": len(unanswerable),
            "answerable_wrongly_refused": sum(r["refused"] for r in answerable),
            "answerable_total": len(answerable),
        },
        "termination_counts": {t: sum(r["termination"] == t for r in per_q)
                               for t in sorted({r["termination"] for r in per_q})},
        "verified_counts": {"true": sum(r["verified"] is True for r in answered),
                            "false": sum(r["verified"] is False for r in answered)},
        "latency_s": {"median": round(statistics.median(latencies), 3), "p95": round(p95, 3),
                      "max": round(max(latencies), 3)},
        "automatic_checks_failed": [r["id"] for r in per_q if r["checks"] and not (
            r["checks"]["citations_in_retained"] and r["checks"]["backticked_identifiers_in_cited_text"])],
        "headline": {
            "retrieval_hit_at_k_answerable": round(
                sum(bool(r["hit_retrieved"]) for r in answerable) / max(len(answerable), 1), 3),
            "retrieval_hit_retained_answerable": round(
                sum(bool(r["hit_retained"]) for r in answerable) / max(len(answerable), 1), 3),
            "unanswerable_refused": f"{sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}",
            "answerable_wrongly_refused": f"{sum(r['refused'] for r in answerable)}/{len(answerable)}",
            "latency_median_s": round(statistics.median(latencies), 3),
            "latency_p95_s": round(p95, 3),
        },
        "per_question": per_q,
    }


def write_groundedness_review(path: Path, questions: list[dict], rows: list[dict]) -> None:
    lines = ["# Groundedness review (manual)", "",
             "For each answered question, check every factual claim against the cited passage.",
             "Mark each claim supported / not supported, and record the result in the report.", ""]
    qmap = {q["id"]: q for q in questions}
    for r in rows:
        if r["refused"]:
            continue
        lines += [f"## {r['id']} ({qmap[r['id']]['category']})", f"Question: {qmap[r['id']]['question']}", "",
                  f"Answer: {r['answer']}", "", "Citations:"]
        lines += [f"- {c['chunk_id']} — {c['location']}" for c in r["citations"]] or ["- (none)"]
        lines += ["", "Supported: [ ] all claims  [ ] partly  [ ] no", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
