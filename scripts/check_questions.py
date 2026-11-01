"""Check that every gold quote appears verbatim in the pinned corpus.

Whitespace is normalised on both sides, so line wraps in the .rst files do
not matter. Everything else (including reST markup) must match exactly.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "data" / "questions.jsonl"
CATEGORY_COUNTS = {
    "single_passage": 9,
    "table_lookup": 4,
    "multi_document": 2,
    "version_dependent": 1,
    "unanswerable": 4,
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main() -> int:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    errors = []
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if len(questions) != 20:
        errors.append(f"expected 20 questions, found {len(questions)}")
    counts: dict[str, int] = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
        if q["category"] == "unanswerable":
            if q["gold_passages"]:
                errors.append(f"{q['id']}: unanswerable must have no gold passages")
            continue
        for p in q["gold_passages"]:
            path = CORPUS / p["doc"]
            if not path.exists():
                errors.append(f"{q['id']}: missing doc {p['doc']}")
                continue
            if norm(p["quote"]) not in norm(path.read_text(encoding="utf-8")):
                errors.append(f"{q['id']}: quote not found in {p['doc']}: {p['quote']!r}")
    if counts != CATEGORY_COUNTS:
        errors.append(f"category counts {counts} != {CATEGORY_COUNTS}")
    for e in errors:
        print("ERROR", e)
    print(f"{len(questions)} questions, categories {counts}, errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
