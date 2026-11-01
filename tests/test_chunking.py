import json
import re
from pathlib import Path

from ragdocs.chunking import chunk_text

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
CFG = {"max_chunk_chars": 1200, "max_entry_chars": 2000}


def _chunks(rel: str, doc: str, version: str):
    from ragdocs.chunking import chunk_file
    return chunk_file(CORPUS / rel, doc=doc, version=version, rel_file=rel, cfg_index=CFG)


def test_grid_table_is_one_chunk_with_caption():
    text = (
        "Format Codes\n============\n\n"
        "Table caption paragraph describing the codes:\n\n"
        "+-------+-----------+\n"
        "| Code  | Meaning   |\n"
        "+=======+===========+\n"
        "| ``%j``| Day of year|\n"
        "+-------+-----------+\n"
        "| ``%U``| Week no.  |\n"
        "+-------+-----------+\n"
    )
    chunks = chunk_text(text, doc="d", version="1", file="d/t.rst", max_chunk_chars=1200, max_entry_chars=2000)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    assert "%j" in tables[0].text and "%U" in tables[0].text
    assert "Table caption paragraph" in tables[0].text
    assert tables[0].section == "Format Codes"


def test_function_entry_keeps_signature_and_version_note():
    text = (
        ".. function:: dumps(obj, *, sort_keys=False)\n\n"
        "   Serialise *obj*.\n\n"
        "   .. versionchanged:: 3.6\n"
        "      Added the sort_keys option.\n"
    )
    chunks = chunk_text(text, doc="d", version="1", file="d/f.rst", max_chunk_chars=1200, max_entry_chars=2000)
    entries = [c for c in chunks if c.chunk_type == "entry"]
    assert len(entries) == 1
    assert entries[0].entry == "dumps"
    assert "dumps(obj, *, sort_keys=False)" in entries[0].text
    assert entries[0].version_notes and entries[0].version_notes[0].startswith("versionchanged 3.6")


def test_long_entry_is_split_and_every_part_repeats_signature():
    body = "\n\n".join(f"   Paragraph {i} " + "word " * 80 for i in range(12))
    text = f".. method:: Thing.run(x)\n\n{body}\n"
    chunks = chunk_text(text, doc="d", version="1", file="d/f.rst", max_chunk_chars=1200, max_entry_chars=1500)
    entries = [c for c in chunks if c.chunk_type == "entry"]
    assert len(entries) > 1
    assert all("Thing.run(x)" in c.text for c in entries)
    assert all(c.entry == "Thing.run" for c in entries)


def test_each_gold_quote_sits_inside_one_chunk():
    norm = lambda t: re.sub(r"\s+", " ", t)
    chunks = []
    for doc, version, rel in [
        ("python-3.12.7", "3.12.7", "python-3.12.7/json.rst"),
        ("python-3.12.7", "3.12.7", "python-3.12.7/datetime.rst"),
        ("python-3.12.7", "3.12.7", "python-3.12.7/string.rst"),
        ("python-3.12.7", "3.12.7", "python-3.12.7/urllib.request.rst"),
        ("requests-2.32.3", "2.32.3", "requests-2.32.3/advanced.rst"),
        ("requests-2.32.3", "2.32.3", "requests-2.32.3/quickstart.rst"),
    ]:
        chunks += _chunks(rel, doc, version)
    for line in (ROOT / "data" / "questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        for p in q["gold_passages"]:
            hits = [c for c in chunks if c.file == p["doc"] and norm(p["quote"]) in norm(c.text)]
            assert hits, f"{q['id']}: gold quote split across chunks or missing"


def test_chunk_ids_are_unique():
    chunks = _chunks("python-3.12.7/json.rst", "python-3.12.7", "3.12.7")
    assert len({c.chunk_id for c in chunks}) == len(chunks)
