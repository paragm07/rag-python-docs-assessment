"""Structure-aware chunking for reStructuredText reference docs.

Three kinds of chunk, chosen per block of the source:

* ``entry``  one API directive (function, method, class, data ...) with its
  body. Kept whole so a signature never loses its parameter descriptions. An
  entry longer than ``max_entry_chars`` is split on blank lines, and every
  piece repeats the signature line.
* ``table``  one grid table, kept whole, plus the paragraph just before it
  (the caption that says what the table is). Cutting a table mid-row destroys
  the cell-to-header mapping that table_lookup questions rely on.
* ``prose``  consecutive paragraphs under the same section, grouped up to
  ``max_chunk_chars``. Paragraphs are never split.

Every chunk records its file, section path, line range and any version notes
("Changed in version", "Deprecated since") so citations can point to the exact
place and the version_dependent questions can cite the note.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENTRY_RE = re.compile(
    r"^(\s*)\.\. (function|method|classmethod|staticmethod|class|data|attribute|"
    r"exception|module|decorator|describe|opcode|option|envvar|cmdoption|"
    r"directive|confval|object)::\s*(.*)$"
)
UNDERLINE_RE = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")
VERSION_RE = re.compile(
    r"^\s*\.\. (versionadded|versionchanged|deprecated|versionremoved)::\s*(\S+)\s*(.*)$"
)
GRID_LINE_RE = re.compile(r"^\s*[+|]")
NAME_RE = re.compile(r"^([\w.]+)")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    version: str
    file: str
    section: str
    anchor: str
    entry: str | None
    chunk_type: str
    start_line: int
    end_line: int
    version_notes: list[str] = field(default_factory=list)
    text: str = ""
    index_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "section"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_title(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    text, under = lines[i], lines[i + 1]
    if not text.strip() or text[:1].isspace():
        return False
    if UNDERLINE_RE.match(text) or ENTRY_RE.match(text) or GRID_LINE_RE.match(text):
        return False
    if not UNDERLINE_RE.match(under):
        return False
    return len(under.strip()) >= len(text.strip()) - 1


def _version_notes(text: str) -> list[str]:
    notes = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m = VERSION_RE.match(line)
        if not m:
            continue
        kind, version, rest = m.group(1), m.group(2), m.group(3).strip()
        note = f"{kind} {version}"
        if rest:
            note += f": {rest}"
        elif idx + 1 < len(lines) and lines[idx + 1].strip():
            note += f": {lines[idx + 1].strip()}"
        notes.append(note)
    return notes


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Split on blank lines, keeping grid tables as single blocks."""
    blocks, cur = [], []
    for line in lines:
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def _pack(blocks: list[list[str]], limit: int) -> list[list[str]]:
    groups, cur, size = [], [], 0
    for block in blocks:
        block_size = sum(len(x) + 1 for x in block)
        if cur and size + block_size > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(block)
        size += block_size
    if cur:
        groups.append(cur)
    return groups


def chunk_text(
    text: str,
    *,
    doc: str,
    version: str,
    file: str,
    max_chunk_chars: int,
    max_entry_chars: int,
) -> list[Chunk]:
    lines = text.splitlines()
    n = len(lines)
    stem = file.rsplit(".", 1)[0]

    section_stack: list[tuple[str, str]] = []  # (underline char, title)
    char_order: list[str] = []

    def current_section() -> str:
        return " > ".join(t for _, t in section_stack) or "(top)"

    units: list[dict] = []  # each: kind, lines, start, section, entry, header
    i = 0
    prev_prose: dict | None = None
    while i < n:
        line = lines[i]
        if _is_title(lines, i):
            ch = lines[i + 1].strip()[0]
            if ch not in char_order:
                char_order.append(ch)
            level = char_order.index(ch)
            section_stack[:] = section_stack[:level]
            section_stack.append((ch, lines[i].strip()))
            i += 2
            prev_prose = None
            continue
        if not line.strip():
            i += 1
            continue
        m = ENTRY_RE.match(line)
        if m:
            indent = _indent(line)
            header = m.group(3).strip()
            j = i + 1
            while header.endswith("\\") and j < n:
                header = header[:-1].rstrip() + " " + lines[j].strip()
                j += 1
            name_m = NAME_RE.match(header)
            entry_name = name_m.group(1) if name_m else header
            body: list[str] = []
            while j < n:
                nxt = lines[j]
                if not nxt.strip():
                    body.append(nxt)
                    j += 1
                    continue
                if _indent(nxt) <= indent:
                    break
                if ENTRY_RE.match(nxt):  # nested entry gets its own chunk
                    break
                body.append(nxt)
                j += 1
            units.append({
                "kind": "entry", "start": i + 1, "end": j, "section": current_section(),
                "entry": entry_name, "header": f"{m.group(2)}:: {header}",
                "body": body,
            })
            i = j
            prev_prose = None
            continue
        if GRID_LINE_RE.match(line) and (line.lstrip().startswith("+") or line.lstrip().startswith("|")):
            j = i
            block: list[str] = []
            while j < n and lines[j].strip() and GRID_LINE_RE.match(lines[j]):
                block.append(lines[j])
                j += 1
            # The paragraph right before a table (usually followed by a blank line) is its caption.
            caption = prev_prose["text_block"] if prev_prose is not None else ""
            cap_start = prev_prose["start"] if prev_prose is not None else i + 1
            if prev_prose is not None:
                units.pop()  # the caption paragraph moves into the table chunk
            units.append({
                "kind": "table", "start": cap_start,
                "end": j, "section": current_section(), "entry": None,
                "header": "", "body": (caption.splitlines() if caption else []) + block,
            })
            i = j
            prev_prose = None
            continue
        # prose paragraph: consecutive non-blank, non-directive lines
        j = i
        para: list[str] = []
        while j < n and lines[j].strip() and not ENTRY_RE.match(lines[j]) \
                and not _is_title(lines, j) and not GRID_LINE_RE.match(lines[j]):
            para.append(lines[j])
            j += 1
        unit = {
            "kind": "prose", "start": i + 1, "end": j, "section": current_section(),
            "entry": None, "header": "", "body": para,
        }
        units.append(unit)
        prev_prose = {"start": i + 1, "end": j, "text_block": "\n".join(para)}
        i = j

    chunks: list[Chunk] = []
    seq = 0

    def emit(kind: str, body_lines: list[str], start: int, end: int, section: str,
             entry: str | None, header: str) -> None:
        nonlocal seq
        raw = "\n".join(([header] if header else []) + body_lines).rstrip()
        if not raw.strip():
            return
        anchor = _slug(entry) if entry else _slug(section.split(" > ")[-1])
        prefix = f"Document: {doc} {version} | Section: {section}"
        if entry:
            prefix += f" | Entry: {entry}"
        chunks.append(Chunk(
            chunk_id=f"{stem}#{seq:04d}", doc=doc, version=version, file=file,
            section=section, anchor=anchor, entry=entry, chunk_type=kind,
            start_line=start, end_line=end, version_notes=_version_notes(raw),
            text=raw, index_text=f"{prefix}\n{raw}",
        ))
        seq += 1

    pending_prose: list[dict] = []

    def flush_prose() -> None:
        if not pending_prose:
            return
        blocks = [b for u in pending_prose for b in _split_blocks(u["body"])]
        for group in _pack(blocks, max_chunk_chars):
            flat = [line for block in group for line in block + [""]][:-1]
            emit("prose", flat, pending_prose[0]["start"], pending_prose[-1]["end"],
                 pending_prose[0]["section"], None, "")
        pending_prose.clear()

    for u in units:
        if u["kind"] == "prose":
            if pending_prose and pending_prose[-1]["section"] != u["section"]:
                flush_prose()
            pending_prose.append(u)
            continue
        flush_prose()
        if u["kind"] == "table":
            emit("table", u["body"], u["start"], u["end"], u["section"], None, "")
        else:
            body_text = "\n".join(u["body"]).rstrip()
            if len(body_text) + len(u["header"]) <= max_entry_chars:
                emit("entry", u["body"], u["start"], u["end"], u["section"],
                     u["entry"], u["header"])
            else:
                blocks = _split_blocks(u["body"])
                for part, group in enumerate(_pack(blocks, max_entry_chars - len(u["header"]) - 40)):
                    flat = [line for block in group for line in block + [""]][:-1]
                    emit("entry", flat, u["start"], u["end"], u["section"], u["entry"],
                         f"{u['header']}  (part {part + 1})")
    flush_prose()
    return chunks


def chunk_file(path: Path, *, doc: str, version: str, rel_file: str, cfg_index: dict) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    return chunk_text(
        text, doc=doc, version=version, file=rel_file,
        max_chunk_chars=cfg_index["max_chunk_chars"],
        max_entry_chars=cfg_index["max_entry_chars"],
    )
