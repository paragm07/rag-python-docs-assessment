"""The question-answering workflow as a LangGraph StateGraph.

Flow::

    retrieve -> grade -+-> generate -> verify -+-> finish            (answered)
                       |                       +-> generate         (regenerate, bounded)
                       |                       +-> finish_unverified (answer with failure disclosed)
                       +-> recover -> retrieve                      (rewrite query, bounded)
                       +-> refuse                                   (no context / retries used up)
    generate -> refuse                                              (model says docs do not cover it)

Termination values written to ``termination``:

* ``answered``            grounded answer, verification passed
* ``refused_no_context``  nothing retrieved
* ``refused_no_answer``   grader kept passages but the generator found no answer in them
* ``retry_limit``         context judged insufficient after all corrective loops
* ``refused_unverified``  answer returned after verification failed (failure disclosed)
* ``graph_limit``         safety net: LangGraph step limit reached (should not happen)

Every loop has a counter in state and a bound from config, so each path ends.
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .prompts import (GENERATE_PROMPT, GENERATE_SCHEMA, GRADE_PROMPT, GRADE_SCHEMA, REWRITE_PROMPT,
                      REWRITE_SCHEMA, VERIFY_PROMPT, VERIFY_SCHEMA)
from .chunking import linearize_tables
from .providers import LLM, parse_json_object
from .retrieval import Index, tokenize


class RagState(TypedDict, total=False):
    question: str            # the user's question; never changed
    query: str               # query used for the current retrieval attempt
    retries: int             # corrective loops taken (rewrite, then search past graded passages)
    candidates: list[str]    # chunk ids from the latest retrieval not seen before, in rank order
    retrieved: list[str]     # every chunk id retrieved so far, first-seen rank order
    sufficient: bool | None  # grader decision for the latest attempt: at least one passage retained
    retained: list[str]      # chunk ids the grader kept for generation (each has evidence)
    evidence: dict[str, str] # chunk id -> the text the grader copied from it; why it was retained
    answer: str              # final answer text (or refusal text)
    citations: list[str]     # chunk ids the answer cites (subset of retained)
    verified: bool | None    # verifier result; None if not reached
    verify_failures: int     # failed verifications so far (regeneration loop)
    refused: bool
    termination: str | None
    note: str                # short reason for the last decision, for logs


def _passage_block(index: Index, ids: list[str], max_chars: int) -> str:
    """Passages as the model sees them, labelled P1, P2 ... in the order given.

    Short labels instead of chunk ids: a 3B model copies "P3" reliably, and on
    CPU every generated token costs about 0.15 s.
    """
    parts = []
    for n, cid in enumerate(ids, start=1):
        row = index.get(cid)
        text = _model_view(row.text)
        body = text if len(text) <= max_chars else text[:max_chars] + " ..."
        parts.append(f"[P{n}] ({row.file}, {row.section})\n{body}")
    return "\n\n".join(parts)


def _model_view(text: str) -> str:
    """The chunk as the model reads it: tables one row per line (see chunking.linearize_tables)."""
    return linearize_tables(text)


def _labels_to_ids(labels: Any, ids: list[str]) -> list[str]:
    """Map the model's passage labels ("P2", "2", 2) back to chunk ids, dropping unknown ones."""
    out: list[str] = []
    for label in labels if isinstance(labels, list) else []:
        m = re.fullmatch(r"\[?P?(\d+)\]?", str(label).strip(), flags=re.IGNORECASE)
        if m and 1 <= int(m.group(1)) <= len(ids) and ids[int(m.group(1)) - 1] not in out:
            out.append(ids[int(m.group(1)) - 1])
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _evidence_norm(text: str) -> str:
    """Lower-case, and fold whitespace and reST markup (backticks, emphasis, table bars)."""
    return re.sub(r"[\s`*|]+", " ", text).strip().lower()


def evidence_in_passage(evidence: str, passage: str, min_chars: int, match_chars: int) -> bool:
    """True when the grader's copied text really comes from the passage.

    The first ``match_chars`` characters of the evidence must appear verbatim
    (after normalisation) in the passage. A paraphrase, a summary or an answer
    from the model's memory fails this, so it cannot make context "sufficient".
    """
    ev = _evidence_norm(evidence)
    return len(ev) >= min_chars and ev[:match_chars] in _evidence_norm(passage)


def unsupported_identifiers(answer: str, cited_text: str) -> list[str]:
    """Backticked identifiers in the answer that no cited passage mentions.

    A qualified name counts as present if any dotted suffix of it is: the docs
    write ``dumps(obj, ...)`` under the ``json`` module, never ``json.dumps``.
    An invented name has no suffix in the text, so it is still caught.
    """
    cited = _norm(cited_text)
    missing = []
    for token in re.findall(r"`([^`]+)`", answer):
        name = _norm(token.split("(")[0]).strip()
        if not name:
            continue
        parts = name.split(".")
        if not any(".".join(parts[i:]) in cited for i in range(len(parts))):
            missing.append(token)
    return missing


def build_graph(cfg: dict[str, Any], index: Index, llm: LLM) -> Any:
    wf = cfg["workflow"]
    rcfg = cfg["retrieval"]
    max_retries = wf["max_retries"]
    max_verify = wf["max_verify_retries"]
    max_passages = wf["max_passages_to_model"]
    passage_chars = wf["max_passage_chars"]

    def retrieve(state: RagState) -> dict:
        # A retry never re-grades a passage: the grader judges against the original
        # question, so a passage rejected once would be rejected again. Skipping seen
        # passages makes each retry search deeper in the ranking at the same prompt size.
        top_k = rcfg["top_k"]
        query = state.get("query", state["question"])
        seen = list(state.get("retrieved", []))
        ranked = index.search(query, top_k + len(seen), rcfg["rrf_k"], rcfg["bm25_weight"])
        cands = [cid for cid in ranked if cid not in seen][:top_k]
        return {"candidates": cands, "retrieved": seen + cands}

    def grade(state: RagState) -> dict:
        # One passage per call, in rank order. The grader must copy the text that answers
        # the question; a passage is retained only if that text is found in it. Grading
        # stops once max_passages carry evidence: the generator needs no more, and on a
        # CPU every passage graded costs seconds.
        cands = state.get("candidates", [])
        retained: list[str] = []
        evidence: dict[str, str] = {}
        graded = 0
        for cid in cands:
            if len(retained) >= max_passages:
                break
            row = index.get(cid)
            shown = _model_view(row.text)[:passage_chars]
            prompt = GRADE_PROMPT.format(question=state["question"], file=row.file, section=row.section,
                                         passage=shown)
            reply = parse_json_object(llm.generate(prompt, schema=GRADE_SCHEMA)) or {}
            quote = str(reply.get("evidence") or "").strip()
            graded += 1
            if evidence_in_passage(quote, shown, wf["min_evidence_chars"], wf["evidence_match_chars"]):
                retained.append(cid)
                evidence[cid] = quote
        sufficient = bool(retained)
        note = (f"grader: {len(retained)} of {graded} graded passages carry evidence" if cands
                else "no passages retrieved")
        return {"sufficient": sufficient, "retained": retained, "evidence": evidence, "note": note}

    def recover(state: RagState) -> dict:
        retries = state.get("retries", 0) + 1
        prompt = REWRITE_PROMPT.format(question=state["question"], previous=state.get("query", state["question"]))
        rewritten = parse_json_object(llm.generate(prompt, schema=REWRITE_SCHEMA)) or {}
        query = str(rewritten.get("query") or "").strip()
        previous = state.get("query", state["question"])
        if _norm(query).lower() in ("", _norm(previous).lower(), _norm(state["question"]).lower()):
            # The model repeated the question. Fall back to its content words, which shifts
            # the BM25 ranking at least; the retry also searches past every passage seen so far.
            query = " ".join(dict.fromkeys(tokenize(state["question"])))
        return {"retries": retries, "query": query, "note": f"recover {retries}: {query[:80]}"}

    def generate(state: RagState) -> dict:
        ids = state.get("retained", [])
        evidence = state.get("evidence", {})
        # On a regeneration, say why the last answer failed; at temperature 0 the same
        # prompt would return the same answer and the retry would be a no-op.
        feedback = (f"A previous answer failed verification ({state.get('note', '')}). "
                    "Write a new answer that states only what the passages say.\n"
                    if state.get("verify_failures", 0) else "")
        prompt = GENERATE_PROMPT.format(question=state["question"],
                                        passages=_passage_block(index, ids, passage_chars),
                                        evidence="\n".join(f"- {evidence[c]}" for c in ids if c in evidence),
                                        feedback=feedback)
        reply = parse_json_object(llm.generate(prompt, schema=GENERATE_SCHEMA)) or {}
        answer = str(reply.get("answer", "")).strip()
        cited = _labels_to_ids(reply.get("citations"), ids)
        if not answer or answer == "NOT_IN_DOCS":
            return {"answer": "", "citations": [], "note": "generator: docs do not answer"}
        return {"answer": answer, "citations": cited, "note": "generated"}

    def verify(state: RagState) -> dict:
        answer = state.get("answer", "")
        cited = state.get("citations", [])
        retained = state.get("retained", [])
        failures = state.get("verify_failures", 0)
        problems = []
        if not cited or any(c not in retained for c in cited):
            problems.append("no valid citation")
        cited_text = " ".join(index.get(c).text for c in cited)
        problems += [f"identifier not in cited passages: {t}"
                     for t in unsupported_identifiers(answer, cited_text)]
        llm_ok = False
        if not problems:
            prompt = VERIFY_PROMPT.format(question=state["question"], answer=answer,
                                          passages=_passage_block(index, cited, passage_chars))
            verdict = parse_json_object(llm.generate(prompt, schema=VERIFY_SCHEMA)) or {}
            llm_ok = bool(verdict.get("supported"))
            if not llm_ok:
                problems.append("unsupported claim: " + str(verdict.get("unsupported", ""))[:120])
        verified = not problems
        return {"verified": verified, "verify_failures": failures if verified else failures + 1,
                "note": "verified" if verified else "; ".join(problems)}

    def refuse(state: RagState) -> dict:
        n = len(state.get("retrieved", []))
        if state.get("sufficient"):
            termination = "refused_no_answer"  # grader kept passages, generator found no answer in them
        elif state.get("retrieved") and state.get("retries", 0) >= max_retries:
            termination = "retry_limit"
        else:
            termination = "refused_no_context"
        text = (f"The documentation held by this assistant does not cover this question. "
                f"Searched for: \"{state.get('query', state['question'])}\". "
                f"Checked {n} passages; none supported an answer.")
        return {"answer": text, "citations": [], "refused": True, "verified": state.get("verified"),
                "termination": termination}

    def finish(state: RagState) -> dict:
        return {"refused": False, "termination": "answered"}

    def finish_unverified(state: RagState) -> dict:
        answer = ("Verification failed for this answer, so treat it with caution. "
                  + state.get("answer", ""))
        return {"answer": answer, "refused": False, "termination": "refused_unverified"}

    def route_after_grade(state: RagState) -> str:
        if state.get("sufficient"):
            return "generate"
        if state.get("retries", 0) < max_retries:
            return "recover"
        return "refuse"

    def route_after_generate(state: RagState) -> str:
        return "verify" if state.get("answer") else "refuse"

    def route_after_verify(state: RagState) -> str:
        if state.get("verified"):
            return "finish"
        if state.get("verify_failures", 0) <= max_verify:
            return "generate"
        return "finish_unverified"

    g = StateGraph(RagState)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("recover", recover),
                     ("generate", generate), ("verify", verify), ("refuse", refuse),
                     ("finish", finish), ("finish_unverified", finish_unverified)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "recover": "recover", "refuse": "refuse"})
    g.add_edge("recover", "retrieve")
    g.add_conditional_edges("generate", route_after_generate, {"verify": "verify", "refuse": "refuse"})
    g.add_conditional_edges("verify", route_after_verify,
                            {"finish": "finish", "generate": "generate",
                             "finish_unverified": "finish_unverified"})
    for node in ("refuse", "finish", "finish_unverified"):
        g.add_edge(node, END)
    # The counters above bound every loop; the step limit is a second, independent net.
    return g.compile().with_config({"recursion_limit": wf["graph_step_limit"]})


def _trace_view(update: dict[str, Any]) -> dict[str, Any]:
    """What a trace line keeps from a node's update: decisions and ids, not passage text."""
    return {k: v for k, v in update.items() if k not in ("retrieved", "question")}


def run_question(graph: Any, question: str) -> dict[str, Any]:
    """Run the compiled graph once and return the result in the results.jsonl shape.

    The result also carries ``trace``: one entry per node visit, in order, with
    the node's wall-clock seconds and what it wrote to state.
    """
    from langgraph.errors import GraphRecursionError

    trace: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    last = time.perf_counter()
    try:
        for mode, chunk in graph.stream({"question": question, "query": question, "retries": 0,
                                         "verify_failures": 0, "retrieved": []},
                                        stream_mode=["updates", "values"]):
            if mode == "values":
                out = chunk
                continue
            now = time.perf_counter()
            for node, update in chunk.items():
                trace.append({"node": node, "seconds": round(now - last, 3), **_trace_view(update or {})})
            last = now
    except GraphRecursionError:
        return {"answer": "The request stopped at the workflow step limit.", "refused": True,
                "citations": [], "retrieved": out.get("retrieved", []), "retained": [], "verified": None,
                "retries": out.get("retries", 0), "termination": "graph_limit", "trace": trace}
    return {
        "trace": trace,
        "answer": out.get("answer", ""),
        "refused": bool(out.get("refused", False)),
        "citations": out.get("citations", []),
        "retrieved": out.get("retrieved", []),
        "retained": out.get("retained", []),
        "verified": out.get("verified"),
        "retries": out.get("retries", 0),
        "termination": out.get("termination") or "unknown",
    }


def render_mermaid(graph: Any) -> str:
    return graph.get_graph().draw_mermaid()
