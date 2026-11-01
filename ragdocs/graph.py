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
* ``refused_no_context``  nothing retrieved, or the model found no answer in context
* ``retry_limit``         context judged insufficient after all corrective loops
* ``refused_unverified``  answer returned after verification failed (failure disclosed)
* ``graph_limit``         safety net: LangGraph step limit reached (should not happen)

Every loop has a counter in state and a bound from config, so each path ends.
"""
from __future__ import annotations

import re
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, VERIFY_PROMPT
from .providers import LLM, parse_json_object
from .retrieval import Index


class RagState(TypedDict, total=False):
    question: str            # the user's question; never changed
    query: str               # query used for the current retrieval attempt
    top_k: int               # passages requested in the current attempt
    retries: int             # corrective loops taken (rewrite / widen retrieval)
    candidates: list[str]    # chunk ids from the latest retrieval, in rank order
    retrieved: list[str]     # every chunk id seen so far, first-seen rank order
    sufficient: bool | None  # grader decision for the latest attempt
    retained: list[str]      # chunk ids the grader kept for generation
    answer: str              # final answer text (or refusal text)
    citations: list[str]     # chunk ids the answer cites (subset of retained)
    verified: bool | None    # verifier result; None if not reached
    verify_failures: int     # failed verifications so far (regeneration loop)
    refused: bool
    termination: str | None
    note: str                # short reason for the last decision, for logs


def _passage_block(index: Index, ids: list[str], max_chars: int = 1500) -> str:
    parts = []
    for cid in ids:
        row = index.get(cid)
        body = row.text if len(row.text) <= max_chars else row.text[:max_chars] + " ..."
        parts.append(f"[{cid}] ({row.file}, {row.section})\n{body}")
    return "\n\n".join(parts)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def build_graph(cfg: dict[str, Any], index: Index, llm: LLM) -> Any:
    wf = cfg["workflow"]
    rcfg = cfg["retrieval"]
    max_retries = wf["max_retries"]
    max_verify = wf["max_verify_retries"]
    max_passages = wf["max_passages_to_model"]

    def retrieve(state: RagState) -> dict:
        top_k = state.get("top_k", rcfg["top_k"])
        query = state.get("query", state["question"])
        cands = index.search(query, top_k, rcfg["rrf_k"])
        seen = list(state.get("retrieved", []))
        for cid in cands:
            if cid not in seen:
                seen.append(cid)
        return {"candidates": cands, "retrieved": seen, "top_k": top_k}

    def grade(state: RagState) -> dict:
        cands = state.get("candidates", [])
        if not cands:
            return {"sufficient": False, "retained": [], "note": "no passages retrieved"}
        prompt = GRADE_PROMPT.format(question=state["question"], passages=_passage_block(index, cands))
        decision = parse_json_object(llm.generate(prompt, json_mode=True)) or {}
        relevant = [c for c in decision.get("relevant", []) if c in cands]
        sufficient = bool(decision.get("sufficient")) and bool(relevant)
        retained = [c for c in cands if c in relevant][:max_passages]
        return {"sufficient": sufficient, "retained": retained if sufficient else [],
                "note": "grader: sufficient" if sufficient else "grader: insufficient"}

    def recover(state: RagState) -> dict:
        retries = state.get("retries", 0) + 1
        prompt = REWRITE_PROMPT.format(question=state["question"], previous=state.get("query", state["question"]))
        rewritten = parse_json_object(llm.generate(prompt, json_mode=True)) or {}
        query = str(rewritten.get("query") or "").strip() or state["question"]
        # Second recovery attempt also widens the retrieval depth.
        top_k = rcfg["widen_top_k"] if retries >= 2 else state.get("top_k", rcfg["top_k"])
        return {"retries": retries, "query": query, "top_k": top_k, "note": f"recover {retries}: {query[:80]}"}

    def generate(state: RagState) -> dict:
        ids = state.get("retained", [])
        prompt = GENERATE_PROMPT.format(question=state["question"], passages=_passage_block(index, ids))
        reply = parse_json_object(llm.generate(prompt, json_mode=True)) or {}
        answer = str(reply.get("answer", "")).strip()
        cited = [c for c in reply.get("citations", []) if c in ids]
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
        cited_text = _norm(" ".join(index.get(c).text for c in cited)) if cited else ""
        for token in re.findall(r"`([^`]+)`", answer):
            needle = _norm(token.rstrip("()").strip())
            if needle and needle not in cited_text:
                problems.append(f"identifier not in cited passages: {token}")
        llm_ok = False
        if not problems:
            prompt = VERIFY_PROMPT.format(answer=answer, passages=_passage_block(index, cited))
            verdict = parse_json_object(llm.generate(prompt, json_mode=True)) or {}
            llm_ok = bool(verdict.get("supported"))
            if not llm_ok:
                problems.append("unsupported claim: " + str(verdict.get("unsupported", ""))[:120])
        verified = not problems
        return {"verified": verified, "verify_failures": failures if verified else failures + 1,
                "note": "verified" if verified else "; ".join(problems)}

    def refuse(state: RagState) -> dict:
        n = len(state.get("retrieved", []))
        if state.get("retrieved"):
            termination = "retry_limit" if state.get("retries", 0) >= max_retries else "refused_no_context"
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
    return g.compile()


def run_question(graph: Any, question: str, recursion_limit: int = 60) -> dict[str, Any]:
    """Run the compiled graph once and return the result in the results.jsonl shape."""
    from langgraph.errors import GraphRecursionError

    try:
        out = graph.invoke({"question": question, "query": question, "retries": 0,
                            "verify_failures": 0, "retrieved": []},
                           config={"recursion_limit": recursion_limit})
    except GraphRecursionError:
        return {"answer": "The request stopped at the workflow step limit.", "refused": True,
                "citations": [], "retrieved": [], "retained": [], "verified": None,
                "retries": 0, "termination": "graph_limit"}
    return {
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
