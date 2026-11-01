"""Prompt templates. Kept in one place so they can be reviewed and versioned.

Passages come first and the instruction last: a 3B model follows the most
recent text best. The grader sees one passage per call; the generator sees
the retained passages labelled P1, P2 ... (see graph._passage_block).

Each prompt has a JSON schema. Ollama constrains decoding to it, so a reply
always parses and the model cannot drift into whitespace or prose.
"""


def _schema(**props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


_LABELS = {"type": "array", "items": {"type": "string"}}
GRADE_SCHEMA = _schema(evidence={"type": "string"})
REWRITE_SCHEMA = _schema(query={"type": "string"})
GENERATE_SCHEMA = _schema(answer={"type": "string"}, citations=_LABELS)
VERIFY_SCHEMA = _schema(supported={"type": "boolean"}, unsupported={"type": "string"})

GRADE_PROMPT = """PASSAGE ({file}, {section}):
{passage}

QUESTION: {question}

Copy from the PASSAGE the one sentence or table row that answers the QUESTION, word for word,
at most 40 words. Copy prose, not a code example. If the passage does not answer the question,
reply with an empty string. Do not use your own knowledge.
Reply with JSON only: {{"evidence": "<copied text, or empty>"}}
"""

REWRITE_PROMPT = """The search below found no passage that answers the QUESTION. Write a different,
shorter search query: the key terms only, 3 to 8 words, no full sentence. Keep exact identifiers,
module names and format codes (for example json.dumps, %j). Do not repeat the question or the
previous query.
Reply with JSON only: {{"query": "<search query>"}}

QUESTION: {question}
PREVIOUS QUERY: {previous}
"""

GENERATE_PROMPT = """PASSAGES:
{passages}

EVIDENCE (sentences a reviewer found in these passages for this question):
{evidence}

QUESTION: {question}

{feedback}Answer the QUESTION using only the passages above, in one or two sentences. An example in a
passage counts as documentation. If the passages do not contain the answer, reply with
{{"answer": "NOT_IN_DOCS", "citations": []}}.
Put any code identifier you mention in backticks, for example `json.dumps`.
Reply with JSON only: {{"answer": "<short answer>", "citations": ["P1", ...]}}
"citations" lists the labels of the passages the answer is taken from.
"""

VERIFY_PROMPT = """PASSAGES:
{passages}

QUESTION: {question}
ANSWER: {answer}

Check the ANSWER against the PASSAGES above. Every factual claim in the answer must be stated
in, or directly follow from, the passages. Reply with JSON only:
{{"supported": true or false, "unsupported": "<the unsupported claim, or empty>"}}
"""
