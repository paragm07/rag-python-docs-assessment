"""Prompt templates. Kept in one place so they can be reviewed and versioned."""

GRADE_PROMPT = """You decide whether the passages below are enough to answer the question.
Use only the passages. Do not use your own knowledge.
Reply with JSON only: {{"sufficient": true or false, "relevant": ["<passage id>", ...]}}
"relevant" lists the ids of passages that contain the information needed. Set "sufficient"
to true only if those passages fully answer the question.

QUESTION: {question}

PASSAGES:
{passages}
"""

REWRITE_PROMPT = """Rewrite the question as a short search query for Python documentation.
Keep exact identifiers, module names and format codes (for example json.dumps, %j).
Reply with JSON only: {{"query": "<search query>"}}

QUESTION: {question}
PREVIOUS QUERY (did not find an answer): {previous}
"""

GENERATE_PROMPT = """Answer the question using only the passages. If the passages do not contain
the answer, reply with {{"answer": "NOT_IN_DOCS", "citations": []}}.
Put any code identifier you mention in backticks, for example `json.dumps`.
Cite the passage ids you used. Reply with JSON only:
{{"answer": "<short answer>", "citations": ["<passage id>", ...]}}

QUESTION: {question}

PASSAGES:
{passages}
"""

VERIFY_PROMPT = """Check the ANSWER against the PASSAGES. Every factual claim in the answer must be
stated in, or directly follow from, the passages. Reply with JSON only:
{{"supported": true or false, "unsupported": "<the unsupported claim, or empty>"}}

ANSWER: {answer}

PASSAGES:
{passages}
"""
