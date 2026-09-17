from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    is_safe: bool
    reason: str

    condensed_query: str

    route: Literal["direct", "rag", "refuse"]  # "web",
    is_from_cache: bool
    cache_score: float

    doc_grade: Literal[
        "relevant",
        "irrelevant",
    ]
    rewrite_query: str
    context: str

    hallucination: bool
    quality_check: bool

    retry_count: int
    answer: str
