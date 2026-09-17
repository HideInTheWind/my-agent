from langgraph.types import Command
from redisvl.extensions.cache.llm import SemanticCache

from ..tools.utils import _last_user_text
from .state import AgentState


def build_cache_node(semantic_cache: SemanticCache | None):
    def cache_node(state: AgentState):
        if semantic_cache is None:
            return Command(
                goto="routeAgent",
                update={
                    "is_from_cache": False,
                    "cache_score": 0.0,
                },
            )
        query = (
            state["condensed_query"]
            or state["rewrite_query"]
            or _last_user_text(state["messages"])
        )
        answer, score = semantic_cache.get(query)
        if not answer:
            return Command(
                goto="routeAgent",
                update={
                    "is_from_cache": False,
                    "cache_score": 0.0,
                },
            )
        return Command(
            goto="output_guard",
            update={
                "answer": answer,
                "is_from_cache": True,
                "cache_score": score,
            },
        )

    return cache_node
