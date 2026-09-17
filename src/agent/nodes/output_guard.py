from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command
from redisvl.extensions.cache.llm import SemanticCache

from ..security.pipline import check_output
from ..tools.utils import _last_user_text
from .state import AgentState


def build_output_guard_node(llm: ChatOpenAI, semantic_cache: SemanticCache | None):
    def output_guard_node(state: AgentState):
        result = check_output(state["answer"], llm)
        safe = result.safe
        reason = result.reason
        if not safe:
            return Command(goto="refuse", update={"reason": reason, "is_safe": safe})
        else:
            if semantic_cache is not None:
                # 存入 key 与 cache_node 查找逻辑保持一致，避免 key 不匹配导致缓存永远 miss
                question = (
                    state.get("condensed_query")
                    or state.get("rewrite_query")
                    or _last_user_text(state["messages"])
                )
                semantic_cache.set(question, state["answer"])
            return Command(
                goto=END,
                update={
                    "messages": [HumanMessage(content=state["answer"])],
                    "is_safe": True,
                    "reason": "",
                },
            )

    return output_guard_node
