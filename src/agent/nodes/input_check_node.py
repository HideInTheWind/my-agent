from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ..security import check_input
from ..tools.utils import _last_user_text
from .state import AgentState


def build_input_check_node(llm: ChatOpenAI):
    def input_check_node(state: AgentState):
        query = _last_user_text(state["messages"])
        result = check_input(query, llm)
        safe = result.safe
        reason = result.reason
        goto = "history_trigger" if safe else "refuse"
        return Command(
            goto=goto,
            update={
                "reason": reason,
                "is_safe": safe,
            },
        )

    return input_check_node
