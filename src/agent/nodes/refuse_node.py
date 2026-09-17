from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from .state import AgentState

GUIDED_REFUSAL = "我目前无法确认这个问题的答案，建议您换个方式提问或咨询相关专业人士。"


def refuse_node(state: AgentState):
    return Command(
        goto=END,
        update={
            "answer": GUIDED_REFUSAL,
            "messages": [HumanMessage(content=GUIDED_REFUSAL)],
        },
    )
