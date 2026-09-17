from langchain_openai import ChatOpenAI
from langgraph.types import Command

from .state import AgentState


def build_direct_node(llm: ChatOpenAI):
    def direct_node(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return Command(
            goto="output_guard",
            update={
                "answer": response.content.strip(),
            },
        )

    return direct_node
