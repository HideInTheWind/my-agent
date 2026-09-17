from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from agent.tools.utils import _extract_content

from .state import AgentState

REWRITE_PROMPT = """
你是问题重写助手；职责：将问题换一种方式表述出来，用来检索文档；

## 规则

- 可参考的历史对话：{history}
"""


def build_rewrite_node(llm: ChatOpenAI):
    def rewrite_node(state: AgentState):
        msgs = state["messages"]
        query = state.get("rewrite_query", "")
        history = "\n".join(
            [
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}：{_extract_content(m.content)}"
                for m in msgs[:-1]
            ]
        )
        prompt = REWRITE_PROMPT.format(
            history=history,
        )
        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=query)]
        )
        return Command(
            goto="rag",
            update={
                "rewrite_query": response.content.strip(),
                "retry_count": state.get("retry_count", 0) + 1,
            },
        )

    return rewrite_node
