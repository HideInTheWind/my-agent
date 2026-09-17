from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ..tools.utils import _last_user_text
from .state import AgentState

CONDENSE_QUERY_SYSTEM_PROMPT = """
你是问题凝缩专家。

## 对话历史
{history_text}

## 用户当前问题
{query}

## 你的任务
结合对话历史，将用户当前问题改写为一个**自含义、完整的独立问题**（不依赖上下文即可理解）。
- 如果用户的问题本身已经完整清晰，则**原样返回**，不要添加任何内容。
- 只输出改写后的问题，不要输出解释。
"""


def build_condense_query_node(llm: ChatOpenAI):
    def condense_query(state: AgentState):

        msgs = state["messages"]
        query = _last_user_text(msgs)
        non_system_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
        history_text = "\n".join(
            [
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{m.content}"
                for m in non_system_msgs[:-1]
            ]
        )
        prompt = CONDENSE_QUERY_SYSTEM_PROMPT.format(
            history_text=history_text, query=query
        )
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return Command(
            goto="cache",
            update={
                "condensed_query": response.content,
                "rewrite_query": response.content,
            },
        )

    return condense_query
