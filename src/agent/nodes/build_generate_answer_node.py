from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from .state import AgentState

GENERATE_ANSWER_SYSTEM_PROMPT = """你是知识助手；职责：根据检索到的内容，回答用户的问题；

## 规则
- 只根据"检索内容"作答，不要编造检索内容以外的信息；
- 如果检索内容不足以回答问题，明确告知用户"暂无相关信息"；
- 回答要简洁、准确；
"""


def build_generate_answer_node(llm: ChatOpenAI):
    def generate_answer_node(state: AgentState):
        query = state.get("rewrite_query", "")
        context = state.get("context", "")

        messages = [
            SystemMessage(content=GENERATE_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"检索内容：\n{context}\n\n用户问题：{query}"),
        ]
        response = llm.invoke(messages)
        return Command[str](
            goto="hallucination_check",
            update={
                "answer": response.content.strip(),
            },
        )

    return generate_answer_node
