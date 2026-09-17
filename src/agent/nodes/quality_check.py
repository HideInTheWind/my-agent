from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from .state import AgentState

QUALITY_CHECK_PROMPT = """
你是质量检查助手；职责：检查答案是否符合质量要求；

# 规则
- 问题：{question}
- 答案：{answer}
- 评判标准：答案是否回答了问题，如果回答了问题，则返回'quality'，否则返回'no_quality'
"""


class QualityCheckResult(BaseModel):
    result: str = Field(description="质量检查结果，只输出'quality'或'no_quality'")
    reason: str = Field(description="质量检查理由")


def build_quality_check_node(llm: ChatOpenAI):
    def quality_check_node(state: AgentState):
        prompt = QUALITY_CHECK_PROMPT.format(
            question=state["rewrite_query"], answer=state["answer"]
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["rewrite_query"]),
        ]
        response = llm.with_structured_output(
            QualityCheckResult, method="function_calling"
        ).invoke(messages)
        retry_count = state.get("retry_count", 0)
        if retry_count >= 3:
            goto = "output_guard"

        else:
            goto = "output_guard" if response.result == "quality" else "rewrite_query"

        return Command(
            goto=goto,
            update={
                "quality_check": response.result == "quality",
            },
        )

    return quality_check_node
