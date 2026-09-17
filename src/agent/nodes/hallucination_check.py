from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from .state import AgentState

HALLUCINATION_CHECK_PROMPT = """
你是幻觉检查助手；职责：检查答案是否产生幻觉；

# 规则
- 上下文：{context}
- 评判标准：答案{answer}采取了上下文中的内容，则返回'no_hallucination'，否则返回'hallucination'
"""


class HallucinationCheckResult(BaseModel):
    result: str = Field(
        description="幻觉检查结果，只输出'hallucination'或'no_hallucination'"
    )
    reason: str = Field(description="幻觉检查理由")


def build_hallucination_check_node(llm: ChatOpenAI):
    def hallucination_check_node(state: AgentState):
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            answer=state["answer"], context=state["context"]
        )
        response = llm.with_structured_output(
            HallucinationCheckResult, method="function_calling"
        ).invoke([SystemMessage(content=prompt), HumanMessage(content=state["answer"])])

        retry_count = state.get("retry_count", 0)
        if response.result == "no_hallucination":
            goto = "quality_check"
        elif retry_count >= 3:
            goto = "output_guard"
        else:
            goto = "rewrite_query"
        return Command(
            goto=goto,
            update={
                "hallucination": response.result == "hallucination",
            },
        )

    return hallucination_check_node
