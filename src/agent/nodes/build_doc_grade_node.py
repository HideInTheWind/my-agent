from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from .state import AgentState

DOC_GRADE_PROMPT = """
你是文档评分助手；职责：根据文档内容，评分文档的质量；

- 检索内容：{context}
- 评分要求：
  - 如果检索内容与问题相关，则评分'relevant'
  - 如果检索内容与问题无关，则评分'irrelevant'

"""


class DocGradeResult(BaseModel):
    grade: str = Field(description="文档质量等级，只输出'relevant'或'irrelevant'")
    reason: str = Field(description="评分理由")


def build_doc_grade_node(llm: ChatOpenAI):
    def doc_grade_node(state: AgentState):
        query = state.get("rewrite_query", "")
        prompt = DOC_GRADE_PROMPT.format(context=state["context"])
        response = llm.with_structured_output(
            DocGradeResult, method="function_calling"
        ).invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        if state.get("retry_count", 0) >= 3:
            goto = "output_guard"

        else:
            goto = (
                "generate_answer" if response.grade == "relevant" else "rewrite_query"
            )

        return Command(
            goto=goto,
            update={
                "doc_grade": response.grade.strip(),
            },
        )

    return doc_grade_node
