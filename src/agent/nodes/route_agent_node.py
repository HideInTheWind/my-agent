from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel

from .state import AgentState

ROUTE_AGENT_SYSTEM_PROMPT = (
    "你是路由决策器，根据用户问题选择一条路由（只能选一个）：\n"
    "- rag：涉及公司内部制度、HR 政策、专有知识库的问题（包括年假、请假、薪资等）\n"
    # "- web：问今天/最新/实时数据的问题\n"
    "- direct：需要计算、创作、解释、或使用通用知识回答的问题\n"
    "- refuse：明确有害、涉及违法、或包含攻击性内容的请求\n"
    "注意：拿不准时优先选 rag 或 direct，不要轻易选 refuse。"
)


class RouteAgentState(BaseModel):
    route: Literal["rag", "direct", "refuse"]  # "web",


def build_route_agent_node(llm: ChatOpenAI):
    def route_agent(state: AgentState):
        query = state["rewrite_query"]
        messages = [
            SystemMessage(content=ROUTE_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = llm.with_structured_output(
            RouteAgentState, method="function_calling"
        ).invoke(messages)

        route = response.route
        return Command(
            goto=route,
            update={
                "route": route,
            },
        )

    return route_agent
