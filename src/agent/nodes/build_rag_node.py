from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from agent.tools.utils import _extract_content, _last_user_text

from .state import AgentState

# 混合检索->粗召回->精排->文档打分

RAG_SYSTEM = (
    "你是文档问答助手，必须调用搜索工具检索文档后再回答，不得凭自身知识直接作答。"
)


def build_rag_node(
    llm_with_tools: ChatOpenAI,
):

    def rag_node(state: AgentState):
        rewrite = state.get("rewrite_query", "")
        last_q = rewrite or _last_user_text(state["messages"])
        state_msgs = state["messages"]
        last_tool_call_idx = next(
            (
                i
                for i in range(len(state_msgs) - 1, -1, -1)
                if isinstance(state_msgs[i], AIMessage)
                and getattr(state_msgs[i], "tool_calls", None)
            ),
            None,
        )
        current_round_msgs = (
            state_msgs[last_tool_call_idx:] if last_tool_call_idx is not None else []
        )
        msgs = [
            SystemMessage(content=RAG_SYSTEM),
            HumanMessage(content=last_q),
        ] + current_round_msgs
        response = llm_with_tools.invoke(msgs)
        # new_retry_count = state.get("retry_count", 0) + 1

        if getattr(response, "tool_calls", None):
            return Command(
                goto="tools",
                update={
                    "messages": [response],
                    # "retry_count": new_retry_count,
                },
            )

        elif not getattr(response, "tool_calls", None) and response.content:
            context = "\n".join(
                [
                    _extract_content(m.content)
                    for m in current_round_msgs
                    if isinstance(m, ToolMessage)
                ]
            )
            if context:
                goto = "doc_grade"
            elif state.get("retry_count", 0) < 3:
                goto = "rewrite_query"
            else:
                goto = "direct"
            return Command(
                goto=goto,
                update={
                    "context": context,
                },
            )
        else:
            goto = "rewrite_query" if state.get("retry_count", 0) < 3 else "direct"
            return Command(
                goto=goto,
            )

    return rag_node
