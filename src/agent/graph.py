"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from redisvl.extensions.cache.llm import SemanticCache

from .mcp_client import MCP_Gateway, MCPClientConfig
from .nodes import (
    AgentState,
    build_cache_node,
    build_condense_query_node,
    build_direct_node,
    build_doc_grade_node,
    build_generate_answer_node,
    build_hallucination_check_node,
    build_history_trigger_node,
    build_input_check_node,
    build_output_guard_node,
    build_quality_check_node,
    build_rag_node,
    build_rewrite_node,
    build_route_agent_node,
    refuse_node,
)


async def build_graph(
    llm: ChatOpenAI,
    *,
    semantic_cache: SemanticCache | None,
    config: MCPClientConfig | None = None,
    checkpointer=None,
):
    gateway = MCP_Gateway(config)
    tools = await gateway.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    builder = StateGraph(AgentState)

    builder.add_node("input_check", build_input_check_node(llm))
    builder.add_node("history_trigger", build_history_trigger_node(llm))
    builder.add_node("condense_query", build_condense_query_node(llm))
    builder.add_node("routeAgent", build_route_agent_node(llm))

    builder.add_node("refuse", refuse_node)
    builder.add_node("cache", build_cache_node(semantic_cache))
    builder.add_node("direct", build_direct_node(llm))
    builder.add_node(
        "rag",
        build_rag_node(
            llm_with_tools,
        ),
    )
    builder.add_node("tools", tool_node)
    builder.add_node("rewrite_query", build_rewrite_node(llm))
    builder.add_node("doc_grade", build_doc_grade_node(llm))
    builder.add_node("generate_answer", build_generate_answer_node(llm))
    builder.add_node("hallucination_check", build_hallucination_check_node(llm))
    builder.add_node("quality_check", build_quality_check_node(llm))
    builder.add_node("output_guard", build_output_guard_node(llm, semantic_cache))

    builder.add_edge(START, "input_check")
    builder.add_edge("tools", "rag")

    graph = builder.compile(checkpointer=checkpointer)
    return graph
