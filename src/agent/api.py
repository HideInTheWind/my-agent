import json
import os
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.mcp_client import MCPClientConfig

from .cache import create_semantic_cache_from_env
from .graph import build_graph
from .tools.load_env import validate_env

system_prompt = "你是知识助手，请基于检索结果准确回答用户问题。"
SAFE_REFUSAL = "抱歉，我无法处理该请求。请换一个问题试试。"


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = validate_env()
    (
        api_key,
        api_url,
        model,
        embedding_api_key,
        embedding_api_url,
        embedding_model,
    ) = env

    app.state.llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=api_url, streaming=True
    )
    embeddings = OpenAIEmbeddings(
        api_key=embedding_api_key,
        base_url=embedding_api_url,
        model=embedding_model,
    )
    app.state.semantic_cache = create_semantic_cache_from_env(embeddings)

    config = MCPClientConfig(allowed_tools=["rag_search"])
    pg_conn = await psycopg.AsyncConnection.connect(
        os.environ["DATABASE_URL"], autocommit=True
    )
    checkpointer = AsyncPostgresSaver(pg_conn)
    await checkpointer.setup()  # 首次运行建表
    app.state.agent = await build_graph(
        app.state.llm,
        semantic_cache=app.state.semantic_cache,
        config=config,
        checkpointer=checkpointer,
    )
    yield
    app.state.agent = None
    app.state.semantic_cache = None
    await pg_conn.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():

    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):

    if request.app.state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    cache_ok = (
        request.app.state.semantic_cache is None
        or request.app.state.semantic_cache.ping()
    )

    return {"status": "ready", "cache_enabled": cache_ok}


class ChatRequest(BaseModel):
    user_input: str = Field(description="用户输入", min_length=1, max_length=2000)
    session_id: str = Field(description="会话ID", default="default")


class ChatResponse(BaseModel):
    answer: str = Field(description="回答")

    blocked: bool = Field(description="是否拦截", default=False)

    block_reason: str = Field(description="拦截原因", default="")

    cached: bool = Field(description="是否命中语义缓存", default=False)
    cache_score: float = Field(description="缓存相似度分数", default=0.0)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    agent = req.app.state.agent
    init_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.user_input),
        ],
        "is_safe": False,
        "reason": "",
        "condensed_query": "",
        "route": "direct",
        "is_from_cache": False,
        "doc_grade": "relevant",
        "rewrite_query": "",
        "context": "",
        "hallucination": False,
        "quality_check": False,
        "retry_count": 0,
        "answer": "",
    }
    result = await agent.ainvoke(
        init_state,
        config={"configurable": {"thread_id": request.session_id}},
    )
    if not result.get("is_safe"):
        return ChatResponse(
            answer=SAFE_REFUSAL,
            blocked=True,
            block_reason=result.get("reason", ""),
        )
    return ChatResponse(
        answer=result.get("answer", ""),
        cached=result.get("is_from_cache", False),
        cache_score=result.get("cache_score", 0.0),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    agent = req.app.state.agent
    node_mappings = {
        "input_check": {"is_safe": False, "reason": ""},
        "history_trigger": None,
        "condense_query": {"condensed_query": ""},
        "cache": {"is_from_cache": False, "cache_score": "", "answer": ""},
        "output_guard": {"is_safe": False, "reason": ""},
        "routeAgent": {"route": ""},
        "rag": None,
        "refuse": None,
        "doc_grade": {"doc_grade": ""},
        "rewrite_query": {"rewrite_query": "", "retry_count": 0},
        "hallucination": {"hallucination": False},
        "quality_check": {"quality_check": False},
    }
    init_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.user_input),
        ],
        "is_safe": False,
        "reason": "",
        "condensed_query": "",
        "route": "direct",
        "is_from_cache": False,
        "doc_grade": "relevant",
        "rewrite_query": "",
        "context": "",
        "hallucination": False,
        "quality_check": False,
        "retry_count": 0,
        "answer": "",
    }

    async def event_generator():
        accumulated = {
            "is_safe": False,
            "reason": "",
            "answer": "",
            "is_from_cache": False,
            "cache_score": 0.0,
        }
        ANSWER_NODES = {"generate_answer", "direct"}
        # token_buffer: list[str] = []
        # tokens_sent = False  # 是否已向客户端发过 token
        # pending_retract = False  # 下一个 token 前是否需要先发 retract 帧
        state = {"tokens_sent": False, "pending_retract": False}
        # ⚠️ 注意：pending_retract 和 tokens_sent 是闭包变量，在 async def event_generator() 内部修改时要用 nonlocal 声明，或者直接用列表/字典包装（推荐）来避免 Python 闭包坑：

        async for event in agent.astream_events(
            init_state,
            config={
                "configurable": {"thread_id": request.session_id},
                "recursion_limit": 60,
            },
            version="v2",
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # ① 立即流式推送 token
            if kind == "on_chat_model_stream" and node in ANSWER_NODES:
                token = event["data"]["chunk"].content
                if token:
                    if state["pending_retract"]:
                        yield f"data: {json.dumps({'type': 'retract'}, ensure_ascii=False)}\n\n"
                        state["pending_retract"] = False
                    yield f"data: {json.dumps({'type': 'token', 'node': node, 'token': token}, ensure_ascii=False)}\n\n"
                    state["tokens_sent"] = True

                    # yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            # ② 节点完成 —— 检测 rewrite_query
            elif kind == "on_chain_end" and node:
                # ✅ 新增：检测重写节点
                if node in ANSWER_NODES:
                    continue
                if node == "rewrite_query" and state["tokens_sent"]:
                    state["pending_retract"] = True
                output = event["data"].get("output")
                if isinstance(output, Command):
                    update = output.update or {}
                elif isinstance(output, dict):
                    update = output
                else:
                    continue
                for k, v in update.items():
                    if k != "messages":
                        accumulated[k] = v
                releting_to_node_keys = node_mappings.get(node, None)
                frame = {
                    "type": "chain",
                    "node": node,
                }
                if releting_to_node_keys:
                    keys = releting_to_node_keys.keys()
                    for key in keys:
                        frame[key] = update.get(key, "")

                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tools', 'node': node}, ensure_ascii=False)}\n\n"
        # ③ 图跑完，发最终帧
        is_safe = accumulated.get("is_safe", True)
        if not is_safe:
            yield f"data: {json.dumps({'type': 'retract', 'final': SAFE_REFUSAL}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'verified'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # 把「生成器函数」变成「生成器对象」
    return StreamingResponse(event_generator(), media_type="text/event-stream")


class CacheStatsResponse(BaseModel):
    hit: int
    miss: int
    stores: int
    total: int
    hit_rate: float


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(req: Request):
    cache = req.app.state.semantic_cache
    if cache is None:
        raise HTTPException(status_code=503, detail="语义缓存服务未初始化")
    return cache.stats()


class CacheClearResponse(BaseModel):
    message: str


@app.delete(
    "/cache/clear",
    response_model=CacheClearResponse,
)
async def cache_clear(req: Request):
    cache = req.app.state.semantic_cache
    if cache is None:
        raise HTTPException(status_code=503, detail="语义缓存服务未初始化")
    cache.clear()
    return CacheClearResponse(message="Cache cleared")
