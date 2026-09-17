# 企业知识库智能问答 Agent

[![Unit Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

> 🚧 代码持续迭代中，功能与接口可能随时变更。

---

## 项目背景

企业内部沉淀了大量制度文档（HR 手册、考勤规范、薪资福利、IT 安全等），但员工需要花费大量时间翻阅才能找到答案，且信息散落在多个文件中，难以快速定位。

本项目为 **云言视界科技有限公司** 打造一套企业级 RAG 智能问答 Agent，解决以下核心痛点：

| 痛点 | 解决方案 |
|------|----------|
| 员工查询制度耗时费力 | 自然语言提问，秒级返回精准答案 |
| 多轮对话上下文丢失 | PostgreSQL 持久化会话记忆 |
| 重复问题反复调用 LLM | Redis 语义缓存，相似问题直接命中 |
| 回答内容不可控、有风险 | 输入/输出双重安全检查 + 敏感词过滤 |
| 检索召回率不足 | 向量检索 + BM25 混合召回 + Rerank 重排序 |

---

## 技术架构

### 请求处理全链路

```mermaid
flowchart TD
    User["用户请求"] --> API["FastAPI\n/chat  /chat/stream"]
    API --> G["LangGraph 状态机"]

    G --> IC["input_check\n安全检查 / 敏感词过滤"]
    IC -->|不安全| RF["refuse\n拒绝回答"]
    IC -->|安全| HT["history_trigger\n判断是否多轮"]
    HT -->|多轮| CQ["condense_query\n问题压缩 & 去指代"]
    HT -->|单轮| RA["routeAgent\n路由决策"]
    CQ --> RA

    RA -->|cache| CA["cache\n语义缓存命中"]
    RA -->|direct| DI["direct\n直接 LLM 生成"]
    RA -->|rag| RAG["rag\nMCP rag_search 工具调用"]
    RA -->|refuse| RF

    RAG --> TN["tools\nMCP Server 执行"]
    TN --> RAG

    RAG --> DG["doc_grade\n文档相关性评估"]
    DG -->|irrelevant| RW["rewrite_query\n问题改写重试"]
    RW --> RAG
    DG -->|relevant| GA["generate_answer\n生成最终答案"]

    GA --> HC["hallucination_check\n幻觉检测"]
    HC -->|幻觉| GA
    HC -->|通过| QC["quality_check\n答案质量评估"]
    QC -->|不合格| GA
    QC -->|合格| OG["output_guard\n输出过滤 & 写缓存"]

    CA --> Response["返回响应"]
    DI --> Response
    OG --> Response
    RF --> Response
```

### 外部依赖概览

```mermaid
flowchart LR
    subgraph LLM_Services ["LLM / Embedding 服务"]
        DS["DeepSeek Chat\n主力 LLM"]
        SF_EMB["SiliconFlow\nBAAI/bge-m3 Embedding"]
        SF_RNK["SiliconFlow\nBAAI/bge-reranker-v2-m3 Rerank"]
    end

    subgraph Storage ["存储层"]
        QD["Qdrant Cloud\n向量数据库"]
        RD["Redis Stack\n语义缓存"]
        PG["PostgreSQL\n会话记忆 Checkpointer"]
    end

    subgraph Observability ["可观测性"]
        LF["Langfuse\nLLM 链路追踪"]
    end

    Agent["Agent 核心"] --> DS
    Agent --> SF_EMB
    Agent --> SF_RNK
    Agent --> QD
    Agent --> RD
    Agent --> PG
    Agent --> LF
```

---

## AI 组件清单

| 组件 | 版本 / 型号 | 用途 |
|------|-------------|------|
| **LangGraph** | `>=1.0.0` | Agent 图编排，管理节点与状态流转 |
| **LangChain OpenAI** | `>=0.3.0` | LLM / Embedding 统一调用接口 |
| **MCP（Model Context Protocol）** | `>=1.30.0` | 将 `rag_search` 封装为标准工具，Agent 通过 MCP Client 调用 |
| **RedisVL SemanticCache** | `>=0.27.2` | 语义相似度缓存（余弦距离阈值 0.15，TTL 3600s） |
| **Qdrant** | `>=1.19.0` | 向量数据库，存储文档 Embedding |
| **jieba + BM25** | `>=0.42.1 / >=0.2.2` | 中文分词 + 关键词检索，与向量检索 5:5 融合 |
| **BAAI/bge-reranker-v2-m3** | — | 混合召回后精排，提升 Top-K 精度 |
| **LangGraph Checkpoint (PostgreSQL)** | `>=3.1.2` | 多轮会话记忆持久化 |
| **Langfuse** | — | LLM 调用链路追踪与可观测性 |
| **FastAPI** | `>=0.141.1` | HTTP API 服务，支持流式（SSE）和非流式两种响应 |

---

## 本地启动

### 前置依赖

- Python 3.10+
- Docker（用于启动 Redis）
- PostgreSQL（本地或 Docker）
- `uv`（推荐）或 `pip`

### 步骤

**1. 克隆项目 & 配置环境变量**

```bash
git clone <repo-url>
cd my-agent
cp .env.example .env
```

编辑 `.env`，填入以下必填项：

```dotenv
# LLM（DeepSeek 或任意 OpenAI 兼容接口）
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Embedding（SiliconFlow）
EMBEDDING_API_KEY=your_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3

# Rerank
RERANK_URL=https://api.siliconflow.cn/v1/rerank
RERANK_MODEL=BAAI/bge-reranker-v2-m3

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_key
QDRANT_COLLECTION=my_collection

# Redis（语义缓存）
REDIS_URL=redis://localhost:6379

# PostgreSQL（会话记忆）
DATABASE_URL=postgresql://user:password@localhost:5432/postgres

# Langfuse（可选，用于追踪）
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

**2. 启动 Redis Stack（语义缓存）**

```bash
docker compose -f docker-compose.cache.yml up -d
```

**3. 启动 PostgreSQL（如果本地没有，可用 Docker）**

```bash
docker run -d \
  --name pg \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:16
```

**4. 安装 Python 依赖**

```bash
# 推荐使用 uv
uv sync

# 或使用 pip
pip install -e .
```

**5. 向量数据库灌库（首次运行）**

将文档放入 `data/` 目录后，运行索引脚本构建 Qdrant 向量索引和本地 BM25 缓存。

**6. 启动 MCP Server（RAG 工具服务）**

```bash
python -m agent.mcp_server
```

**7. 启动 FastAPI 服务**

```bash
uvicorn agent.api:app --reload --host 0.0.0.0 --port 8000
```

**8. 验证服务**

```bash
# 健康检查
curl http://localhost:8000/health

# 就绪检查
curl http://localhost:8000/ready

# 发送问题（非流式）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "年假有几天？", "session_id": "test-001"}'
```

也可以访问 [http://localhost:8000/docs](http://localhost:8000/docs) 使用 Swagger UI 交互式测试。

---

## 项目结构

```
my-agent/
├── src/agent/
│   ├── api.py                  # FastAPI 入口，/chat /chat/stream 接口
│   ├── graph.py                # LangGraph 图构建
│   ├── nodes/                  # 各节点实现（安全检查、路由、RAG、生成等）
│   │   └── state.py            # AgentState 定义
│   ├── mcp_server/             # MCP Server（暴露 rag_search 工具）
│   │   └── tools/
│   │       ├── rag_search.py   # 混合检索 + Rerank
│   │       └── vector_store.py # Qdrant + BM25 初始化
│   ├── mcp_client/             # MCP Client（Agent 侧工具绑定）
│   ├── cache/                  # RedisVL 语义缓存封装
│   ├── security/               # 敏感词规则 & 输出过滤
│   └── middleware/             # 审计中间件
├── data/                       # 原始知识库文档
├── docker-compose.cache.yml    # Redis Stack 启动配置
├── langgraph.json              # LangGraph 图入口声明
└── pyproject.toml              # 项目依赖
```

---

## API 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务存活检查 |
| `GET` | `/ready` | Agent + 缓存就绪检查 |
| `POST` | `/chat` | 同步问答，返回完整 `ChatResponse` |
| `POST` | `/chat/stream` | SSE 流式问答，逐 token 推送 |
| `GET` | `/cache/stats` | 查看语义缓存命中统计 |
| `DELETE` | `/cache/clear` | 清空语义缓存 |

---

> 🚧 代码持续迭代中，功能与接口可能随时变更。
