import os

import requests
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from agent.mcp_server.tools.vector_store import (
    get_bm25,
    get_vec_store,
    refresh_bm25_cache,
)

from ..mcp_instance import mcp

load_dotenv()
RECALL_K = int(os.getenv("RAG_RECALL_K", "10"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_collection")


def ensemble_retriever():
    bm25 = get_bm25(RECALL_K)

    vector_store = get_vec_store(COLLECTION_NAME)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": RECALL_K})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        unique.append(doc)
    return unique


def rerank(query: str, documents: list[Document]) -> list[Document]:
    rerank_url = os.getenv("RERANK_URL")
    rerank_model = os.getenv("RERANK_MODEL")
    embedding_key = os.getenv("EMBEDDING_API_KEY")
    if not rerank_url or not rerank_model:
        raise ValueError("RERANK_URL is not set")

    docs_text = [doc.page_content.strip() for doc in documents]
    if not docs_text:
        return []
    payload = {
        "model": rerank_model,
        "query": query,
        "documents": docs_text,
        "top_n": RECALL_K,
        "return_documents": True,
    }

    headers = {
        "Authorization": f"Bearer {embedding_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url=rerank_url, json=payload, headers=headers)
    response.raise_for_status()
    hits = build_hits(response.json(), documents)
    return hits


def build_hits(data: dict, documents: list[Document]) -> list[dict]:
    hits = []
    for item in data.get("results", []):
        idx = item.get("index")
        text = item["document"]["text"]
        if not text:
            continue
        hits.append(
            {
                "content": text,
                "source": documents[idx].metadata.get("source", ""),
                "score": item.get("relevance_score", 0),
            }
        )
    if not hits:
        return "未检索到相关文档。"
    return "\n\n".join(f"[{h['source']}] {h['content']}" for h in hits)


@mcp.tool()
def rag_search(query: str, top_k: int = 3):

    retriever = ensemble_retriever()
    candidates = retriever.invoke(query)
    if not candidates:
        return []
    unique_docs = _dedupe_documents(candidates)
    hits = rerank(query, unique_docs)
    return hits


@mcp.tool()
def refresh_bm25(recall_k: int = RECALL_K):
    """
    更新本地知识库（运行 index_docs.py）后调用此工具，
    刷新 BM25 索引缓存，使新文档立即参与关键词检索，无需重启服务。
    """
    refresh_bm25_cache(recall_k)
    return f"BM25 缓存已刷新（recall_k={recall_k}），新知识库已生效。"
