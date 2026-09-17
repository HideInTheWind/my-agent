import os
from functools import lru_cache

import jieba
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

load_dotenv()


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_collection")


@lru_cache
def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set")
    return QdrantClient(url=url, api_key=api_key)


@lru_cache
def get_embeddings():
    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL")
    embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=model)
    return embeddings


def ensure_collection_exists(collection_name: str, vector_size: int):
    client = get_qdrant_client()
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


# ---------------------------------------------------------------------------
# 默认情况下，payload 只是跟着点存着，并没有为某个字段建查询索引。
# create_payload_index 就是告诉 Qdrant：请给某个 payload 字段建索引，之后按这个字段过滤会快很多。
# collection_name：在哪张向量表上建索引
# field_name="metadata.source"：要索引的路径：payload 里 metadata 对象下的 source 字段
# field_schema=PayloadSchemaType.KEYWORD：按 精确字符串匹配 来索引（等值过滤），不是全文分词
# ---------------------------------------------------------------------------


def create_source_payload_index(collection_name: str):
    """在 metadata.source 字段上建 keyword 索引，加速按文件名过滤删除。"""
    client = get_qdrant_client()
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def delete_by_source(collection_name: str, source_name: str):
    """删除 collection 中所有 metadata.source == source_name 的 point。
    Returns:
        删除的 point 数量（Qdrant 返回的 operation_id，实际删除数需从日志确认）。
    """
    client = get_qdrant_client()
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source", match=MatchValue(value=source_name)
                    )
                ]
            )
        ),
    )


@lru_cache
def get_vec_store(collection_name: str):
    client = get_qdrant_client()
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store


def load_documents_from_qdrant():
    client = get_qdrant_client()
    documents: list[Document] = []
    offset = None
    while True:
        # scroll 适合「导出 / 重建索引 / 全量加载」这类场景。
        # # 从某张表中按分页加载全量数据
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,  # 要读取的 Qdrant collection 名称
            limit=100,  # 每页最多返回 100 条，避免一次拉太多导致内存或超时问题
            offset=offset,  # 上一页的结束点，第一次传 None 表示从头开始
            with_payload=True,  # 回每条 point 的 payload（文本、metadata 等业务字段）
            with_vectors=False,  # 不返回 embedding 向量。这里只重建 Document，不需要向量，设为 False 可减少网络传输和内存占用
        )

        for point in points:
            payload = point.payload or {}
            text = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            source = metadata.get(
                "source",
            )
            chunk_id = metadata.get("chunk_id", "")
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "chunk_id": chunk_id,
                    },
                )
            )
        if offset is None:
            break

    return documents


@lru_cache
def get_bm25(recall_k: int = 10):
    documents = load_documents_from_qdrant()
    if not documents:
        raise ValueError(
            f"Qdrant 表 {COLLECTION_NAME} 为空，请先运行 python scripts/index_docs.py"
        )
        # 【用户可调】BM25 分词方式
    #   jieba.lcut            — 精确模式（默认），适合短查询
    #   jieba.lcut_for_search — 搜索模式，切得更细，适合长文本召回
    #   自定义词典：在此行上方加 jieba.load_userdict("data/custom_dict.txt")

    bm25 = BM25Retriever.from_documents(
        documents=documents,
        k=recall_k,
        preprocess_func=jieba.lcut,  # ← 改分词方式在这里
    )
    return bm25


def refresh_bm25_cache(recall_k: int = 10):
    """清除 BM25 内存缓存，并立即用 Qdrant 最新数据重建索引。"""
    get_bm25.cache_clear()  # 1. 清除旧缓存
    get_bm25(recall_k)
