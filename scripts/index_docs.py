import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.mcp_server.tools.vector_store import (
    create_source_payload_index,
    delete_by_source,
    ensure_collection_exists,
    get_embeddings,
    get_vec_store,
)

load_dotenv()

DATA_DIR = Path(__file__).parents[1] / "data"
MANIFEST_PATH = DATA_DIR / ".index_manifest.json"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文件哈希
# ---------------------------------------------------------------------------


# "rb" = read + binary，以二进制模式打开文件，计算哈希必须用这个，否则不同系统的换行符处理会导致哈希不一致。
# b""    bytes，字节串（二进制内容）
# lambda 是 Python 的匿名函数语法，等价于：def read_chunk(): return f.read(8192)
# f.read(8192) 表示从文件中读取最多 8192 字节（8 KB）
# 每次调用这个函数，都会从当前文件指针位置继续往后读
# 当文件读完时，f.read() 返回 b""（空字节串）
# iter() 的哨兵值用法，语法是：iter(callable, sentinel)
# 含义：反复调用 callable()，直到返回值等于 sentinel 为止，每次调用的返回值作为迭代元素。
# 所以 iter(lambda: f.read(8192), b"") 的完整行为是：
# 第1次调用 lambda → 读取 8192 字节 → 返回 chunk1（非空）→ 继续
# 第2次调用 lambda → 读取 8192 字节 → 返回 chunk2（非空）→ 继续
# ...
# 最后一次调用    → 文件读完了 → 返回 b""（空）→ 停止迭代
def file_md5(path: Path) -> str:
    # 创建一个 MD5 哈希计算器对象 h，你可以不断往里面"喂"数据，最后它输出一个 固定 32 位的十六进制字符串（即文件指纹）。
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# manifest 读写
# ---------------------------------------------------------------------------
# 返回一个 dict[str, str]，即 { 文件名: 上次索引时的MD5哈希值 }，类似于：
# {
#     "langchain_guide.md": "a3f2c8d9e1b4...",
#     "api_reference.pdf": "7c4e9f2a0b3...",
#     "notes.txt": "1d5e8a3b7f2..."
# }


def load_manifest() -> dict[str, str]:
    """返回 {filename: md5hex} 映射。"""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, str]):
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 文档处理
# ---------------------------------------------------------------------------


def load_local_documents(file_path: Path) -> list[Document]:
    path = str(file_path).lower()
    if path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif path.endswith(".txt") or path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return loader.load()


def split_documents(
    documents: list[Document], chunk_size: int = 300, chunk_overlap: int = 30
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def enrich_documents(chunks: list[Document], file_path: Path) -> list[Document]:
    doc_type = file_path.suffix.lstrip(".")
    source = file_path.name

    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", -1)
        chunk.metadata.update(
            {
                "source": source,
                "doc_type": doc_type,
                "chunk_id": f"{source}_p{page}_c{i}",
                "page": page,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    if not COLLECTION_NAME:
        raise SystemExit("QDRANT_COLLECTION 环境变量未设置")

    manifest = load_manifest()
    vec_store = get_vec_store(COLLECTION_NAME)
    collection_initialized = False
    total_added = 0
    total_deleted_files = 0

    # ── 扫描磁盘上的文件，处理新增 / 变更 ──────────────────────────────────
    current_files: set[str] = set()

    for file_path in DATA_DIR.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file_path.name.startswith("."):  # 跳过 .index_manifest.json 等隐藏文件
            continue

        source_name = file_path.name
        current_files.add(source_name)
        current_hash = file_md5(file_path)

        if manifest.get(source_name) == current_hash:
            logger.info(f"[跳过] {source_name}（内容未变更）")
            continue

        # 文件有变更或是新文件 → 删旧 chunk，写新 chunk
        if source_name in manifest:
            logger.info(f"[变更] {source_name} → 删除旧 chunk …")
            delete_by_source(COLLECTION_NAME, source_name)

        documents = load_local_documents(file_path)
        if not documents:
            logger.warning(f"[跳过] {source_name} 加载后为空")
            continue

        chunks = split_documents(documents)
        enriched_chunks = enrich_documents(chunks, file_path)

        # 首次写入前确保 collection 存在，并建 payload 索引
        if not collection_initialized:
            sample_vec = get_embeddings().embed_documents(
                [enriched_chunks[0].page_content]
            )
            size = len(sample_vec[0])
            ensure_collection_exists(COLLECTION_NAME, size)
            create_source_payload_index(COLLECTION_NAME)
            collection_initialized = True

        vec_store.add_documents(enriched_chunks)
        manifest[source_name] = current_hash
        total_added += len(enriched_chunks)
        logger.info(f"[完成] {source_name} → 写入 {len(enriched_chunks)} 条 chunk")

    # ── 清理磁盘已删除的文件在 Qdrant 中的残留 ─────────────────────────────
    stale_files = set(manifest.keys()) - current_files
    for source_name in stale_files:
        logger.info(f"[清理] {source_name} 已从磁盘删除 → 从 Qdrant 移除旧 chunk")
        delete_by_source(COLLECTION_NAME, source_name)
        del manifest[source_name]
        total_deleted_files += 1

    save_manifest(manifest)

    logger.info(
        f"增量索引完成：新增/更新 {total_added} 条 chunk，"
        f"清理已删除文件 {total_deleted_files} 个"
    )


if __name__ == "__main__":
    main()
