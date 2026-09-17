import logging
import os

import redis
from langchain_openai import OpenAIEmbeddings
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomVectorizer

logger = logging.getLogger(__name__)


def build_vectorizer(embeddings: OpenAIEmbeddings) -> CustomVectorizer:
    # CustomVectorizer 只接受一个固定签名的回调：所以这里要闭包
    def generate_embeddings(text_input, **kwargs):
        if isinstance(text_input, str):
            return embeddings.embed_query(text_input)
        return embeddings.embed_documents(text_input)

    # CustomVectorizer：把你自己的「文本 → 向量」函数，包装成 RedisVL 能用的向量化器
    return CustomVectorizer(generate_embeddings)


class SemanticCacheService:
    def __init__(
        self,
        redis_url: str,
        embeddings: OpenAIEmbeddings,
        *,
        name: str = "my_agent_semantic_cache",
        distance_threshold: float = 0.15,
        ttl: int = 3600,
    ):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._stats_key = f"{name}_stats"
        self._cache = SemanticCache(
            redis_url=redis_url,
            name=name,
            distance_threshold=distance_threshold,
            vectorizer=build_vectorizer(embeddings),
        )
        self._cache.set_ttl(ttl)

    def ping(self):
        try:
            return self._redis.ping()
        except redis.RedisError:
            return False

    def get(self, user_input: str) -> tuple[str | None, float]:
        try:
            result = self._cache.check(
                prompt=user_input, return_fields=["response", "vector_distance"]
            )
        except Exception:
            self._incr("miss")
            return None, 0.0
        if not result:
            self._incr("miss")
            return None, 0.0
        self._incr("hit")
        hit = result[0]
        answer = hit.get("response", None)
        distance = hit.get("vector_distance", 0.0)
        score = max(0.0, 1 - distance)
        return answer, score

    def set(self, user_input: str, answer: str):
        try:
            self._cache.store(prompt=user_input, response=answer)
            self._incr("stores")
        except Exception as exc:
            logger.warning("cache store failed: %s", exc)

    def _incr(self, key: str):
        self._redis.hincrby(self._stats_key, key, 1)

    def stats(self):
        result = self._redis.hgetall(self._stats_key)
        hit = int(result.get("hit", 0))
        miss = int(result.get("miss", 0))
        stores = int(result.get("stores", 0))
        total = hit + miss
        total = hit + miss
        return {
            "hit": hit,
            "miss": miss,
            "stores": stores,
            "total": total,
            "hit_rate": hit / total if total > 0 else 0.0,
        }

    def clear(self):
        self._redis.delete(self._stats_key)
        self._cache.clear()


def create_semantic_cache_from_env(embeddings: OpenAIEmbeddings):
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.warning("REDIS_URL is not set")
        return None
    service = SemanticCacheService(
        redis_url=redis_url,
        embeddings=embeddings,
        name=os.getenv("SEMANTIC_CACHE_NAME", "my_agent_semantic_cache"),
        distance_threshold=float(os.getenv("SEMANTIC_CACHE_DISTANCE_THRESHOLD", 0.15)),
        ttl=int(os.getenv("SEMANTIC_CACHE_TTL", 3600)),
    )
    if not service:
        logger.warning("Failed to create semantic cache service")
        return None
    return service
