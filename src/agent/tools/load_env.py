import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_env():
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_api_url = os.getenv("EMBEDDING_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    return (
        api_key,
        api_url,
        model,
        embedding_api_key,
        embedding_api_url,
        embedding_model,
    )


def validate_env():
    env = get_env()
    names = [
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ]
    if not all(env):
        missing = [n for n, v in zip(names, env) if not v]
        logger.warning(f"缺少环境变量: {missing}")
        return None
    return env
