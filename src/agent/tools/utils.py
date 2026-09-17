import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _has_system_message(messages: list) -> bool:
    return any(isinstance(m, SystemMessage) for m in messages)


def _prepend_system(messages: list, system_prompt: str) -> list:
    if any(isinstance(m, SystemMessage) for m in messages):
        return list(messages)
    return [SystemMessage(content=system_prompt), *messages]


def _token_count(messages: list) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = 0
    for message in messages:
        content = _extract_content(message.content)
        tokens += len(enc.encode(content)) + 4
    return tokens


# LangChain 的消息 content 字段实际上有两种类型：


# 场景	content 类型	示例
# 纯文本消息
# str
# "你好"
# 工具调用结果 / AI 多步骤响应
# list
# [{"type": "text", "text": "..."}, ...]
def _extract_content(content) -> str:
    """统一提取 message.content 的文本，兼容 str 和 list 两种格式"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()
