import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

INJECTION_PATTERNS = [
    r"忽略(之前|上面|前面)的?(所有|全部)?指令",
    r"ignore\s+(previous|above|before)\s+?instructions",
    r"你现在是.{0, 30}(没有限制|无限制|无约束)",
    r"system\s*prompt",
    r"repeat\s+(your|the)\s+(system|instructions)",
]


LEAK_PATTERNS = ["system\s*prompt*", "<\|im_start\>", "hidden_rules"]


def rule_check(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "输入为空"
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "命中拦截规则"
    return True, "未命中拦截规则"


def rule_check_leak(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "输入为空"
    for pattern in LEAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "命中泄露规则"
    return True, "未命中泄露规则"


class LLMCheck(BaseModel):
    is_safe: bool
    reason: str


def llm_check(text: str, system_prompt: str, llm: ChatOpenAI):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    response = (
        llm.bind(temperature=0.0)
        .with_structured_output(LLMCheck, method="function_calling")
        .invoke(messages)
    )
    if response.is_safe:
        return True, response.reason
    else:
        return False, response.reason
