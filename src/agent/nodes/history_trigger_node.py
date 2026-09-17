from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ..tools.utils import _token_count
from .state import AgentState

MAX_BUFFER_TOKEN = 150
KEEP_RECENT = 10

COMPRESS_SYSTEM_PROMPT = """
你是对话压缩专家；擅长将长对话压缩，保留关键信息;

- 对话内容：{history_text} 
"""


def build_history_trigger_node(llm: ChatOpenAI):
    def compress_history(history_text: str) -> str:
        prompt = COMPRESS_SYSTEM_PROMPT.format(history_text=history_text)
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content

    def history_trigger_node(state: AgentState):
        msgs = state["messages"]
        no_system_messages = [m for m in msgs if not isinstance(m, SystemMessage)]
        current_token_count = _token_count(no_system_messages)
        if current_token_count <= MAX_BUFFER_TOKEN:
            return Command(
                goto="condense_query",
            )
        system_messages = [m for m in msgs if isinstance(m, SystemMessage)]
        to_compress = no_system_messages[:-KEEP_RECENT]
        if not to_compress:
            return Command(
                goto="condense_query",
            )
        kept = no_system_messages[-KEEP_RECENT:]

        compress_text = "\n".join(
            [
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{m.content}"
                for m in to_compress
            ]
        )
        new_summary = compress_history(compress_text)
        new_messages = system_messages + [SystemMessage(content=new_summary)] + kept
        remove_msgs = [
            RemoveMessage(id=m.id) for m in to_compress if m not in new_messages if m.id
        ]
        return Command(
            goto="condense_query",
            update={
                "messages": remove_msgs + new_messages,
            },
        )

    return history_trigger_node
