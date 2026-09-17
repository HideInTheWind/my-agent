import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass
class MCPClientConfig:
    server_name: str = "mcp_server"
    mcp_url: str | None = field(default_factory=lambda: os.getenv("MCP_SERVER_URL"))
    allowed_tools: list[str] | None = None
    stdio_command: list[str] | None = None
    cwd: str | None = None


PROJECT_SRC = Path(__file__).parents[2]


class MCP_Gateway:
    def __init__(self, config: MCPClientConfig | None = None):
        self._config = config or MCPClientConfig()
        cwd = self._config.cwd or PROJECT_SRC  # 子进程在 src 下运行
        # 以模块方式启动 MCP Server
        args = self._config.stdio_command or [
            "-m",
            "agent.mcp_server",
        ]
        # mcp_url = self.config.mcp_url
        self._client = MultiServerMCPClient(
            {
                self._config.server_name: {
                    "transport": "stdio",
                    "args": args,
                    "cwd": cwd,
                    "command": sys.executable,
                }
            }
        )

    async def get_tools(self):
        tools = await self._client.get_tools()
        if self._config.allowed_tools:
            tools = [tool for tool in tools if tool.name in self._config.allowed_tools]
        return tools
