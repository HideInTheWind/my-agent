from .mcp_instance import mcp
from .tools import register_all_tools

register_all_tools()

if __name__ == "__main__":
    mcp.run(transport="stdio")
