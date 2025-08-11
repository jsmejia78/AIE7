"""MCP client utilities for tool integration with LangGraph agents."""

import json
import asyncio
from typing import Any, Dict
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPClient:
    def __init__(self):

        self.servers = {
            "mcp-server": {
                "command" : "uv",
                "args" : ["--directory", "/home/jsmejia/the_ai_eng_bootcamp/code/AIE7/14_LangGraph_Platform/mcp/", "run", "server.py"],
                "transport" : "stdio",
            }
        }
        # MCP client
        self.mcp_client = MultiServerMCPClient(self.servers)
        # Get tools synchronously
        self.tools = asyncio.run(self.mcp_client.get_tools())

    def get_tools(self):
        return self.tools

MCP_client = MCPClient()
    
    
    