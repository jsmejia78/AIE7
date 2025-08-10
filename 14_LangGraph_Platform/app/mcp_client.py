"""MCP client utilities for tool integration with LangGraph agents."""

import json
import asyncio
from typing import Any, Dict
from langchain_core.tools import tool
from fastmcp import Client


@tool
def stock_info(ticker: str) -> str:
    """
    Get stock information for a given ticker symbol using the MCP server.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        A string containing stock information
    """
    try:
        # Call the MCP server served by Cursor
        result = asyncio.run(_call_mcp_stock_info(ticker))
        return result
    except Exception as e:
        return json.dumps({"error": f"Failed to get stock info: {str(e)}"})


async def _call_mcp_stock_info(ticker: str) -> str:
    """Call the Cursor-served MCP server."""
    try:
        # Connect to the MCP server using the name from your Cursor configuration
        async with Client("mcp-server") as client:  # <- Use the server name from your config
            result = await client.call_tool("stock_info", {"ticker": ticker.upper()})
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"MCP call failed: {str(e)}"})