from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import os
from typing import Any, Dict
from stock_info import StockInfo

load_dotenv()

mcp = FastMCP("mcp-server")

@mcp.tool()
def stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get consolidated stock information about a stock TICKER
    """
    return StockInfo(ticker).get_info()

if __name__ == "__main__":
    mcp.run(transport="stdio")