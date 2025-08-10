from fastmcp import Client
from pprint import pprint
import json

async def main():
    # Connect via stdio to a local script
    async with Client("mcp-server") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        result = await client.call_tool("stock_info", {"ticker": "AAPL"})
        print(f"Result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())