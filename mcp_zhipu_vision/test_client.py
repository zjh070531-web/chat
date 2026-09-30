# -*- coding: utf-8 -*-
"""用官方 mcp 客户端验证 server 能正常启动并调用 vision 工具"""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.environ["ZHIPU_API_KEY"] = "64cbf83935ba42448ae928e3df0df03e.InWxyAXpEkGQkYWU"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command=r"C:\Users\31026\.workbuddy\binaries\python\envs\default\Scripts\python.exe",
        args=[r"D:\chatting\mcp_zhipu_vision\server.py"],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            r = await s.call_tool("vision", {
                "image": r"C:\Users\31026\.workbuddy\clipboard-images\clipboard-2026-09-29T15-27-29-055Z-155c1e0c.png",
                "prompt": "一句话描述图片内容",
            })
            print("result:", r.content[0].text[:200])

asyncio.run(main())
