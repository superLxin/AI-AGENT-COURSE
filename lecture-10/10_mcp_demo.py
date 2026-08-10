"""
第 10 讲：MCP 协议 —— Agent 的"USB 接口"
============================================
演示 MCP（Model Context Protocol）的基本用法。
MCP 让 Agent 可以通过统一协议连接任意工具/数据源。

前置条件：pip install mcp
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def mcp_demo():
    """
    MCP 演示：连接文件系统服务器，让 Agent 能读写文件。

    实际运行需要先安装 MCP server：
        npm install -g @modelcontextprotocol/server-filesystem

    这是一个概念演示——展示 MCP 的核心思想：
    不需要为每个工具写 @tool，直接连社区已写好的 MCP Server。
    """
    print("=" * 60)
    print("MCP（Model Context Protocol）概念演示")
    print("=" * 60)
    print("""
MCP 是什么？
───────────
Agent 要调用工具 → 传统方式：每个工具手动写 @tool
Agent 要调用工具 → MCP 方式：连接现成的 MCP Server

类比：
  传统 = 每个设备都要写驱动
  MCP  = USB 即插即用

MCP Server 暴露三样东西：
  - Tools:     可调用的函数（读文件、发邮件、查数据库...）
  - Resources: 可读取的数据（文档、配置...）
  - Prompts:   可复用的提示模板

社区已有的 MCP Server：
  - 文件系统 (filesystem)
  - GitHub (github)
  - Slack (slack)
  - PostgreSQL (postgres)
  - Brave Search (brave-search)
  - ...100+ 个
    """)

    print("代码示例（需要先 npm install MCP server）：")
    print("""
from mcp import ClientSession, StdioServerParameters

# 连接文件系统 MCP Server
server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
)

async with ClientSession(server) as session:
    # 发现可用工具——不需要自己写文件操作代码！
    tools = await session.list_tools()
    # tools = [
    #   {"name": "read_file", "description": "读取文件内容"},
    #   {"name": "write_file", "description": "写入文件"},
    #   {"name": "list_directory", "description": "列出目录"},
    # ]

    # 调用工具
    result = await session.call_tool("read_file", {"path": "/readme.md"})
    print(result)
    """)

    print("=" * 60)
    print("💡 关键思想：")
    print("  工具生态化——不自己造轮子，接入社区现成的 MCP Server")
    print("  你的 Agent 只需要成为一个 MCP Client")
    print()
    print("🔗 MCP 与可插拔 Skill 的关系：")
    print("  MCP Server 暴露的三样东西中，'Prompts' 就是 Skill 的工业级实现。")
    print("  04 讲我们硬编码了 Skill，10_travel_agent_full.py 里演进为")
    print("  从 skills/ 目录自动扫描 .md 文件——这本质上就是 MCP Prompts 的")
    print("  简化版：可复用的提示模板，按需发现，按需加载。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(mcp_demo())
