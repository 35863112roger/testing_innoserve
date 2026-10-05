import asyncio
import json

from backend.mcp_client.gateway import MCPToolGateway
from mcp_servers.internal_tools import mcp


async def main() -> None:
    gateway = MCPToolGateway(mcp)

    tools = await gateway.list_tools()
    tool_names = [
        tool.get("name")
        for tool in tools
    ]

    print("已註冊 MCP Tools：")
    print(
        json.dumps(
            tool_names,
            ensure_ascii=False,
            indent=2,
        )
    )

    if "analyze_thread_risk" not in tool_names:
        raise RuntimeError(
            "找不到 analyze_thread_risk MCP Tool"
        )

    result = await gateway.call_tool(
        "analyze_thread_risk",
        {
            "thread_id": "mcp-manual-test",
            "root_text": (
                "原始貼文正在討論某位使用者"
                "在社群平台上的發言。"
            ),
            "statistics": {
                "general": 20,
                "friendly": 2,
                "harassment": 2,
                "cyberbullying": 1,
            },
            "risk_nodes": [
                {
                    "target_id": "comment-1",
                    "label": "harassment",
                    "confidence": 0.94,
                    "parent_text": "我只是表達自己的意見。",
                    "text": "你這種人根本沒有資格講話。",
                },
                {
                    "target_id": "comment-2",
                    "label": "cyberbullying",
                    "confidence": 0.89,
                    "parent_text": "請不要再攻擊當事人。",
                    "text": "大家每天一起去他的帳號留言。",
                },
                {
                    "target_id": "comment-3",
                    "label": "general",
                    "confidence": 0.99,
                    "parent_text": "討論內容。",
                    "text": "這筆應該被 MCP Tool 排除。",
                },
            ],
        },
    )

    print("\nMCP Tool 回傳：")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())