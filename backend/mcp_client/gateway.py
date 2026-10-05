import asyncio
import json
from typing import Any

from mcp import Client


class MCPToolGateway:
    def __init__(self, server: Any) -> None:
        self.server = server

    async def list_tools(self) -> list[dict[str, Any]]:
        async with Client(self.server) as client:
            result = await client.list_tools()
            return [self._dump(tool) for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            async with Client(self.server) as client:
                result = await client.call_tool(name, arguments)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            details = self._exception_messages(exc)
            raise RuntimeError(f"MCP 工具 {name} 失敗：{'；'.join(details)}") from exc
        else:
            if getattr(result, "is_error", False):
                messages = [
                    str(getattr(item, "text", item))
                    for item in result.content
                ]
                raise RuntimeError(f"MCP 工具 {name} 失敗：{' '.join(messages)}")
            structured = getattr(result, "structured_content", None)
            if structured is not None:
                return structured
            if len(result.content) == 1:
                text = getattr(result.content[0], "text", None)
                if text:
                    try:
                        decoded = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(decoded, dict):
                            return decoded
            return {"content": [self._dump(item) for item in result.content]}

    @classmethod
    def _exception_messages(cls, exc: BaseException) -> list[str]:
        nested = getattr(exc, "exceptions", None)
        if nested:
            messages: list[str] = []
            for child in nested:
                messages.extend(cls._exception_messages(child))
            return list(dict.fromkeys(messages))
        message = str(exc).strip()
        return [message or exc.__class__.__name__]

    @staticmethod
    def _dump(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        return {"value": str(value)}
