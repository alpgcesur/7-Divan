"""Tool registry for Divan advisors."""

from __future__ import annotations

from langchain_core.tools import BaseTool

TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """Register a tool in the global registry."""
    TOOL_REGISTRY[tool.name] = tool


def get_tools_for_advisor(tool_names: list[str]) -> list[BaseTool]:
    """Resolve a list of tool names to actual tool instances."""
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]


def list_available_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(TOOL_REGISTRY.keys())


def ensure_tools_registered() -> None:
    """Import base tools module to trigger registration. Idempotent."""
    if not TOOL_REGISTRY:
        import divan.tools.base  # noqa: F401
