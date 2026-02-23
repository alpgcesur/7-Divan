"""Tests for the tool registry and tool-enabled advisors."""

from pathlib import Path

import pytest

from divan.advisor import Advisor, load_persona
from divan.tools import (
    TOOL_REGISTRY,
    ensure_tools_registered,
    get_tools_for_advisor,
    list_available_tools,
    register_tool,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    """Ensure tools are registered before each test."""
    ensure_tools_registered()


class TestToolRegistry:
    def test_tools_registered(self):
        """All 5 core tools should be registered."""
        assert len(TOOL_REGISTRY) >= 5
        expected = {"web_search", "read_file", "list_files", "grep_search", "run_command"}
        assert expected.issubset(set(TOOL_REGISTRY.keys()))

    def test_list_available_tools(self):
        tools = list_available_tools()
        assert "web_search" in tools
        assert "read_file" in tools
        assert "run_command" in tools

    def test_get_tools_for_advisor_all(self):
        tools = get_tools_for_advisor(["web_search", "read_file"])
        assert len(tools) == 2
        assert tools[0].name == "web_search"
        assert tools[1].name == "read_file"

    def test_get_tools_for_advisor_empty(self):
        tools = get_tools_for_advisor([])
        assert tools == []

    def test_get_tools_for_advisor_unknown_ignored(self):
        tools = get_tools_for_advisor(["web_search", "nonexistent_tool"])
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_ensure_tools_registered_idempotent(self):
        count_before = len(TOOL_REGISTRY)
        ensure_tools_registered()
        assert len(TOOL_REGISTRY) == count_before


class TestPersonaToolsLoading:
    """Test that persona files with tools field load correctly."""

    def test_contrarian_has_tools(self):
        persona = load_persona(Path("personas/contrarian.md"))
        assert persona.tools == ["web_search"]

    def test_operator_has_full_toolkit(self):
        persona = load_persona(Path("personas/operator.md"))
        assert "web_search" in persona.tools
        assert "read_file" in persona.tools
        assert "list_files" in persona.tools
        assert "grep_search" in persona.tools
        assert "run_command" in persona.tools

    def test_bas_vezir_has_no_tools(self):
        persona = load_persona(Path("personas/bas_vezir.md"))
        assert persona.tools == []

    def test_visionary_has_web_search(self):
        persona = load_persona(Path("personas/visionary.md"))
        assert persona.tools == ["web_search"]


class TestToolResolution:
    """Test resolving persona tool names to actual tool instances."""

    def test_resolve_operator_tools(self):
        persona = load_persona(Path("personas/operator.md"))
        tools = get_tools_for_advisor(persona.tools)
        assert len(tools) == 5
        tool_names = {t.name for t in tools}
        assert tool_names == {"web_search", "read_file", "list_files", "grep_search", "run_command"}

    def test_resolve_contrarian_tools(self):
        persona = load_persona(Path("personas/contrarian.md"))
        tools = get_tools_for_advisor(persona.tools)
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_resolve_empty_tools(self):
        persona = load_persona(Path("personas/bas_vezir.md"))
        tools = get_tools_for_advisor(persona.tools)
        assert tools == []


class TestCoreTools:
    """Test that the core tools execute correctly."""

    def test_read_file_existing(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nline 2")
        from divan.tools.base import read_file

        result = read_file.invoke({"path": str(test_file)})
        assert "hello world" in result
        assert "line 2" in result

    def test_read_file_missing(self):
        from divan.tools.base import read_file

        result = read_file.invoke({"path": "/nonexistent/file.txt"})
        assert "not found" in result.lower()

    def test_list_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        from divan.tools.base import list_files

        result = list_files.invoke({"path": str(tmp_path), "pattern": "*.py"})
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_grep_search(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    return 'world'\n")
        from divan.tools.base import grep_search

        result = grep_search.invoke({"pattern": "hello", "path": str(tmp_path)})
        assert "hello" in result

    def test_run_command(self):
        from divan.tools.base import run_command

        result = run_command.invoke({"command": "echo test123"})
        assert "test123" in result

    def test_run_command_timeout(self):
        from divan.tools.base import run_command

        result = run_command.invoke({"command": "sleep 100", "timeout_seconds": 1})
        assert "timed out" in result.lower()
