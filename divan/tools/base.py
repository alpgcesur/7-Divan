"""Core tools available to Divan advisors."""

from __future__ import annotations

import glob as globlib
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from divan.tools import register_tool

MAX_OUTPUT_CHARS = 15000
MAX_FILE_LINES = 500


@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns top results with titles, URLs, and snippets.

    Use this to find current data, market information, competitor details,
    news, statistics, or any factual information relevant to the question.

    Args:
        query: The search query string.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        output_parts = []
        for r in results:
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            output_parts.append(f"**{title}**\n{href}\n{body}")

        return "\n\n---\n\n".join(output_parts)
    except ImportError:
        return "Web search unavailable: duckduckgo-search package not installed."
    except Exception as e:
        return f"Search error: {e}"


@tool
def read_file(path: str, max_lines: int = MAX_FILE_LINES) -> str:
    """Read the contents of a local file.

    Use this to examine code, config files, documents, or any text file
    relevant to the question being deliberated.

    Args:
        path: Absolute or relative path to the file.
        max_lines: Maximum number of lines to read (default 500).
    """
    try:
        filepath = Path(path).expanduser().resolve()
        if not filepath.exists():
            return f"File not found: {path}"
        if not filepath.is_file():
            return f"Not a file: {path}"
        if filepath.stat().st_size > 1_000_000:
            return f"File too large (>{1_000_000} bytes): {path}"

        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        truncated = lines[:max_lines]
        content = "\n".join(truncated)

        if total > max_lines:
            content += f"\n\n[Truncated: showing {max_lines} of {total} lines]"

        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files and directories at a given path, with optional glob pattern.

    Use this to explore directory structures, find relevant files, or
    understand project layout.

    Args:
        path: Directory path to list (default: current directory).
        pattern: Glob pattern to filter results (e.g., "*.py", "**/*.md").
    """
    try:
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"Path not found: {path}"
        if not base.is_dir():
            return f"Not a directory: {path}"

        matches = sorted(globlib.glob(str(base / pattern), recursive=True))
        if not matches:
            return f"No files matching '{pattern}' in {path}"

        # Limit output
        entries = []
        for m in matches[:100]:
            p = Path(m)
            rel = p.relative_to(base) if p.is_relative_to(base) else p
            suffix = "/" if p.is_dir() else ""
            entries.append(f"{rel}{suffix}")

        result = "\n".join(entries)
        if len(matches) > 100:
            result += f"\n\n[Showing 100 of {len(matches)} results]"

        return result
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def grep_search(pattern: str, path: str = ".", file_pattern: str = "") -> str:
    """Search file contents by regex pattern.

    Use this to find specific code, configuration values, references,
    or text patterns across files.

    Args:
        pattern: Regex pattern to search for.
        path: Directory or file to search in (default: current directory).
        file_pattern: Optional glob to filter files (e.g., "*.py").
    """
    try:
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"Path not found: {path}"

        regex = re.compile(pattern, re.IGNORECASE)
        matches = []

        if base.is_file():
            files = [base]
        else:
            glob_pat = file_pattern if file_pattern else "**/*"
            files = [f for f in base.glob(glob_pat) if f.is_file()]

        for filepath in files[:500]:
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for line_num, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = filepath.relative_to(base) if filepath.is_relative_to(base) else filepath
                        matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                        if len(matches) >= 50:
                            break
            except (PermissionError, UnicodeDecodeError):
                continue
            if len(matches) >= 50:
                break

        if not matches:
            return f"No matches for pattern '{pattern}' in {path}"

        result = "\n".join(matches)
        if len(matches) >= 50:
            result += "\n\n[Results truncated at 50 matches]"

        return result
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    except Exception as e:
        return f"Error searching: {e}"


@tool
def run_command(command: str, timeout_seconds: int = 30) -> str:
    """Run a shell command and return its output.

    Use this for quick system queries, checking versions, running simple
    scripts, or gathering system information. Commands are limited by timeout.

    Args:
        command: The shell command to execute.
        timeout_seconds: Maximum execution time in seconds (default 30, max 60).
    """
    timeout_seconds = min(timeout_seconds, 60)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        if not output.strip():
            output = "[no output]"

        # Truncate if too long
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n\n[Truncated at {MAX_OUTPUT_CHARS} chars]"

        return output
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_seconds} seconds."
    except Exception as e:
        return f"Error running command: {e}"


# Register all tools on import
for _tool in [web_search, read_file, list_files, grep_search, run_command]:
    register_tool(_tool)
