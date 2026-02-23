"""Rich-based display layer for Divan deliberations."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from rich.rule import Rule

from divan.advisor import Advisor
from divan.session import Session, build_advisor_debate_history, build_advisor_history, build_synthesis_history
from divan.synthesis import build_synthesis_prompt


console = Console()

MAX_TOOL_ITERATIONS = 5


def render_header(question: str) -> Panel:
    """Render the Divan header panel."""
    content = Text.from_markup(
        f"[bold bright_yellow]D İ V A N[/bold bright_yellow]\n"
        f"[dim]Personal Advisory Council[/dim]\n\n"
        f'[italic]"{question}"[/italic]'
    )
    return Panel(
        content,
        border_style="bright_yellow",
        padding=(1, 2),
    )


def render_advisor_panel(
    advisor: Advisor,
    content: str,
    streaming: bool = False,
    error: str = "",
    elapsed: float | None = None,
) -> Panel:
    """Render an advisor's response panel."""
    subtitle = advisor.title
    if elapsed is not None:
        subtitle = f"{advisor.title} [dim]({elapsed:.1f}s)[/dim]"

    if error:
        body = Text.from_markup(
            f"[bold red]✗ Error[/bold red]\n\n{error}"
        )
        return Panel(
            body,
            title=f"{advisor.icon}  {advisor.name}",
            subtitle=subtitle,
            border_style="red",
            padding=(1, 2),
        )

    if streaming and not content:
        body = Spinner("dots", text="Deliberating...")
    elif content:
        body = Markdown(content)
    else:
        body = Text("No response received", style="dim italic")

    return Panel(
        body,
        title=f"{advisor.icon}  {advisor.name}",
        subtitle=subtitle,
        border_style=advisor.color,
        padding=(1, 2),
    )


def render_synthesis_panel(content: str, streaming: bool = False) -> Panel:
    """Render the Bas Vezir synthesis panel."""
    if streaming and not content:
        body = Spinner("dots", text="Synthesizing council deliberations...")
    elif content:
        body = Markdown(content)
    else:
        body = Text("Awaiting council deliberations...", style="dim")

    return Panel(
        body,
        title="👁  Bas Vezir",
        subtitle="Grand Vizier",
        border_style="bright_yellow",
        padding=(1, 2),
    )


def _format_tool_call(name: str, args: dict) -> str:
    """Format a tool call for display in the advisor panel."""
    if name == "web_search":
        query = args.get("query", "")
        return f"🔍 Searching: \"{query}\"..."
    elif name == "read_file":
        path = args.get("path", "")
        return f"📄 Reading: {path}..."
    elif name == "list_files":
        path = args.get("path", ".")
        pattern = args.get("pattern", "*")
        return f"📂 Listing: {path}/{pattern}..."
    elif name == "grep_search":
        pattern = args.get("pattern", "")
        return f"🔎 Grepping: \"{pattern}\"..."
    elif name == "run_command":
        cmd = args.get("command", "")
        return f"⚡ Running: {cmd}..."
    else:
        return f"🔧 {name}..."


def _extract_text_content(content: str | list) -> str:
    """Extract text from AIMessage content, which may be a string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


async def _run_advisor_with_tools(
    advisor: Advisor,
    messages: list,
    model: BaseChatModel,
    tools: list[BaseTool],
    buffer_ref: dict[str, str],
) -> None:
    """Run an advisor with tools using an agentic invoke loop.

    Tool usage lines are appended to the buffer so the display shows progress.
    The final response text replaces the tool lines.
    """
    tool_map = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    tool_lines: list[str] = []
    hit_iteration_limit = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        result = await model_with_tools.ainvoke(messages)
        messages.append(result)

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            # Show tool usage in buffer
            display_line = _format_tool_call(tc["name"], tc["args"])
            tool_lines.append(display_line)
            buffer_ref[advisor.id] = "\n".join(tool_lines) + "\n"

            # Execute tool
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                try:
                    output = await tool_fn.ainvoke(tc["args"])
                except Exception as e:
                    output = f"Tool error: {e}"
            else:
                output = f"Unknown tool: {tc['name']}"

            messages.append(ToolMessage(
                content=str(output),
                tool_call_id=tc["id"],
            ))

        if iteration == MAX_TOOL_ITERATIONS - 1:
            hit_iteration_limit = True

    # Extract text content (handles both str and list[dict] content formats)
    final_content = _extract_text_content(result.content)

    if hit_iteration_limit and not final_content:
        final_content = f"[Advisor reached tool iteration limit ({MAX_TOOL_ITERATIONS}) without producing a final response]"

    if tool_lines:
        tool_header = "\n".join(tool_lines)
        buffer_ref[advisor.id] = f"{tool_header}\n\n{final_content}"
    else:
        buffer_ref[advisor.id] = final_content


async def stream_advisor(
    advisor: Advisor,
    question: str,
    model: BaseChatModel,
) -> tuple[Advisor, str]:
    """Stream a single advisor's response and return the full text."""
    messages = [
        SystemMessage(content=advisor.system_prompt),
        HumanMessage(content=question),
    ]
    chunks = []
    try:
        async for chunk in model.astream(messages):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                chunks.append(chunk.content)
    except Exception as e:
        chunks = [f"[Advisor error: {e}]"]

    return advisor, "".join(chunks)


async def run_deliberation_streaming(
    question: str,
    advisors: list[Advisor],
    synthesizer: Advisor,
    advisor_model: BaseChatModel,
    synthesis_model: BaseChatModel,
    session: Session | None = None,
    round_num: int = 1,
    total_rounds: int = 1,
    context_pairs: list[dict] | None = None,
    advisor_tools: dict[str, list[BaseTool]] | None = None,
    advisor_memory_texts: dict[str, str] | None = None,
    synthesis_memory_text: str = "",
    attachments: list | None = None,
) -> dict[str, str]:
    """Run the full deliberation with streaming display.

    Returns dict mapping advisor IDs to their responses, plus 'synthesis' key.

    If session is provided, advisors receive their per-advisor history and
    Bas Vezir receives the full deliberation history.

    advisor_tools: optional dict mapping advisor ID to resolved tool instances.
    Advisors with tools use an invoke loop (no streaming during tool use),
    advisors without tools stream normally.
    """
    # Build the enriched question with context if provided
    if context_pairs or attachments:
        from divan.context import format_context_for_advisors

        enriched_question = format_context_for_advisors(
            question, context_pairs, attachments=attachments
        )
    else:
        enriched_question = question

    # Print header (only on round 1)
    if round_num == 1:
        console.print()
        console.print(render_header(question))
        console.print()

    # Print round header if multi-round
    if total_rounds > 1:
        console.print(Rule(
            f"Round {round_num} of {total_rounds}",
            style="bright_yellow",
        ))
        console.print()

    # Phase 1: All advisors deliberate in parallel with streaming
    buffers: dict[str, str] = {a.id: "" for a in advisors}
    errors: dict[str, str] = {}
    timings: dict[str, float] = {}
    advisor_map: dict[str, Advisor] = {a.id: a for a in advisors}
    completed: set[str] = set()

    def build_display() -> Group:
        panels = []
        for advisor in advisors:
            panels.append(render_advisor_panel(
                advisor,
                buffers[advisor.id],
                streaming=advisor.id not in completed,
                error=errors.get(advisor.id, ""),
                elapsed=timings.get(advisor.id),
            ))
        return Group(*panels)

    async def stream_one_advisor(advisor: Advisor) -> None:
        start_time = time.monotonic()

        # Build message history: system prompt (with memory) + conversation history + current question
        system_prompt = advisor.system_prompt
        if advisor_memory_texts and advisor.id in advisor_memory_texts:
            memory_text = advisor_memory_texts[advisor.id]
            if memory_text:
                system_prompt = f"{memory_text}\n\n{system_prompt}"

        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=system_prompt),
        ]

        if session:
            # For debate rounds (round 2+), use debate history that includes synthesis
            if round_num > 1:
                history = build_advisor_debate_history(session, advisor.id)
            else:
                history = build_advisor_history(session, advisor.id)
            messages.extend(history)

        # Add the current question (enriched with context if available)
        messages.append(HumanMessage(content=enriched_question))

        # Check if this advisor has tools
        tools = (advisor_tools or {}).get(advisor.id)

        if tools:
            # Two-phase: invoke loop with tools, no streaming
            try:
                await _run_advisor_with_tools(advisor, messages, advisor_model, tools, buffers)
            except Exception as e:
                errors[advisor.id] = str(e)
        else:
            # Pure streaming, no tools
            try:
                async for chunk in advisor_model.astream(messages):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        text = _extract_text_content(chunk.content)
                        if text:
                            buffers[advisor.id] += text
            except Exception as e:
                errors[advisor.id] = str(e)

        # Detect silent failures: advisor completed but produced nothing
        if not buffers[advisor.id].strip() and advisor.id not in errors:
            errors[advisor.id] = "Advisor completed but produced no response (possible model or tool issue)"

        timings[advisor.id] = time.monotonic() - start_time
        completed.add(advisor.id)

    with Live(build_display(), console=console, refresh_per_second=8) as live:
        tasks = [asyncio.create_task(stream_one_advisor(a)) for a in advisors]
        gather_task = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))

        # Refresh display while advisors are streaming
        while not gather_task.done():
            live.update(build_display())
            await asyncio.sleep(0.1)

        # Ensure all tasks have resolved and render final state
        await gather_task
        live.update(build_display())

    console.print()

    # Post-deliberation error summary
    if errors:
        failed_names = []
        for aid, err_msg in errors.items():
            advisor = advisor_map[aid]
            failed_names.append(f"  {advisor.icon} {advisor.name}: {err_msg}")
        error_summary = "\n".join(failed_names)
        console.print(Panel(
            Text.from_markup(
                f"[bold yellow]Warning:[/bold yellow] {len(errors)} advisor(s) failed:\n\n{error_summary}\n\n"
                f"[dim]Tip: retry with --advisors to target specific advisors[/dim]"
            ),
            border_style="yellow",
            title="⚠  Advisor Errors",
            padding=(1, 2),
        ))
        console.print()

    # Phase 2: Bas Vezir synthesis with streaming
    previous_rounds = ""
    if session:
        previous_rounds = build_synthesis_history(session)

    # Build advisor responses for synthesis, noting any failures
    advisor_entries = []
    for advisor in advisors:
        aid = advisor.id
        response = buffers[aid]
        if aid in errors and not response.strip():
            response = f"[This advisor did not respond due to an error: {errors[aid]}]"
        elif not response.strip() and aid not in errors:
            response = "[This advisor returned no response.]"
        advisor_entries.append({
            "name": advisor_map[aid].name,
            "title": advisor_map[aid].title,
            "icon": advisor_map[aid].icon,
            "response": response,
        })

    synthesis_prompt = build_synthesis_prompt(
        enriched_question,
        advisor_entries,
        previous_rounds=previous_rounds,
        round_num=round_num,
        past_verdicts=synthesis_memory_text,
    )

    synthesis_buffer = ""
    synthesis_messages = [
        SystemMessage(content=synthesizer.system_prompt),
        HumanMessage(content=synthesis_prompt),
    ]

    with Live(
        render_synthesis_panel("", streaming=True),
        console=console,
        refresh_per_second=8,
    ) as live:
        try:
            async for chunk in synthesis_model.astream(synthesis_messages):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    text = _extract_text_content(chunk.content)
                    if text:
                        synthesis_buffer += text
                        live.update(render_synthesis_panel(synthesis_buffer, streaming=True))
        except Exception as e:
            synthesis_buffer += f"\n[Synthesis error: {e}]"

        live.update(render_synthesis_panel(synthesis_buffer))

    console.print()

    # Build result (include error marker in buffer for failed advisors)
    result = {}
    for aid in buffers:
        if aid in errors and not buffers[aid].strip():
            result[aid] = f"[Error: {errors[aid]}]"
        else:
            result[aid] = buffers[aid]
    result["synthesis"] = synthesis_buffer
    return result
