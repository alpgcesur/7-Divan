"""Rich-based display layer for Divan deliberations."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from divan.advisor import Advisor
from divan.session import Session, build_advisor_history, build_synthesis_history
from divan.synthesis import build_synthesis_prompt


console = Console()


def render_header(question: str) -> Panel:
    """Render the Divan header panel."""
    content = Text.from_markup(
        f"[bold]DİVAN[/bold] [dim]Personal Advisory Council[/dim]\n\n"
        f'[italic]"{question}"[/italic]'
    )
    return Panel(
        content,
        border_style="bright_yellow",
        padding=(1, 2),
    )


def render_advisor_panel(advisor: Advisor, content: str, streaming: bool = False) -> Panel:
    """Render an advisor's response panel."""
    if streaming and not content:
        body = Spinner("dots", text="Deliberating...")
    elif content:
        body = Markdown(content)
    else:
        body = Text("Waiting...", style="dim")

    return Panel(
        body,
        title=f"{advisor.icon}  {advisor.name}",
        subtitle=advisor.title,
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
) -> dict[str, str]:
    """Run the full deliberation with streaming display.

    Returns dict mapping advisor IDs to their responses, plus 'synthesis' key.

    If session is provided, advisors receive their per-advisor history and
    Bas Vezir receives the full deliberation history.
    """
    # Print header
    console.print()
    console.print(render_header(question))
    console.print()

    # Phase 1: All advisors deliberate in parallel with streaming
    buffers: dict[str, str] = {a.id: "" for a in advisors}
    advisor_map: dict[str, Advisor] = {a.id: a for a in advisors}
    completed: set[str] = set()

    def build_display() -> Group:
        panels = []
        for advisor in advisors:
            panels.append(render_advisor_panel(
                advisor,
                buffers[advisor.id],
                streaming=advisor.id not in completed,
            ))
        return Group(*panels)

    async def stream_one_advisor(advisor: Advisor) -> None:
        # Build message history: system prompt + conversation history + current question
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=advisor.system_prompt),
        ]

        if session:
            # Add per-advisor history (previous Q&A pairs for this advisor only)
            history = build_advisor_history(session, advisor.id)
            messages.extend(history)

        # Add the current question
        messages.append(HumanMessage(content=question))

        try:
            async for chunk in advisor_model.astream(messages):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    buffers[advisor.id] += chunk.content
        except Exception as e:
            buffers[advisor.id] += f"\n[Error: {e}]"
        completed.add(advisor.id)

    with Live(build_display(), console=console, refresh_per_second=8) as live:
        tasks = [asyncio.create_task(stream_one_advisor(a)) for a in advisors]

        while not all(a.id in completed for a in advisors):
            live.update(build_display())
            await asyncio.sleep(0.1)

        # Final update
        live.update(build_display())
        await asyncio.gather(*tasks, return_exceptions=True)

    console.print()

    # Phase 2: Bas Vezir synthesis with streaming
    previous_rounds = ""
    if session:
        previous_rounds = build_synthesis_history(session)

    synthesis_prompt = build_synthesis_prompt(
        question,
        [
            {
                "name": advisor_map[aid].name,
                "title": advisor_map[aid].title,
                "icon": advisor_map[aid].icon,
                "response": buffers[aid],
            }
            for aid in [a.id for a in advisors]
        ],
        previous_rounds=previous_rounds,
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
                    synthesis_buffer += chunk.content
                    live.update(render_synthesis_panel(synthesis_buffer, streaming=True))
        except Exception as e:
            synthesis_buffer += f"\n[Synthesis error: {e}]"

        live.update(render_synthesis_panel(synthesis_buffer))

    console.print()

    # Build result
    result = {aid: buffers[aid] for aid in buffers}
    result["synthesis"] = synthesis_buffer
    return result
