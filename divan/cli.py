"""CLI entry point for Divan."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from divan.advisor import Advisor, get_advisors, get_synthesizer, load_all_personas
from divan.config import get_settings
from divan.templates import find_template, load_all_templates
from divan.display import run_deliberation_streaming
from divan.export import export_session_markdown
from divan.memory import (
    format_advisor_memory,
    format_verdict_memory_for_synthesis,
    generate_and_save_memories,
    load_advisor_memories,
    load_verdict_memories,
)
from divan.models import create_advisor_model, create_synthesis_model
from divan.tools import ensure_tools_registered, get_tools_for_advisor
from divan.session import (
    Session,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
    save_advisor_response,
    save_question,
    save_synthesis,
)

console = Console()


def _resolve_advisor_tools(advisors: list[Advisor]) -> dict[str, list] | None:
    """Build a mapping of advisor ID to resolved tool instances.

    Returns None if no advisor has tools configured.
    """
    ensure_tools_registered()
    tools_map = {}
    for advisor in advisors:
        if advisor.tools:
            resolved = get_tools_for_advisor(advisor.tools)
            if resolved:
                tools_map[advisor.id] = resolved
    return tools_map or None


def _run_deliberation(
    question: str,
    advisors,
    synthesizer,
    advisor_llm,
    synthesis_llm,
    session: Session | None,
    round_num: int = 1,
    total_rounds: int = 1,
    context_pairs: list[dict] | None = None,
    advisor_memory_texts: dict[str, str] | None = None,
    synthesis_memory_text: str = "",
) -> dict[str, str]:
    """Run a single deliberation round, saving results to session."""
    # Save question to session
    if session:
        save_question(session.id, question)

    # Resolve tools for advisors
    advisor_tools = _resolve_advisor_tools(advisors)

    result = asyncio.run(
        run_deliberation_streaming(
            question=question,
            advisors=advisors,
            synthesizer=synthesizer,
            advisor_model=advisor_llm,
            synthesis_model=synthesis_llm,
            session=session,
            round_num=round_num,
            total_rounds=total_rounds,
            context_pairs=context_pairs,
            advisor_tools=advisor_tools,
            advisor_memory_texts=advisor_memory_texts,
            synthesis_memory_text=synthesis_memory_text,
        )
    )

    # Save responses to session
    if session:
        for advisor in advisors:
            save_advisor_response(
                session.id,
                advisor_id=advisor.id,
                name=advisor.name,
                title=advisor.title,
                icon=advisor.icon,
                content=result[advisor.id],
            )
        save_synthesis(session.id, result["synthesis"])

        # Reload session so future rounds see the new entries
        reloaded = load_session(session.id)
        session.entries = reloaded.entries

    return result


def _all_flags_set(
    model_override: str | None,
    advisor_filter: str | None,
    continue_latest: bool,
    session_id: str | None,
) -> bool:
    """Check if enough CLI flags are set to skip TUI entirely."""
    has_session = continue_latest or session_id is not None
    has_advisors = advisor_filter is not None
    has_model = model_override is not None
    return has_session and has_advisors and has_model


@click.command()
@click.argument("question", required=False)
@click.option("--model", "model_override", default=None, help="Model for advisors (provider:model)")
@click.option("--synthesis-model", default=None, help="Model for Bas Vezir synthesis")
@click.option("--advisors", "advisor_filter", default=None, help="Comma-separated advisor IDs to use")
@click.option("--list", "list_personas", is_flag=True, help="List available personas")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save output as markdown")
@click.option("-c", "continue_latest", is_flag=True, help="Continue most recent session")
@click.option("--session", "session_id", default=None, help="Continue a specific session by ID")
@click.option("--history", "show_history", is_flag=True, help="List past sessions")
@click.option("--no-tui", is_flag=True, help="Disable interactive TUI menu")
@click.option("--rounds", "num_rounds", default=None, type=click.IntRange(min=1), help="Number of debate rounds")
@click.option("--no-context", is_flag=True, help="Skip clarifying questions step")
@click.option("--template", "template_id", default=None, help="Use a pre-configured template (ID or name)")
@click.option("--list-templates", is_flag=True, help="List available templates")
def main(
    question: str | None,
    model_override: str | None,
    synthesis_model: str | None,
    advisor_filter: str | None,
    list_personas: bool,
    output: str | None,
    continue_latest: bool,
    session_id: str | None,
    show_history: bool,
    no_tui: bool,
    num_rounds: int | None,
    no_context: bool,
    template_id: str | None,
    list_templates: bool,
) -> None:
    """Divan: Personal Advisory Council.

    Pose a question and receive deliberations from multiple AI advisors
    with distinct worldviews, synthesized into a decision brief.
    """
    # Load settings
    overrides = {}
    if model_override:
        overrides["advisor_model"] = model_override
    if synthesis_model:
        overrides["synthesis_model"] = synthesis_model

    settings = get_settings(**overrides)

    # Handle --list
    if list_personas:
        personas = load_all_personas(settings.personas_dir)
        table = Table(title="Divan Personas")
        table.add_column("Order", style="dim")
        table.add_column("Icon")
        table.add_column("Name", style="bold")
        table.add_column("Title")
        table.add_column("ID", style="dim")
        table.add_column("Role", style="dim")

        for p in personas:
            role = "Synthesizer" if p.is_synthesizer else "Advisor"
            table.add_row(str(p.order), p.icon, p.name, p.title, p.id, role)

        console.print(table)
        return

    # Handle --list-templates
    if list_templates:
        templates = load_all_templates(settings.templates_dir)
        if not templates:
            console.print("[dim]No templates found.[/dim]")
            return

        table = Table(title="Divan Templates")
        table.add_column("Icon")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("Advisors", style="dim")
        table.add_column("Rounds", style="dim", justify="center")

        for t in templates:
            advisors_str = ", ".join(t.advisors)
            rounds_str = str(t.rounds) if t.rounds else "default"
            table.add_row(t.icon, t.id, t.name, t.description, advisors_str, rounds_str)

        console.print(table)
        return

    # Handle --history
    if show_history:
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No past sessions found.[/dim]")
            return

        table = Table(title="Divan Sessions")
        table.add_column("Date", style="dim")
        table.add_column("Rounds", style="dim", justify="center")
        table.add_column("First Question", style="bold")
        table.add_column("Session ID", style="dim")

        for s in sessions:
            dt = datetime.fromtimestamp(s.created_at, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
            table.add_row(date_str, str(s.num_rounds), s.first_question[:80], s.id[:8] + "...")

        console.print(table)
        return

    # Resolve template if provided via CLI
    cli_template = None
    if template_id:
        cli_template = find_template(settings.templates_dir, template_id)
        if cli_template is None:
            console.print(f"[red]Error:[/red] Template not found: {template_id}")
            console.print("Use --list-templates to see available templates.")
            raise SystemExit(1)

    # Determine if we should use interactive TUI
    is_interactive = sys.stdin.isatty() and not no_tui
    has_question = question is not None
    all_flags = _all_flags_set(model_override, advisor_filter, continue_latest, session_id)

    # If interactive and not all flags provided, use TUI
    if is_interactive and not (has_question and all_flags):
        from divan.tui import run_interactive_setup

        tui_config = run_interactive_setup(
            question=question,
            settings=settings,
            skip_session=continue_latest or session_id is not None,
            skip_advisors=advisor_filter is not None or cli_template is not None,
            skip_models=model_override is not None,
            skip_rounds=num_rounds is not None or (cli_template is not None and cli_template.rounds is not None),
            skip_template=cli_template is not None,
            template=cli_template,
        )

        question = tui_config.question
        advisors = tui_config.advisors
        synthesizer = get_synthesizer(settings.personas_dir)

        # Use TUI rounds if not set via CLI
        if num_rounds is None:
            num_rounds = tui_config.rounds

        # Update settings with TUI selections for model creation
        tui_overrides = {}
        if model_override is None:
            tui_overrides["advisor_model"] = tui_config.advisor_model
        if synthesis_model is None:
            tui_overrides["synthesis_model"] = tui_config.synthesis_model
        if tui_overrides:
            settings = get_settings(**{**overrides, **tui_overrides})

        # Session: CLI flags take priority over TUI selection
        if session_id:
            session = _resolve_session_by_id(session_id)
        elif continue_latest:
            session = _resolve_latest_session()
        else:
            session = tui_config.session

    else:
        # Non-interactive path (original behavior)
        # Resolve session
        if session_id:
            session = _resolve_session_by_id(session_id)
        elif continue_latest:
            session = _resolve_latest_session()
        else:
            session = create_session()

        # Get question from argument or stdin
        if question is None:
            if not sys.stdin.isatty():
                question = sys.stdin.read().strip()
            else:
                console.print("[red]Error:[/red] Please provide a question.")
                console.print("Usage: divan \"Your question here\"")
                raise SystemExit(1)

        if not question:
            console.print("[red]Error:[/red] Question cannot be empty.")
            raise SystemExit(1)

        # Load advisors
        advisors = get_advisors(settings.personas_dir)
        synthesizer = get_synthesizer(settings.personas_dir)

        # Apply template advisor filter
        if cli_template:
            advisor_map = {a.id: a for a in advisors}
            advisors = [advisor_map[aid] for aid in cli_template.advisors if aid in advisor_map]
            if not advisors:
                console.print(f"[red]Error:[/red] No advisors match template: {cli_template.name}")
                raise SystemExit(1)
            if cli_template.rounds is not None and num_rounds is None:
                num_rounds = cli_template.rounds

        # Filter advisors if requested
        if advisor_filter:
            filter_ids = {s.strip() for s in advisor_filter.split(",")}
            advisors = [a for a in advisors if a.id in filter_ids]
            if not advisors:
                console.print(f"[red]Error:[/red] No advisors match filter: {advisor_filter}")
                console.print("Use --list to see available advisor IDs.")
                raise SystemExit(1)
        elif question:
            # Smart advisor selection in non-interactive mode
            try:
                from divan.advisor_selector import select_advisors

                advisor_llm_tmp = create_advisor_model(settings)
                selected_ids = asyncio.run(select_advisors(question, advisors, advisor_llm_tmp))
                if selected_ids and len(selected_ids) < len(advisors):
                    advisors = [a for a in advisors if a.id in selected_ids]
                    names = ", ".join(f"{a.icon} {a.name}" for a in advisors)
                    console.print(f"[dim]Selected advisors: {names}[/dim]")
            except Exception:
                pass  # Fall back to all advisors

    # Create models
    try:
        advisor_llm = create_advisor_model(settings)
        synthesis_llm = create_synthesis_model(settings)
    except Exception as e:
        console.print(f"[red]Error creating models:[/red] {e}")
        console.print("Check your API keys in .env file. See .env.example for format.")
        raise SystemExit(1)

    # Default rounds to 1 if not set
    if num_rounds is None:
        num_rounds = 1

    # Memory: determine if enabled and load if so
    memory_enabled = True  # default for non-interactive mode
    tui_was_used = is_interactive and not (has_question and all_flags)
    if tui_was_used:
        memory_enabled = tui_config.memory_enabled

    advisor_memory_texts: dict[str, str] | None = None
    synthesis_memory_text = ""
    if memory_enabled:
        try:
            verdicts = load_verdict_memories(limit=3)
            synthesis_memory_text = format_verdict_memory_for_synthesis(verdicts)
            advisor_memory_texts = {}
            for advisor in advisors:
                memories = load_advisor_memories(advisor.id, limit=5)
                text = format_advisor_memory(advisor.id, memories, verdicts)
                if text:
                    advisor_memory_texts[advisor.id] = text
            if not advisor_memory_texts:
                advisor_memory_texts = None
        except Exception as e:
            console.print(f"[dim]Skipping memory loading: {e}[/dim]")
            advisor_memory_texts = None
            synthesis_memory_text = ""

    # Context gathering (pre-deliberation clarifying questions)
    context_pairs: list[dict] | None = None
    if is_interactive and not no_context:
        from divan.context import generate_clarifying_questions
        from divan.session import save_context
        from divan.tui import prompt_context_answers

        try:
            with console.status("[dim]Generating clarifying questions...[/dim]"):
                clarifying_qs = asyncio.run(
                    generate_clarifying_questions(question, advisor_llm)
                )
            if clarifying_qs:
                context_pairs = prompt_context_answers(clarifying_qs)
                if context_pairs and session:
                    save_context(session.id, context_pairs)
        except Exception as e:
            console.print(f"[dim]Skipping context gathering: {e}[/dim]")

    # Run deliberation rounds
    for round_idx in range(1, num_rounds + 1):
        result = _run_deliberation(
            question, advisors, synthesizer, advisor_llm, synthesis_llm, session,
            round_num=round_idx,
            total_rounds=num_rounds,
            context_pairs=context_pairs if round_idx == 1 else None,
            advisor_memory_texts=advisor_memory_texts if round_idx == 1 else None,
            synthesis_memory_text=synthesis_memory_text if round_idx == 1 else "",
        )

    # Generate and save memories after deliberation
    if memory_enabled and session:
        try:
            # Build advisor_responses dict from the last round's result
            advisor_responses = {a.id: result[a.id] for a in advisors if a.id in result}
            with console.status("[dim]Saving memories...[/dim]"):
                asyncio.run(
                    generate_and_save_memories(
                        session_id=session.id,
                        question=question,
                        advisor_responses=advisor_responses,
                        synthesis=result.get("synthesis", ""),
                        model=advisor_llm,
                    )
                )
        except Exception as e:
            console.print(f"[dim]Memory save skipped: {e}[/dim]")

    # Interactive follow-up loop
    if sys.stdin.isatty():
        while True:
            console.print()
            try:
                followup = console.input("[dim]Follow-up question (Enter to exit):[/dim] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not followup:
                break

            result = _run_deliberation(followup, advisors, synthesizer, advisor_llm, synthesis_llm, session)

    # Save output after all rounds (captures full deliberation)
    if output and session:
        _save_output(output, session, settings.advisor_model, settings.synthesis_model)

    if session:
        console.print(f"\n[dim]Session saved: {session.id[:8]}...[/dim]")


def _resolve_session_by_id(session_id: str) -> Session:
    """Load a session by ID or prefix match."""
    try:
        session = load_session(session_id)
    except FileNotFoundError:
        sessions = list_sessions()
        matches = [s for s in sessions if s.id.startswith(session_id)]
        if len(matches) == 1:
            session = load_session(matches[0].id)
        else:
            console.print(f"[red]Error:[/red] Session not found: {session_id}")
            raise SystemExit(1)
    console.print(f"[dim]Continuing session {session.id[:8]}... ({session.num_rounds} previous rounds)[/dim]")
    return session


def _resolve_latest_session() -> Session:
    """Load the most recent session or error."""
    session = get_latest_session()
    if session is None:
        console.print("[red]Error:[/red] No previous sessions found.")
        raise SystemExit(1)
    console.print(f"[dim]Continuing session {session.id[:8]}... ({session.num_rounds} previous rounds)[/dim]")
    return session


def _save_output(
    output: str,
    session: Session,
    advisor_model: str,
    synthesis_model: str,
) -> None:
    """Save full session as a polished markdown decision brief."""
    output_path = Path(output)
    export_session_markdown(
        session=session,
        advisor_model=advisor_model,
        synthesis_model=synthesis_model,
        output_path=output_path,
    )
    console.print(f"\n[dim]Deliberation saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
