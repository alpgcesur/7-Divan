"""CLI entry point for Divan."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from divan.advisor import get_advisors, get_synthesizer, load_all_personas
from divan.config import get_settings
from divan.display import run_deliberation_streaming
from divan.export import export_session_markdown
from divan.models import create_advisor_model, create_synthesis_model
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
) -> dict[str, str]:
    """Run a single deliberation round, saving results to session."""
    # Save question to session
    if session:
        save_question(session.id, question)

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
            skip_advisors=advisor_filter is not None,
            skip_models=model_override is not None,
            skip_rounds=num_rounds is not None,
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

        # Filter advisors if requested
        if advisor_filter:
            filter_ids = {s.strip() for s in advisor_filter.split(",")}
            advisors = [a for a in advisors if a.id in filter_ids]
            if not advisors:
                console.print(f"[red]Error:[/red] No advisors match filter: {advisor_filter}")
                console.print("Use --list to see available advisor IDs.")
                raise SystemExit(1)

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
        )

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
