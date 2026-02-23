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

    # Resolve session
    session: Session | None = None
    if session_id:
        try:
            session = load_session(session_id)
        except FileNotFoundError:
            # Try prefix match
            sessions = list_sessions()
            matches = [s for s in sessions if s.id.startswith(session_id)]
            if len(matches) == 1:
                session = load_session(matches[0].id)
            else:
                console.print(f"[red]Error:[/red] Session not found: {session_id}")
                raise SystemExit(1)
        console.print(f"[dim]Continuing session {session.id[:8]}... ({session.num_rounds} previous rounds)[/dim]")
    elif continue_latest:
        session = get_latest_session()
        if session is None:
            console.print("[red]Error:[/red] No previous sessions found.")
            raise SystemExit(1)
        console.print(f"[dim]Continuing session {session.id[:8]}... ({session.num_rounds} previous rounds)[/dim]")
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

    # Run first deliberation
    result = _run_deliberation(question, advisors, synthesizer, advisor_llm, synthesis_llm, session)

    # Save output if requested
    if output:
        _save_output(output, question, advisors, result)

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

    if session:
        console.print(f"\n[dim]Session saved: {session.id[:8]}...[/dim]")


def _save_output(output: str, question: str, advisors, result: dict[str, str]) -> None:
    """Save deliberation result as markdown."""
    output_path = Path(output)
    md_lines = [f"# Divan Deliberation\n\n"]
    md_lines.append(f"> {question}\n\n")

    for advisor in advisors:
        md_lines.append(f"## {advisor.icon} {advisor.name} ({advisor.title})\n\n")
        md_lines.append(f"{result[advisor.id]}\n\n")

    md_lines.append(f"## 👁 Bas Vezir (Grand Vizier)\n\n")
    md_lines.append(f"{result['synthesis']}\n")

    output_path.write_text("".join(md_lines), encoding="utf-8")
    console.print(f"\n[dim]Deliberation saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
