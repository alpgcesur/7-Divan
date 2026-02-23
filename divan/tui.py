"""Interactive TUI menu for Divan using InquirerPy + Rich."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from InquirerPy.utils import InquirerPyStyle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from divan.advisor import Advisor, get_advisors, load_all_personas
from divan.config import DivanSettings
from divan.session import (
    Session,
    SessionSummary,
    create_session,
    list_sessions,
    load_session,
)

console = Console()

# Ottoman-inspired theme: gold highlights, warm tones
DIVAN_STYLE = InquirerPyStyle({
    "questionmark": "#e5a00d bold",
    "answermark": "#e5a00d bold",
    "answer": "#e5a00d",
    "input": "#ffffff",
    "question": "#ffffff bold",
    "answered_question": "#808080",
    "instruction": "#808080",
    "long_instruction": "#808080",
    "pointer": "#e5a00d bold",
    "checkbox": "#e5a00d",
    "separator": "#808080",
    "skipped": "#808080",
    "validator": "#ff5555",
    "marker": "#e5a00d",
    "fuzzy_prompt": "#e5a00d",
    "fuzzy_info": "#808080",
    "fuzzy_border": "#e5a00d",
    "fuzzy_match": "#e5a00d bold",
})

ADVISOR_MODELS = [
    "google_genai:gemini-2.5-flash",
    "openai:gpt-5-mini-2025-08-07",
    "anthropic:claude-sonnet-4-6",
]

SYNTHESIS_MODELS = [
    "openai:gpt-5.1-2025-11-13",
    "anthropic:claude-sonnet-4-6",
    "google_genai:gemini-2.5-flash",
]


@dataclass
class TUIConfig:
    question: str
    session: Session | None
    advisors: list[Advisor]
    advisor_model: str
    synthesis_model: str
    rounds: int = 1


def _print_banner() -> None:
    """Print the Divan welcome banner."""
    banner = Text()
    banner.append("D I V A N", style="bold bright_yellow")
    banner.append("\n")
    banner.append("Personal Advisory Council", style="dim")
    console.print(Panel(
        banner,
        border_style="bright_yellow",
        padding=(1, 2),
    ))
    console.print()


def _print_section(label: str) -> None:
    """Print a dim section divider."""
    console.print(f"  [dim]{label}[/dim]")


def _print_config_summary(config: TUIConfig) -> None:
    """Print a styled summary of the selected configuration."""
    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 2, 0, 4),
    )
    table.add_column("key", style="dim", width=16)
    table.add_column("value", style="bright_yellow")

    advisor_names = ", ".join(f"{a.icon} {a.name}" for a in config.advisors)
    table.add_row("Advisors", advisor_names)
    table.add_row("Advisor model", config.advisor_model)
    table.add_row("Synthesis model", config.synthesis_model)
    rounds_label = f"{config.rounds} round{'s' if config.rounds > 1 else ''}"
    if config.rounds == 1:
        rounds_label += " (standard)"
    table.add_row("Debate rounds", rounds_label)

    if config.session:
        rounds = config.session.num_rounds
        label = f"Continuing ({rounds} previous rounds)" if rounds > 0 else "New session"
        table.add_row("Session", label)

    console.print()
    console.print(Panel(
        table,
        title="[bold]Configuration[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 0),
    ))
    console.print()


def prompt_question() -> str:
    """Prompt user for their question."""
    _print_section("What would you like the council to deliberate?")
    console.print()
    question = inquirer.text(
        message="Your question:",
        style=DIVAN_STYLE,
        validate=lambda val: len(val.strip()) > 0,
        invalid_message="Question cannot be empty.",
        long_instruction="Be specific. The more context you give, the better the advice.",
    ).execute()
    console.print()
    return question.strip()


def prompt_session_mode(sessions: list[SessionSummary]) -> Session | None:
    """Prompt user to choose new session or continue existing one."""
    if not sessions:
        return create_session()

    choices = [
        Choice(value="new", name="  New session"),
    ]

    # Show the latest session as "Continue latest"
    latest = sessions[0]
    dt = datetime.fromtimestamp(latest.created_at, tz=timezone.utc)
    date_str = dt.strftime("%b %d, %H:%M")
    q_preview = latest.first_question[:50]
    if len(latest.first_question) > 50:
        q_preview += "..."
    choices.append(Choice(
        value=f"continue:{latest.id}",
        name=f"  Continue latest ({latest.num_rounds}r) \"{q_preview}\"",
    ))

    if len(sessions) > 1:
        choices.append(Separator("  ──────────"))
        choices.append(Choice(value="pick", name="  Browse all sessions..."))

    result = inquirer.select(
        message="Session",
        choices=choices,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    if result == "new":
        return create_session()
    elif result == "pick":
        return _prompt_session_picker(sessions)
    else:
        session_id = result.split(":", 1)[1]
        return load_session(session_id)


def _prompt_session_picker(sessions: list[SessionSummary]) -> Session | None:
    """Let user pick from all past sessions."""
    choices = []
    for s in sessions:
        dt = datetime.fromtimestamp(s.created_at, tz=timezone.utc)
        date_str = dt.strftime("%b %d, %H:%M")
        q_preview = s.first_question[:50]
        if len(s.first_question) > 50:
            q_preview += "..."
        choices.append(Choice(
            value=s.id,
            name=f"  [{date_str}]  {s.num_rounds}r  {q_preview}",
        ))

    session_id = inquirer.select(
        message="Select a session",
        choices=choices,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    return load_session(session_id)


def prompt_advisors(available: list[Advisor]) -> list[Advisor]:
    """Multi-select checkbox for advisor selection. All pre-selected."""
    choices = [
        Choice(
            value=a.id,
            name=f"  {a.icon}  {a.name} ({a.title})",
            enabled=True,
        )
        for a in available
    ]

    selected_ids = inquirer.checkbox(
        message="Advisors",
        choices=choices,
        style=DIVAN_STYLE,
        instruction="Space to toggle, Enter to confirm",
        validate=lambda val: len(val) > 0,
        invalid_message="Select at least one advisor.",
    ).execute()
    console.print()

    return [a for a in available if a.id in selected_ids]


def prompt_context_answers(questions: list[str]) -> list[dict[str, str]]:
    """Present clarifying questions and collect answers.

    Empty answers are skipped. Returns only answered pairs.
    """
    _print_section("Clarifying questions (press Enter to skip)")
    console.print()

    pairs: list[dict[str, str]] = []
    for q in questions:
        answer = inquirer.text(
            message=q,
            style=DIVAN_STYLE,
            long_instruction="Leave empty to skip this question.",
        ).execute()

        answer = answer.strip()
        if answer:
            pairs.append({"question": q, "answer": answer})

    console.print()
    return pairs


def prompt_rounds() -> int:
    """Prompt user to select the number of debate rounds."""
    choices = [
        Choice(value=1, name="  1 round (standard)"),
        Choice(value=2, name="  2 rounds (advisors respond to synthesis)"),
        Choice(value=3, name="  3 rounds (deep deliberation)"),
        Separator("  ──────────"),
        Choice(value="__custom__", name="  Custom..."),
    ]

    result = inquirer.select(
        message="Debate rounds",
        choices=choices,
        default=1,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    if result == "__custom__":
        custom = inquirer.number(
            message="How many rounds?",
            style=DIVAN_STYLE,
            min_allowed=1,
            validate=lambda val: val is not None and int(val) >= 1,
            invalid_message="Must be at least 1.",
        ).execute()
        console.print()
        return int(custom)

    return result


def prompt_model(label: str, default: str, presets: list[str]) -> str:
    """Single-select for model with Custom... option."""
    choices = []
    for model in presets:
        provider, name = model.split(":", 1)
        tag = " (default)" if model == default else ""
        choices.append(Choice(
            value=model,
            name=f"  {name}{tag}  [dim]{provider}[/dim]" if False else f"  {name}{tag}",
        ))
    choices.append(Separator("  ──────────"))
    choices.append(Choice(value="__custom__", name="  Custom..."))

    default_value = default if default in presets else presets[0]

    result = inquirer.select(
        message=label,
        choices=choices,
        default=default_value,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    if result == "__custom__":
        custom = inquirer.text(
            message=f"{label} (provider:model_name)",
            style=DIVAN_STYLE,
            validate=lambda val: ":" in val,
            invalid_message="Format: provider:model_name (e.g., openai:gpt-5.2)",
        ).execute().strip()
        console.print()
        return custom

    return result


def run_interactive_setup(
    question: str | None,
    settings: DivanSettings,
    skip_session: bool = False,
    skip_advisors: bool = False,
    skip_models: bool = False,
    skip_rounds: bool = False,
) -> TUIConfig:
    """Run the full interactive setup flow.

    Parameters control which prompts to show (skipped when CLI flags provide values).
    """
    _print_banner()

    # Question
    if question is None:
        question = prompt_question()

    # Session
    session: Session | None = None
    if not skip_session:
        sessions = list_sessions()
        session = prompt_session_mode(sessions)
    else:
        session = create_session()

    # Advisors
    available = get_advisors(settings.personas_dir)
    if not skip_advisors:
        advisors = prompt_advisors(available)
    else:
        advisors = available

    # Models
    if not skip_models:
        advisor_model = prompt_model(
            "Advisor model",
            settings.advisor_model,
            ADVISOR_MODELS,
        )
        synthesis_model = prompt_model(
            "Synthesis model",
            settings.synthesis_model,
            SYNTHESIS_MODELS,
        )
    else:
        advisor_model = settings.advisor_model
        synthesis_model = settings.synthesis_model

    # Rounds
    rounds = 1
    if not skip_rounds:
        rounds = prompt_rounds()

    config = TUIConfig(
        question=question,
        session=session,
        advisors=advisors,
        advisor_model=advisor_model,
        synthesis_model=synthesis_model,
        rounds=rounds,
    )

    _print_config_summary(config)

    return config
