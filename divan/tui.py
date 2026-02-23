"""Interactive TUI menu for Divan using InquirerPy + Rich."""

from __future__ import annotations

import asyncio
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

from divan.advisor import (
    Advisor,
    get_advisors,
    load_all_personas,
    load_persona,
    next_advisor_order,
    slugify_name,
    write_persona_file,
)
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
    tools_customized: bool = False
    memory_enabled: bool = True


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

    # Tool summary
    tool_advisors = [a for a in config.advisors if a.tools]
    if tool_advisors:
        tool_parts = [f"{a.icon} {len(a.tools)}" for a in tool_advisors]
        table.add_row("Tools", ", ".join(tool_parts) + " tools enabled")
    else:
        table.add_row("Tools", "disabled")

    table.add_row("Advisor model", config.advisor_model)
    table.add_row("Synthesis model", config.synthesis_model)
    rounds_label = f"{config.rounds} round{'s' if config.rounds > 1 else ''}"
    if config.rounds == 1:
        rounds_label += " (standard)"
    table.add_row("Debate rounds", rounds_label)

    # Memory status
    from divan.memory import count_deliberations
    mem_count = count_deliberations()
    if config.memory_enabled and mem_count > 0:
        table.add_row("Memory", f"enabled ({mem_count} past deliberations)")
    elif config.memory_enabled:
        table.add_row("Memory", "enabled (no past deliberations)")
    else:
        table.add_row("Memory", "disabled")

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


def prompt_memory() -> bool:
    """Prompt user to manage cross-session memory.

    Returns True if memory should be enabled for this session.
    """
    from divan.memory import (
        clear_all_memories,
        count_deliberations,
        load_verdict_memories,
    )

    num = count_deliberations()

    if num == 0:
        # No memories yet, just inform and enable by default
        _print_section("Memory: no past deliberations yet (will start recording)")
        console.print()
        return True

    choices = [
        Choice(
            value="use",
            name=f"  Use memory ({num} past deliberation{'s' if num != 1 else ''})",
        ),
        Separator("  ──────────"),
        Choice(value="view", name="  View past verdicts..."),
        Choice(value="disable", name="  Disable memory for this session"),
        Choice(value="clear", name="  Clear all memory"),
    ]

    result = inquirer.select(
        message="Memory",
        choices=choices,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    if result == "use":
        return True
    elif result == "disable":
        return False
    elif result == "clear":
        deleted = clear_all_memories()
        console.print(f"  [dim]Cleared {deleted} memory file{'s' if deleted != 1 else ''}.[/dim]")
        console.print()
        return False
    elif result == "view":
        verdicts = load_verdict_memories(limit=10)
        if not verdicts:
            console.print("  [dim]No past verdicts found.[/dim]")
            console.print()
            return True

        table = Table(
            show_header=True,
            show_edge=False,
            padding=(0, 2, 0, 4),
        )
        table.add_column("Question", style="bold", max_width=40)
        table.add_column("Verdict", style="bright_yellow")
        table.add_column("Summary", style="dim", max_width=50)

        for v in verdicts:
            q = v.question[:37] + "..." if len(v.question) > 40 else v.question
            table.add_row(q, v.verdict, v.summary)

        console.print(table)
        console.print()

        # After viewing, ask again
        return prompt_memory()

    return True


CREATE_ADVISOR_SENTINEL = "__create_advisor__"

PERSONA_GENERATION_PROMPT = """\
You are creating a new AI advisor persona for a personal advisory council called "the Divan" \
(inspired by the Ottoman Divan-i Humayun). The council has multiple advisors with distinct \
worldviews who deliberate on a user's question in parallel.

The existing advisors are:
- The Contrarian (Muhalif): stress-tests ideas, finds flaws, plays devil's advocate
- The Operator (Sadrazam): focuses on execution, shipping, practical next steps
- The Visionary (Kahin): thinks 3-5 years out, connects to larger trends
- The Customer (Musteri): role-plays as the potential buyer/user

Based on the user's description below, generate a COMPLETE advisor persona as a JSON object \
with these exact fields:

{{
  "name": "The [Name]",
  "title": "[Ottoman/Turkish-style title]",
  "icon": "[single emoji that fits the role]",
  "color": "[one of: red, blue, green, purple, cyan, magenta, yellow, white]",
  "system_prompt": "[full system prompt, see structure below]"
}}

The system_prompt MUST follow this exact structure:
1. Opening paragraph: "You are [Name] ([Title]), ..." describing their role on the Divan.
2. "## Your approach" section: 2-3 paragraphs on how they think and analyze.
3. "## How you respond" section: 5 bullet points starting with "-".
4. "## Your signature questions (always address at least one)" section: 3 bullet questions.
5. "## Your style" section: 1 paragraph on tone, voice, personality.
6. Closing: "You speak in first person, directly to the user, as if you're a real advisor \
sitting across the table. Keep your response focused and under 400 words."

Rules:
- NEVER use em dashes or en dashes anywhere. Use commas, periods, or colons instead.
- The advisor MUST produce meaningfully different output from the existing advisors listed above.
- Be opinionated and specific. Balance comes from multiple perspectives, not from one balanced advisor.
- Keep the system_prompt under 400 words.
- Pick a color that is NOT red, blue, purple, or green (those are taken by existing advisors).
- Output ONLY the JSON object. No explanation, no markdown code fences.

User's description:
{description}"""


async def _generate_advisor_persona(description: str, model) -> dict:
    """Use an LLM to generate a complete advisor persona from a description."""
    import json

    from langchain_core.messages import HumanMessage

    prompt_text = PERSONA_GENERATION_PROMPT.format(description=description)
    response = await model.ainvoke([HumanMessage(content=prompt_text)])
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    return json.loads(raw)


def run_advisor_creator(
    personas_dir: str,
    advisor_model_spec: str,
    settings: DivanSettings,
) -> Advisor | None:
    """Interactive wizard to create a new custom advisor persona.

    The user describes what kind of advisor they want in plain language.
    The LLM generates the name, title, icon, color, and full system prompt.

    Returns the created Advisor, or None if creation fails.
    """
    console.print()
    console.print(Panel(
        "[bold]Create New Advisor[/bold]\n[dim]Describe the perspective you're missing on your council[/dim]",
        border_style="bright_yellow",
        padding=(1, 2),
    ))
    console.print()

    description = inquirer.text(
        message="Describe your advisor:",
        style=DIVAN_STYLE,
        validate=lambda val: len(val.strip()) > 0,
        invalid_message="Description cannot be empty.",
        long_instruction="e.g., \"Someone who looks at everything through money, incentives, and opportunity costs\"",
    ).execute()
    if not description:
        return None
    description = description.strip()

    # Auto-compute order
    order = next_advisor_order(personas_dir)

    # Generate full persona via LLM
    console.print()
    from divan.models import create_model
    try:
        with console.status("[dim]Generating advisor persona...[/dim]"):
            model = create_model(advisor_model_spec, settings, max_tokens=2000)
            persona_data = asyncio.run(
                _generate_advisor_persona(description, model)
            )
    except Exception as e:
        console.print(f"[red]Error generating persona:[/red] {e}")
        return None

    name = persona_data["name"]
    title = persona_data["title"]
    icon = persona_data["icon"]
    color = persona_data["color"]
    system_prompt = persona_data["system_prompt"]

    # Write file
    filepath = write_persona_file(
        personas_dir=personas_dir,
        name=name,
        title=title,
        icon=icon,
        color=color,
        order=order,
        system_prompt=system_prompt,
    )

    # Load it back to verify
    advisor = load_persona(filepath)

    console.print()
    console.print(Panel(
        f"[bold]{icon}  {name}[/bold] ({title})\n"
        f"[dim]Saved to {filepath}[/dim]",
        title="[bold green]Advisor Created[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()

    return advisor


def prompt_advisors(
    available: list[Advisor],
    personas_dir: str | None = None,
    advisor_model_spec: str | None = None,
    settings: DivanSettings | None = None,
) -> list[Advisor]:
    """Advisor selection with optional create-new-advisor flow.

    Shows a pre-prompt with "Create new advisor..." option when the creation
    params are provided. Then shows the standard checkbox for selection.
    """
    can_create = personas_dir is not None and advisor_model_spec is not None and settings is not None

    # Pre-prompt: offer to create a new advisor before selection
    if can_create:
        pre_choices = [
            Choice(value="select", name="  Select from existing advisors"),
            Separator("  ──────────"),
            Choice(value="create", name="  + Create new advisor..."),
        ]
        action = inquirer.select(
            message="Advisors",
            choices=pre_choices,
            style=DIVAN_STYLE,
            pointer="  \u25b8",
        ).execute()
        console.print()

        if action == "create":
            new_advisor = run_advisor_creator(personas_dir, advisor_model_spec, settings)
            # Reload and re-prompt regardless (new advisor will appear if created)
            refreshed = get_advisors(personas_dir)
            return prompt_advisors(refreshed, personas_dir, advisor_model_spec, settings)

    # Standard checkbox selection
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


def prompt_tools(advisors: list[Advisor]) -> list[Advisor]:
    """Prompt user to manage tool assignments for advisors.

    Shows a top-level choice: use defaults or customize. If customizing,
    shows per-advisor tool toggles.

    Returns updated advisor list (tools field may be modified).
    """
    from divan.tools import ensure_tools_registered, list_available_tools

    ensure_tools_registered()
    all_tools = list_available_tools()

    if not all_tools:
        return advisors

    # Check if any advisor has tools
    any_tools = any(a.tools for a in advisors)

    # Build summary of current tool assignments
    tool_summary_parts = []
    for a in advisors:
        if a.tools:
            tool_summary_parts.append(f"{a.icon} {len(a.tools)} tools")
        else:
            tool_summary_parts.append(f"{a.icon} no tools")

    choices = [
        Choice(
            value="default",
            name=f"  Use defaults ({', '.join(tool_summary_parts)})",
        ),
        Separator("  ──────────"),
        Choice(value="customize", name="  Customize tools..."),
        Choice(value="disable", name="  Disable all tools"),
    ]

    result = inquirer.select(
        message="Tools",
        choices=choices,
        style=DIVAN_STYLE,
        pointer="  \u25b8",
    ).execute()
    console.print()

    if result == "default":
        return advisors
    elif result == "disable":
        for a in advisors:
            a.tools = []
        return advisors

    # Customize: per-advisor tool selection
    for advisor in advisors:
        tool_choices = [
            Choice(
                value=tool_name,
                name=f"  {tool_name}",
                enabled=tool_name in advisor.tools,
            )
            for tool_name in all_tools
        ]

        selected = inquirer.checkbox(
            message=f"{advisor.icon} {advisor.name} tools",
            choices=tool_choices,
            style=DIVAN_STYLE,
            instruction="Space to toggle, Enter to confirm",
        ).execute()
        advisor.tools = selected
        console.print()

    return advisors


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

    # Memory
    memory_enabled = prompt_memory()

    # Advisors
    available = get_advisors(settings.personas_dir)
    if not skip_advisors:
        advisors = prompt_advisors(
            available,
            personas_dir=settings.personas_dir,
            advisor_model_spec=settings.advisor_model,
            settings=settings,
        )
    else:
        advisors = available

    # Tools
    advisors = prompt_tools(advisors)

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
        memory_enabled=memory_enabled,
    )

    _print_config_summary(config)

    return config
