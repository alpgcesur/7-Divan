"""Polished markdown export for Divan deliberation sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from divan.session import Session


def export_session_markdown(
    session: Session,
    advisor_model: str = "",
    synthesis_model: str = "",
    output_path: Path | str | None = None,
) -> str:
    """Export a session as a polished markdown decision brief.

    Walks session.entries in order. Each "question" entry starts a new round.
    Collects "advisor_response" entries until a "synthesis" entry closes that round.
    Handles "context" entries gracefully if present.

    For single-round sessions, omits the "Round 1" header.

    Args:
        session: The session to export.
        advisor_model: Display string for the advisor model used.
        synthesis_model: Display string for the synthesis model used.
        output_path: If provided, writes the markdown to this file.

    Returns:
        The full markdown string.
    """
    rounds = _collect_rounds(session)
    total_rounds = len(rounds)

    lines: list[str] = []

    # Title
    lines.append("# Divan Deliberation\n")

    # Quote the first question
    if rounds:
        lines.append(f"> {rounds[0]['question']}\n")

    # Metadata line
    meta_parts = []
    dt = datetime.fromtimestamp(session.created_at, tz=timezone.utc)
    meta_parts.append(f"**Date:** {dt.strftime('%Y-%m-%d')}")
    meta_parts.append(f"**Session:** {session.id[:8]}")
    meta_parts.append(f"**Rounds:** {total_rounds}")
    lines.append("  |  ".join(meta_parts) + "  ")

    model_parts = []
    if advisor_model:
        model_parts.append(f"**Advisor model:** {advisor_model}")
    if synthesis_model:
        model_parts.append(f"**Synthesis:** {synthesis_model}")
    if model_parts:
        lines.append("  |  ".join(model_parts) + "\n")
    else:
        lines.append("")

    # Rounds
    for i, round_data in enumerate(rounds, 1):
        lines.append("---\n")

        # Round header (only if multi-round)
        if total_rounds > 1:
            lines.append(f"## Round {i}\n")

            # If this round has a different question (follow-up), show it
            if i > 1:
                lines.append(f"> {round_data['question']}\n")

        # Context (from Feature 3, if present)
        if round_data.get("context_pairs"):
            lines.append("### Additional Context\n")
            for pair in round_data["context_pairs"]:
                lines.append(f"- **Q:** {pair['question']}")
                lines.append(f"  **A:** {pair['answer']}\n")

        # Advisor responses
        for resp in round_data["advisors"]:
            lines.append(f"### {resp['icon']} {resp['name']} ({resp['title']})\n")
            lines.append(f"{resp['content']}\n")

        # Synthesis
        if round_data.get("synthesis"):
            lines.append(f"### 👁 Bas Vezir (Grand Vizier)\n")
            lines.append(f"{round_data['synthesis']}\n")

    result = "\n".join(lines)

    if output_path is not None:
        path = Path(output_path)
        path.write_text(result, encoding="utf-8")

    return result


def _collect_rounds(session: Session) -> list[dict]:
    """Walk session entries and group them into rounds.

    Each round is a dict with:
        question: str
        advisors: list[dict] with keys name, title, icon, content
        synthesis: str | None
        context_pairs: list[dict] | None
    """
    rounds: list[dict] = []
    current: dict | None = None

    for entry in session.entries:
        if entry["type"] == "question":
            # Start a new round
            current = {
                "question": entry["content"],
                "advisors": [],
                "synthesis": None,
                "context_pairs": None,
            }
            rounds.append(current)
        elif entry["type"] == "context" and current is not None:
            current["context_pairs"] = entry.get("pairs", [])
        elif entry["type"] == "advisor_response" and current is not None:
            current["advisors"].append({
                "name": entry["name"],
                "title": entry["title"],
                "icon": entry["icon"],
                "content": entry["content"],
            })
        elif entry["type"] == "synthesis" and current is not None:
            current["synthesis"] = entry["content"]

    return rounds
