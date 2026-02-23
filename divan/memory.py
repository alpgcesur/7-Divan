"""Cross-session memory for Divan advisors.

Stores per-advisor insights and shared verdicts as JSONL files in .divan/memory/.
Memory is automatically generated after each deliberation and injected into
advisor system prompts at deliberation time.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from divan.session import MEMORY_DIR


# --- Data models ---


@dataclass
class AdvisorMemory:
    session_id: str
    timestamp: float
    question: str
    key_insight: str
    tags: list[str]


@dataclass
class VerdictMemory:
    session_id: str
    timestamp: float
    question: str
    verdict: str
    summary: str
    tags: list[str]


# --- Storage paths ---


def _advisor_memory_path(advisor_id: str) -> Path:
    return MEMORY_DIR / f"{advisor_id}.jsonl"


def _verdicts_path() -> Path:
    return MEMORY_DIR / "_verdicts.jsonl"


def ensure_memory_dir() -> None:
    """Create .divan/memory/ if it doesn't exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# --- Loading ---


def load_advisor_memories(advisor_id: str, limit: int = 5) -> list[AdvisorMemory]:
    """Load recent memories for a specific advisor.

    Returns up to `limit` most recent entries, newest first.
    """
    path = _advisor_memory_path(advisor_id)
    if not path.exists():
        return []

    entries: list[AdvisorMemory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(AdvisorMemory(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            question=data["question"],
            key_insight=data["key_insight"],
            tags=data.get("tags", []),
        ))

    # Sort by timestamp descending, return most recent
    entries.sort(key=lambda m: m.timestamp, reverse=True)
    return entries[:limit]


def load_verdict_memories(limit: int = 3) -> list[VerdictMemory]:
    """Load recent council verdicts.

    Returns up to `limit` most recent verdicts, newest first.
    """
    path = _verdicts_path()
    if not path.exists():
        return []

    entries: list[VerdictMemory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(VerdictMemory(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            question=data["question"],
            verdict=data["verdict"],
            summary=data["summary"],
            tags=data.get("tags", []),
        ))

    entries.sort(key=lambda m: m.timestamp, reverse=True)
    return entries[:limit]


def count_deliberations() -> int:
    """Count total past deliberations by counting verdict entries."""
    path = _verdicts_path()
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


# --- Formatting for injection ---


def format_advisor_memory(advisor_id: str, memories: list[AdvisorMemory], verdicts: list[VerdictMemory]) -> str:
    """Format memories into text to prepend to an advisor's system prompt.

    Returns empty string if no memories exist.
    """
    if not memories and not verdicts:
        return ""

    parts = ["## Your memory of past Divan sessions\n"]

    if memories:
        parts.append("### Your past insights")
        for m in memories:
            tags_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            parts.append(f'- On "{m.question}"{tags_str}: {m.key_insight}')
        parts.append("")

    if verdicts:
        parts.append("### Recent council decisions")
        for v in verdicts:
            parts.append(f'- "{v.question}" -> {v.verdict}: {v.summary}')
        parts.append("")

    parts.append(
        "Use these memories naturally. Reference past advice when relevant, "
        "note how situations have evolved, and build on previous insights. "
        "Do not list memories mechanically.\n"
    )

    return "\n".join(parts)


def format_verdict_memory_for_synthesis(verdicts: list[VerdictMemory]) -> str:
    """Format past verdicts for the Bas Vezir synthesis prompt.

    Returns empty string if no verdicts exist.
    """
    if not verdicts:
        return ""

    parts = ["## Past council decisions\n"]
    for v in verdicts:
        parts.append(f'- "{v.question}" -> {v.verdict}: {v.summary}')
    parts.append("")
    parts.append(
        "Reference past decisions when relevant. Note patterns, evolution of thinking, "
        "and whether previous advice was followed.\n"
    )
    return "\n".join(parts)


# --- Saving ---


def save_advisor_memory(advisor_id: str, memory: AdvisorMemory) -> None:
    """Append a single memory entry for an advisor."""
    ensure_memory_dir()
    path = _advisor_memory_path(advisor_id)
    entry = {
        "session_id": memory.session_id,
        "timestamp": memory.timestamp,
        "question": memory.question,
        "key_insight": memory.key_insight,
        "tags": memory.tags,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_verdict_memory(verdict: VerdictMemory) -> None:
    """Append a single verdict entry."""
    ensure_memory_dir()
    path = _verdicts_path()
    entry = {
        "session_id": verdict.session_id,
        "timestamp": verdict.timestamp,
        "question": verdict.question,
        "verdict": verdict.verdict,
        "summary": verdict.summary,
        "tags": verdict.tags,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --- LLM-based memory extraction ---


MEMORY_EXTRACTION_PROMPT = """\
You are extracting key memories from a Divan advisory council deliberation.

The user asked: "{question}"

Here are the advisor responses and the final synthesis:

{advisor_section}

SYNTHESIS (Bas Vezir):
{synthesis}

Extract a JSON object with:
1. "advisor_insights": a dict mapping advisor ID to their single most important insight (1 sentence max).
2. "verdict": the council's overall verdict in 3-5 words (e.g., "DO IT BUT VALIDATE FIRST").
3. "summary": a 1-sentence summary of the council's collective advice.
4. "tags": 2-3 lowercase topic tags for this deliberation.

Output ONLY the JSON object. No markdown fences, no explanation.

Example:
{{"advisor_insights": {{"contrarian": "Warned about 3-month runway being insufficient.", "operator": "Suggested shipping an MVP in one weekend."}}, "verdict": "VALIDATE BEFORE COMMITTING", "summary": "Council agreed the idea has potential but needs customer validation before any career changes.", "tags": ["startup", "career"]}}"""


async def extract_memories(
    question: str,
    advisor_responses: dict[str, str],
    synthesis: str,
    model: BaseChatModel,
) -> dict:
    """Use an LLM to extract key memories from a deliberation.

    Returns parsed dict with advisor_insights, verdict, summary, tags.
    """
    advisor_section = ""
    for advisor_id, response in advisor_responses.items():
        advisor_section += f"\n{advisor_id.upper()}:\n{response}\n"

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        question=question,
        advisor_section=advisor_section,
        synthesis=synthesis,
    )

    result = await model.ainvoke([HumanMessage(content=prompt)])
    raw = result.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    return json.loads(raw)


async def generate_and_save_memories(
    session_id: str,
    question: str,
    advisor_responses: dict[str, str],
    synthesis: str,
    model: BaseChatModel,
) -> None:
    """Extract memories from a deliberation and save them.

    This is the main entry point called after each deliberation completes.
    Makes one LLM call, then writes per-advisor + verdict JSONL entries.
    """
    try:
        extracted = await extract_memories(question, advisor_responses, synthesis, model)
    except Exception:
        # Memory extraction failure should never break the main flow
        return

    now = time.time()
    tags = extracted.get("tags", [])

    # Save per-advisor insights
    insights = extracted.get("advisor_insights", {})
    for advisor_id, insight in insights.items():
        save_advisor_memory(advisor_id, AdvisorMemory(
            session_id=session_id,
            timestamp=now,
            question=question,
            key_insight=insight,
            tags=tags,
        ))

    # Save verdict
    save_verdict_memory(VerdictMemory(
        session_id=session_id,
        timestamp=now,
        question=question,
        verdict=extracted.get("verdict", ""),
        summary=extracted.get("summary", ""),
        tags=tags,
    ))


# --- Memory management ---


def clear_all_memories() -> int:
    """Delete all memory files. Returns count of files deleted."""
    if not MEMORY_DIR.exists():
        return 0
    count = 0
    for f in MEMORY_DIR.glob("*.jsonl"):
        f.unlink()
        count += 1
    return count


def clear_advisor_memory(advisor_id: str) -> bool:
    """Delete memory file for a specific advisor. Returns True if file existed."""
    path = _advisor_memory_path(advisor_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_memory_files() -> list[tuple[str, int]]:
    """List memory files with entry counts. Returns [(filename, count), ...]."""
    if not MEMORY_DIR.exists():
        return []
    results = []
    for f in MEMORY_DIR.glob("*.jsonl"):
        count = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        results.append((f.stem, count))
    results.sort()
    return results
