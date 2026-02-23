"""Session management for Divan conversations.

Handles persistence of multi-turn deliberations to .divan/sessions/ as JSONL files.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

DIVAN_DIR = Path(".divan")
SESSIONS_DIR = DIVAN_DIR / "sessions"
MEMORY_DIR = DIVAN_DIR / "memory"


@dataclass
class SessionSummary:
    id: str
    first_question: str
    created_at: float
    num_rounds: int


@dataclass
class Session:
    id: str
    created_at: float
    entries: list[dict] = field(default_factory=list)

    @property
    def questions(self) -> list[str]:
        return [e["content"] for e in self.entries if e["type"] == "question"]

    @property
    def num_rounds(self) -> int:
        return len(self.questions)

    @property
    def first_question(self) -> str:
        questions = self.questions
        return questions[0] if questions else ""


def ensure_divan_dir() -> None:
    """Create .divan/sessions/ and .divan/memory/ if they don't exist."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    # Create placeholder memory file
    memory_file = MEMORY_DIR / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("# Divan Memory\n\nPlaceholder for future memory feature.\n")


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.jsonl"


def create_session() -> Session:
    """Create a new empty session and ensure directory exists."""
    ensure_divan_dir()
    session_id = str(uuid.uuid4())
    now = time.time()
    session = Session(id=session_id, created_at=now)
    # Write empty file to reserve the ID
    _session_path(session_id).touch()
    return session


def load_session(session_id: str) -> Session:
    """Load a session from its JSONL file."""
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    entries: list[dict] = []
    created_at = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        entries.append(entry)
        if created_at == 0.0:
            created_at = entry.get("timestamp", 0.0)

    return Session(id=session_id, created_at=created_at, entries=entries)


def append_entry(session_id: str, entry: dict) -> None:
    """Append one JSONL line to a session file."""
    ensure_divan_dir()
    if "timestamp" not in entry:
        entry["timestamp"] = time.time()
    path = _session_path(session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def list_sessions() -> list[SessionSummary]:
    """List all sessions, most recent first."""
    if not SESSIONS_DIR.exists():
        return []

    summaries = []
    for jsonl_file in SESSIONS_DIR.glob("*.jsonl"):
        session_id = jsonl_file.stem
        first_question = ""
        created_at = 0.0
        num_rounds = 0

        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["type"] == "question":
                num_rounds += 1
                if not first_question:
                    first_question = entry["content"]
                    created_at = entry.get("timestamp", 0.0)

        if first_question:
            summaries.append(SessionSummary(
                id=session_id,
                first_question=first_question,
                created_at=created_at,
                num_rounds=num_rounds,
            ))

    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_latest_session() -> Session | None:
    """Get the most recently created session, or None."""
    summaries = list_sessions()
    if not summaries:
        return None
    return load_session(summaries[0].id)


def build_advisor_history(session: Session, advisor_id: str) -> list[HumanMessage | AIMessage]:
    """Build per-advisor message history.

    Each advisor sees: Q1, their response to Q1, Q2, their response to Q2, ...
    Advisors never see other advisors' responses.
    """
    messages: list[HumanMessage | AIMessage] = []
    for entry in session.entries:
        if entry["type"] == "question":
            messages.append(HumanMessage(content=entry["content"]))
        elif entry["type"] == "advisor_response" and entry["advisor_id"] == advisor_id:
            messages.append(AIMessage(content=entry["content"]))
    return messages


def build_synthesis_history(session: Session) -> str:
    """Build full context string for Bas Vezir across all rounds.

    Includes all questions, all advisor responses, and all previous syntheses.
    """
    parts: list[str] = []
    round_num = 0

    for entry in session.entries:
        if entry["type"] == "question":
            round_num += 1
            parts.append(f"--- Round {round_num} ---")
            parts.append(f'Question: "{entry["content"]}"')
        elif entry["type"] == "advisor_response":
            parts.append(f"{entry['icon']}  {entry['name']} ({entry['title']}):\n{entry['content']}")
        elif entry["type"] == "synthesis":
            parts.append(f"Previous Bas Vezir Synthesis:\n{entry['content']}")

    return "\n\n".join(parts)


def save_question(session_id: str, question: str) -> None:
    """Save a question entry to the session."""
    append_entry(session_id, {
        "type": "question",
        "content": question,
    })


def save_advisor_response(
    session_id: str,
    advisor_id: str,
    name: str,
    title: str,
    icon: str,
    content: str,
) -> None:
    """Save an advisor response entry to the session."""
    append_entry(session_id, {
        "type": "advisor_response",
        "advisor_id": advisor_id,
        "name": name,
        "title": title,
        "icon": icon,
        "content": content,
    })


def save_synthesis(session_id: str, content: str) -> None:
    """Save a synthesis entry to the session."""
    append_entry(session_id, {
        "type": "synthesis",
        "content": content,
    })
