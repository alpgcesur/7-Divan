"""Tests for Divan cross-session memory system."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from divan.memory import (
    AdvisorMemory,
    VerdictMemory,
    clear_advisor_memory,
    clear_all_memories,
    count_deliberations,
    format_advisor_memory,
    format_verdict_memory_for_synthesis,
    list_memory_files,
    load_advisor_memories,
    load_verdict_memories,
    save_advisor_memory,
    save_verdict_memory,
)
from divan.session import MEMORY_DIR


@pytest.fixture(autouse=True)
def clean_memory(tmp_path, monkeypatch):
    """Redirect MEMORY_DIR to a temp directory for each test."""
    test_memory_dir = tmp_path / "memory"
    test_memory_dir.mkdir()
    monkeypatch.setattr("divan.memory.MEMORY_DIR", test_memory_dir)
    yield test_memory_dir


class TestAdvisorMemory:
    def test_save_and_load_advisor_memory(self, clean_memory):
        mem = AdvisorMemory(
            session_id="test-session-1",
            timestamp=time.time(),
            question="Should I build X?",
            key_insight="Warned about insufficient runway.",
            tags=["startup", "career"],
        )
        save_advisor_memory("contrarian", mem)

        loaded = load_advisor_memories("contrarian", limit=5)
        assert len(loaded) == 1
        assert loaded[0].question == "Should I build X?"
        assert loaded[0].key_insight == "Warned about insufficient runway."
        assert loaded[0].tags == ["startup", "career"]

    def test_load_returns_newest_first(self, clean_memory):
        for i in range(5):
            save_advisor_memory("operator", AdvisorMemory(
                session_id=f"session-{i}",
                timestamp=1000.0 + i,
                question=f"Question {i}",
                key_insight=f"Insight {i}",
                tags=[],
            ))

        loaded = load_advisor_memories("operator", limit=3)
        assert len(loaded) == 3
        assert loaded[0].question == "Question 4"  # newest
        assert loaded[2].question == "Question 2"  # oldest in limit

    def test_load_respects_limit(self, clean_memory):
        for i in range(10):
            save_advisor_memory("visionary", AdvisorMemory(
                session_id=f"s-{i}",
                timestamp=float(i),
                question=f"Q{i}",
                key_insight=f"I{i}",
                tags=[],
            ))

        loaded = load_advisor_memories("visionary", limit=5)
        assert len(loaded) == 5

    def test_load_empty_advisor(self, clean_memory):
        loaded = load_advisor_memories("nonexistent")
        assert loaded == []


class TestVerdictMemory:
    def test_save_and_load_verdict(self, clean_memory):
        verdict = VerdictMemory(
            session_id="test-session-1",
            timestamp=time.time(),
            question="Should I pivot?",
            verdict="DO IT BUT VALIDATE",
            summary="Council agreed: validate first.",
            tags=["startup"],
        )
        save_verdict_memory(verdict)

        loaded = load_verdict_memories(limit=3)
        assert len(loaded) == 1
        assert loaded[0].verdict == "DO IT BUT VALIDATE"
        assert loaded[0].summary == "Council agreed: validate first."

    def test_count_deliberations(self, clean_memory):
        assert count_deliberations() == 0

        for i in range(3):
            save_verdict_memory(VerdictMemory(
                session_id=f"s-{i}",
                timestamp=float(i),
                question=f"Q{i}",
                verdict="YES",
                summary="Do it.",
                tags=[],
            ))

        assert count_deliberations() == 3


class TestFormatting:
    def test_format_advisor_memory_empty(self):
        result = format_advisor_memory("contrarian", [], [])
        assert result == ""

    def test_format_advisor_memory_with_data(self):
        memories = [
            AdvisorMemory(
                session_id="s1",
                timestamp=1000.0,
                question="Build X?",
                key_insight="Runway is too short.",
                tags=["startup"],
            ),
        ]
        verdicts = [
            VerdictMemory(
                session_id="s1",
                timestamp=1000.0,
                question="Build X?",
                verdict="WAIT",
                summary="Not yet.",
                tags=[],
            ),
        ]
        result = format_advisor_memory("contrarian", memories, verdicts)
        assert "Your memory of past Divan sessions" in result
        assert "Runway is too short." in result
        assert "Build X?" in result
        assert "WAIT" in result

    def test_format_verdict_memory_for_synthesis_empty(self):
        assert format_verdict_memory_for_synthesis([]) == ""

    def test_format_verdict_memory_for_synthesis_with_data(self):
        verdicts = [
            VerdictMemory(
                session_id="s1",
                timestamp=1000.0,
                question="Pivot?",
                verdict="YES",
                summary="Do it now.",
                tags=[],
            ),
        ]
        result = format_verdict_memory_for_synthesis(verdicts)
        assert "Past council decisions" in result
        assert "Pivot?" in result
        assert "YES" in result


class TestMemoryManagement:
    def test_clear_all_memories(self, clean_memory):
        save_advisor_memory("contrarian", AdvisorMemory(
            session_id="s1", timestamp=1.0, question="Q", key_insight="I", tags=[],
        ))
        save_verdict_memory(VerdictMemory(
            session_id="s1", timestamp=1.0, question="Q", verdict="V", summary="S", tags=[],
        ))

        deleted = clear_all_memories()
        assert deleted == 2
        assert load_advisor_memories("contrarian") == []
        assert load_verdict_memories() == []

    def test_clear_advisor_memory(self, clean_memory):
        save_advisor_memory("operator", AdvisorMemory(
            session_id="s1", timestamp=1.0, question="Q", key_insight="I", tags=[],
        ))

        assert clear_advisor_memory("operator") is True
        assert clear_advisor_memory("operator") is False  # already deleted
        assert load_advisor_memories("operator") == []

    def test_list_memory_files(self, clean_memory):
        save_advisor_memory("contrarian", AdvisorMemory(
            session_id="s1", timestamp=1.0, question="Q", key_insight="I", tags=[],
        ))
        save_advisor_memory("contrarian", AdvisorMemory(
            session_id="s2", timestamp=2.0, question="Q2", key_insight="I2", tags=[],
        ))
        save_verdict_memory(VerdictMemory(
            session_id="s1", timestamp=1.0, question="Q", verdict="V", summary="S", tags=[],
        ))

        files = list_memory_files()
        names = [f[0] for f in files]
        assert "_verdicts" in names
        assert "contrarian" in names

        # contrarian should have 2 entries
        contrarian_entry = next(f for f in files if f[0] == "contrarian")
        assert contrarian_entry[1] == 2
