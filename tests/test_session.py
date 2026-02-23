"""Tests for Divan session persistence and history building."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from divan.session import (
    Session,
    append_entry,
    build_advisor_debate_history,
    build_advisor_history,
    build_synthesis_history,
    create_session,
    list_sessions,
    load_session,
    save_advisor_response,
    save_question,
    save_synthesis,
)


@pytest.fixture(autouse=True)
def _use_tmp_dir(monkeypatch, tmp_path):
    """Redirect .divan/ to a temp directory for all tests."""
    import divan.session as mod

    monkeypatch.setattr(mod, "DIVAN_DIR", tmp_path / ".divan")
    monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path / ".divan" / "sessions")
    monkeypatch.setattr(mod, "MEMORY_DIR", tmp_path / ".divan" / "memory")


class TestSessionCRUD:
    def test_create_session(self):
        session = create_session()
        assert session.id
        assert session.created_at > 0
        assert session.entries == []

    def test_save_and_load_session(self):
        session = create_session()
        save_question(session.id, "Should I build X?")
        save_advisor_response(
            session.id,
            advisor_id="contrarian",
            name="The Contrarian",
            title="Muhalif",
            icon="\u2694\ufe0f",
            content="This will fail.",
        )
        save_synthesis(session.id, "The council has spoken.")

        loaded = load_session(session.id)
        assert len(loaded.entries) == 3
        assert loaded.entries[0]["type"] == "question"
        assert loaded.entries[0]["content"] == "Should I build X?"
        assert loaded.entries[1]["type"] == "advisor_response"
        assert loaded.entries[1]["advisor_id"] == "contrarian"
        assert loaded.entries[2]["type"] == "synthesis"

    def test_load_nonexistent_session(self):
        with pytest.raises(FileNotFoundError):
            load_session("nonexistent-uuid")

    def test_list_sessions(self):
        s1 = create_session()
        save_question(s1.id, "First question")

        s2 = create_session()
        save_question(s2.id, "Second question")

        sessions = list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0].first_question == "Second question"
        assert sessions[1].first_question == "First question"

    def test_list_sessions_empty(self):
        sessions = list_sessions()
        assert sessions == []

    def test_session_properties(self):
        session = create_session()
        save_question(session.id, "Q1")
        save_question(session.id, "Q2")

        loaded = load_session(session.id)
        assert loaded.num_rounds == 2
        assert loaded.first_question == "Q1"
        assert loaded.questions == ["Q1", "Q2"]

    def test_jsonl_roundtrip(self):
        """Each line in the JSONL file is valid JSON."""
        session = create_session()
        save_question(session.id, "Test question")
        save_advisor_response(
            session.id,
            advisor_id="operator",
            name="The Operator",
            title="Sadrazam",
            icon="\u2699\ufe0f",
            content="Ship it.",
        )

        import divan.session as mod

        path = mod.SESSIONS_DIR / f"{session.id}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        for line in lines:
            entry = json.loads(line)
            assert "type" in entry
            assert "timestamp" in entry


class TestHistoryBuilding:
    def _build_multi_round_session(self) -> Session:
        """Create a session with 2 rounds of deliberation."""
        session = create_session()

        # Round 1
        save_question(session.id, "Should I build X?")
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "It will fail.")
        save_advisor_response(session.id, "operator", "The Operator", "Sadrazam", "\u2699\ufe0f", "Ship fast.")
        save_advisor_response(session.id, "visionary", "The Visionary", "Kahin", "\ud83d\udd2d", "Think big.")
        save_synthesis(session.id, "Round 1 synthesis.")

        # Round 2
        save_question(session.id, "What about timing?")
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "Too late.")
        save_advisor_response(session.id, "operator", "The Operator", "Sadrazam", "\u2699\ufe0f", "Now or never.")
        save_advisor_response(session.id, "visionary", "The Visionary", "Kahin", "\ud83d\udd2d", "Timing is right.")
        save_synthesis(session.id, "Round 2 synthesis.")

        return load_session(session.id)

    def test_advisor_sees_only_own_responses(self):
        session = self._build_multi_round_session()
        history = build_advisor_history(session, "contrarian")

        # Should see: Q1, contrarian response 1, Q2, contrarian response 2
        assert len(history) == 4
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "Should I build X?"
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "It will fail."
        assert isinstance(history[2], HumanMessage)
        assert history[2].content == "What about timing?"
        assert isinstance(history[3], AIMessage)
        assert history[3].content == "Too late."

    def test_advisor_does_not_see_other_advisors(self):
        session = self._build_multi_round_session()
        history = build_advisor_history(session, "operator")

        # Only operator's responses, not contrarian's or visionary's
        contents = [m.content for m in history if isinstance(m, AIMessage)]
        assert "Ship fast." in contents
        assert "Now or never." in contents
        assert "It will fail." not in contents
        assert "Think big." not in contents

    def test_synthesis_history_includes_everything(self):
        session = self._build_multi_round_session()
        history = build_synthesis_history(session)

        assert "Should I build X?" in history
        assert "What about timing?" in history
        assert "It will fail." in history
        assert "Ship fast." in history
        assert "Think big." in history
        assert "Round 1 synthesis." in history
        assert "Round 2 synthesis." in history
        assert "Round 1" in history
        assert "Round 2" in history

    def test_empty_session_advisor_history(self):
        session = create_session()
        loaded = load_session(session.id)
        history = build_advisor_history(loaded, "contrarian")
        assert history == []

    def test_empty_session_synthesis_history(self):
        session = create_session()
        loaded = load_session(session.id)
        history = build_synthesis_history(loaded)
        assert history == ""


class TestDebateHistory:
    def _build_multi_round_session(self) -> Session:
        """Create a session with 2 rounds including synthesis."""
        session = create_session()

        # Round 1
        save_question(session.id, "Should I build X?")
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "It will fail.")
        save_advisor_response(session.id, "operator", "The Operator", "Sadrazam", "\u2699\ufe0f", "Ship fast.")
        save_synthesis(session.id, "The council is divided.")

        return load_session(session.id)

    def test_debate_history_includes_synthesis(self):
        session = self._build_multi_round_session()
        history = build_advisor_debate_history(session, "contrarian")

        # Should see: Q, contrarian response, synthesis as HumanMessage
        assert len(history) == 3
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "Should I build X?"
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "It will fail."
        assert isinstance(history[2], HumanMessage)
        assert "The council is divided." in history[2].content

    def test_debate_history_synthesis_is_human_message(self):
        """Synthesis should be injected as HumanMessage, not AIMessage."""
        session = self._build_multi_round_session()
        history = build_advisor_debate_history(session, "contrarian")

        synthesis_msg = history[2]
        assert isinstance(synthesis_msg, HumanMessage)
        assert "Bas Vezir" in synthesis_msg.content

    def test_debate_history_excludes_other_advisors(self):
        session = self._build_multi_round_session()
        history = build_advisor_debate_history(session, "operator")

        contents = [m.content for m in history]
        assert any("Ship fast." in c for c in contents)
        assert not any("It will fail." in c for c in contents)

    def test_debate_history_empty_session(self):
        session = create_session()
        loaded = load_session(session.id)
        history = build_advisor_debate_history(loaded, "contrarian")
        assert history == []

    def test_debate_history_two_rounds(self):
        session = create_session()

        # Round 1
        save_question(session.id, "Q1")
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "R1 response")
        save_synthesis(session.id, "R1 synthesis")

        # Round 2
        save_question(session.id, "Q1")
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "R2 response")
        save_synthesis(session.id, "R2 synthesis")

        loaded = load_session(session.id)
        history = build_advisor_debate_history(loaded, "contrarian")

        # Q1, R1 response, R1 synthesis, Q1 (again), R2 response, R2 synthesis
        assert len(history) == 6
        assert isinstance(history[0], HumanMessage)  # Q1
        assert isinstance(history[1], AIMessage)      # R1 response
        assert isinstance(history[2], HumanMessage)   # R1 synthesis
        assert isinstance(history[3], HumanMessage)   # Q1 (round 2)
        assert isinstance(history[4], AIMessage)      # R2 response
        assert isinstance(history[5], HumanMessage)   # R2 synthesis
