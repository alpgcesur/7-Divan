"""Tests for Divan polished markdown export."""

from __future__ import annotations

import pytest

from divan.export import export_session_markdown
from divan.session import (
    create_session,
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


def _add_round(session_id: str, question: str) -> None:
    """Add a complete round (question + 2 advisors + synthesis) to a session."""
    save_question(session_id, question)
    save_advisor_response(session_id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "It will fail.")
    save_advisor_response(session_id, "operator", "The Operator", "Sadrazam", "\u2699\ufe0f", "Ship it fast.")
    save_synthesis(session_id, "The council recommends caution.")


class TestExportSingleRound:
    def test_contains_title_and_question(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "# Divan Deliberation" in md
        assert "> Should I build X?" in md

    def test_contains_metadata(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        md = export_session_markdown(
            session,
            advisor_model="openai:gpt-5-mini",
            synthesis_model="anthropic:claude-sonnet-4-6",
        )
        assert f"**Session:** {session.id[:8]}" in md
        assert "**Rounds:** 1" in md
        assert "**Advisor model:** openai:gpt-5-mini" in md
        assert "**Synthesis:** anthropic:claude-sonnet-4-6" in md

    def test_single_round_no_round_header(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "## Round 1" not in md

    def test_contains_advisor_responses(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "### \u2694\ufe0f The Contrarian (Muhalif)" in md
        assert "It will fail." in md
        assert "### \u2699\ufe0f The Operator (Sadrazam)" in md
        assert "Ship it fast." in md

    def test_contains_synthesis(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "### 👁 Bas Vezir (Grand Vizier)" in md
        assert "The council recommends caution." in md


class TestExportMultiRound:
    def test_multi_round_has_round_headers(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        _add_round(session.id, "What about timing?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "## Round 1" in md
        assert "## Round 2" in md
        assert "**Rounds:** 2" in md

    def test_multi_round_shows_followup_question(self):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        _add_round(session.id, "What about timing?")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "> What about timing?" in md


class TestExportFileOutput:
    def test_writes_to_file(self, tmp_path):
        session = create_session()
        _add_round(session.id, "Should I build X?")
        session = load_session(session.id)

        output_file = tmp_path / "brief.md"
        result = export_session_markdown(session, output_path=output_file)

        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == result

    def test_returns_string_without_file(self):
        session = create_session()
        _add_round(session.id, "Test")
        session = load_session(session.id)

        result = export_session_markdown(session)
        assert isinstance(result, str)
        assert "# Divan Deliberation" in result


class TestExportWithContext:
    def test_handles_context_entries(self):
        """Context entries from Feature 3 should be rendered gracefully."""
        import divan.session as mod

        session = create_session()
        save_question(session.id, "Should I build X?")
        mod.append_entry(session.id, {
            "type": "context",
            "pairs": [
                {"question": "What is your runway?", "answer": "6 months"},
                {"question": "Do you have cofounders?", "answer": "No"},
            ],
        })
        save_advisor_response(session.id, "contrarian", "The Contrarian", "Muhalif", "\u2694\ufe0f", "Risky.")
        save_synthesis(session.id, "Proceed with caution.")
        session = load_session(session.id)

        md = export_session_markdown(session)
        assert "### Additional Context" in md
        assert "What is your runway?" in md
        assert "6 months" in md
