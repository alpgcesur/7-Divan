"""Tests for Divan context gathering."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from divan.context import format_context_for_advisors, generate_clarifying_questions
from divan.session import (
    create_session,
    load_session,
    save_context,
)


@pytest.fixture(autouse=True)
def _use_tmp_dir(monkeypatch, tmp_path):
    """Redirect .divan/ to a temp directory for all tests."""
    import divan.session as mod

    monkeypatch.setattr(mod, "DIVAN_DIR", tmp_path / ".divan")
    monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path / ".divan" / "sessions")
    monkeypatch.setattr(mod, "MEMORY_DIR", tmp_path / ".divan" / "memory")


class TestGenerateClarifyingQuestions:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self):
        questions = ["What is your runway?", "Do you have a cofounder?"]
        fake_model = FakeListChatModel(responses=[json.dumps(questions)])

        result = await generate_clarifying_questions("Should I start a company?", fake_model)
        assert result == questions

    @pytest.mark.asyncio
    async def test_limits_to_three_questions(self):
        questions = ["Q1?", "Q2?", "Q3?", "Q4?"]
        fake_model = FakeListChatModel(responses=[json.dumps(questions)])

        result = await generate_clarifying_questions("test", fake_model)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        fake_model = FakeListChatModel(responses=["not valid json"])

        result = await generate_clarifying_questions("test", fake_model)
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self):
        questions = ["What is your budget?"]
        fenced = f"```json\n{json.dumps(questions)}\n```"
        fake_model = FakeListChatModel(responses=[fenced])

        result = await generate_clarifying_questions("test", fake_model)
        assert result == questions


class TestFormatContext:
    def test_formats_with_pairs(self):
        pairs = [
            {"question": "What is your runway?", "answer": "6 months"},
            {"question": "Do you have cofounders?", "answer": "No"},
        ]
        result = format_context_for_advisors("Should I start a company?", pairs)

        assert 'The user asks: "Should I start a company?"' in result
        assert "Additional context:" in result
        assert "Q: What is your runway?" in result
        assert "A: 6 months" in result
        assert "Q: Do you have cofounders?" in result
        assert "A: No" in result
        assert "Consider this context" in result

    def test_returns_original_when_no_pairs(self):
        result = format_context_for_advisors("Should I do X?", [])
        assert result == "Should I do X?"

    def test_returns_original_when_none(self):
        # Empty list case
        result = format_context_for_advisors("test", [])
        assert result == "test"


class TestSaveContext:
    def test_save_and_load_context(self):
        session = create_session()
        pairs = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        save_context(session.id, pairs)

        loaded = load_session(session.id)
        assert len(loaded.entries) == 1
        assert loaded.entries[0]["type"] == "context"
        assert loaded.entries[0]["pairs"] == pairs

    def test_context_entry_has_timestamp(self):
        session = create_session()
        save_context(session.id, [{"question": "Q?", "answer": "A"}])

        loaded = load_session(session.id)
        assert "timestamp" in loaded.entries[0]
