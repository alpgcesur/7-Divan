"""Tests for Divan engine, persona loading, and synthesis."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from divan.advisor import (
    Advisor,
    get_advisors,
    get_synthesizer,
    load_all_personas,
    load_persona,
)
from divan.engine import build_deliberation_graph
from divan.synthesis import build_synthesis_prompt

PERSONAS_DIR = Path(__file__).parent.parent / "personas"


# --- Persona loading tests ---


class TestPersonaLoading:
    def test_load_single_persona(self):
        persona = load_persona(PERSONAS_DIR / "contrarian.md")
        assert persona.id == "contrarian"
        assert persona.name == "The Contrarian"
        assert persona.title == "Muhalif"
        assert persona.icon == "\u2694\ufe0f"
        assert persona.color == "red"
        assert persona.order == 1
        assert not persona.is_synthesizer
        assert len(persona.system_prompt) > 100

    def test_load_all_personas_sorted(self):
        personas = load_all_personas(PERSONAS_DIR)
        assert len(personas) == 5
        orders = [p.order for p in personas]
        assert orders == sorted(orders)

    def test_get_advisors_excludes_synthesizer(self):
        advisors = get_advisors(PERSONAS_DIR)
        assert len(advisors) == 4
        assert all(not a.is_synthesizer for a in advisors)

    def test_get_synthesizer(self):
        synth = get_synthesizer(PERSONAS_DIR)
        assert synth.is_synthesizer
        assert synth.id == "bas_vezir"
        assert synth.name == "Ba\u015f Vezir"

    def test_persona_node_names_unique(self):
        personas = load_all_personas(PERSONAS_DIR)
        node_names = [p.node_name for p in personas]
        assert len(node_names) == len(set(node_names))

    def test_no_em_dashes_in_personas(self):
        """No persona file should contain em dashes."""
        for md_file in PERSONAS_DIR.glob("*.md"):
            content = md_file.read_text()
            assert "\u2014" not in content, f"Em dash found in {md_file.name}"
            assert "\u2013" not in content, f"En dash found in {md_file.name}"


# --- Synthesis prompt tests ---


class TestSynthesisPrompt:
    def test_build_synthesis_prompt(self):
        responses = [
            {"name": "The Contrarian", "title": "Muhalif", "icon": "\u2694\ufe0f", "response": "This will fail."},
            {"name": "The Operator", "title": "Sadrazam", "icon": "\u2699\ufe0f", "response": "Build it now."},
        ]
        prompt = build_synthesis_prompt("Should I do X?", responses)

        assert "Should I do X?" in prompt
        assert "The Contrarian" in prompt
        assert "This will fail." in prompt
        assert "The Operator" in prompt
        assert "Build it now." in prompt
        assert "Divan Karari" in prompt

    def test_synthesis_prompt_includes_all_advisors(self):
        advisors = get_advisors(PERSONAS_DIR)
        responses = [
            {"name": a.name, "title": a.title, "icon": a.icon, "response": f"Response from {a.name}"}
            for a in advisors
        ]
        prompt = build_synthesis_prompt("test question", responses)
        for a in advisors:
            assert a.name in prompt


# --- Graph structure tests ---


class TestGraphStructure:
    def test_graph_builds_without_error(self):
        advisors = get_advisors(PERSONAS_DIR)
        synthesizer = get_synthesizer(PERSONAS_DIR)
        fake_model = FakeListChatModel(responses=["test"])

        graph = build_deliberation_graph(advisors, synthesizer, fake_model, fake_model)
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        advisors = get_advisors(PERSONAS_DIR)
        synthesizer = get_synthesizer(PERSONAS_DIR)
        fake_model = FakeListChatModel(responses=["test"])

        graph = build_deliberation_graph(advisors, synthesizer, fake_model, fake_model)

        node_names = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", synthesizer.node_name}
        expected.update(a.node_name for a in advisors)
        assert expected == node_names


# --- Integration test with fake models ---


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_deliberation_with_fake_model(self):
        advisors = get_advisors(PERSONAS_DIR)
        synthesizer = get_synthesizer(PERSONAS_DIR)

        # FakeListChatModel cycles through responses
        responses = [f"Response from advisor {i}" for i in range(len(advisors))]
        responses.append("Synthesis complete.")

        fake_model = FakeListChatModel(responses=responses)

        graph = build_deliberation_graph(advisors, synthesizer, fake_model, fake_model)
        result = await graph.ainvoke({"query": "Should I build X?", "advisor_responses": []})

        assert result["query"] == "Should I build X?"
        assert len(result["advisor_responses"]) == len(advisors)
        assert result["synthesis"]  # not empty
