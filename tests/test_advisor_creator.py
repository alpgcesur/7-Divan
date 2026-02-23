"""Tests for custom advisor creation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from divan.advisor import (
    Advisor,
    load_persona,
    next_advisor_order,
    slugify_name,
    write_persona_file,
)


class TestSlugifyName:
    def test_strips_the_prefix(self):
        assert slugify_name("The Economist") == "economist"

    def test_lowercase(self):
        assert slugify_name("Economist") == "economist"

    def test_spaces_to_underscores(self):
        assert slugify_name("The Risk Analyst") == "risk_analyst"

    def test_hyphens_to_underscores(self):
        assert slugify_name("Risk-Analyst") == "risk_analyst"

    def test_strips_whitespace(self):
        assert slugify_name("  The  Economist  ") == "economist"

    def test_the_prefix_case_insensitive(self):
        assert slugify_name("the economist") == "economist"

    def test_special_chars_removed(self):
        assert slugify_name("The Data & AI Expert") == "data__ai_expert"

    def test_no_the_prefix(self):
        assert slugify_name("Philosopher") == "philosopher"


class TestNextAdvisorOrder:
    def test_returns_next_order(self, tmp_path: Path):
        """With advisors at orders 1,2 it should return 3."""
        _write_minimal_persona(tmp_path, "a", order=1)
        _write_minimal_persona(tmp_path, "b", order=2)
        assert next_advisor_order(tmp_path) == 3

    def test_empty_dir_returns_1(self, tmp_path: Path):
        assert next_advisor_order(tmp_path) == 1

    def test_ignores_synthesizer(self, tmp_path: Path):
        _write_minimal_persona(tmp_path, "a", order=1)
        _write_minimal_persona(tmp_path, "synth", order=99, is_synthesizer=True)
        assert next_advisor_order(tmp_path) == 2


class TestWritePersonaFile:
    def test_writes_valid_file(self, tmp_path: Path):
        filepath = write_persona_file(
            personas_dir=tmp_path,
            name="The Economist",
            title="Iktisat Naziri",
            icon="📊",
            color="cyan",
            order=5,
            system_prompt="You are The Economist, an advisor on the Divan.\n\n## Your approach\n\nAnalyze everything.",
        )

        assert filepath.exists()
        assert filepath.name == "economist.md"

    def test_roundtrip(self, tmp_path: Path):
        """Write a persona file, load it back, verify all fields match."""
        filepath = write_persona_file(
            personas_dir=tmp_path,
            name="The Economist",
            title="Iktisat Naziri",
            icon="📊",
            color="cyan",
            order=5,
            system_prompt="You are The Economist, an advisor on the Divan.\n\n## Your approach\n\nAnalyze everything.",
        )

        advisor = load_persona(filepath)
        assert advisor.id == "economist"
        assert advisor.name == "The Economist"
        assert advisor.title == "Iktisat Naziri"
        assert advisor.icon == "📊"
        assert advisor.color == "cyan"
        assert advisor.order == 5
        assert not advisor.is_synthesizer
        assert "Analyze everything" in advisor.system_prompt

    def test_file_content_has_frontmatter(self, tmp_path: Path):
        filepath = write_persona_file(
            personas_dir=tmp_path,
            name="Test Advisor",
            title="Test",
            icon="🧪",
            color="red",
            order=1,
            system_prompt="System prompt body here.",
        )

        content = filepath.read_text()
        assert content.startswith("---\n")
        assert "name: Test Advisor" in content
        assert "System prompt body here." in content


def _write_minimal_persona(
    directory: Path, name: str, order: int, is_synthesizer: bool = False
) -> Path:
    """Write a minimal persona .md file for testing."""
    synth_line = f"is_synthesizer: {'true' if is_synthesizer else 'false'}\n"
    content = (
        f"---\n"
        f"name: {name}\n"
        f"title: Title\n"
        f'icon: "🔧"\n'
        f"color: red\n"
        f"order: {order}\n"
        f"{synth_line}"
        f"---\n\n"
        f"System prompt for {name}.\n"
    )
    filepath = directory / f"{name}.md"
    filepath.write_text(content)
    return filepath
