"""Advisor model and persona loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Advisor:
    id: str
    name: str
    title: str
    icon: str
    color: str
    order: int
    system_prompt: str
    is_synthesizer: bool = False

    @property
    def node_name(self) -> str:
        """Name used as LangGraph node identifier."""
        return self.id


def load_persona(filepath: Path) -> Advisor:
    """Load a single persona from a markdown file with YAML frontmatter."""
    text = filepath.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(f"Persona file {filepath} missing YAML frontmatter")

    # Split frontmatter from body
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Persona file {filepath} has malformed frontmatter")

    frontmatter = yaml.safe_load(parts[1])
    system_prompt = parts[2].strip()

    return Advisor(
        id=filepath.stem,
        name=frontmatter["name"],
        title=frontmatter["title"],
        icon=frontmatter["icon"],
        color=frontmatter["color"],
        order=frontmatter["order"],
        system_prompt=system_prompt,
        is_synthesizer=frontmatter.get("is_synthesizer", False),
    )


def load_all_personas(personas_dir: str | Path) -> list[Advisor]:
    """Load all persona files from directory, sorted by order."""
    personas_path = Path(personas_dir)
    if not personas_path.exists():
        raise FileNotFoundError(f"Personas directory not found: {personas_path}")

    advisors = []
    for md_file in personas_path.glob("*.md"):
        advisors.append(load_persona(md_file))

    advisors.sort(key=lambda a: a.order)
    return advisors


def get_advisors(personas_dir: str | Path) -> list[Advisor]:
    """Get all non-synthesizer advisors, sorted by order."""
    return [a for a in load_all_personas(personas_dir) if not a.is_synthesizer]


def get_synthesizer(personas_dir: str | Path) -> Advisor:
    """Get the synthesizer (Bas Vezir)."""
    synthesizers = [a for a in load_all_personas(personas_dir) if a.is_synthesizer]
    if not synthesizers:
        raise ValueError("No synthesizer persona found (needs is_synthesizer: true in frontmatter)")
    return synthesizers[0]
