"""Advisor model and persona loader."""

import re
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
    tools: list[str] = field(default_factory=list)

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
        tools=frontmatter.get("tools", []),
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


def next_advisor_order(personas_dir: str | Path) -> int:
    """Return max non-synthesizer order + 1. If advisors have orders 1,2,3,4 returns 5."""
    advisors = get_advisors(personas_dir)
    if not advisors:
        return 1
    return max(a.order for a in advisors) + 1


def slugify_name(name: str) -> str:
    """Convert advisor name to file-safe ID.

    'The Economist' -> 'economist'. Strips 'The ' prefix, lowercases,
    replaces spaces/hyphens with underscores, removes other special chars.
    """
    slug = name.strip()
    if slug.lower().startswith("the "):
        slug = slug[4:]
    slug = slug.strip().lower()
    slug = re.sub(r"[\s\-]+", "_", slug)
    slug = re.sub(r"[^\w]", "", slug)
    return slug


def write_persona_file(
    personas_dir: str | Path,
    name: str,
    title: str,
    icon: str,
    color: str,
    order: int,
    system_prompt: str,
) -> Path:
    """Write a persona markdown file with YAML frontmatter + body.

    Returns the path to the written file.
    """
    file_id = slugify_name(name)
    filepath = Path(personas_dir) / f"{file_id}.md"

    frontmatter = {
        "name": name,
        "title": title,
        "icon": icon,
        "color": color,
        "order": order,
    }

    content = "---\n"
    content += yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    content += "\n---\n\n"
    content += system_prompt.strip()
    content += "\n"

    filepath.write_text(content, encoding="utf-8")
    return filepath
