"""Template loader for pre-configured Divan compositions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DivanTemplate:
    id: str  # filename stem
    name: str
    description: str
    icon: str
    advisors: list[str] = field(default_factory=list)  # advisor IDs
    rounds: int | None = None  # None = use default


def load_template(filepath: Path) -> DivanTemplate:
    """Load a single template from a YAML file."""
    data = yaml.safe_load(filepath.read_text(encoding="utf-8"))

    return DivanTemplate(
        id=filepath.stem,
        name=data["name"],
        description=data.get("description", ""),
        icon=data.get("icon", ""),
        advisors=data.get("advisors", []),
        rounds=data.get("rounds"),
    )


def load_all_templates(templates_dir: str | Path) -> list[DivanTemplate]:
    """Load all template YAML files from directory, sorted by name."""
    templates_path = Path(templates_dir)
    if not templates_path.exists():
        return []

    templates = []
    for yaml_file in sorted(templates_path.glob("*.yaml")):
        try:
            templates.append(load_template(yaml_file))
        except Exception:
            continue  # skip malformed templates

    return templates


def find_template(templates_dir: str | Path, query: str) -> DivanTemplate | None:
    """Find a template by ID or name (case-insensitive)."""
    templates = load_all_templates(templates_dir)
    query_lower = query.lower()
    for t in templates:
        if t.id == query_lower or t.name.lower() == query_lower:
            return t
    return None
