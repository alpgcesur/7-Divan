"""Smart advisor selection via LLM analysis of the user's question."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from divan.advisor import Advisor

SELECTOR_PROMPT = """\
You are a routing system for an advisory council called the Divan. Given a user's question, \
decide which advisors are relevant.

Available advisors:
{advisor_list}

Rules:
- Select at least 2 advisors
- Only include advisors whose perspective is genuinely relevant to this question
- For product/startup questions, include customer and financial advisors
- For career, personal, or technical questions, exclude product-specific advisors (Customer, Defterdar) unless the question directly involves products or finances
- The Contrarian and Operator are relevant to almost any decision

Return ONLY a JSON array of advisor IDs. Example: ["contrarian", "operator", "visionary"]

User's question:
{question}"""


async def select_advisors(
    question: str,
    advisors: list[Advisor],
    model: BaseChatModel,
) -> list[str]:
    """Use a fast LLM call to determine which advisors are relevant to the question.

    Returns a list of advisor IDs. Falls back to all advisors on failure.
    """
    all_ids = [a.id for a in advisors]

    advisor_lines = []
    for a in advisors:
        desc = a.description or a.title
        advisor_lines.append(f"- {a.id}: {a.name} ({a.title}). {desc}")

    prompt_text = SELECTOR_PROMPT.format(
        advisor_list="\n".join(advisor_lines),
        question=question,
    )

    try:
        response = await model.ainvoke([HumanMessage(content=prompt_text)])
        content = response.content
        raw = content if isinstance(content, str) else "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
        raw = raw.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        selected = json.loads(raw)

        if not isinstance(selected, list):
            return all_ids

        # Filter to valid IDs only
        valid = [s for s in selected if s in all_ids]

        # Enforce minimum of 2
        if len(valid) < 2:
            return all_ids

        return valid

    except Exception:
        return all_ids
