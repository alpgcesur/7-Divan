"""Pre-deliberation context gathering for Divan.

Generates clarifying questions based on the user's question, then formats
the answers as structured context for advisors.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

CONTEXT_SYSTEM_PROMPT = """You are a pre-deliberation analyst for an advisory council. Your job is to identify 2-3 critical clarifying questions that would help advisors give much better advice on the user's question.

Rules:
- Return ONLY a JSON array of strings, nothing else
- Each question should be short and specific
- Focus on facts the advisors need (numbers, constraints, context) not opinions
- Do not ask more than 3 questions
- Do not wrap in markdown code blocks

Example output:
["What is your current monthly revenue?", "Do you have a technical co-founder?", "What is your runway in months?"]"""


async def generate_clarifying_questions(
    question: str,
    model: BaseChatModel,
) -> list[str]:
    """Generate 2-3 clarifying questions for the given user question.

    Args:
        question: The user's original question.
        model: The LLM to use for generating questions.

    Returns:
        A list of 2-3 clarifying question strings.
    """
    messages = [
        SystemMessage(content=CONTEXT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    response = await model.ainvoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        questions = json.loads(content)
        if isinstance(questions, list):
            return [str(q) for q in questions[:3]]
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def format_context_for_advisors(
    original_question: str,
    context_pairs: list[dict[str, str]],
) -> str:
    """Format the original question with gathered context for advisors.

    Args:
        original_question: The user's original question.
        context_pairs: List of dicts with 'question' and 'answer' keys.

    Returns:
        An enriched question string with context prepended.
    """
    if not context_pairs:
        return original_question

    parts = [f'The user asks: "{original_question}"']
    parts.append("")
    parts.append("Additional context:")

    for pair in context_pairs:
        parts.append(f"- Q: {pair['question']}")
        parts.append(f"  A: {pair['answer']}")

    parts.append("")
    parts.append("Consider this context in your deliberation.")

    return "\n".join(parts)
