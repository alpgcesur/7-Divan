"""Synthesis prompt builder for Bas Vezir."""


def build_synthesis_prompt(
    question: str,
    advisor_responses: list[dict],
    previous_rounds: str = "",
) -> str:
    """Build the prompt that Bas Vezir uses to synthesize all advisor responses.

    Args:
        question: The current user question.
        advisor_responses: List of dicts with keys: name, title, icon, response.
        previous_rounds: Optional context from prior deliberation rounds.
    """
    sections = []
    for resp in advisor_responses:
        sections.append(f"## {resp['icon']}  {resp['name']} ({resp['title']}):\n{resp['response']}")

    advisor_text = "\n\n".join(sections)

    parts = []
    if previous_rounds:
        parts.append(
            "This is an ongoing Divan deliberation. Here is the context from previous rounds:\n\n"
            f"{previous_rounds}\n\n"
            "--- Current Round ---\n"
        )

    parts.append(
        f'The user asked the Divan: "{question}"\n\n'
        f"Here are the council's deliberations:\n\n"
        f"{advisor_text}\n\n"
        f"Now synthesize these perspectives into your Divan Karari."
    )

    if previous_rounds:
        parts.append(
            "\n\nBuild on the previous rounds. Reference how perspectives have evolved "
            "and highlight what is new in this round."
        )

    return "".join(parts)
