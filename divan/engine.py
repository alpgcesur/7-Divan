"""LangGraph deliberation engine with parallel fan-out/fan-in."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from divan.advisor import Advisor
from divan.synthesis import build_synthesis_prompt


class AdvisorResponse(TypedDict):
    advisor_id: str
    name: str
    title: str
    icon: str
    color: str
    response: str


class DivanState(TypedDict):
    query: str
    advisor_responses: Annotated[list[AdvisorResponse], operator.add]
    synthesis: str


def make_advisor_node(advisor: Advisor, model: BaseChatModel):
    """Create an async graph node for a single advisor."""

    async def advisor_node(state: DivanState) -> dict[str, Any]:
        try:
            messages = [
                SystemMessage(content=advisor.system_prompt),
                HumanMessage(content=state["query"]),
            ]
            result = await model.ainvoke(messages)
            response_text = result.content
        except Exception as e:
            response_text = f"[Advisor error: {e}]"

        return {
            "advisor_responses": [
                AdvisorResponse(
                    advisor_id=advisor.id,
                    name=advisor.name,
                    title=advisor.title,
                    icon=advisor.icon,
                    color=advisor.color,
                    response=response_text,
                )
            ]
        }

    return advisor_node


def make_synthesis_node(synthesizer: Advisor, model: BaseChatModel):
    """Create an async graph node for the Bas Vezir synthesis."""

    async def synthesis_node(state: DivanState) -> dict[str, Any]:
        prompt = build_synthesis_prompt(
            state["query"],
            [
                {
                    "name": r["name"],
                    "title": r["title"],
                    "icon": r["icon"],
                    "response": r["response"],
                }
                for r in state["advisor_responses"]
            ],
        )

        messages = [
            SystemMessage(content=synthesizer.system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            result = await model.ainvoke(messages)
            return {"synthesis": result.content}
        except Exception as e:
            return {"synthesis": f"[Synthesis error: {e}]"}

    return synthesis_node


def build_deliberation_graph(
    advisors: list[Advisor],
    synthesizer: Advisor,
    advisor_model: BaseChatModel,
    synthesis_model: BaseChatModel,
) -> StateGraph:
    """Build the LangGraph deliberation graph.

    Structure: START -> [all advisors in parallel] -> synthesis -> END
    """
    graph = StateGraph(DivanState)

    # Add advisor nodes
    for advisor in advisors:
        graph.add_node(advisor.node_name, make_advisor_node(advisor, advisor_model))
        graph.add_edge(START, advisor.node_name)
        graph.add_edge(advisor.node_name, synthesizer.node_name)

    # Add synthesis node
    graph.add_node(synthesizer.node_name, make_synthesis_node(synthesizer, synthesis_model))
    graph.add_edge(synthesizer.node_name, END)

    return graph.compile()
