"""LangGraph deliberation engine with parallel fan-out/fan-in."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from divan.advisor import Advisor
from divan.synthesis import build_synthesis_prompt


def _extract_text_content(content: str | list) -> str:
    """Extract text from AIMessage content, which may be a string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


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


MAX_TOOL_ITERATIONS = 5


def make_advisor_node(
    advisor: Advisor,
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    memory_text: str = "",
):
    """Create an async graph node for a single advisor.

    If tools are provided, the advisor runs an agentic loop: invoke the model,
    execute any tool calls, feed results back, repeat until the model responds
    with plain text or the iteration limit is reached.

    memory_text: optional formatted memory to prepend to system prompt.
    """

    async def advisor_node(state: DivanState) -> dict[str, Any]:
        try:
            system_prompt = advisor.system_prompt
            if memory_text:
                system_prompt = f"{memory_text}\n\n{system_prompt}"

            messages: list = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["query"]),
            ]

            if tools:
                tool_map = {t.name: t for t in tools}
                model_with_tools = model.bind_tools(tools)

                for _ in range(MAX_TOOL_ITERATIONS):
                    result = await model_with_tools.ainvoke(messages)
                    messages.append(result)

                    if not result.tool_calls:
                        break

                    for tc in result.tool_calls:
                        tool_fn = tool_map.get(tc["name"])
                        if tool_fn:
                            try:
                                output = await tool_fn.ainvoke(tc["args"])
                            except Exception as e:
                                output = f"Tool error: {e}"
                        else:
                            output = f"Unknown tool: {tc['name']}"

                        messages.append(ToolMessage(
                            content=str(output),
                            tool_call_id=tc["id"],
                        ))

                response_text = _extract_text_content(result.content)
            else:
                result = await model.ainvoke(messages)
                response_text = _extract_text_content(result.content)
        except Exception as e:
            response_text = f"✗ Advisor error: {e}"

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
            return {"synthesis": f"✗ Synthesis error: {e}"}

    return synthesis_node


def build_deliberation_graph(
    advisors: list[Advisor],
    synthesizer: Advisor,
    advisor_model: BaseChatModel,
    synthesis_model: BaseChatModel,
    advisor_tools: dict[str, list[BaseTool]] | None = None,
    advisor_memory_texts: dict[str, str] | None = None,
) -> StateGraph:
    """Build the LangGraph deliberation graph.

    Structure: START -> [all advisors in parallel] -> synthesis -> END

    advisor_tools: optional dict mapping advisor ID to list of tool instances.
    advisor_memory_texts: optional dict mapping advisor ID to formatted memory text.
    """
    graph = StateGraph(DivanState)

    # Add advisor nodes
    for advisor in advisors:
        tools = (advisor_tools or {}).get(advisor.id)
        memory = (advisor_memory_texts or {}).get(advisor.id, "")
        graph.add_node(
            advisor.node_name,
            make_advisor_node(advisor, advisor_model, tools=tools, memory_text=memory),
        )
        graph.add_edge(START, advisor.node_name)
        graph.add_edge(advisor.node_name, synthesizer.node_name)

    # Add synthesis node
    graph.add_node(synthesizer.node_name, make_synthesis_node(synthesizer, synthesis_model))
    graph.add_edge(synthesizer.node_name, END)

    return graph.compile()
