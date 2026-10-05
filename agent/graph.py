"""LangGraph agent: search the graph, then answer with citations.

Flow: agent node (Fireworks chat model with the three tools) ↔ tools node,
until the model stops calling tools; then a final node produces a
structured answer (answer + sources + graph cross-references) enforced by
a JSON schema — the citation contract from AGENTS.md.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Generator, Literal

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_fireworks import ChatFireworks
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import config
from .prompt import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, TOOLS
from .viz import save_viz_html

TOOL_BY_NAME = {t.__name__: t for t in TOOLS}

ANSWER_SCHEMA = {
    "name": "cited_answer",
    "description": "Answer to a UK tax question, with mandatory citations.",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"}},
                    "required": ["title", "url"]}},
            "graph_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_entity": {"type": "string"},
                        "relationship": {"type": "string"},
                        "to_entity": {"type": "string"}},
                    "required": ["from_entity", "relationship",
                                  "to_entity"]}},
        },
        "required": ["answer", "sources", "graph_refs"],
    },
}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], lambda a, b: a + b]
    tool_rounds: int
    # Must be declared here: LangGraph drops update keys that aren't
    # channels in the state schema, which is why the answer vanished.
    answer: dict


MAX_TOOL_ROUNDS = 8


def _llm() -> ChatFireworks:
    # The SDK appends /v1/ itself, so strip a trailing .../v1 from the
    # configured base URL (https://api.fireworks.ai/inference/v1 -> .../inference).
    base = config.FIREWORKS_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return ChatFireworks(
        model=config.FIREWORKS_MODEL,
        api_key=config.FIREWORKS_API_KEY,
        base_url=base,
        temperature=0,
        max_tokens=4096,
    )


def agent_node(state: AgentState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    ai: AIMessage = _llm().bind_tools(TOOL_SCHEMAS).invoke(msgs)
    return {"messages": [ai]}


def tools_node(state: AgentState) -> dict:
    out: list[ToolMessage] = []
    rounds = state.get("tool_rounds", 0) + 1
    for call in state["messages"][-1].tool_calls:
        if rounds > MAX_TOOL_ROUNDS:
            out.append(ToolMessage(
                content="Tool round limit reached — produce your final "
                        "cited answer now with what you have.",
                tool_call_id=call["id"]))
            continue
        fn = TOOL_BY_NAME.get(call["name"])
        try:
            result = fn(**(call.get("args") or {})) if fn else \
                f"Unknown tool {call['name']}"
        except Exception as e:                      # noqa: BLE001
            result = json.dumps({"error": str(e)})
        out.append(ToolMessage(content=str(result)[:20000],
                               tool_call_id=call["id"]))
    return {"messages": out, "tool_rounds": rounds}


def final_node(state: AgentState) -> dict:
    """Structured final answer — the citation contract, enforced by schema."""
    # Flatten the conversation: tool messages become user-role context so
    # the chat API can't choke on unmatched tool_call frames.
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            msgs.append({"role": "assistant", "content": m.content})
        elif isinstance(m, ToolMessage):
            msgs.append({"role": "user",
                         "content": f"Tool result ({m.name}): "
                                     f"{m.content[:4000]}"})
    msgs.append({"role": "user",
                 "content":
                     "Now produce your final cited answer using the "
                     "cited_answer schema. Every claim must cite the "
                     "section URLs you actually used. If the graph did "
                     "not contain the answer, say so plainly in `answer` "
                     "and leave sources empty."})

    resp = requests.post(
        config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {config.FIREWORKS_API_KEY}"},
        json={
            "model": config.FIREWORKS_MODEL,
            "temperature": 0,
            "messages": msgs,
            "response_format": {
                "type": "json_schema",
                "json_schema": ANSWER_SCHEMA,
            },
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content) if content else None
    if not isinstance(parsed, dict) or not parsed.get("answer"):
        # Model ignored the schema — surface its raw text as the answer
        # rather than failing, with empty citations.
        parsed = {"answer": content or "(model returned an empty answer)",
                  "sources": [], "graph_refs": []}
    parsed.setdefault("sources", [])
    parsed.setdefault("graph_refs", [])
    return {"answer": parsed}


def _jsonable(m: Any) -> dict:  # retained for potential reuse
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content}
    if isinstance(m, AIMessage):
        return {"role": "assistant", "content": m.content or ""}
    return {"role": "assistant", "content": str(m)}


def _route(state: AgentState) -> Literal["tools", "final"]:
    return "tools" if state["messages"][-1].tool_calls else "final"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("final", final_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route)
    g.add_edge("tools", "agent")
    g.add_edge("final", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(question: str) -> Generator[dict, None, None]:
    """Run the agent, yielding SSE-friendly event dicts.

    Events: {type: "tool_call", name, args}
            {type: "answer", answer, sources, graph_refs}
    """
    state: dict[str, Any] = {"messages": [HumanMessage(content=question)]}
    for step in get_graph().stream(state, stream_mode="updates"):
        for node, update in step.items():
            if node == "agent" and update.get("messages"):
                ai = update["messages"][-1]
                if getattr(ai, "tool_calls", None):
                    for call in ai.tool_calls:
                        yield {"type": "tool_call", "name": call["name"],
                               "args": call.get("args", {})}
            if node == "final" and update and update.get("answer"):
                answer = update["answer"]
                # Best-effort visualization of the answering subgraph.
                viz_id = save_viz_html(answer)
                if viz_id:
                    answer = {**answer, "viz_id": viz_id}
                yield {"type": "answer", **answer}
