"""Build a neo4j_viz visualization of the subgraph that answered a question.

Given the agent's final answer (graph_refs + sources), query Neo4j for the
cited Entity/Section nodes and the relationships between them, render them
with neo4j_viz (VisualizationGraph.render -> self-contained HTML), and write
the HTML to static/viz/<id>.html for the app to serve in an iframe.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from neo4j_viz import Node, Relationship, VisualizationGraph

from .db import run_read_cypher
from .tools import _driver

STATIC_VIZ = Path(__file__).resolve().parent.parent / "static" / "viz"

ENTITY_COLOR = "#2f6fed"   # matches the app accent color
SECTION_COLOR = "#1a9e6e"
MAX_SECTIONS = 12


def _cap(text: str, n: int = 40) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def build_graph(answer: dict) -> VisualizationGraph | None:
    """Assemble the visualization graph from the answer's citations."""
    refs = answer.get("graph_refs") or []
    urls = [s.get("url") for s in (answer.get("sources") or []) if s.get("url")]
    entity_names = sorted({r["from_entity"] for r in refs if r.get("from_entity")} |
                          {r["to_entity"] for r in refs if r.get("to_entity")})
    if not entity_names and not urls:
        return None

    nodes: list[Node] = []
    rels: list[Relationship] = []
    node_ids: set[str] = set()
    section_urls: set[str] = set()

    def add_node(nid: str, caption: str, color: str, size: int, url: str = ""):
        if nid in node_ids:
            return
        node_ids.add(nid)
        nodes.append(Node(id=nid, caption=caption, color=color, size=size,
                          properties={"url": url}))

    driver = _driver()

    # Entity -> Entity relationships traversed in the answer
    if entity_names:
        rows = run_read_cypher(
            driver,
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE a.name IN $names AND b.name IN $names "
            "RETURN a.name AS a, type(r) AS rel, b.name AS b, "
            "elementId(r) AS rid LIMIT 50",
            {"names": entity_names})
        for row in rows:
            add_node(f"e:{row['a']}", row["a"], ENTITY_COLOR, 24)
            add_node(f"e:{row['b']}", row["b"], ENTITY_COLOR, 24)
            rels.append(Relationship(id=f"r:{row['rid']}",
                                     source=f"e:{row['a']}",
                                     target=f"e:{row['b']}",
                                     caption=row["rel"]))

    # Cited sections, plus any of those entities MENTIONED_IN them
    if urls:
        rows = run_read_cypher(
            driver,
            "MATCH (s:Section) WHERE s.url IN $urls "
            "RETURN s.url AS url, s.heading AS heading "
            "ORDER BY s.heading LIMIT $max",
            {"urls": urls, "max": MAX_SECTIONS})
        for row in rows:
            url = row["url"]
            section_urls.add(url)
            add_node(f"s:{url}", _cap(row["heading"]), SECTION_COLOR, 18, url)
        if section_urls and entity_names:
            rows = run_read_cypher(
                driver,
                "MATCH (e:Entity)-[m:MENTIONED_IN]->(s:Section) "
                "WHERE e.name IN $names AND s.url IN $urls "
                "RETURN e.name AS e, s.url AS url, elementId(m) AS mid "
                "LIMIT 60",
                {"names": entity_names, "urls": sorted(section_urls)})
            for row in rows:
                add_node(f"e:{row['e']}", row["e"], ENTITY_COLOR, 24)
                rels.append(Relationship(
                    id=f"r:{row['mid']}", source=f"e:{row['e']}",
                    target=f"s:{row['url']}", caption="MENTIONED_IN"))

    if not nodes:
        return None
    return VisualizationGraph(nodes=nodes, relationships=rels)


def save_viz_html(answer: dict) -> str | None:
    """Render the answering subgraph to static/viz/<id>.html.

    Returns the id (for the /viz/<id> URL) or None if there is nothing to
    show or rendering fails.
    """
    try:
        vg = build_graph(answer)
        if vg is None:
            return None
        # height must be viewport-relative: the bundle applies it to a div
        # directly under an unsized <body>, so "100%" collapses to 0px.
        html = vg.render(width="100%", height="100vh",
                         layout="forcedirected").data
    except Exception:                                # noqa: BLE001
        # Visualization is best-effort: never fail the answer for it.
        return None
    STATIC_VIZ.mkdir(parents=True, exist_ok=True)
    viz_id = uuid.uuid4().hex[:12]
    (STATIC_VIZ / f"{viz_id}.html").write_text(html)
    return viz_id
