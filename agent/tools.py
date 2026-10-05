"""Agent tools: semantic search, id-based graph expansion, read-only Cypher.

The intended flow is search -> expand/read by id. semantic_search returns
section `id`s and entity `key`s; expand and read_sections take those ids
directly, so the agent never has to re-find nodes by text matching.
"""
from __future__ import annotations

import json
import os

import vector_search
from . import config
from .db import run_read_cypher

# vector_search calls sys.exit(2/3) on connection/embedding failure —
# surface those as tool errors instead of killing the service.

MAX_IDS = 10            # per expand / read_sections call
RELS_PER_NODE = 25      # cap on hub entities (e.g. 'united kingdom')
SECTION_CHARS = 12000   # p99 section length is ~10k chars


def semantic_search(query: str, kind: str = "both", top: int = 5) -> str:
    """Semantic (vector) search over the HMRC manuals graph.

    kind: 'sections' (full manual text), 'entities' (tax concepts),
    or 'both'. Section hits carry an `id` (section_id); entity hits carry
    a `key`. Pass those to expand / read_sections — do not re-find the
    same nodes with text-matching Cypher. Use this FIRST.
    """
    kinds = ["sections", "entities"] if kind == "both" else [kind]
    top = min(max(int(top), 1), 10)
    os.environ.setdefault("NEO4J_DATABASE", config.NEO4J_DATABASE)
    try:
        v = vector_search.embed(query)          # one embedding, both indexes
        out = {k: vector_search.search(k, query, top=top, snippet=400, v=v)
               for k in kinds}
    except SystemExit as e:
        raise RuntimeError(f"vector search failed: {e}") from e
    return json.dumps(out, ensure_ascii=False)


def _id_list(value, name: str) -> list[str]:
    """Accept a JSON list, a bare list, or a single id from the model."""
    if isinstance(value, str):
        value = value.strip()
        try:
            value = json.loads(value) if value.startswith("[") else [value]
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} is not a valid JSON list: {e}") from e
    ids = [str(v) for v in (value or []) if str(v).strip()]
    return list(dict.fromkeys(ids))[:MAX_IDS]


# Relationships whose evidence quote appears in the section's own text —
# i.e. facts actually stated in that section, with that section as the
# citation. (`created_from` on relationships is a manual slug, not a
# section id, so section-level provenance comes from the evidence match.)
EXPAND_SECTIONS_CYPHER = """
UNWIND $ids AS sid
MATCH (s:Section {section_id: sid})
OPTIONAL MATCH (parent:Section)-[:HAS_CHILD]->(s)
CALL (s) {
  MATCH (a:Entity)-[:MENTIONED_IN]->(s)
  MATCH (a)-[r]->(b:Entity)-[:MENTIONED_IN]->(s)
  WITH r, a, b, [q IN r.evidence WHERE s.text CONTAINS q] AS ev
  WHERE size(ev) > 0
  RETURN collect({from: a.name, rel: type(r), to: b.name,
                  evidence: ev[..2]})[..$per] AS rels
}
CALL (s) {
  MATCH (e:Entity)-[:MENTIONED_IN]->(s)
  RETURN collect({key: e.key, name: e.name, type: e.type})[..20] AS entities
}
RETURN s.section_id AS id, s.heading AS heading, s.url AS url,
       parent.section_id AS parent, entities, rels
"""

# 1-hop neighbourhood of an entity, strongest relationships first. Each
# relationship is cited by a section mentioning both ends whose text holds
# the evidence quote; failing that, by the manual it was extracted from.
EXPAND_ENTITIES_CYPHER = """
UNWIND $keys AS k
MATCH (e:Entity {key: k})
CALL (e) {
  MATCH (e)-[r]-(o:Entity)
  WITH e, r, o ORDER BY coalesce(r.mentions, 0) DESC LIMIT $per
  OPTIONAL MATCH (doc:Document {slug: r.created_from})
  CALL (e, r, o) {
    OPTIONAL MATCH (e)-[:MENTIONED_IN]->(s:Section)<-[:MENTIONED_IN]-(o)
    WHERE any(q IN r.evidence WHERE s.text CONTAINS q)
    RETURN s LIMIT 1
  }
  RETURN collect({from: startNode(r).name, rel: type(r), to: endNode(r).name,
                  other_key: o.key, evidence: r.evidence[..2],
                  section: s.section_id,
                  url: coalesce(s.url, doc.url)}) AS rels
}
RETURN e.key AS key, e.name AS name, e.type AS type,
       e.definition AS definition, rels
"""


def expand(section_ids: str = "[]", entity_keys: str = "[]") -> str:
    """Expand search hits by id into their graph neighbourhood, in one call.

    section_ids: JSON list of section `id`s from semantic_search. Returns
      each section's heading/url/parent, the entities mentioned in it, and
      the entity relationships stated in that section (with evidence).
    entity_keys: JSON list of entity `key`s from semantic_search. Returns
      each entity's definition and its strongest relationships, each with
      evidence and a citation `url` (plus `section` id when known).
    Use this straight after semantic_search instead of writing Cypher.
    """
    sids = _id_list(section_ids, "section_ids")
    keys = _id_list(entity_keys, "entity_keys")
    out: dict = {}
    if sids:
        out["sections"] = run_read_cypher(
            _driver(), EXPAND_SECTIONS_CYPHER,
            {"ids": sids, "per": RELS_PER_NODE})
    if keys:
        out["entities"] = run_read_cypher(
            _driver(), EXPAND_ENTITIES_CYPHER,
            {"keys": keys, "per": RELS_PER_NODE})
    if not out:
        return json.dumps({"error": "pass section_ids and/or entity_keys"})
    return json.dumps(out, ensure_ascii=False)


def read_sections(section_ids: str) -> str:
    """Fetch full manual sections by id (heading, url, full text) in one call.

    section_ids: JSON list of section ids (max 10). Search snippets are only
    400 chars — read the whole section before citing it.
    """
    ids = _id_list(section_ids, "section_ids")
    rows = run_read_cypher(
        _driver(),
        "UNWIND $ids AS sid MATCH (s:Section {section_id: sid}) "
        "RETURN s.section_id AS id, s.heading AS heading, s.url AS url, "
        "left(s.text, $chars) AS text, size(s.text) > $chars AS truncated",
        {"ids": ids, "chars": SECTION_CHARS})
    missing = sorted(set(ids) - {r["id"] for r in rows})
    return json.dumps({"sections": rows, "missing": missing},
                      ensure_ascii=False)


def cypher(query: str, params_json: str = "{}") -> str:
    """Run a READ-ONLY Cypher query against the Neo4j HMRC graph.

    Only for traversals expand can't do. Anchor on ids from semantic_search
    — `MATCH (s:Section {section_id: $id})`, `MATCH (e:Entity {key: $key})`
    (both indexed). Do NOT use CONTAINS / toLower on s.text or e.name to
    find nodes: it scans 84k sections, is slow and noisy, and duplicates
    the vector search.

    The graph model:
      (Document)-[:HAS_SECTION]->(Section)-[:HAS_CHILD]->(Section)
      (Section)-[:NEXT]->(Section)
      (Entity)-[:MENTIONED_IN]->(Section)
      (Document)-[:DISCUSSES]->(Entity)
      Entity->Entity rels: ADMINISTERED_BY, APPLIES_TO, CALCULATED_FROM,
        DEFINED_BY, EXAMPLE_OF, EXCLUDES, EXEMPTS, GOVERNED_BY,
        HAS_DEADLINE, HAS_RATE, HAS_THRESHOLD, MEASURED_BY, PART_OF,
        PENALTY_FOR, REFERENCES, RELATED_TO, RELIEVES, REPORTED_ON,
        REQUIRES, SUBJECT_TO, SUPERSEDES — each carrying `evidence`
        (list of quotes), `mentions`, and `created_from` (the manual slug,
        = Document.slug — NOT a section id).
    Sections have heading, section_id, text, url, position.
    Entities have key (lowercased name, unique), name, type, definition.
    Write queries are rejected.
    """
    try:
        params = json.loads(params_json or "{}")
    except json.JSONDecodeError as e:
        return f"params_json is not valid JSON: {e}"
    rows = run_read_cypher(_driver(), query, params)
    return json.dumps(rows, ensure_ascii=False, default=str)


_DRIVER = None


def _driver():
    """Lazily-created shared driver for the Cypher tools."""
    global _DRIVER
    if _DRIVER is None:
        from .db import get_driver
        _DRIVER = get_driver()
    return _DRIVER


TOOLS = [semantic_search, expand, read_sections, cypher]
_ID_LIST = {"type": "string",
            "description": 'JSON list of ids, e.g. ["DT15600", "DT15602"]'}
TOOL_SCHEMAS = [
    {"type": "function",
     "function": {
         "name": "semantic_search",
         "description": semantic_search.__doc__.strip(),
         "parameters": {
             "type": "object",
             "properties": {
                 "query": {"type": "string"},
                 "kind": {"type": "string",
                          "enum": ["sections", "entities", "both"]},
                 "top": {"type": "integer", "default": 5}},
             "required": ["query"]}}},
    {"type": "function",
     "function": {
         "name": "expand",
         "description": expand.__doc__.strip(),
         "parameters": {
             "type": "object",
             "properties": {"section_ids": _ID_LIST,
                            "entity_keys": {
                                **_ID_LIST,
                                "description": "JSON list of entity keys, "
                                               'e.g. ["portugal"]'}}}}},
    {"type": "function",
     "function": {
         "name": "read_sections",
         "description": read_sections.__doc__.strip(),
         "parameters": {
             "type": "object",
             "properties": {"section_ids": _ID_LIST},
             "required": ["section_ids"]}}},
    {"type": "function",
     "function": {
         "name": "cypher",
         "description": cypher.__doc__.strip(),
         "parameters": {
             "type": "object",
             "properties": {
                 "query": {"type": "string"},
                 "params_json": {"type": "string",
                                 "description": "JSON object of $params"}},
             "required": ["query"]}}},
]
