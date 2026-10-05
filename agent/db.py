"""Shared Neo4j driver and a read-only Cypher guard for the agent tools."""
from __future__ import annotations

import re

from neo4j import GraphDatabase

from . import config

WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|LOAD\s+CSV|"
    r"FOREACH|CALL\s+dbms\.admin)\b", re.IGNORECASE)


def get_driver():
    """One pooled driver for the whole service (lifespan-managed)."""
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def run_read_cypher(driver, query: str, params: dict | None = None,
                    limit: int = 100) -> list[dict]:
    """Run a Cypher query, refusing anything that mutates the graph.

    Read-only by construction: mutation keywords are rejected and the
    query runs against the same database as the vector search.
    """
    stripped = re.sub(r"//.*$", "", query)          # strip line comments
    if WRITE_RE.search(stripped):
        raise ValueError(
            "Only read-only Cypher is allowed "
            "(CREATE/MERGE/DELETE/SET/REMOVE/DROP are rejected).")
    with driver.session(database=config.NEO4J_DATABASE) as session:
        rows = session.run(query, params or {}).data()
    return rows[:limit]
