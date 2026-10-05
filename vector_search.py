#!/usr/bin/env python3
"""
Similarity search over the HMRC manuals graph in Neo4j.

A thin, agent-friendly tool: it embeds a query with the Fireworks embeddings
model (same model/dims as the loader, so vectors are comparable) and runs a
vector search against the `section_embedding` / `entity_embedding` indexes.
Results are printed as JSON by default — one object per line or a single
array — so an agent can consume them without parsing prose.

Usage
-----
    python3 vector_search.py "relief for selling a business"
    python3 vector_search.py "how is aggregates levy charged?" \\
        --kind sections --top 5 --expand
    python3 vector_search.py "partnership" --kind both

Config comes from .env (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD /
NEO4J_DATABASE, FIREWORKS_API_KEY / _BASE_URL / _EMBEDDING_MODEL /
_EMBEDDING_DIMS); the defaults match load_hmrc_to_neo4j.py exactly — the
indexes only return sensible scores if the query vector comes from the same
model at the same dimensions as the stored ones.

Exit codes: 0 results (possibly empty), 2 config/connection errors,
3 embedding API errors.

Importable: `embed(query)`, `search(kind, query, top, expand)` return plain
Python lists of dicts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIMS = 1024

# Same flags as the loader: query-side vectors must match the index.
INDEX_FOR = {"sections": "section_embedding", "entities": "entity_embedding"}

SEARCH_CYPHER = {
    "sections": """
        CALL db.index.vector.queryNodes($index, $top, $v)
        YIELD node, score
        OPTIONAL MATCH (node)-[:MENTIONED_IN]->(e:Entity)
        OPTIONAL MATCH (doc:Document)-[:HAS_SECTION]->(node)
        WITH node, score,
             collect(DISTINCT e.name)[..10] AS entities,
             head(collect(DISTINCT doc.slug)) AS manual
        RETURN score, node.section_id AS id, node.heading AS title,
               node.url AS url, manual, entities,
               left(node.text, $snippet) AS text
    """,
    "entities": """
        CALL db.index.vector.queryNodes($index, $top, $v)
        YIELD node, score
        OPTIONAL MATCH (doc:Document)-[:DISCUSSES]->(node)
        WITH node, score, head(collect(DISTINCT doc.slug)) AS manual
        OPTIONAL MATCH (node)-[r]->(other:Entity)
        WITH node, score, manual,
             [x IN collect({type: type(r), name: other.name})
              WHERE x.type <> 'MENTIONED_IN' AND x.type <> 'DISCUSSES'
              | x][..10] AS related
        RETURN score, node.key AS key, node.name AS title, node.type AS type,
               node.aliases AS aliases, manual, related,
               left(node.definition, $snippet) AS text
    """,
}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            os.environ.setdefault(key, val)


def embed(text: str) -> list[float]:
    """Embed one query string with the Fireworks embeddings API."""
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL",
                              "https://api.fireworks.ai/inference/v1")
    model = os.environ.get("FIREWORKS_EMBEDDING_MODEL",
                           DEFAULT_EMBEDDING_MODEL)
    dims = int(os.environ.get("FIREWORKS_EMBEDDING_DIMS")
               or DEFAULT_EMBEDDING_DIMS) or None
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is not set (put it in .env)")

    payload = {"model": model, "input": [text.strip()]}
    if dims:
        payload["dimensions"] = dims
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())["data"]
            return data[0]["embedding"]
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            if e.code not in (408, 429, 500, 502, 503, 504):
                print(f"embedding API error: HTTP {e.code} {detail}",
                      file=sys.stderr)
                sys.exit(3)
            last = e
        except Exception as e:                      # timeout, reset, bad JSON
            last = e
        time.sleep(min(20.0, 2 ** attempt))
    print(f"embedding failed: {last}", file=sys.stderr)
    sys.exit(3)


def search(kind: str, query: str, top: int = 5,
           snippet: int = 400, v: list[float] | None = None) -> list[dict]:
    """Vector-search one index. kind: 'sections' or 'entities'.

    Pass a precomputed query vector `v` to reuse one embedding across
    both indexes.
    """
    if kind not in SEARCH_CYPHER:
        raise ValueError(f"kind must be one of {list(SEARCH_CYPHER)}")
    if v is None:
        v = embed(query)
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", ""),
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "")))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"cannot connect to Neo4j: {e}", file=sys.stderr)
        sys.exit(2)
    with driver.session(database=os.environ.get("NEO4J_DATABASE")
                        or "neo4j") as session:
        try:
            rows = session.run(SEARCH_CYPHER[kind],
                               index=INDEX_FOR[kind], top=top, v=v,
                               snippet=snippet).data()
        finally:
            driver.close()
    for r in rows:
        r["score"] = round(r["score"], 4)
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Similarity search over the HMRC manuals graph "
                    "(embeds the query with Fireworks, searches the "
                    "Neo4j vector indexes).")
    ap.add_argument("query", help="natural-language query")
    ap.add_argument("--kind", default="both",
                    choices=["sections", "entities", "both"],
                    help="which vector index to search (default both)")
    ap.add_argument("--top", type=int, default=5,
                    help="results per index (default 5)")
    ap.add_argument("--text", action="store_true",
                    help="human-readable output instead of JSON")
    ap.add_argument("--env", default=".env", help="path to .env")
    args = ap.parse_args()

    load_dotenv(args.env)

    kinds = ["sections", "entities"] if args.kind == "both" else [args.kind]
    v = embed(args.query)
    out = {}
    for kind in kinds:
        out[kind] = search(kind, args.query, top=args.top, v=v)

    if args.text:
        for kind, rows in out.items():
            print(f"== {kind} ==")
            for r in rows:
                title = r.pop("title", None) or r.get("id") or "?"
                print(f"  {r['score']:<7} {title}")
                if r.get("manual"):
                    print(f"          manual: {r['manual']}")
                if r.get("entities"):
                    print(f"          entities: {', '.join(r['entities'])}")
                if r.get("type"):
                    print(f"          type: {r['type']}")
                if r.get("text"):
                    print(f"          {r['text'][:200]}…")
            if not rows:
                print("  (no results)")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
