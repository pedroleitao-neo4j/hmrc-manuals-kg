#!/usr/bin/env python3
"""
Compute graph centralities for the extracted entity graph and write them
back onto Entity nodes as plain data:

  Entity.pagerank      global importance of the concept in the extracted
                      knowledge graph (normalised so max = 1.0)
  Entity.community    id of its community (Louvain via GDS, or WCC via the
                      Python fallback)
  Entity.community_size  size of that community

Two backends (--backend auto, the default):

  gds     Aura Graph Analytics - the Cypher API runs inside AuraDB, no
          separate credentials. Projects the filtered entity graph into a
          remote session, runs gds.pageRank.mutate / gds.louvain.mutate /
          gds.wcc.mutate, writes the properties back, then drops the graph
          (the session has a TTL, so a crash still tears it down).
  python  plain-Python weighted PageRank + union-find WCC over the edges
          pulled from Neo4j - used when the GDS Cypher API is not enabled
          on the instance.

The agent never runs algorithms at query time: it just reads these
properties (see agent/tools.py - centrality is routing signal, not citable
fact - there is no section URL behind a PageRank score).

Noise control - the raw extraction graph is dominated by hub entities that
are central only because they are mentioned everywhere. The projection
therefore excludes:
  - fallback / navigational relationship types (RELATED_TO, REFERENCES,
    EXAMPLE_OF) that carry the least semantics
  - relationship types that mostly connect hub nodes (ADMINISTERED_BY,
    GOVERNED_BY - how "HMRC" and "United Kingdom" accumulate degree)
  - hub-prone entity types (Organisation, Jurisdiction) on either endpoint
    (Python backend; GDS projections cannot filter nodes by property)
and weights each edge by its `mentions`.

Recompute after re-enriching or reloading the graph; the writes are SETs,
so the script is idempotent. Entities with no qualifying edges are left
untouched (no pagerank property).

Connection config comes from .env (NEO4J_URI / NEO4J_USERNAME /
NEO4J_PASSWORD / NEO4J_DATABASE); CLI flags win.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    print("the `neo4j` driver is not installed:  pip install neo4j",
          file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------- #
# .env (same minimal loader as the other scripts)
# --------------------------------------------------------------------------- #
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


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# Projection filters: hub noise out of the centrality signal.
EXCLUDED_REL_TYPES = ["RELATED_TO", "REFERENCES", "EXAMPLE_OF",
                      "ADMINISTERED_BY", "GOVERNED_BY"]
EXCLUDED_ENTITY_TYPES = ["Organisation", "Jurisdiction"]

DAMPING = 0.85
MAX_ITERS = 50
TOL = 1e-8
WRITE_BATCH = 5000

GDS_GRAPH = "hmrc-entity-centralities"

# Post-processing on the database: normalise pagerank to max 1.0 and size
# each community. (SETs, so reruns are idempotent.)
NORMALISE_CYPHER = """
MATCH (e:Entity) WHERE e.pagerank IS NOT NULL
WITH max(e.pagerank) AS m
MATCH (e:Entity) WHERE e.pagerank IS NOT NULL
SET e.pagerank = e.pagerank / m
"""
COMMUNITY_SIZE_CYPHER = """
MATCH (e:Entity) WHERE e.community IS NOT NULL
WITH e.community AS c, count(*) AS n
MATCH (e2:Entity {community: c})
SET e2.community_size = n
"""

EDGES_CYPHER = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE NOT type(r) IN $excl_rels
  AND NOT a.type IN $excl_types
  AND NOT b.type IN $excl_types
RETURN a.key AS src, b.key AS tgt, coalesce(r.mentions, 1) AS w
"""

WRITE_CYPHER = """
UNWIND $rows AS row
MATCH (e:Entity {key: row.key})
SET e.pagerank = row.pagerank,
    e.community = row.community,
    e.community_size = row.community_size
"""


# --------------------------------------------------------------------------- #
# Python backend: weighted PageRank (power iteration) + WCC
# --------------------------------------------------------------------------- #
def build_graph(rows: list[dict]) -> dict[str, dict]:
    """Fold the edge rows into an adjacency structure keyed by entity key.

    Parallel edges between the same pair (different relationship types)
    sum their weights, so a fact stated by several manuals counts once per
    manual, not once per relationship type.
    """
    adj: dict[str, dict[str, float]] = {}
    nodes: set[str] = set()
    for r in rows:
        src, tgt = r["src"], r["tgt"]
        if not src or not tgt or src == tgt:
            continue
        nodes.add(src)
        nodes.add(tgt)
        adj.setdefault(src, {})
        adj[src][tgt] = adj[src].get(tgt, 0.0) + max(float(r["w"] or 1), 1.0)
    return {"nodes": nodes, "adj": adj}


def pagerank(graph: dict, damping: float = DAMPING,
             max_iters: int = MAX_ITERS, tol: float = TOL) -> dict[str, float]:
    """Power iteration. Edge weights normalise to transition probabilities;
    dangling mass (nodes with no out-edges) is redistributed uniformly.
    """
    nodes = graph["nodes"]
    adj = graph["adj"]
    n = len(nodes)
    if n == 0:
        return {}
    rank = {k: 1.0 / n for k in nodes}
    total_w = {k: sum(adj.get(k, {}).values()) for k in nodes}
    for it in range(max_iters):
        dangling = sum(rank[k] for k in nodes if total_w[k] == 0)
        base = (1 - damping) / n + damping * dangling / n
        nxt = {}
        for k in nodes:
            nxt[k] = base
        for src, targets in adj.items():
            share = damping * rank[src] / total_w[src]
            for tgt, w in targets.items():
                nxt[tgt] += share * w
        delta = sum(abs(nxt[k] - rank[k]) for k in nodes)
        rank = nxt
        if delta < tol:
            log(f"    · converged after {it + 1} iterations (delta {delta:.2e})")
            break
    top = max(rank.values()) or 1.0
    return {k: v / top for k, v in rank.items()}


def weak_components(graph: dict) -> tuple[dict[str, int], dict[int, int]]:
    """Union-find over the undirected view; returns (node->component id,
    component id -> size). Component ids are 0.. ordered by size desc, so
    id 0 is the giant component."""
    parent: dict[str, str] = {k: k for k in graph["nodes"]}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    for src, targets in graph["adj"].items():
        for tgt in targets:
            rs, rt = find(src), find(tgt)
            if rs != rt:
                parent[rs] = rt

    sizes: dict[str, int] = {}
    for k in parent:
        sizes[find(k)] = sizes.get(find(k), 0) + 1
    order = {root: i for i, root in
             enumerate(sorted(sizes, key=lambda r: -sizes[r]))}
    return ({k: order[find(k)] for k in parent},
            {i: sizes[root] for root, i in order.items()})


def run_python(session, top: int, dry_run: bool) -> None:
    log("· pulling filtered entity->entity edges ...")
    rows = session.run(EDGES_CYPHER, {
        "excl_rels": EXCLUDED_REL_TYPES,
        "excl_types": EXCLUDED_ENTITY_TYPES,
    }).data()
    log(f"  {len(rows)} edges pulled")

    graph = build_graph(rows)
    log(f"· projection: {len(graph['nodes'])} entities, "
        f"{sum(len(t) for t in graph['adj'].values())} weighted edges "
        f"(excluded rels {EXCLUDED_REL_TYPES}, "
        f"entity types {EXCLUDED_ENTITY_TYPES})")

    log("· pagerank (power iteration) ...")
    pr = pagerank(graph)
    log("· weakly-connected components ...")
    comp, comp_sizes = weak_components(graph)
    log(f"  {len(comp_sizes)} components; largest = {comp_sizes.get(0, 0)}")

    if dry_run:
        _print_top(session, sorted(pr.items(), key=lambda kv: -kv[1])[:top],
                   {k: comp[k] for k in pr})
        return

    log("· writing pagerank / community / community_size to Entity nodes ...")
    results = [{"key": k, "pagerank": pr[k], "community": comp[k],
                "community_size": comp_sizes[comp[k]]} for k in graph["nodes"]]
    written = 0
    for i in range(0, len(results), WRITE_BATCH):
        batch = results[i:i + WRITE_BATCH]
        session.execute_write(
            lambda tx, b=batch: tx.run(WRITE_CYPHER, {"rows": b}).consume())
        written += len(batch)
        log(f"  {written}/{len(results)}")
    _print_top(session, sorted(pr.items(), key=lambda kv: -kv[1])[:top],
               comp, to_stderr=True)


# --------------------------------------------------------------------------- #
# GDS backend: Aura Graph Analytics Cypher API
# --------------------------------------------------------------------------- #
def gds_available(session) -> bool:
    """The Aura Graph Analytics Cypher API registers gds.* procedures on the
    AuraDB instance; gds.graph.list() is the cheap probe."""
    try:
        session.run("CALL gds.graph.list() YIELD graphName RETURN graphName")\
            .consume()
        return True
    except Exception:
        return False


def run_gds(session, memory_gb: float, top: int, dry_run: bool) -> None:
    # The 22 relationship types minus the excluded ones.
    rel_types = ["DEFINED_BY", "APPLIES_TO", "EXCLUDES", "EXEMPTS",
                 "RELIEVES", "REQUIRES", "CALCULATED_FROM", "SUBJECT_TO",
                 "PART_OF", "SUPERSEDES", "REPLACED_BY", "HAS_RATE",
                 "HAS_THRESHOLD", "HAS_DEADLINE", "QUALIFIES_FOR",
                 "PENALTY_FOR", "REPORTED_ON", "MEASURED_BY"]
    log(f"· projecting Entity + {len(rel_types)} relationship types "
        f"(mentions-weighted) into a GDS session ({memory_gb:g}GB, 30min ttl)")
    session.run(f"""
        CALL gds.graph.project(
          $name, 'Entity', $relTypes,
          {{ relationshipProperties: ['mentions'],
             memory: $mem,
             ttl: toString(duration({{ minutes: 30 }})) }})
        YIELD nodeCount, relationshipCount
        RETURN nodeCount, relationshipCount
    """, {"name": GDS_GRAPH, "relTypes": rel_types,
          "mem": f"{memory_gb:g}GB"}).consume()

    try:
        if dry_run:
            log("· gds.pageRank.stream (nothing written) ...")
            rows = session.run("""
                CALL gds.pageRank.stream($name, {
                    relationshipWeightProperty: 'mentions',
                    dampingFactor: $d })
                YIELD nodeId, score
                WITH nodeId, score ORDER BY score DESC LIMIT $top
                WITH gds.util.asNode(nodeId) AS e, score
                WHERE e.type IS NOT NULL AND NOT e.type IN $excl_types
                RETURN e.key AS key, e.name AS name, e.type AS type,
                       score ORDER BY score DESC
            """, {"name": GDS_GRAPH, "d": DAMPING, "top": top * 4,
                  "excl_types": EXCLUDED_ENTITY_TYPES}).data()
            for r in rows[:top]:
                print(f"{r['score'] / rows[0]['score']:10.4f}  "
                      f"{r['name']}  [{r['type']}]")
            log("(dry run - nothing written)")
            return

        log("· gds.pageRank.mutate ...")
        session.run("""
            CALL gds.pageRank.mutate($name, {
                relationshipWeightProperty: 'mentions',
                dampingFactor: $d, mutateProperty: 'pagerank' })
            YIELD ranIterations
            RETURN ranIterations
        """, {"name": GDS_GRAPH, "d": DAMPING}).consume()
        log("· gds.louvain.mutate ...")
        session.run("""
            CALL gds.louvain.mutate($name, { mutateProperty: 'community' })
            YIELD communityCount, modularity
            RETURN communityCount, modularity
        """, {"name": GDS_GRAPH}).consume()
        log("· gds.wcc.mutate ...")
        session.run("""
            CALL gds.wcc.mutate($name, { mutateProperty: 'component' })
            YIELD componentCount
            RETURN componentCount
        """, {"name": GDS_GRAPH}).consume()

        log("· writing properties back to AuraDB ...")
        session.run("""
            CALL gds.graph.nodeProperties.write($name,
                ['pagerank', 'community'], ['Entity'], {})
            YIELD propertiesWritten
            RETURN propertiesWritten
        """, {"name": GDS_GRAPH}).consume()

        log("· normalising pagerank, sizing communities ...")
        session.run(NORMALISE_CYPHER).consume()
        session.run(COMMUNITY_SIZE_CYPHER).consume()
    finally:
        # Dropping the graph tears the session down - no lingering cost.
        log("· dropping the GDS graph (frees the session) ...")
        session.run("CALL gds.graph.drop($name, false) YIELD graphName "
                    "RETURN graphName", {"name": GDS_GRAPH}).consume()

    log("· top entities now in the database:")
    for r in session.run("""
            MATCH (e:Entity) WHERE e.pagerank IS NOT NULL
            RETURN e.name AS name, e.pagerank AS pagerank,
                   e.community AS community
            ORDER BY e.pagerank DESC LIMIT $top
        """, {"top": top}):
        log(f"    {r['pagerank']:8.4f}  community {r['community']:>6}  "
            f"{r['name']}")


def _print_top(session, top_items, comp, to_stderr: bool = False) -> None:
    out = log if to_stderr else print
    keys = [k for k, _ in top_items]
    names = {r["key"]: r["name"] for r in session.run(
        "UNWIND $keys AS k MATCH (e:Entity {key: k}) "
        "RETURN e.key AS key, e.name AS name", {"keys": keys}).data()}
    out(f"{'pagerank':>10}  {'community':>9}  entity")
    for k, v in top_items:
        out(f"{v:10.4f}  {comp.get(k, -1):>9}  {names.get(k, k)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute entity-graph PageRank + communities, "
                    "write to Entity nodes")
    ap.add_argument("--backend", choices=["auto", "gds", "python"],
                    default="auto",
                    help="gds = Aura Graph Analytics Cypher API, python = "
                         "in-process; auto = gds when available "
                         "(default auto)")
    ap.add_argument("--memory", type=float, default=2.0,
                    help="GDS session memory in GB (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the top entities, write nothing")
    ap.add_argument("--top", type=int, default=50,
                    help="how many top entities to print (default 50)")
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    ap.add_argument("--database", default=os.environ.get("NEO4J_DATABASE",
                                                         "neo4j"))
    args = ap.parse_args()

    load_dotenv()
    uri = args.uri or os.environ.get("NEO4J_URI")
    password = args.password or os.environ.get("NEO4J_PASSWORD")
    database = args.database or os.environ.get("NEO4J_DATABASE", "neo4j")
    if not (uri and password):
        ap.error("need NEO4J_URI / NEO4J_PASSWORD (set them in .env or flags)")

    driver = GraphDatabase.driver(uri, auth=(args.user or "neo4j", password))
    driver.verify_connectivity()

    with driver.session(database=database) as session:
        backend = args.backend
        if backend == "auto":
            backend = "gds" if gds_available(session) else "python"
            log(f"· backend: {backend} (auto-detected)")
        else:
            if backend == "gds" and not gds_available(session):
                log("! GDS Cypher API not available on this instance - "
                    "the gds.graph.* procedures are not registered "
                    "(Aura Graph Analytics not enabled?)")
                sys.exit(2)

        if backend == "gds":
            run_gds(session, args.memory, args.top, args.dry_run)
        else:
            run_python(session, args.top, args.dry_run)
    driver.close()


if __name__ == "__main__":
    main()
