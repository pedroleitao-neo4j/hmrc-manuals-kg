#!/usr/bin/env python3
"""
Load the enriched HMRC manual JSON into Neo4j (Aura).

Input : the per-manual JSON written by enrich_hmrc_manuals.py
        (`sections` + `enrichment` blocks; the flat `pages` schema is
        handled too, with pages as sections).
Output : a graph with two layers:

  Lexical structure (the documents themselves)
    (:Document)-[:HAS_SECTION]->(:Section)
    (:Section)-[:HAS_CHILD]->(:Section)          from `child_sections`
    (:Section)-[:NEXT]->(:Section)               sibling order in the crawl
    (:Section)-[:CROSS_REFERENCES]->(:Section)   resolved by section_id
    (:Section)-[:HAS_ATTACHMENT]->(:Attachment)  where present
    Section.text -> Section.embedding           Fireworks embeddings, cached

  Extracted knowledge (from the `enrichment` block)
    (:Entity)              merged globally on the normalised `key`, so the
                           same concept extracted from several manuals joins
                           into one node
    (:Document)-[:DISCUSSES {salience}]->(:Entity)
    (:Section)-[:MENTIONS]->(:Entity)            via entity section_ids
    (:Entity)-[:<RELTYPE> {evidence, mentions}]->(:Entity)   e.g. APPLIES_TO

Embeddings
----------
Sections with text (skipping contents pages and near-empty stubs) and entity
definitions (name + definition) are embedded with the Fireworks embeddings
API (FIREWORKS_EMBEDDING_MODEL, default qwen3-embedding-8b, reduced to
FIREWORKS_EMBEDDING_DIMS=1024 of its native 4096 via MRL). The vector lands on
the node as `embedding`, with `section_embedding` / `entity_embedding` vector
indexes (cosine) created over them. Entities are global nodes merged across
manuals, so their embeddings share one cache file regardless of which manual
first mentioned them.
Embeddings are cached to EMBED_CACHEDIR/<slug>.jsonl keyed by a hash of
model + text, so — like the enrichment chunk cache — reruns and interrupted
runs replay for free. --no-embed skips all of it.

Every write is a MERGE, so the loader is idempotent: rerun it, or rerun it
after re-enriching a single manual, and only the differences land. Entities
from different manuals with the same key union their aliases/section_ids on
the relationship side; the node keeps the first-seen name/definition unless a
later manual has a longer one.

Connection config comes from .env (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
/ NEO4J_DATABASE); CLI flags win.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from neo4j import GraphDatabase
except ImportError:
    print("the `neo4j` driver is not installed:  pip install neo4j",
          file=sys.stderr)
    sys.exit(2)

GRAPH_SCHEMA_VERSION = 1


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


# --------------------------------------------------------------------------- #
# Fireworks embeddings
# --------------------------------------------------------------------------- #
DEFAULT_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
# Qwen3 embeddings are MRL-trained at 4096 dims; reduced vectors keep most of
# the quality at a quarter of the storage/query cost.
DEFAULT_EMBEDDING_DIMS = 1024
EMBEDDING_BATCH = 64                     # inputs per /embeddings request


class EmbedCache:
    """Append-only JSONL of text_hash -> embedding, one file per manual.

    Keyed on model + text (the same discipline as the enrichment chunk cache),
    so reruns after an interrupt — or after editing only the graph logic —
    replay the cache and cost nothing.
    """

    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["embedding"]
                    except Exception:
                        continue                  # truncated tail: ignore

    def get(self, key: str) -> list[float] | None:
        return self.data.get(key)

    def put(self, key: str, embedding: list[float]) -> None:
        with self._lock:
            self.data[key] = embedding
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "embedding": embedding}) + "\n")


class Embedder:
    """Fireworks /embeddings/v1 with batching, retries and the disk cache."""

    def __init__(self, api_key: str, model: str, base_url: str,
                 cachedir: str, dims: int | None = None,
                 batch: int = EMBEDDING_BATCH, retries: int = 6,
                 timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.dims_requested = dims
        self.url = base_url.rstrip("/") + "/embeddings"
        self.cachedir = cachedir
        self.batch = batch
        self.retries = retries
        self.timeout = timeout
        self.calls = 0
        self.cached_hits = 0
        self.dims: int | None = None    # learned from the first response
        self._caches: dict[str, EmbedCache] = {}

    def _cache(self, slug: str) -> EmbedCache:
        if slug not in self._caches:
            self._caches[slug] = EmbedCache(
                os.path.join(self.cachedir, f"{slug}.jsonl"))
        return self._caches[slug]

    @staticmethod
    def key(text: str, model: str, dims: int | None) -> str:
        h = hashlib.sha256()
        h.update(f"{model}\0{dims}\0".encode())
        h.update(text.encode())
        return h.hexdigest()[:32]

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        if self.dims_requested:
            payload["dimensions"] = self.dims_requested
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data=body, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                data = sorted(payload.get("data") or [], key=lambda d: d.get("index", 0))
                vectors = [d["embedding"] for d in data]
                if len(vectors) != len(texts):
                    raise ValueError(f"got {len(vectors)} embeddings "
                                     f"for {len(texts)} inputs")
                return vectors
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise                       # 400/401/404: fatal, don't loop
            except Exception as e:              # timeout, reset, bad JSON
                last = e
            time.sleep(min(30.0, 2 ** attempt + 1))
        raise RuntimeError(f"embeddings failed after {self.retries} attempts: {last}")

    def embed(self, slug: str, texts: list[str]) -> list[list[float]]:
        """Embed texts cache-first, in batches; preserves input order."""
        cache = self._cache(slug)
        keys = [self.key(t, self.model, self.dims_requested) for t in texts]
        out: list[list[float] | None] = [cache.get(k) for k in keys]
        for v in out:
            if v is not None:
                self.cached_hits += 1

        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), self.batch):
            idxs = todo[start:start + self.batch]
            vectors = self._request([texts[i] for i in idxs])
            self.calls += 1
            for i, vec in zip(idxs, vectors):
                out[i] = vec
                cache.put(keys[i], vec)

        result = [v for v in out if v is not None]
        if result:
            self.dims = self.dims or len(result[0])
        return result


# --------------------------------------------------------------------------- #
# Document model
# --------------------------------------------------------------------------- #
def doc_items(doc: dict) -> tuple[list[dict], str]:
    """(sections-or-pages, item_key) for either enrichment input schema."""
    if "sections" in doc:
        return doc["sections"], "sections"
    if "pages" in doc:
        return doc["pages"], "pages"
    raise ValueError("unrecognised input: no `sections` or `pages`")


def section_id(item: dict) -> str | None:
    return item.get("section_id") or item.get("base_path") or None


# --------------------------------------------------------------------------- #
# Cypher
# --------------------------------------------------------------------------- #
# One statement per structural hop; MERGE everywhere so reruns are no-ops.
CYPHER = {
    "constraints": [
        "CREATE CONSTRAINT doc_slug IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.slug IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS "
        "FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE CONSTRAINT attachment_id IF NOT EXISTS "
        "FOR (a:Attachment) REQUIRE a.url IS UNIQUE",
        "CREATE CONSTRAINT entity_key IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    ],
    "document": """
        MERGE (d:Document {slug: $slug})
        SET d.title = $title,
            d.url = $url,
            d.description = $description,
            d.format = $format,
            d.organisations = $organisations,
            d.first_published_at = $first_published_at,
            d.public_updated_at = $public_updated_at,
            d.schema_version = $schema_version
        RETURN d.slug AS slug
    """,
    # parent_slug lets section_ids stay document-scoped: an id like AGL1100 is
    # prefixed per manual so it's globally unique in practice, but the MATCH
    # through the Document makes that an invariant instead of an assumption.
    "sections": """
        MATCH (d:Document {slug: $slug})
        UNWIND $sections AS s
        MERGE (sec:Section {section_id: s.section_id})
        SET sec.title = s.title,
            sec.heading = s.heading,
            sec.url = s.url,
            sec.base_path = s.base_path,
            sec.text = s.text,
            sec.description = s.description,
            sec.word_count = s.word_count,
            sec.char_count = s.char_count,
            sec.is_contents_page = s.is_contents_page,
            sec.withdrawn = s.withdrawn,
            sec.first_published_at = s.first_published_at,
            sec.public_updated_at = s.public_updated_at,
            sec.updated_at = s.updated_at,
            sec.content_id = s.content_id,
            sec.position = s.position,
            sec.embedding = CASE WHEN s.embedding IS NULL
                THEN sec.embedding ELSE s.embedding END,
            sec.embedding_model = CASE WHEN s.embedding_model IS NULL
                THEN sec.embedding_model ELSE s.embedding_model END
        MERGE (d)-[:HAS_SECTION]->(sec)
        WITH collect(sec) AS secs
        UNWIND range(0, size(secs) - 2) AS i
        WITH secs[i] AS a, secs[i + 1] AS b
        MERGE (a)-[:NEXT]->(b)
        RETURN count(*) AS nexts
    """,
    "children": """
        UNWIND $rows AS row
        MATCH (parent:Section {section_id: row.parent})
        MATCH (child:Section {section_id: row.child})
        MERGE (parent)-[:HAS_CHILD]->(child)
        RETURN count(*) AS children
    """,
    "cross_references": """
        UNWIND $rows AS row
        MATCH (src:Section {section_id: row.src})
        MATCH (tgt:Section {section_id: row.tgt})
        MERGE (src)-[:CROSS_REFERENCES]->(tgt)
        RETURN count(*) AS xrefs
    """,
    "attachments": """
        UNWIND $rows AS row
        MATCH (s:Section {section_id: row.section_id})
        MERGE (a:Attachment {url: row.url})
        SET a.title = row.title,
            a.content_type = row.content_type,
            a.file_size = row.file_size
        MERGE (s)-[:HAS_ATTACHMENT]->(a)
        RETURN count(*) AS attachments
    """,
    # Entities merge globally on `key`: the whole point of the closed enums in
    # extraction is a graph that joins across manuals.
    "entities": """
        UNWIND $entities AS e
        MERGE (ent:Entity {key: e.key})
        ON CREATE SET ent.created_from = $slug
        SET ent.name = CASE
                WHEN ent.name IS NULL OR size(ent.name) < size(e.name)
                THEN e.name ELSE ent.name END,
            ent.type = coalesce(ent.type, e.type),
            ent.definition = CASE
                WHEN ent.definition IS NULL
                  OR size(ent.definition) < size(e.definition)
                THEN e.definition ELSE ent.definition END,
            ent.aliases = [x IN apoc.coll.toSet(
                coalesce(ent.aliases, []) + e.aliases) | x],
            ent.embedding = CASE WHEN e.embedding IS NULL
                THEN ent.embedding ELSE e.embedding END
        WITH ent, e
        MATCH (d:Document {slug: $slug})
        MERGE (d)-[di:DISCUSSES]->(ent)
        SET di.salience = CASE
                WHEN coalesce(di.salience_rank, 0) < e.salience_rank
                THEN e.salience ELSE coalesce(di.salience, e.salience) END,
            di.salience_rank = coalesce(e.salience_rank,
                                       coalesce(di.salience_rank, 0)),
            di.mentions = coalesce(di.mentions, 0) + e.mentions
        WITH ent, e
        UNWIND (CASE WHEN size(e.section_ids) > 0 THEN e.section_ids
                     ELSE [null] END) AS sid
        WITH ent, e, sid WHERE sid IS NOT NULL
        MATCH (s:Section {section_id: sid})
        MERGE (ent)-[:MENTIONED_IN]->(s)
        RETURN count(DISTINCT ent) AS entities
    """,
    # Relationship types are the enum values from extraction, so the edge type
    # carries the semantics directly: (e:Entity)-[:APPLIES_TO]->(e:Entity). The
    # type is baked into a per-type template because plain Cypher can't MERGE
    # a dynamic relationship type (that would need apoc.create.relationship).
    "relationship": """
        UNWIND $relationships AS r
        MATCH (src:Entity {key: r.source_key})
        MATCH (tgt:Entity {key: r.target_key})
        MERGE (src)-[rel:__TYPE__]->(tgt)
        ON CREATE SET rel.created_from = $slug
        SET rel.evidence = [x IN apoc.coll.toSet(
                coalesce(rel.evidence, []) + r.evidence) | x],
            rel.mentions = coalesce(r.mentions, 1)
        RETURN count(*) AS relationships
    """,
}


# apoc may not be enabled on every Aura instance; these variants do without it.
CYPHER_NO_APOC = {
    **CYPHER,
    "entities": """
        UNWIND $entities AS e
        MERGE (ent:Entity {key: e.key})
        ON CREATE SET ent.created_from = $slug
        SET ent.name = CASE
                WHEN ent.name IS NULL OR size(ent.name) < size(e.name)
                THEN e.name ELSE ent.name END,
            ent.type = coalesce(ent.type, e.type),
            ent.definition = CASE
                WHEN ent.definition IS NULL
                  OR size(ent.definition) < size(e.definition)
                THEN e.definition ELSE ent.definition END,
            ent.aliases = e.aliases,
            ent.embedding = CASE WHEN e.embedding IS NULL
                THEN ent.embedding ELSE e.embedding END
        WITH ent, e
        MATCH (d:Document {slug: $slug})
        MERGE (d)-[di:DISCUSSES]->(ent)
        SET di.salience = e.salience, di.mentions = e.mentions
        WITH ent, e
        UNWIND (CASE WHEN size(e.section_ids) > 0 THEN e.section_ids
                     ELSE [null] END) AS sid
        WITH ent, e, sid WHERE sid IS NOT NULL
        MATCH (s:Section {section_id: sid})
        MERGE (ent)-[:MENTIONED_IN]->(s)
        RETURN count(DISTINCT ent) AS entities
    """,
    "relationship": """
        UNWIND $relationships AS r
        MATCH (src:Entity {key: r.source_key})
        MATCH (tgt:Entity {key: r.target_key})
        MERGE (src)-[rel:__TYPE__]->(tgt)
        ON CREATE SET rel.created_from = $slug
        SET rel.evidence = r.evidence, rel.mentions = r.mentions
        RETURN count(*) AS relationships
    """,
}


def relationship_queries(cy: dict) -> dict[str, str]:
    """One concrete Cypher statement per relationship type in the enum."""
    import enrich_hmrc_manuals as enrich
    return {rtype: cy["relationship"].replace("__TYPE__", rtype)
            for rtype in enrich.RELATIONSHIP_TYPES}


# --------------------------------------------------------------------------- #
# Per-document load
# --------------------------------------------------------------------------- #
def _clean_props(item: dict, position: int) -> dict:
    """Whitelist + default the section properties we push to the graph."""
    return {
        "section_id": item.get("section_id") or item.get("base_path"),
        "title": item.get("title") or item.get("heading") or "",
        "heading": item.get("heading") or "",
        "url": item.get("url") or "",
        "base_path": item.get("base_path") or "",
        "text": (item.get("text") or "").strip(),
        "description": item.get("description") or "",
        "word_count": item.get("word_count"),
        "char_count": item.get("char_count"),
        "is_contents_page": bool(item.get("is_contents_page", False)),
        "withdrawn": bool(item.get("withdrawn", False)),
        "first_published_at": item.get("first_published_at"),
        "public_updated_at": item.get("public_updated_at"),
        "updated_at": item.get("updated_at"),
        "content_id": item.get("content_id"),
        "position": position,
    }


def load_document(session, path: str, cy: dict, relq: dict,
                  embedder: "Embedder | None" = None,
                  min_embed_words: int = 10) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)

    items, _ = doc_items(doc)
    meta = doc.get("manual") or {}
    slug = meta.get("slug") or os.path.splitext(os.path.basename(path))[0]
    enr = doc.get("enrichment") or {}

    sections = []
    rows_children, rows_xref, rows_attach = [], [], []
    known_ids = set()
    for i, item in enumerate(items):
        sid = section_id(item)
        if not sid:
            continue
        known_ids.add(sid)
        sections.append(_clean_props(item, len(sections)))
        for child in item.get("child_sections") or []:
            cid = child.get("section_id") if isinstance(child, dict) else None
            if cid:
                rows_children.append({"parent": sid, "child": cid})
        for x in item.get("cross_references") or []:
            # dict with a target section_id, or a bare id string
            xid = x.get("section_id") if isinstance(x, dict) else x
            if xid:
                rows_xref.append({"src": sid, "tgt": xid})
        for att in item.get("attachments") or []:
            url = att.get("url") if isinstance(att, dict) else att
            if not url:
                continue
            rows_attach.append({
                "section_id": sid,
                "url": url,
                "title": att.get("title") if isinstance(att, dict) else None,
                "content_type": att.get("content_type") if isinstance(att, dict) else None,
                "file_size": att.get("file_size") if isinstance(att, dict) else None,
            })

    # Only wire structure we can actually resolve inside this document.
    rows_children = [r for r in rows_children
                     if r["child"] in known_ids and r["child"] != r["parent"]]
    rows_xref = [r for r in rows_xref
                 if r["tgt"] in known_ids and r["tgt"] != r["src"]]

    session.execute_write(lambda tx: tx.run(cy["document"], {
        "slug": slug,
        "title": meta.get("title") or slug,
        "url": meta.get("url") or "",
        "description": meta.get("description") or "",
        "format": meta.get("format") or "hmrc_manual",
        "organisations": meta.get("organisations") or [],
        "first_published_at": meta.get("first_published_at"),
        "public_updated_at": meta.get("public_updated_at"),
        "schema_version": GRAPH_SCHEMA_VERSION,
    }).consume())

    # Embed eligible sections before they hit the graph: heading prepended so
    # the vector carries what the section is, not just its body.
    if embedder is not None:
        eligible = [s for s in sections
                    if not s["is_contents_page"]
                    and len(s["text"].split()) >= min_embed_words]
        if eligible:
            texts = [f"{s['heading'] or s['title']}\n\n{s['text']}"
                     if s["heading"] or s["title"] else s["text"]
                     for s in eligible]
            vectors = embedder.embed(slug, texts)
            for s, vec in zip(eligible, vectors):
                s["embedding"] = vec
                s["embedding_model"] = embedder.model

    session.execute_write(lambda tx: tx.run(cy["sections"], {
        "slug": slug, "sections": sections,
    }).consume())
    if rows_children:
        session.execute_write(lambda tx: tx.run(
            cy["children"],
            {"slug": slug, "rows": rows_children}).consume())
    if rows_xref:
        session.execute_write(lambda tx: tx.run(
            cy["cross_references"], {"rows": rows_xref}).consume())
    if rows_attach:
        session.execute_write(lambda tx: tx.run(
            cy["attachments"], {"rows": rows_attach}).consume())

    counts = {"sections": len(sections),
              "children": len(rows_children),
              "cross_references": len(rows_xref),
              "attachments": len(rows_attach),
              "entities": 0, "relationships": 0}

    entities = enr.get("entities") or []
    if entities:
        # Keep alias union across documents in the loader itself — Cypher-side
        # unioning needs apoc, and a Python dict is one line.
        rows = [{
            "key": e["key"],
            "name": e.get("name") or "",
            "type": e.get("type") or "Other",
            "definition": e.get("definition") or "",
            "aliases": sorted(set(e.get("aliases") or [])),
            "section_ids": [sid for sid in (e.get("section_ids") or [])
                            if sid in known_ids],
            "salience": e.get("salience") or "mentioned",
            "salience_rank": {"primary": 3, "secondary": 2,
                              "mentioned": 1}[e.get("salience") or "mentioned"],
            "mentions": e.get("mentions", 1),
        } for e in entities if e.get("key")]

        # Embed each entity's name + definition. Entities are global nodes
        # merged across manuals, so their embeddings live in one shared cache
        # file — the same entity from a second manual is a cache hit.
        if embedder is not None and rows:
            texts = [f"{r['name']}\n\n{r['definition']}".strip()
                    for r in rows]
            vectors = embedder.embed("_entities", texts)
            for r, vec in zip(rows, vectors):
                r["embedding"] = vec
                r["embedding_model"] = embedder.model

        rec = session.execute_write(
            lambda tx: tx.run(cy["entities"],
                              {"slug": slug, "entities": rows}).single())
        counts["entities"] = rec["entities"] if rec else 0

    rels = enr.get("relationships") or []
    if rels:
        by_type: dict[str, list[dict]] = {}
        for r in rels:
            if not (r.get("source_key") and r.get("target_key")):
                continue
            by_type.setdefault(r.get("type") or "RELATED_TO", []).append({
                "source_key": r["source_key"],
                "target_key": r["target_key"],
                "evidence": r.get("evidence") or [],
                "mentions": r.get("mentions", 1),
            })
        n = 0
        for rtype, rows in by_type.items():
            stmt = relq.get(rtype)
            if stmt is None:
                print(f"  ! {slug}: unknown relationship type {rtype!r}, "
                      f"skipping {len(rows)} edge(s)", file=sys.stderr)
                continue
            rec = session.execute_write(
                lambda tx, s=stmt, rw=rows: tx.run(
                    s, {"slug": slug, "relationships": rw}).single())
            n += rec["relationships"] if rec else 0
        counts["relationships"] = n

    return counts


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Load enriched HMRC manual JSON into Neo4j Aura.")
    ap.add_argument("-i", "--indir", default="hmrc_manuals_enriched")
    ap.add_argument("--manuals", nargs="*",
                    help="only these slugs (e.g. aggregates-levy)")
    ap.add_argument("--limit", type=int, help="only the first N documents")
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI"))
    ap.add_argument("--username", default=os.environ.get("NEO4J_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    ap.add_argument("--database", default=os.environ.get("NEO4J_DATABASE")
                    or "neo4j")
    ap.add_argument("--env", default=".env", help="path to .env (default .env)")
    ap.add_argument("--no-apoc", action="store_true",
                    help="use Cypher variants that avoid apoc (Aura without "
                         "the apoc plugin; disables alias/evidence unioning)")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embedding section text (embeddings are "
                         "on by default when FIREWORKS_API_KEY is set)")
    ap.add_argument("--embedding-model",
                    default=os.environ.get("FIREWORKS_EMBEDDING_MODEL")
                    or DEFAULT_EMBEDDING_MODEL,
                    help=f"Fireworks embeddings model "
                         f"(default {DEFAULT_EMBEDDING_MODEL})")
    ap.add_argument("--embedding-dims", type=int,
                    default=int(os.environ.get("FIREWORKS_EMBEDDING_DIMS")
                                or DEFAULT_EMBEDDING_DIMS),
                    help=f"vector dimensions, for MRL models that support "
                         f"reduction (default {DEFAULT_EMBEDDING_DIMS}; 0 "
                         f"uses the model's native size)")
    ap.add_argument("--embed-cachedir", default=".embed_cache",
                    help="per-manual embedding cache; makes reruns free")
    ap.add_argument("--embed-batch", type=int, default=EMBEDDING_BATCH,
                    help=f"inputs per embeddings request (default "
                         f"{EMBEDDING_BATCH})")
    ap.add_argument("--min-embed-words", type=int, default=10,
                    help="skip sections with fewer words than this "
                         "(default 10)")
    ap.add_argument("--clear", action="store_true",
                    help="delete the whole graph before loading (destructive)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and count without connecting to Neo4j")
    args = ap.parse_args()

    load_dotenv(args.env)
    uri = args.uri or os.environ.get("NEO4J_URI", "")
    username = args.username or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = args.password or os.environ.get("NEO4J_PASSWORD", "")
    database = args.database or "neo4j"
    if not uri and not args.dry_run:
        print("NEO4J_URI is not set (put it in .env or pass --uri)",
              file=sys.stderr)
        return 2
    if uri and not uri.startswith(("neo4j+s://", "neo4j+ssc://", "bolt://")):
        # Aura refuses plain neo4j:// on the same host; catch it before the
        # driver spends ten seconds discovering that.
        print(f"NEO4J_URI should use neo4j+s:// for Aura, got {uri!r}",
              file=sys.stderr)
        return 2

    files = sorted(f for f in os.listdir(args.indir)
                   if f.endswith(".json") and not f.startswith("_"))
    if args.manuals:
        want = {m.lower().removesuffix(".json") for m in args.manuals}
        files = [f for f in files if f[:-5].lower() in want]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no input JSON matched in {args.indir}", file=sys.stderr)
        return 1

    if args.dry_run:
        total = {"sections": 0, "entities": 0, "relationships": 0}
        for f in files:
            with open(os.path.join(args.indir, f), encoding="utf-8") as fh:
                doc = json.load(fh)
            items, _ = doc_items(doc)
            enr = doc.get("enrichment") or {}
            total["sections"] += len(items)
            total["entities"] += enr.get("entity_count", 0)
            total["relationships"] += enr.get("relationship_count", 0)
        print(f"would load {len(files)} document(s): {total}")
        return 0

    fw_key = os.environ.get("FIREWORKS_API_KEY", "")
    fw_base = os.environ.get("FIREWORKS_BASE_URL",
                             "https://api.fireworks.ai/inference/v1")
    embedder = None
    if not args.no_embed:
        if not fw_key:
            print("embedding disabled: FIREWORKS_API_KEY is not set "
                  "(or pass --no-embed to silence this)", file=sys.stderr)
        else:
            embedder = Embedder(fw_key, args.embedding_model, fw_base,
                                args.embed_cachedir,
                                dims=args.embedding_dims or None,
                                batch=args.embed_batch)
            print(f"embedding sections with {args.embedding_model} "
                  f"(cache: {args.embed_cachedir})")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"cannot connect to {uri}: {e}", file=sys.stderr)
        return 2

    cy = CYPHER_NO_APOC if args.no_apoc else CYPHER
    with driver.session(database=database) as session:
        for stmt in cy["constraints"]:
            session.run(stmt).consume()

        if args.clear:
            # One MATCH (n) DETACH DELETE over the whole graph blows Aura's
            # dbms.memory.transaction.total.max (~800MB with 1024-dim vectors
            # on every node) — so delete in bounded batches instead, each a
            # small transaction.
            print("clearing graph (--clear)…")
            batch = 5000
            total = 0
            while True:
                deleted = session.run(
                    f"MATCH (n) WITH n LIMIT {batch} "
                    "DETACH DELETE n RETURN count(*) AS n").single()["n"]
                total += deleted
                if not deleted:
                    break
            print(f"  · deleted {total:,} nodes")

        # Probe once: if apoc isn't available, fall back for the rest of the run.
        if not args.no_apoc:
            try:
                session.run("RETURN apoc.coll.toSet([1,1]) AS x").single()
            except Exception:
                print("apoc not available — switching to no-apoc variants "
                      "(alias/evidence unioning disabled)")
                cy = CYPHER_NO_APOC

        relq = relationship_queries(cy)

        t0 = time.time()
        bar = (tqdm(files, unit="doc", desc="manuals") if tqdm
               else None)
        totals = {"sections": 0, "entities": 0, "relationships": 0,
                  "children": 0, "cross_references": 0, "attachments": 0}
        for f in files:
            path = os.path.join(args.indir, f)
            try:
                counts = load_document(session, path, cy, relq,
                                       embedder, args.min_embed_words)
            except Exception as e:
                print(f"  ! {f} failed: {e}", file=sys.stderr)
                continue
            for k in totals:
                totals[k] += counts.get(k, 0)
            if bar:
                bar.set_postfix(ents=totals["entities"],
                                rels=totals["relationships"])
            elif not tqdm:
                print(f"  + {f[:-5]}: {counts['sections']} sections, "
                      f"{counts['entities']} entities, "
                      f"{counts['relationships']} relationships")
            if bar:
                bar.update(1)
        if bar:
            bar.close()

        if embedder is not None and embedder.dims:
            # Vector index over the section embeddings; dimension is learned
            # from the first real response so switching embedding models
            # needs no flag. Neo4j 5.13+ syntax.
            for label, name in (("Section", "section_embedding"),
                                ("Entity", "entity_embedding")):
                session.run(
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.embedding) "
                    "OPTIONS {indexConfig: {"
                    "`vector.dimensions`: $dims, "
                    "`vector.similarity_function`: 'cosine'}}",
                    dims=embedder.dims).consume()

    # Summary straight from the database, not from our own counters.
    with driver.session(database=database) as session:
        summary = session.run("""
            MATCH (d:Document) WITH count(d) AS documents
            OPTIONAL MATCH (:Document)-[:HAS_SECTION]->(s:Section)
            WITH documents, count(DISTINCT s) AS sections
            OPTIONAL MATCH (:Entity) WITH documents, sections, count(*) AS entities
            RETURN documents, sections, entities
        """).single()
        rels = session.run(
            "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n").single()["n"]
        mentions = session.run(
            "MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS n").single()["n"]
        embedded = session.run(
            "MATCH (s:Section) WHERE s.embedding IS NOT NULL "
            "RETURN count(s) AS n").single()["n"]
        embedded_ents = session.run(
            "MATCH (e:Entity) WHERE e.embedding IS NOT NULL "
            "RETURN count(e) AS n").single()["n"]

    driver.close()
    print(f"\ndone in {time.time() - t0:.1f}s — "
          f"{summary['documents']} documents, {summary['sections']} sections, "
          f"{summary['entities']} entities, {rels} entity rels, "
          f"{mentions} MENTIONED_IN, {embedded} embedded sections, "
          f"{embedded_ents} embedded entities"
          + (f" ({embedder.calls} embedding calls, "
             f"{embedder.cached_hits} cached)" if embedder is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
