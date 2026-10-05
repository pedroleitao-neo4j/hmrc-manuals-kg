# AGENTS.md — Searching the HMRC Manuals Knowledge Graph

This project contains a Neo4j knowledge graph of the HMRC internal manuals
(crawled from `/hmrc-internal-manuals/...` on GOV.UK, enriched with entities and
relationships, and loaded with vector embeddings). Two search mechanisms are
available; both should be used when answering tax questions, and answers must
always cite sources (URLs to the original manual sections) and include graph
cross-references.

## 1. Vector (semantic) search — `vector_search.py`

For natural-language questions ("what relief applies when selling a
business?"), start with semantic search:

```bash
python3 vector_search.py "relief for selling a business"                 # both indexes
python3 vector_search.py "how is aggregates levy charged?" --kind sections --top 5
python3 vector_search.py "partnership" --kind entities
python3 vector_search.py "..." --text     # human-readable output instead of JSON
```

- `--kind`: `sections` (full manual text, index `section_embedding`), `entities`
  (extracted tax concepts, index `entity_embedding`), or `both` (default).
- Output is JSON (default): each hit has `score`, `title`, `url`, `manual`,
  `entities`/`related`, and a text snippet.
- Config comes from `.env` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
  `NEO4J_DATABASE`, `FIREWORKS_API_KEY`, etc.). Exit code 2 = connection
  problem, 3 = embedding API problem.
- Run from the project root so `.env` is found.

## 2. Graph search — `neo4j_hmrc_graph` MCP endpoint

Use the MCP tools for structured/relational queries once vector search has
identified relevant nodes:

- `get-schema` — labels, relationship types, property keys.
- `read-cypher` — run read-only Cypher.

### Graph model

- **`Document`** (manual cover page) — `title`, `slug`, `url`, `description`.
- **`Section`** (a manual page) — `heading`, `section_id`, `text`, `url`,
  `position`, `embedding`. Structure: `(Document)-[:HAS_SECTION]->(Section)`,
  `(Section)-[:HAS_CHILD]->(Section)`, `(Section)-[:NEXT]->(Section)`.
- **`Entity`** (extracted tax concept) — `name`, `type`, `definition`,
  `aliases`, `embedding`, `key`. Linked to sections via `(Entity)-[:MENTIONED_IN]->(Section)`
  and to documents via `(Document)-[:DISCUSSES]->(Entity)` with `mentions`,
  `salience`, `salience_rank`.
- **Entity→Entity relationships** (all carry `created_from`, `evidence`,
  `mentions`): `ADMINISTERED_BY`, `APPLIES_TO`, `CALCULATED_FROM`, `DEFINED_BY`,
  `EXAMPLE_OF`, `EXCLUDES`, `EXEMPTS`, `GOVERNED_BY`, `HAS_DEADLINE`,
  `HAS_RATE`, `HAS_THRESHOLD`, `MEASURED_BY`, `PART_OF`, `PENALTY_FOR`,
  `REFERENCES`, `RELATED_TO`, `RELIEVES`, `REPORTED_ON`, `REQUIRES`,
  `SUBJECT_TO`, `SUPERSEDES`. `evidence` is a list of quotes;
  `created_from` is the **manual slug** (= `Document.slug`, e.g.
  `double-taxation-relief`), not a section id.

### Search first, then traverse by id

Vector search already returns node identifiers: section hits carry `id`
(= `section_id`) and entity hits carry `key`. Both are indexed. Anchor
every follow-up Cypher on them, using `{section_id: $id}` or `{key: $key}`.
**Do not** re-find the same nodes with `CONTAINS` / `toLower` / `=~` on
`s.text`, `s.heading` or `e.name`. Those predicates can't use an index,
so they scan all ~84k sections and ~237k entities, return noisy matches,
and repeat the vector search you already ran. If the hits miss part of
the question, run another vector search with a rephrased query.

### Useful query patterns

Explore a concept found by vector search (`key` from an entity hit):

```cypher
MATCH (e:Entity {key: $key})
OPTIONAL MATCH (e)-[r]-(other:Entity)
RETURN e.name, e.type, e.definition,
       type(r) AS rel, other.name, other.key, r.evidence, r.created_from
ORDER BY r.mentions DESC LIMIT 25
```

Relationships stated in a section found by vector search (`id` from a
section hit). The evidence check keeps only facts quoted in that
section's text:

```cypher
MATCH (s:Section {section_id: $id})
MATCH (a:Entity)-[:MENTIONED_IN]->(s)
MATCH (a)-[r]->(b:Entity)-[:MENTIONED_IN]->(s)
WHERE any(q IN r.evidence WHERE s.text CONTAINS q)
RETURN a.name, type(r), b.name, r.evidence, s.url
```

Find the manual sections that mention a concept:

```cypher
MATCH (e:Entity {key: $key})-[:MENTIONED_IN]->(s:Section)
RETURN s.heading, s.section_id, s.url ORDER BY s.position
```

Get a manual's section outline:

```cypher
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
WHERE d.slug = $slug
RETURN s.position, s.section_id, s.heading, s.url ORDER BY s.position
```

Follow the citation chain from a relationship to its evidence. Cite the
section that mentions both ends and quotes the evidence. If no section
qualifies, fall back to the manual named by `created_from`:

```cypher
MATCH (a:Entity {key: $key})-[r]-(b:Entity)
WHERE type(r) IN ['HAS_RATE','HAS_THRESHOLD','RELIEVES','EXEMPTS']
OPTIONAL MATCH (d:Document {slug: r.created_from})
CALL (a, b, r) {
  OPTIONAL MATCH (a)-[:MENTIONED_IN]->(s:Section)<-[:MENTIONED_IN]-(b)
  WHERE any(q IN r.evidence WHERE s.text CONTAINS q)
  RETURN s LIMIT 1
}
RETURN a.name, type(r), b.name, r.evidence, coalesce(s.url, d.url) AS url
```

## 3. Recommended workflow for tax questions

1. Run `vector_search.py` on the question (both kinds) to find candidate
   sections/entities, and note their `id` / `key`.
2. Use `read-cypher`, anchored on those ids/keys, to pull the full section
   `text`, related entities, and the relationships' `evidence`.
3. Answer with:
   - **Sources**: the `url` of every section or relationship evidence used.
     Use the evidence-matched section URL, or the manual URL from
     `created_from` (a `Document.slug`).
   - **Graph cross-references**: the entity names and relationship types
     traversed (e.g. *Aggregates Levy `-[:SUBJECT_TO]->` Registered Trader*).
4. If the answer isn't in the graph, say so — don't guess.

## Project layout (for context)

- `crawl_hmrc_manuals.py` — crawler; `enrich_hmrc_to_neo4j.py` —
  entity/relationship enrichment; `load_hmrc_to_neo4j.py` — graph loader
  (embeds with the same Fireworks model as `vector_search.py` —
  `qwen3-embedding-8b`, 1024 dims; query vectors must match).
- `hmrc_manuals/` and `hmrc_manuals_enriched/` — raw and enriched crawl JSON.
- `formats.md` — notes on GOV.UK content formats (HMRC).
