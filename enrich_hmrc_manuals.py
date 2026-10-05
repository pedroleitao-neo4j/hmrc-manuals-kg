#!/usr/bin/env python3
"""
Enrich crawled HMRC manual JSON with tax/accounting entities and relationships.

Input  : the per-manual JSON written by crawl_hmrc_manuals.py
         (schema "hmrc_manual_crawl", and "hmrc_publications_crawl" for the
         flat guidance formats — both are handled).
Output : the same document plus an `enrichment` block, and per-section
         `entity_mentions` / `relationship_refs` back-pointers, so the graph
         and the prose stay in one file.

Extraction
----------
Sections are packed into chunks under a word budget and sent to a Fireworks
chat completion with `response_format: json_schema`, so the model is
constrained by the grammar in EXTRACTION_SCHEMA rather than asked nicely for
JSON. Entity and relationship *types* are closed enums in that schema — an
open vocabulary produces a different ontology per manual and nothing joins up
across 320 files.

Every entity carries the `section_id`s it was seen in, so an entity extracted
from four chunks becomes one node with four mentions. Relationships are
deduplicated on (source, type, target) with their evidence quotes unioned.

Chunk size vs reasoning budget
------------------------------
The default model is a reasoning model: on a measured 1,200-word chunk it spent
9,197 of 11,801 completion tokens on reasoning before emitting the object. The
same prompt at 3,000 words consumed a 40,000-token ceiling entirely in
reasoning and returned empty content (finish_reason "length"). So --chunk-words
and --max-tokens are coupled: raising the chunk size without raising the
ceiling turns every request into a paid non-answer. 1,200 / 32,000 is the
tested-safe pair, but reasoning length varies with how dense the text is, so
some chunks truncate anyway. Those are halved and retried (--max-splits)
rather than retried as-is — an identical request would truncate identically.

Cost control
------------
Chunk results are cached to CACHEDIR/<slug>.jsonl keyed by a hash of the chunk
text + model + prompt version. Re-running after an interrupt (or after editing
only the merge logic) replays the cache and costs nothing. The corpus is ~24M
words; a full pass is a paid, multi-hour job, so resumability is not optional.

Config comes from .env (FIREWORKS_API_KEY / _MODEL / _BASE_URL); CLI flags win.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:                               # degrade to plain prints
    tqdm = None

PROMPT_VERSION = 1          # bump to invalidate the chunk cache
SCHEMA_VERSION = 1

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    """Print without corrupting an active tqdm bar (or plainly if tqdm is absent)."""
    with _print_lock:
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg, flush=True)


def _bar(iterable=None, **kw):
    """tqdm wrapper that falls back to the bare iterable when tqdm is missing."""
    if tqdm is None:
        return iterable if iterable is not None else _NoBar(0)
    return tqdm(iterable, **kw) if iterable is not None else tqdm(**kw)


class _NoBar:
    """Minimal tqdm stand-in when tqdm isn't installed: count, update silently."""

    def __init__(self, total=None, **kw):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def set_description(self, *a):
        pass

    def set_description_str(self, *a):
        pass

    def set_postfix(self, *a, **kw):
        pass

    def refresh(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #
def load_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE loader. Real environment always wins."""
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
# Ontology
# --------------------------------------------------------------------------- #
# Closed enums: the point of extraction is a graph that joins across manuals,
# and free-text types don't join.
ENTITY_TYPES = [
    "Tax", "Duty", "Levy", "Relief", "Allowance", "Exemption", "Deduction",
    "Credit", "Charge", "Penalty", "Rate", "Threshold", "Legislation",
    "StatutoryProvision", "CaseLaw", "Regulation", "Treaty", "Form",
    "Scheme", "Procedure", "Obligation", "Deadline", "Organisation",
    "TaxpayerType", "Asset", "Income", "Expense", "AccountingConcept",
    "AccountingStandard", "FinancialInstrument", "Transaction", "Valuation",
    "Jurisdiction", "Definition", "Threshold_Test", "Other",
]

RELATIONSHIP_TYPES = [
    "DEFINED_BY", "GOVERNED_BY", "APPLIES_TO", "EXCLUDES", "EXEMPTS",
    "RELIEVES", "REQUIRES", "CALCULATED_FROM", "SUBJECT_TO", "PART_OF",
    "SUPERSEDES", "REPLACED_BY", "REFERENCES", "HAS_RATE", "HAS_THRESHOLD",
    "HAS_DEADLINE", "QUALIFIES_FOR", "ADMINISTERED_BY", "PENALTY_FOR",
    "REPORTED_ON", "MEASURED_BY", "EXAMPLE_OF", "RELATED_TO",
]

# json_schema response format. `strict`-style: every property required,
# additionalProperties false — Fireworks compiles this to a grammar.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Canonical name of the concept, as HMRC "
                                       "would write it (e.g. 'Business Asset "
                                       "Disposal Relief', 'TCGA 1992 s.165').",
                    },
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "definition": {
                        "type": "string",
                        "description": "One or two sentences grounded in the "
                                       "supplied text. Do not invent detail.",
                    },
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Abbreviations/synonyms used in the text.",
                    },
                    "section_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The HMRC section codes (e.g. CG15250) "
                                       "from the supplied text where this "
                                       "appears. Use only codes shown.",
                    },
                    "salience": {
                        "type": "string",
                        "enum": ["primary", "secondary", "mentioned"],
                        "description": "primary = the text is substantially "
                                       "about it; mentioned = in passing.",
                    },
                },
                "required": ["name", "type", "definition", "aliases",
                             "section_ids", "salience"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Entity name, exactly as in entities[].name.",
                    },
                    "type": {"type": "string", "enum": RELATIONSHIP_TYPES},
                    "target": {
                        "type": "string",
                        "description": "Entity name, exactly as in entities[].name.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Short quote or close paraphrase from the "
                                       "text supporting this relationship.",
                    },
                    "section_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Section codes the evidence came from.",
                    },
                },
                "required": ["source", "type", "target", "evidence", "section_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a UK tax and accounting knowledge engineer building a graph from "
    "HMRC internal manuals. You extract concepts and the relationships between "
    "them, strictly grounded in the text you are given.\n\n"
    "Rules:\n"
    "1. Extract only what the text supports. Never introduce tax facts from "
    "your own knowledge, and never guess a rate, threshold or date.\n"
    "2. Prefer canonical HMRC naming. Expand abbreviations in `name` and put "
    "the abbreviation in `aliases` (name: 'Capital Gains Tax', alias: 'CGT').\n"
    "3. Statutory references are entities: use the citation as the name "
    "(e.g. 'TCGA 1992 s.165', 'Finance Act 2020'), type StatutoryProvision "
    "or Legislation.\n"
    "4. Skip navigational and administrative noise: contents lists, 'this page "
    "has been archived', page numbering, internal HMRC team names with no tax "
    "meaning.\n"
    "5. Both endpoints of every relationship MUST appear in `entities`. Drop a "
    "relationship rather than inventing an endpoint for it.\n"
    "6. `section_ids` must be codes that literally appear in the supplied text.\n"
    "7. Deduplicate within your answer: one object per distinct concept, with "
    "its section_ids merged.\n"
    "Return nothing but the structured object."
)


def build_user_prompt(manual_title: str, manual_desc: str, chunk: list[dict]) -> str:
    parts = [
        f"HMRC manual: {manual_title}",
    ]
    if manual_desc:
        parts.append(f"Manual description: {manual_desc}")
    parts.append(
        "\nExtract tax and accounting concepts and their relationships from the "
        "following manual sections.\n"
    )
    for s in chunk:
        sid = s.get("section_id") or s.get("base_path") or ""
        head = s.get("heading") or s.get("title") or ""
        parts.append(f"\n=== [{sid}] {head} ===")
        desc = (s.get("description") or "").strip()
        if desc and desc not in (s.get("text") or ""):
            parts.append(desc)
        parts.append((s.get("text") or "").strip())
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Fireworks client
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Process-wide minimum spacing between requests."""

    def __init__(self, rps: float):
        self.min_interval = (1.0 / rps) if rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
        if wait:
            time.sleep(wait)

    def penalise(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class ExtractionError(Exception):
    pass


class TruncatedResponse(ExtractionError):
    """Hit max_tokens before the object closed — the chunk is too big to answer."""


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _parse_json_payload(content: str) -> dict:
    """Decode the model's message content.

    Reasoning models sometimes wrap the answer in <think> or a ```json fence
    even under a grammar, so strip both before falling back to a brace scan.
    """
    text = _THINK.sub("", content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ExtractionError(f"no JSON in response: {text[:200]!r}")


class Fireworks:
    """Chat completions with schema-constrained output, retry and backoff."""

    def __init__(self, api_key: str, model: str, base_url: str,
                 rps: float = 4.0, retries: int = 6, timeout: int = 900,
                 temperature: float = 0.0, max_tokens: int = 32000,
                 reasoning_effort: str | None = None):
        self.api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.limiter = RateLimiter(rps)
        self.retries = retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.throttled = 0
        self._lock = threading.Lock()
        # Fireworks accepts the OpenAI `json_schema` form; older deployments
        # only take `json_object` + `schema`. Detected once, on first 400.
        self._response_format_mode = "json_schema"

    def _response_format(self) -> dict:
        if self._response_format_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "hmrc_extraction",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                },
            }
        return {"type": "json_object", "schema": EXTRACTION_SCHEMA}

    def extract(self, system: str, user: str) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            payload_req = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": self._response_format(),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.reasoning_effort:
                payload_req["reasoning_effort"] = self.reasoning_effort
            body = json.dumps(payload_req).encode("utf-8")

            self.limiter.acquire()
            try:
                req = urllib.request.Request(self.url, data=body, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:400]
                except Exception:
                    pass
                last = ExtractionError(f"HTTP {e.code}: {detail}")
                if e.code == 400 and self._response_format_mode == "json_schema" \
                        and "json_schema" in detail:
                    # deployment predates the OpenAI-style field; switch once
                    log("    · falling back to json_object+schema response format")
                    self._response_format_mode = "json_object"
                    continue
                if e.code in (408, 409, 429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    wait = float(retry_after) if (retry_after or "").isdigit() \
                        else min(90.0, (2 ** attempt) + random.random() * 3)
                    if e.code == 429:
                        with self._lock:
                            self.throttled += 1
                        self.limiter.penalise(wait)
                    time.sleep(wait)
                    continue
                raise last                                  # 401/403/404: fatal
            except Exception as e:                          # timeout, reset, bad JSON
                last = e
                time.sleep(min(90.0, (2 ** attempt) + random.random() * 3))
                continue

            usage = payload.get("usage") or {}
            with self._lock:
                self.calls += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                self.completion_tokens += usage.get("completion_tokens", 0) or 0

            choices = payload.get("choices") or []
            if not choices:
                last = ExtractionError("empty choices")
                continue
            msg = choices[0].get("message") or {}
            finish = choices[0].get("finish_reason")
            try:
                result = _parse_json_payload(msg.get("content") or "")
            except (ExtractionError, json.JSONDecodeError) as e:
                if finish == "length":
                    raise TruncatedResponse(
                        "response truncated at max_tokens") from e
                last = e
                continue
            if not isinstance(result, dict):
                last = ExtractionError("response was not an object")
                continue
            result.setdefault("entities", [])
            result.setdefault("relationships", [])
            return result
        raise ExtractionError(str(last))


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_sections(sections: list[dict], chunk_words: int,
                   min_words: int) -> list[list[dict]]:
    """Pack sections into chunks under a word budget, in document order.

    A section longer than the budget becomes its own chunk (split further by
    words, keeping the section header on each piece so the model still knows
    which section_id the content belongs to).
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_words = 0

    for s in sections:
        if s.get("error") or s.get("is_contents_page"):
            continue
        text = (s.get("text") or "").strip()
        if len((text + " " + (s.get("description") or "")).split()) < min_words:
            continue

        words = text.split()
        if len(words) > chunk_words:
            if cur:
                chunks.append(cur)
                cur, cur_words = [], 0
            for i in range(0, len(words), chunk_words):
                part = dict(s)
                part["text"] = " ".join(words[i:i + chunk_words])
                if i:
                    part["heading"] = f"{s.get('heading') or s.get('title') or ''} " \
                                      f"(continued {i // chunk_words + 1})"
                chunks.append([part])
            continue

        if cur_words + len(words) > chunk_words and cur:
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += len(words)

    if cur:
        chunks.append(cur)
    return chunks


def split_chunk(chunk: list[dict]) -> list[list[dict]] | None:
    """Halve a chunk for a truncation retry, or None if it can't shrink further.

    Multi-section chunks split on a section boundary; a lone oversized section
    splits by words, keeping its identity on both halves so section_ids still
    resolve.
    """
    if len(chunk) > 1:
        mid = len(chunk) // 2
        return [chunk[:mid], chunk[mid:]]

    s = chunk[0]
    words = (s.get("text") or "").split()
    if len(words) < 200:                       # already tiny: splitting won't help
        return None
    mid = len(words) // 2
    a, b = dict(s), dict(s)
    a["text"] = " ".join(words[:mid])
    b["text"] = " ".join(words[mid:])
    base = s.get("heading") or s.get("title") or ""
    b["heading"] = f"{base} (continued)"
    return [[a], [b]]


def chunk_key(model: str, chunk: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(f"{PROMPT_VERSION}\0{model}\0".encode())
    for s in chunk:
        h.update((s.get("section_id") or s.get("base_path") or "").encode())
        h.update(b"\0")
        h.update((s.get("text") or "").encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def norm(name: str) -> str:
    """Collapse casing/punctuation/whitespace so 'the CGT relief' joins 'CGT Relief'."""
    s = (name or "").strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"[^a-z0-9'&./§\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


SALIENCE_RANK = {"primary": 3, "secondary": 2, "mentioned": 1}


def merge_results(results: list[dict], valid_section_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Fold per-chunk extractions into one entity list and one edge list.

    Entities key on normalised name; the longest definition and the highest
    salience win, aliases and section_ids union. Relationships key on
    (source, type, target) and are dropped if either endpoint is unknown —
    the prompt forbids dangling edges, but the merge is where it's enforced.
    """
    ents: dict[str, dict] = {}
    alias_to_key: dict[str, str] = {}

    for res in results:
        for e in res.get("entities") or []:
            name = (e.get("name") or "").strip()
            key = norm(name)
            if not key or len(key) < 2:
                continue
            sids = [s for s in (e.get("section_ids") or [])
                    if s in valid_section_ids] if valid_section_ids else \
                   list(e.get("section_ids") or [])
            aliases = [a.strip() for a in (e.get("aliases") or []) if a and a.strip()]
            cur = ents.get(key)
            if cur is None:
                ents[key] = {
                    "key": key,
                    "name": name,
                    "type": e.get("type") or "Other",
                    "definition": (e.get("definition") or "").strip(),
                    "aliases": sorted({a for a in aliases if norm(a) != key}),
                    "section_ids": sorted(set(sids)),
                    "salience": e.get("salience") or "mentioned",
                    "mentions": 1,
                }
            else:
                cur["mentions"] += 1
                if len(e.get("definition") or "") > len(cur["definition"]):
                    cur["definition"] = (e.get("definition") or "").strip()
                    cur["name"] = name or cur["name"]
                if SALIENCE_RANK.get(e.get("salience"), 0) > \
                        SALIENCE_RANK.get(cur["salience"], 0):
                    cur["salience"] = e["salience"]
                cur["aliases"] = sorted(set(cur["aliases"]) |
                                        {a for a in aliases if norm(a) != key})
                cur["section_ids"] = sorted(set(cur["section_ids"]) | set(sids))

    for key, e in ents.items():
        for a in e["aliases"]:
            alias_to_key.setdefault(norm(a), key)

    def resolve(name: str) -> str | None:
        k = norm(name)
        if k in ents:
            return k
        return alias_to_key.get(k)

    rels: dict[tuple, dict] = {}
    dropped = 0
    for res in results:
        for r in res.get("relationships") or []:
            src = resolve(r.get("source") or "")
            tgt = resolve(r.get("target") or "")
            rtype = r.get("type")
            if not src or not tgt or src == tgt or rtype not in RELATIONSHIP_TYPES:
                dropped += 1
                continue
            sids = [s for s in (r.get("section_ids") or [])
                    if s in valid_section_ids] if valid_section_ids else \
                   list(r.get("section_ids") or [])
            k = (src, rtype, tgt)
            cur = rels.get(k)
            evidence = (r.get("evidence") or "").strip()
            if cur is None:
                rels[k] = {
                    "source": ents[src]["name"],
                    "source_key": src,
                    "type": rtype,
                    "target": ents[tgt]["name"],
                    "target_key": tgt,
                    "evidence": [evidence] if evidence else [],
                    "section_ids": sorted(set(sids)),
                    "mentions": 1,
                }
            else:
                cur["mentions"] += 1
                if evidence and evidence not in cur["evidence"]:
                    cur["evidence"].append(evidence)
                cur["section_ids"] = sorted(set(cur["section_ids"]) | set(sids))

    entities = sorted(ents.values(),
                      key=lambda e: (-SALIENCE_RANK.get(e["salience"], 0),
                                     -e["mentions"], e["name"].lower()))
    relationships = sorted(rels.values(),
                           key=lambda r: (-r["mentions"], r["source"].lower(),
                                          r["type"], r["target"].lower()))
    if dropped:
        log(f"    · dropped {dropped} relationship(s) with unresolved endpoints")
    return entities, relationships


def attach_to_sections(items: list[dict], entities: list[dict],
                       relationships: list[dict]) -> None:
    """Write back-pointers into each section so the prose carries its graph."""
    by_sid: dict[str, dict] = {}
    for s in items:
        sid = s.get("section_id")
        if sid:
            by_sid[sid] = s
        s.pop("entity_mentions", None)
        s.pop("relationship_refs", None)

    for e in entities:
        for sid in e["section_ids"]:
            s = by_sid.get(sid)
            if s is not None:
                s.setdefault("entity_mentions", []).append(e["key"])
    for i, r in enumerate(relationships):
        for sid in r["section_ids"]:
            s = by_sid.get(sid)
            if s is not None:
                s.setdefault("relationship_refs", []).append(i)

    for s in items:
        if "entity_mentions" in s:
            s["entity_mentions"] = sorted(set(s["entity_mentions"]))
        if "relationship_refs" in s:
            s["relationship_refs"] = sorted(set(s["relationship_refs"]))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class ChunkCache:
    """Append-only JSONL of chunk_key -> extraction, one file per manual."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["result"]
                    except Exception:
                        continue                       # truncated tail: ignore

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, result: dict) -> None:
        with self._lock:
            self.data[key] = result
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "result": result},
                                    ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Per-document enrichment
# --------------------------------------------------------------------------- #
def doc_items(doc: dict) -> tuple[list[dict], str, str, str]:
    """Return (sections, item_key, title, description) for either input schema."""
    if "sections" in doc:
        meta = doc.get("manual") or {}
        return (doc["sections"], "sections",
                meta.get("title") or "", meta.get("description") or "")
    if "pages" in doc:
        fmt = doc.get("format") or "publications"
        return (doc["pages"], "pages",
                f"HMRC {fmt} publications", "")
    raise ExtractionError("unrecognised input: no `sections` or `pages`")


def enrich_file(fw: Fireworks, path: str, outdir: str, args) -> dict:
    slug = os.path.splitext(os.path.basename(path))[0]
    outpath = os.path.join(outdir, f"{slug}.json")
    if os.path.exists(outpath) and not args.force:
        log(f"  = {slug}: exists, skipping (use --force)")
        return {"slug": slug, "skipped": True}

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    items, item_key, title, desc = doc_items(doc)

    chunks = chunk_sections(items, args.chunk_words, args.min_words)
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    if not chunks:
        log(f"  · {slug}: no extractable text, skipping")
        return {"slug": slug, "skipped": True, "reason": "empty"}

    cache = ChunkCache(os.path.join(args.cachedir, f"{slug}.jsonl"))
    keys = [chunk_key(fw.model, c) for c in chunks]
    todo = [i for i, k in enumerate(keys) if cache.get(k) is None]
    log(f"  · {slug}: {len(chunks)} chunks ({len(chunks) - len(todo)} cached, "
        f"{len(todo)} to extract)")

    if args.dry_run:
        words = sum(len((s.get('text') or '').split()) for c in chunks for s in c)
        return {"slug": slug, "chunks": len(chunks), "to_extract": len(todo),
                "words": words, "dry_run": True}

    failures = 0
    splits = 0
    counter_lock = threading.Lock()

    def extract_chunk(chunk: list[dict], label: str, depth: int = 0) -> dict | None:
        """Extract one chunk, halving and retrying if the answer didn't fit.

        A truncated response means the model spent its whole budget reasoning
        about too much text — the same request would truncate again, so retry
        on smaller inputs rather than on the identical one.
        """
        nonlocal splits
        try:
            return fw.extract(SYSTEM_PROMPT, build_user_prompt(title, desc, chunk))
        except TruncatedResponse as e:
            halves = split_chunk(chunk) if depth < args.max_splits else None
            if halves is None:
                log(f"    ! {slug} {label}: {e}, cannot split further")
                return None
            with counter_lock:
                splits += 1
            log(f"    · {slug} {label}: truncated, retrying as 2 smaller chunks")
            parts = [extract_chunk(h, f"{label}.{j}", depth + 1)
                     for j, h in enumerate(halves)]
            parts = [p for p in parts if p]
            if not parts:
                return None
            merged: dict = {"entities": [], "relationships": []}
            for p in parts:
                merged["entities"].extend(p.get("entities") or [])
                merged["relationships"].extend(p.get("relationships") or [])
            return merged
        except ExtractionError as e:
            log(f"    ! {slug} {label}: {e}")
            return None

    if todo:
        def run(i: int) -> dict | None:
            res = extract_chunk(chunks[i], f"chunk {i}")
            if res is not None:
                cache.put(keys[i], res)
            return res

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run, i) for i in todo]
            pbar = _bar(total=len(futures), desc=f"    {slug[:30]}",
                        unit="chunk", leave=False, smoothing=0.1)
            for fut in as_completed(futures):
                if fut.result() is None:
                    failures += 1
                pbar.update(1)                    # ticks as each chunk lands
                pbar.set_postfix(fail=failures, splits=splits,
                                 throttled=fw.throttled,
                                 calls=fw.calls,
                                 in_tok=f"{fw.prompt_tokens:,}",
                                 out_tok=f"{fw.completion_tokens:,}",
                                 refresh=False)
            pbar.close()

    results = [cache.get(k) for k in keys]
    results = [r for r in results if r]

    valid_ids = {s.get("section_id") for s in items if s.get("section_id")}
    entities, relationships = merge_results(results, valid_ids)
    attach_to_sections(items, entities, relationships)

    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    rel_counts: dict[str, int] = {}
    for r in relationships:
        rel_counts[r["type"]] = rel_counts.get(r["type"], 0) + 1

    doc["enrichment"] = {
        "schema": "hmrc_manual_enrichment",
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": fw.model,
        "provider": "fireworks",
        "enriched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks_total": len(chunks),
        "chunks_extracted": len(results),
        "chunks_failed": failures,
        "chunks_split_on_truncation": splits,
        "chunk_words": args.chunk_words,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entity_type_counts": dict(sorted(type_counts.items(),
                                          key=lambda kv: -kv[1])),
        "relationship_type_counts": dict(sorted(rel_counts.items(),
                                                key=lambda kv: -kv[1])),
        "entities": entities,
        "relationships": relationships,
    }
    doc[item_key] = items

    os.makedirs(outdir, exist_ok=True)
    tmp = outpath + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, outpath)                       # atomic: no truncated files

    log(f"  + {slug}: {len(entities)} entities, {len(relationships)} relationships"
        f"{f', {failures} chunk(s) failed' if failures else ''} -> {outpath}")
    return {"slug": slug, "chunks": len(chunks), "chunks_failed": failures,
            "chunks_split": splits,
            "entities": len(entities), "relationships": len(relationships)}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract tax/accounting entities and relationships from "
                    "crawled HMRC manual JSON using the Fireworks AI API.")
    ap.add_argument("-i", "--indir", default="hmrc_manuals",
                    help="directory of crawled manual JSON (default hmrc_manuals)")
    ap.add_argument("-o", "--outdir", default="hmrc_manuals_enriched")
    ap.add_argument("--cachedir", default=".enrich_cache",
                    help="per-manual chunk cache (JSONL); makes reruns free")
    ap.add_argument("--manuals", nargs="*",
                    help="only these slugs (e.g. aggregates-levy)")
    ap.add_argument("--limit", type=int, help="only the first N input files")
    ap.add_argument("--chunk-words", type=int, default=1200,
                    help="word budget per request (default 1200; see the note "
                         "on reasoning budget in the module docstring before "
                         "raising it)")
    ap.add_argument("--min-words", type=int, default=25,
                    help="skip sections shorter than this (default 25)")
    ap.add_argument("--max-chunks", type=int,
                    help="cap chunks per manual — for sampling large manuals")
    ap.add_argument("--max-splits", type=int, default=2,
                    help="how many times a truncated chunk may be halved and "
                         "retried (default 2, i.e. down to a quarter)")
    ap.add_argument("-w", "--workers", type=int,
                    default=int(os.environ.get("FIREWORKS_WORKERS", 4) or 4),
                    help="concurrent extraction requests (default 4)")
    ap.add_argument("--rps", type=float,
                    default=float(os.environ.get("FIREWORKS_RPS", 4) or 4),
                    help="global request-rate ceiling (default 4)")
    ap.add_argument("--model", help="override FIREWORKS_MODEL")
    ap.add_argument("--base-url", help="override FIREWORKS_BASE_URL")
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("FIREWORKS_TEMPERATURE", 0.0) or 0.0))
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("FIREWORKS_MAX_TOKENS", 32000) or 32000),
                    help="completion ceiling; must cover reasoning tokens too "
                         "(default 32000)")
    ap.add_argument("--reasoning-effort",
                    default=os.environ.get("FIREWORKS_REASONING_EFFORT") or None,
                    help="pass through to the API (e.g. low/medium/high) for "
                         "models that support it; omitted by default")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-request timeout in seconds (default 900; "
                         "reasoning passes are slow)")
    ap.add_argument("--env", default=".env", help="path to .env (default .env)")
    ap.add_argument("--force", action="store_true",
                    help="re-enrich manuals whose output already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="report chunk/word counts without calling the API")
    args = ap.parse_args()

    load_dotenv(args.env)
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    model = args.model or os.environ.get("FIREWORKS_MODEL") or DEFAULT_MODEL
    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_BASE_URL
    if not api_key and not args.dry_run:
        print("FIREWORKS_API_KEY is not set (put it in .env or the environment)",
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

    fw = Fireworks(api_key, model, base_url, rps=args.rps,
                   temperature=args.temperature, max_tokens=args.max_tokens,
                   reasoning_effort=args.reasoning_effort, timeout=args.timeout)

    log(f"model {model} @ {base_url}")
    log(f"{len(files)} document(s) from {args.indir} -> {args.outdir}\n")

    t0 = time.time()
    summary: list[dict] = []
    dbar = _bar(total=len(files), desc="manuals", unit="doc",
                disable=not files)
    for i, name in enumerate(files, 1):
        path = os.path.join(args.indir, name)
        if tqdm is None:
            log(f"[{i}/{len(files)}] {name}")
        dbar.set_description_str(f"manuals · {name[:-5][:40]}")
        try:
            summary.append(enrich_file(fw, path, args.outdir, args))
        except KeyboardInterrupt:
            log("interrupted (chunk cache retained — rerun to resume)")
            break
        except Exception as e:
            log(f"  ! {name} failed: {e}")
            summary.append({"slug": name[:-5], "error": str(e)})
        dbar.update(1)
        dbar.set_postfix(docs=i, ents=sum(s.get("entities", 0) for s in summary
                                          if s.get("entities") is not None),
                         calls=fw.calls, in_tok=f"{fw.prompt_tokens:,}",
                         out_tok=f"{fw.completion_tokens:,}", refresh=False)
    dbar.close()

    elapsed = time.time() - t0
    done = [s for s in summary if s.get("entities") is not None]
    idx = {
        "schema": "hmrc_enrichment_index",
        "enriched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "documents": len(done),
        "documents_skipped": sum(1 for s in summary if s.get("skipped")),
        "documents_failed": sum(1 for s in summary if s.get("error")),
        "entities": sum(s.get("entities", 0) for s in done),
        "relationships": sum(s.get("relationships", 0) for s in done),
        "chunks_failed": sum(s.get("chunks_failed", 0) for s in done),
        "chunks_split_on_truncation": sum(s.get("chunks_split", 0) for s in done),
        "api_calls": fw.calls,
        "prompt_tokens": fw.prompt_tokens,
        "completion_tokens": fw.completion_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "results": summary,
    }
    if not args.dry_run:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "_enrichment_index.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(idx, fh, ensure_ascii=False, indent=1)

    log(f"\ndone in {elapsed/60:.1f} min — {idx['documents']} documents, "
        f"{idx['entities']:,} entities, {idx['relationships']:,} relationships, "
        f"{fw.calls:,} API calls, "
        f"{fw.prompt_tokens:,} prompt + {fw.completion_tokens:,} completion tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
