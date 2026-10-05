#!/usr/bin/env python3
"""
Crawl HMRC manuals and guidance from GOV.UK into JSON files.

Discovery
---------
GOV.UK Search API:
  /api/search.json?filter_format=hmrc_manual                 -> the 253 internal manuals
  /api/search.json?filter_format=hmrc_manual_section
                  &filter_manual=<manual base_path>          -> every section
  /api/search.json?filter_format=manual                      -> 68 manuals from the
                  &filter_organisations=hm-revenue-customs      other Manuals app
                  &filter_format=manual_section                (Tax Agent's Handbook,
                  &filter_manual=<manual base_path>            notices manuals, ...)
                                                              -> their sections
  /api/search.json?filter_format=<publication format>        -> flat guidance pages
                  &filter_organisations=hm-revenue-customs      (detailed_guide, notice,
                                                                answer, ...)

The manual's own Content API page lists ONLY its top-level sections (e.g. the
Capital Gains Manual shows 17 of its 5,423) because sections nest arbitrarily
deep via child_section_groups. Enumerating by search avoids a recursive descent
and is authoritative for the full set.

Content
-------
Each section is fetched from /api/content<base_path>, which returns real HTML in
details.body. We extract text structurally: headings are kept, list items get
bullets, and table rows are emitted pipe-joined so tabular values survive
(a naive get_text() collapses a rates table into meaningless loose lines).

Publication formats (detailed_guide, guidance, notice, ...) are mostly
/government/publications/ pages whose body is only a short summary — the real
content is in PDF attachments. For those we record the body text AND an
attachment index (title, url, content type) per page.

Output: OUTDIR/<manual-slug>.json per manual, OUTDIR/<format>.json per
publication format (see build_manual_json / build_publications_json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup, NavigableString, Tag

BASE = "https://www.gov.uk"
SEARCH = BASE + "/api/search.json"
CONTENT = BASE + "/api/content"

UA = "bdo-kb-crawler/1.0 (+pedro.leitao@neo4j.com; HMRC manual ingest for semantic layer)"
PAGE = 500            # search page size (API tolerates 1500; 500 is a safe middle)
SCHEMA_VERSION = 1
HMRC_ORG = "hm-revenue-customs"

# manual-family formats -> (section search format, needs org filter on listing)
# `hmrc_manual` is HMRC-only by definition; `manual` (the other Manuals app,
# e.g. the Tax Agent's Handbook) is shared across government, so its listing
# must be scoped to HMRC or we'd crawl other departments' manuals too.
MANUAL_TYPES: dict[str, tuple[str, bool]] = {
    "hmrc_manual": ("hmrc_manual_section", False),
    "manual": ("manual_section", True),
}

# flat guidance/publication formats: one JSON file each, org-filtered.
# These pages typically carry a short summary body + PDF attachments.
PUBLICATION_FORMATS = ["detailed_guide", "guidance", "notice",
                       "statutory_guidance", "answer"]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Process-wide minimum spacing between requests (token bucket, 1 token)."""

    def __init__(self, rps: float):
        self.min_interval = (1.0 / rps) if rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait < 0:
                wait = 0.0
            self._next = max(now, self._next) + self.min_interval
        if wait:
            time.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """After a 429, push the whole fleet's next slot back."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class RateLimited(Exception):
    """Ran out of retries on a throttled/failing request."""


class Fetcher:
    """GET + JSON decode with retry/backoff on 429, 5xx and transport errors."""

    def __init__(self, rps: float = 12.0, retries: int = 8, timeout: int = 45):
        self.limiter = RateLimiter(rps)
        self.retries = retries
        self.timeout = timeout
        self.calls = 0
        self.throttles = 0
        self._lock = threading.Lock()

    def get_json(self, url: str, required: bool = False) -> dict | None:
        """Fetch and decode JSON.

        required=True raises RateLimited instead of returning None when the
        request can't be completed — used for section *listings*, where a
        dropped page would silently shrink a manual.
        """
        last = None
        for attempt in range(self.retries):
            self.limiter.acquire()
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": UA,
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = r.read()
                with self._lock:
                    self.calls += 1
                return json.loads(payload)
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 404:
                    return None                       # genuinely gone; don't retry
                if e.code in (429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    wait = float(retry_after) if (retry_after or "").isdigit() else \
                        min(60.0, (2 ** attempt) + random.random() * 2)
                    if e.code == 429:
                        with self._lock:
                            self.throttles += 1
                        # back the whole crawl off, not just this thread
                        self.limiter.penalise(wait)
                    time.sleep(wait)
                    continue
                break                                  # other 4xx: unretryable
            except Exception as e:                     # timeout, reset, bad JSON
                last = e
                time.sleep(min(60.0, (2 ** attempt) + random.random() * 2))
        if required:
            raise RateLimited(f"{url} ({last})")
        log(f"    ! giving up on {url} ({last})")
        return None


# --------------------------------------------------------------------------- #
# HTML -> structured text
# --------------------------------------------------------------------------- #
_WS = re.compile(r"[ \t ​]+")
_BLANKS = re.compile(r"\n{3,}")

BLOCK = {"p", "div", "section", "article", "blockquote", "figcaption",
         "h1", "h2", "h3", "h4", "h5", "h6"}


def _clean(s: str) -> str:
    return _WS.sub(" ", s.replace(" ", " ")).strip()


def html_to_text(html: str) -> str:
    """Flatten section HTML to plain text, preserving document structure.

    Tables are the reason this isn't just get_text(): HMRC puts rates and
    allowances in tables, and a naive flatten turns each cell into its own
    line, destroying the row association. Here each row becomes one
    pipe-joined line.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    for bad in soup.find_all(["script", "style"]):
        bad.decompose()

    out: list[str] = []

    def emit(text: str) -> None:
        text = _clean(text)
        if text:
            out.append(text)

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                # loose text directly under a container
                if not isinstance(child, Tag):
                    txt = _clean(str(child))
                    if txt:
                        out.append(txt)
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()

            if name == "table":
                rows = []
                for tr in child.find_all("tr"):
                    cells = [_clean(c.get_text(" ", strip=True))
                             for c in tr.find_all(["th", "td"])]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    out.append("")
                    out.extend(rows)
                    out.append("")
                continue

            if name in ("ul", "ol"):
                out.append("")
                for i, li in enumerate(child.find_all("li", recursive=False), 1):
                    marker = "-" if name == "ul" else f"{i}."
                    emit(f"{marker} {li.get_text(' ', strip=True)}")
                out.append("")
                continue

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                out.append("")
                emit(child.get_text(" ", strip=True))
                out.append("")
                continue

            if name == "br":
                out.append("")
                continue

            if name in BLOCK:
                # containers that hold only inline content become one line
                if child.find(list(BLOCK | {"table", "ul", "ol"})) is None:
                    emit(child.get_text(" ", strip=True))
                    out.append("")
                else:
                    walk(child)
                continue

            # inline (span, a, strong, em, ...) — merge into current line
            if child.find(["table", "ul", "ol"] + list(BLOCK)) is not None:
                walk(child)
            else:
                txt = _clean(child.get_text(" ", strip=True))
                if txt:
                    if out and out[-1] and not out[-1].endswith(("|",)):
                        out[-1] = _clean(out[-1] + " " + txt)
                    else:
                        out.append(txt)

    walk(soup)
    text = "\n".join(out)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def extract_links(html: str) -> list[dict]:
    """Cross-references out of a section — useful graph edges later."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        label = _clean(a.get_text(" ", strip=True))
        if not href or href.startswith(("#", "mailto:", "tel:")):
            continue
        key = (href, label)
        if key in seen:
            continue
        seen.add(key)
        links.append({"href": href, "text": label})
    return links


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def search_page(f: Fetcher, params: dict, required: bool = False) -> dict | None:
    return f.get_json(SEARCH + "?" + urllib.parse.urlencode(params, doseq=True),
                      required=required)


def _search_all(f: Fetcher, params: dict) -> tuple[list[dict], int]:
    """Walk every page of a search listing, de-dup on link. Returns (docs, total)."""
    docs, start, total = [], 0, 0
    while True:
        page = search_page(f, {**params, "count": PAGE, "start": start},
                           required=True)
        if not page:
            break
        results = page.get("results", [])
        docs.extend(results)
        total = page.get("total", total)
        start += len(results)
        if not results or start >= total:
            break
    out, seen = [], set()
    for d in docs:
        if d.get("link") and d["link"] not in seen:
            seen.add(d["link"])
            out.append(d)
    return out, total


def list_manuals(f: Fetcher, fmt: str = "hmrc_manual") -> list[dict]:
    """Every manual document of `fmt`, paginated defensively."""
    params = {
        "filter_format": fmt,
        "fields": ["link", "title", "description",
                   "public_timestamp", "organisations"],
        "order": "title",
    }
    if MANUAL_TYPES[fmt][1]:
        params["filter_organisations"] = HMRC_ORG
    return _search_all(f, params)[0]


def list_sections(f: Fetcher, manual_link: str,
                  section_fmt: str = "hmrc_manual_section") -> tuple[list[dict], int]:
    """Every section of one manual (flat; nesting is rebuilt from base_path).

    Returns (sections, reported_total). Raises RateLimited if any page of the
    listing can't be fetched — a silently short listing would look like a
    successful crawl of a smaller manual, which is worse than a loud failure.
    """
    return _search_all(f, {
        "filter_format": section_fmt,
        "filter_manual": manual_link,
        "fields": ["link", "title", "public_timestamp"],
        "order": "title",
    })


def list_publications(f: Fetcher, fmt: str) -> tuple[list[dict], int]:
    """Every HMRC publication of a flat format (guidance, notice, ...)."""
    return _search_all(f, {
        "filter_format": fmt,
        "filter_organisations": HMRC_ORG,
        "fields": ["link", "title", "description",
                   "public_timestamp", "organisations"],
        "order": "title",
    })


# --------------------------------------------------------------------------- #
# Section fetch
# --------------------------------------------------------------------------- #
def fetch_section(f: Fetcher, link: str) -> dict | None:
    doc = f.get_json(CONTENT + link)
    if not doc:
        return {"base_path": link, "error": "fetch_failed"}

    details = doc.get("details", {}) or {}
    body_html = details.get("body") or ""
    text = html_to_text(body_html)

    # child sections listed on this page (contents pages carry the tree)
    children = []
    for group in details.get("child_section_groups", []) or []:
        for cs in group.get("child_sections", []) or []:
            children.append({
                "section_id": cs.get("section_id"),
                "title": cs.get("title"),
                "base_path": cs.get("base_path"),
                "group": group.get("title") or None,
            })

    title = doc.get("title") or ""
    # HMRC titles look like "CG15250 - Expenditure: incidental costs"
    sec_id = details.get("section_id")
    heading = title
    if sec_id and title.upper().startswith(sec_id.upper()):
        heading = title[len(sec_id):].lstrip(" -–—:").strip() or title

    attachments = [
        {"title": a.get("title"),
         "url": a.get("url"),
         "content_type": a.get("content_type")}
        for a in (details.get("attachments") or [])
    ]

    return {
        "section_id": sec_id,
        "base_path": doc.get("base_path") or link,
        "url": BASE + (doc.get("base_path") or link),
        "title": title,
        "heading": heading,
        "description": doc.get("description") or "",
        "text": text,
        "attachments": attachments,
        "char_count": len(text),
        "word_count": len(text.split()),
        "is_contents_page": bool(children) and not text,
        "child_sections": children,
        "cross_references": extract_links(body_html),
        "first_published_at": doc.get("first_published_at"),
        "public_updated_at": doc.get("public_updated_at"),
        "updated_at": doc.get("updated_at"),
        "withdrawn": bool(doc.get("withdrawn_notice")),
        "content_id": doc.get("content_id"),
    }


def manual_slug(link: str) -> str:
    return link.rstrip("/").split("/")[-1] or "manual"


def build_manual_json(manual: dict, manual_doc: dict | None,
                      sections: list[dict], stats: dict,
                      fmt: str = "hmrc_manual") -> dict:
    link = manual["link"]
    org_names = []
    for o in (manual.get("organisations") or []):
        if isinstance(o, dict):
            org_names.append(o.get("title") or o.get("slug"))
    ok = [s for s in sections if not s.get("error")]
    return {
        "schema": "hmrc_manual_crawl",
        "schema_version": SCHEMA_VERSION,
        "manual": {
            "slug": manual_slug(link),
            "format": fmt,
            "title": manual.get("title") or (manual_doc or {}).get("title"),
            "base_path": link,
            "url": BASE + link,
            "description": manual.get("description")
                           or (manual_doc or {}).get("description") or "",
            "organisations": [o for o in org_names if o],
            "first_published_at": (manual_doc or {}).get("first_published_at"),
            "public_updated_at": (manual_doc or {}).get("public_updated_at")
                                 or manual.get("public_timestamp"),
            "withdrawn": bool((manual_doc or {}).get("withdrawn_notice")),
        },
        "crawl": {
            "source": "gov.uk Content API (/api/content) + Search API (/api/search.json)",
            "crawled_at": stats["crawled_at"],
            "section_count_reported": stats["reported"],
            "section_count_fetched": len(ok),
            "section_count_failed": len(sections) - len(ok),
            "total_words": sum(s.get("word_count", 0) for s in ok),
            "total_chars": sum(s.get("char_count", 0) for s in ok),
        },
        "sections": sections,
    }


def crawl_manual(f: Fetcher, manual: dict, outdir: str,
                 workers: int, force: bool, fmt: str = "hmrc_manual") -> dict:
    link = manual["link"]
    slug = manual_slug(link)
    path = os.path.join(outdir, f"{slug}.json")

    if os.path.exists(path) and not force:
        log(f"  = {slug}: exists, skipping (use --force to refetch)")
        return {"slug": slug, "skipped": True}

    section_fmt = MANUAL_TYPES[fmt][0]
    manual_doc = f.get_json(CONTENT + link)
    listing, reported = list_sections(f, link, section_fmt)
    if reported and len(listing) < reported:
        # incomplete enumeration: refuse to write a half manual
        raise RateLimited(
            f"{slug}: listed {len(listing)} of {reported} sections — refusing partial write")
    log(f"  · {slug}: {len(listing)} sections")

    sections: list[dict] = []
    if listing:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_section, f, s["link"]): s["link"]
                       for s in listing}
            done = 0
            for fut in as_completed(futures):
                try:
                    sections.append(fut.result())
                except Exception as e:                  # never lose a manual to one section
                    sections.append({"base_path": futures[fut],
                                     "error": f"exception: {e}"})
                done += 1
                if done % 500 == 0:
                    log(f"      {slug}: {done}/{len(listing)}")

    sections.sort(key=lambda s: (s.get("section_id") or "", s.get("base_path") or ""))

    payload = build_manual_json(manual, manual_doc, sections, {
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reported": reported or len(listing),
    }, fmt=fmt)

    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)                               # atomic: no truncated files

    c = payload["crawl"]
    log(f"  + {slug}: {c['section_count_fetched']} ok, "
        f"{c['section_count_failed']} failed, {c['total_words']:,} words -> {path}")
    return {"slug": slug, **c}


def crawl_publications(f: Fetcher, fmt: str, outdir: str,
                       workers: int, force: bool) -> dict:
    """Crawl one flat publication format (guidance, notice, ...) to a single JSON.

    These pages carry a short summary body and, usually, the real content as
    PDF attachments — both are recorded per page. Sections and manuals share
    the same fetch path (fetch_section), so no separate extractor is needed.
    """
    path = os.path.join(outdir, f"{fmt}.json")
    if os.path.exists(path) and not force:
        log(f"  = {fmt}: exists, skipping (use --force to refetch)")
        return {"format": fmt, "skipped": True}

    listing, reported = list_publications(f, fmt)
    if reported and len(listing) < reported:
        raise RateLimited(
            f"{fmt}: listed {len(listing)} of {reported} pages — refusing partial write")
    log(f"  · {fmt}: {len(listing)} pages")

    pages: list[dict] = []
    if listing:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_section, f, p["link"]): p["link"]
                       for p in listing}
            done = 0
            for fut in as_completed(futures):
                try:
                    pages.append(fut.result())
                except Exception as e:
                    pages.append({"base_path": futures[fut],
                                  "error": f"exception: {e}"})
                done += 1
                if done % 500 == 0:
                    log(f"      {fmt}: {done}/{len(listing)}")

    pages.sort(key=lambda p: p.get("base_path") or "")
    ok = [p for p in pages if not p.get("error")]
    payload = {
        "schema": "hmrc_publications_crawl",
        "schema_version": SCHEMA_VERSION,
        "format": fmt,
        "crawl": {
            "source": "gov.uk Content API (/api/content) + Search API (/api/search.json)",
            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "page_count_reported": reported or len(listing),
            "page_count_fetched": len(ok),
            "page_count_failed": len(pages) - len(ok),
            "attachment_count": sum(len(p.get("attachments") or []) for p in ok),
            "total_words": sum(p.get("word_count", 0) for p in ok),
        },
        "pages": pages,
    }
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)

    c = payload["crawl"]
    log(f"  + {fmt}: {c['page_count_fetched']} ok, {c['page_count_failed']} failed, "
        f"{c['attachment_count']} attachments, {c['total_words']:,} words -> {path}")
    return {"format": fmt, "pages_fetched": c["page_count_fetched"],
            "pages_failed": c["page_count_failed"],
            "total_words": c["total_words"], "attachment_count": c["attachment_count"]}


def main() -> int:
    all_formats = list(MANUAL_TYPES) + PUBLICATION_FORMATS
    ap = argparse.ArgumentParser(description="Crawl HMRC manuals to JSON (one file per manual).")
    ap.add_argument("-o", "--outdir", default="hmrc_manuals")
    ap.add_argument("-w", "--workers", type=int, default=8,
                    help="concurrent section fetches (default 8; be polite)")
    ap.add_argument("--rps", type=float, default=12.0,
                    help="global request-rate ceiling across all threads (default 12)")
    ap.add_argument("--limit", type=int, help="only the first N manuals per manual format")
    ap.add_argument("--manuals", nargs="*",
                    help="crawl only these slugs (e.g. capital-gains-manual)")
    ap.add_argument("--formats", nargs="*", default=all_formats,
                    metavar="FORMAT", choices=[["all"]] + all_formats,
                    help="which formats to crawl: manual formats (%s) and/or "
                         "publication formats (%s); 'all' (default) for everything"
                         % (", ".join(MANUAL_TYPES), ", ".join(PUBLICATION_FORMATS)))
    ap.add_argument("--force", action="store_true",
                    help="refetch outputs whose JSON already exists")
    ap.add_argument("--list-only", action="store_true",
                    help="print the discovered manuals and exit")
    args = ap.parse_args()
    fmts = all_formats if "all" in args.formats else args.formats
    manual_fmts = [x for x in MANUAL_TYPES if x in fmts]
    pub_fmts = [x for x in PUBLICATION_FORMATS if x in fmts]

    os.makedirs(args.outdir, exist_ok=True)
    f = Fetcher(rps=args.rps)

    t0 = time.time()
    summary: list[dict] = []
    deferred: list[tuple[str, dict]] = []        # (format, manual) for the retry pass

    for fmt in manual_fmts:
        log(f"discovering manuals (filter_format={fmt}) ...")
        manuals = list_manuals(f, fmt)
        log(f"found {len(manuals)} manuals")

        if args.manuals:
            want = {m.lower() for m in args.manuals}
            manuals = [m for m in manuals if manual_slug(m["link"]).lower() in want]
            log(f"filtered to {len(manuals)}")
        if args.limit:
            manuals = manuals[:args.limit]

        if args.list_only:
            for m in manuals:
                print(f"{manual_slug(m['link']):55s} {m.get('title','')}")
            continue

        for i, m in enumerate(manuals, 1):
            log(f"[{i}/{len(manuals)}] {m.get('title')}")
            try:
                summary.append(crawl_manual(f, m, args.outdir, args.workers,
                                            args.force, fmt=fmt))
            except KeyboardInterrupt:
                log("interrupted")
                break
            except RateLimited as e:
                log(f"  ! {manual_slug(m['link'])} throttled: {e} (will retry at end)")
                deferred.append((fmt, m))
            except Exception as e:
                log(f"  ! {manual_slug(m['link'])} failed: {e}")
                summary.append({"slug": manual_slug(m["link"]), "error": str(e)})

    if args.list_only:
        return 0

    for fmt in pub_fmts:
        try:
            summary.append(crawl_publications(f, fmt, args.outdir,
                                              args.workers, args.force))
        except KeyboardInterrupt:
            log("interrupted")
            break
        except Exception as e:
            log(f"  ! {fmt} failed: {e}")
            summary.append({"format": fmt, "error": str(e)})

    # second pass for anything throttled on the first attempt, at a gentler rate
    if deferred:
        log(f"\nretrying {len(deferred)} throttled item(s) at reduced rate ...")
        f.limiter = RateLimiter(max(2.0, args.rps / 3))
        for fmt, m in deferred:
            slug = manual_slug(m["link"])
            try:
                summary.append(crawl_manual(f, m, args.outdir, max(2, args.workers // 2),
                                            args.force, fmt=fmt))
            except Exception as e:
                log(f"  ! {slug} failed on retry: {e}")
                summary.append({"slug": slug, "error": str(e)})

    elapsed = time.time() - t0
    manuals_done = [s for s in summary if s.get("section_count_fetched") is not None]
    pages_done = [s for s in summary if s.get("pages_fetched") is not None]
    idx = {
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manuals": len(manuals_done),
        "sections_fetched": sum(s.get("section_count_fetched", 0) for s in manuals_done),
        "sections_failed": sum(s.get("section_count_failed", 0) for s in manuals_done),
        "publication_formats": len(pages_done),
        "publication_pages_fetched": sum(s.get("pages_fetched", 0) for s in pages_done),
        "publication_pages_failed": sum(s.get("pages_failed", 0) for s in pages_done),
        "attachments_indexed": sum(s.get("attachment_count", 0) for s in pages_done),
        "total_words": (sum(s.get("total_words", 0) for s in manuals_done)
                        + sum(s.get("total_words", 0) for s in pages_done)),
        "http_requests": f.calls,
        "throttle_events": f.throttles,
        "elapsed_seconds": round(elapsed, 1),
        "results": summary,
    }
    with open(os.path.join(args.outdir, "_index.json"), "w", encoding="utf-8") as fh:
        json.dump(idx, fh, ensure_ascii=False, indent=1)

    log(f"\ndone in {elapsed/60:.1f} min — "
        f"{idx['sections_fetched']:,} sections, "
        f"{idx['publication_pages_fetched']:,} publication pages, "
        f"{idx['attachments_indexed']:,} attachments, "
        f"{idx['total_words']:,} words, {f.calls:,} requests, "
        f"{f.throttles} throttle events")
    log(f"index: {os.path.join(args.outdir, '_index.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
