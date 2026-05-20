"""
MCP server for EAGV3 Session 6.

Nine tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file

web_search:  Tavily → Exa → Firecrawl → DuckDuckGo fallback chain.
             Hard-capped at 5 results.
fetch_url:   httpx fast-path (plain HTTP, ~3 s) → crawl4ai fallback for JS-heavy pages.
Usage for all search providers is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import html.parser as _html_parser
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

load_dotenv(Path(__file__).parent / ".env")

mcp = FastMCP("eagv3-s6-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "exa": {"count": 0, "errors": 0},
        "firecrawl": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "exa", "firecrawl", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _exa_search(query: str, max_results: int) -> list[dict]:
    from exa_py import Exa

    client = Exa(api_key=os.environ["EXA_API_KEY"])
    # search() returns text contents by default in exa-py 2.x
    resp = client.search(
        query,
        num_results=max_results,
        contents={"text": {"max_characters": 500}},
    )
    return [
        {
            "title": r.title or "",
            "url": r.url,
            "snippet": (getattr(r, "text", None) or "")[:500],
        }
        for r in resp.results
    ]


def _firecrawl_search(query: str, max_results: int) -> list[dict]:
    from firecrawl import V1FirecrawlApp

    app = V1FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    resp = app.search(query, limit=max_results)
    rows: list[dict] = resp.data if resp.success else []
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("description", "") or r.get("markdown", ""))[:500],
        }
        for r in rows[:max_results]
    ]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    hits: list[dict] = []
    with DDGS() as ddgs:
        for backend in ("auto", "html", "lite"):
            try:
                hits = list(ddgs.text(query, max_results=max_results, backend=backend))
            except Exception:
                hits = []
            if hits:
                break
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]


_HTTPX_TIMEOUT = 12       # seconds for plain-HTTP fast-path
_MAX_FETCH_CHARS = 20_000  # keep intro + key sections; enough for any fact extraction


def _html_to_text(html_str: str) -> str:
    """Strip HTML tags to plain text using stdlib html.parser (no extra deps)."""

    class _Extractor(_html_parser.HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip_depth = 0
            self._SKIP = {"script", "style", "nav", "footer", "head", "noscript"}

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in self._SKIP:
                self._skip_depth += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in self._SKIP and self._skip_depth:
                self._skip_depth -= 1

        def handle_data(self, data: str) -> None:
            if not self._skip_depth:
                stripped = data.strip()
                if stripped:
                    self.parts.append(stripped)

    p = _Extractor()
    p.feed(html_str)
    return "\n".join(p.parts)


async def _httpx_fetch(url: str) -> dict | None:
    """Fast-path fetch via plain HTTP (no headless browser).

    Returns a fetch_url-compatible dict on success, or None if the response
    is unsuitable (binary, empty, <200 chars) so the caller falls back to
    crawl4ai.
    """
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_HTTPX_TIMEOUT) as client:
            r = await client.get(url, headers=_HEADERS)
        if r.status_code >= 400:
            return {
                "status": r.status_code,
                "content_type": "",
                "length_bytes": 0,
                "text": f"[HTTP {r.status_code}]",
            }
        ctype = r.headers.get("content-type", "")
        if "html" in ctype or not ctype:
            text = _html_to_text(r.text)
        elif "json" in ctype or "text" in ctype:
            text = r.text
        else:
            return None  # binary — let crawl4ai handle it
        text = text[:_MAX_FETCH_CHARS]
        if len(text.strip()) < 200:
            return None  # too little content — crawl4ai may do better
        return {
            "status": r.status_code,
            "content_type": ctype,
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
        }
    except Exception:
        return None


async def _crawl4ai_fetch(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
    # serializes as {} because its real field is private. Pull the raw string
    # out and force a plain str so FastMCP serializes correctly.
    md = r.markdown
    raw = (
        getattr(md, "raw_markdown", None)
        or getattr(md, "fit_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return snippets from multiple sources.

    USE FOR: current events, weather forecasts, news, sports scores, live prices,
    general knowledge, finding URLs, anything where multiple short snippets are
    sufficient.  Best default choice when you need external information.

    Provider chain (automatic fallback): Tavily → Exa → Firecrawl → DuckDuckGo.
    Hard-capped at 5 results.

    PREFER OVER fetch_url when: snippets are enough, you don't have a specific
    URL yet, or you need speed (fetch_url is 3–30 s; web_search is ~1–2 s).

    Example: web_search("Tokyo weather this Saturday", 3)
    """
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))

    # 1. Tavily — best quality, paid (soft cap 950/mo)
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")

    # 2. Exa — neural search, paid (free tier 1 000/mo)
    if os.environ.get("EXA_API_KEY"):
        try:
            results = _exa_search(query, max_results)
            if results:
                _bump("exa")
                return results
        except Exception:
            _bump("exa", "errors")

    # 3. Firecrawl — scraping-based search, paid (free tier 500 credits/mo)
    if os.environ.get("FIRECRAWL_API_KEY"):
        try:
            results = _firecrawl_search(query, max_results)
            if results:
                _bump("firecrawl")
                return results
        except Exception:
            _bump("firecrawl", "errors")

    # 4. DuckDuckGo — free, no key, rate-limited; last resort
    results = _ddg_search(query, max_results)
    _bump("duckduckgo")
    return results


@mcp.tool()
async def fetch_url(url: str) -> dict:
    """Fetch the full text content of a specific URL.

    USE FOR: reading the complete content of a known URL — Wikipedia articles,
    documentation pages, blog posts, API responses.  Returns full page text,
    not just a snippet.

    PREFER OVER web_search when: you already have the exact URL and need the
    full body (e.g. after a web_search returned the URL).

    NOTE: Takes 3–30 seconds.  Avoid for weather sites, social media, or any
    page that requires JavaScript to render content — use web_search instead.

    Example: fetch_url("https://en.wikipedia.org/wiki/Claude_Shannon")
    """
    # Phase 1 — fast path: plain HTTP via httpx (works for Wikipedia, news sites, docs, APIs)
    result = await _httpx_fetch(url)
    if result and result.get("text", "").strip():
        return result
    # Phase 2 — slow path: headless Chromium via crawl4ai (for JS-rendered SPAs)
    return await _crawl4ai_fetch(url)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Get the current date and time in any IANA timezone.

    USE FOR: any query about the current time, date, or day of the week.
    Always call this for real-time temporal questions — never guess the time.

    The 'timezone' parameter must be a valid IANA name:
      "UTC", "America/New_York", "America/Los_Angeles", "Europe/London",
      "Europe/Paris", "Asia/Tokyo", "Asia/Kolkata", "Australia/Sydney"

    Example: get_time("Asia/Tokyo")
    """
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using live exchange rates.

    USE FOR: currency conversion, exchange rate queries, "how much is X in Y
    currency".  Rates are live from frankfurter.dev (ECB reference data).
    Always call this for currency questions — never use stale training-data rates.

    Parameters use ISO-4217 currency codes: USD, EUR, GBP, JPY, INR, AUD, CAD…

    Example: currency_convert(100, "USD", "EUR")
    """
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox.

    USE FOR: recalling previously saved facts or data.  User-requested
    persistent facts are saved under the memory/ directory — check there first
    before saying information is unavailable.

    Common pattern: if a prior run saved a fact, read it back with
      read_file("memory/<descriptive_key>.txt")

    Use list_dir("memory") first if you're unsure what memory files exist.

    Example: read_file("memory/moms_birthday.txt")
    """
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List files and directories inside the sandbox.

    USE FOR: discovering what files exist before reading them.  In particular,
    call list_dir("memory") to see what facts have been previously saved by
    the user — then use read_file to retrieve the relevant one.

    Example: list_dir("memory")
    """
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox with given content.

    USE FOR: durably saving any fact or data the user wants to remember.
    Save to memory/<descriptive_key>.txt so it can be recalled later:
      create_file("memory/moms_birthday.txt", "May 15, 2026")

    Auto-creates parent directories — no need to create memory/ first.

    IMPORTANT: raises an error if the file already exists.
    Use update_file instead when the file may already exist.

    Example: create_file("memory/project_deadline.txt", "2026-06-01")
    """
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing file in the sandbox with new content.

    USE FOR: updating a previously saved fact when the file already exists.
    If you're not sure whether the file exists, use this instead of
    create_file — it raises an error only if the file is missing.

    Example: update_file("memory/moms_birthday.txt", "May 16, 2026 (corrected)")
    """
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find and replace text within an existing sandbox file.

    USE FOR: making targeted edits without rewriting the entire file content.
    Preferred over update_file when only part of the content needs to change.
    Set replace_all=True to replace every occurrence of the search string.

    Example: edit_file("memory/notes.txt", "old value", "new value")
    """
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
