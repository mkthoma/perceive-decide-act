"""MCP server exposing tools over stdio transport.

Search: Tavily (TAVILY_API_KEY in .env) with DDGS fallback.
Crawl:  crawl4ai (async, JS-capable) with httpx fallback.
File tools operate within sandbox/.
Weather: Open-Meteo (free, no API key required).
Memory: save_memory() writes durable facts to state/memory.json.

After first install, run the crawl4ai setup to install Playwright browsers:
  uv run crawl4ai-setup

Start with: uv run python mcp_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("agent-tools")

_SANDBOX = Path("sandbox")
_SANDBOX.mkdir(parents=True, exist_ok=True)

_STATE_DIR = Path("state")
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_MEMORY_FILE = _STATE_DIR / "memory.json"

_TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# WMO weather interpretation codes → human-readable descriptions
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Heavy thunderstorm",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _safe_path(rel: str) -> Path:
    """Resolve path inside sandbox, reject traversal attempts."""
    target = (_SANDBOX / rel).resolve()
    if not str(target).startswith(str(_SANDBOX.resolve())):
        raise ValueError(f"Path traversal blocked: {rel!r}")
    return target


class _TextExtractor(HTMLParser):
    """Strips HTML tags, keeping visible text from non-boilerplate elements."""

    _SKIP = {"script", "style", "nav", "footer", "head", "noscript", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _format_search_results(results: list[dict], url_key: str, snippet_key: str) -> str:
    lines: list[str] = []
    for r in results:
        lines.append(f"Title: {r.get('title', '(no title)')}")
        lines.append(f"URL:   {r.get(url_key, '')}")
        lines.append(f"Snippet: {r.get(snippet_key, '')}")
        lines.append("")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Tools                                                                         #
# --------------------------------------------------------------------------- #

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs, and snippets.

    Uses Tavily when TAVILY_API_KEY is set (higher quality, AI-extracted
    snippets). Falls back to DuckDuckGo (DDGS) when the key is absent.
    Use this to discover relevant pages before fetching their content.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1–10).
    """
    max_results = max(1, min(10, max_results))

    if _TAVILY_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=_TAVILY_KEY)
            resp = client.search(query, max_results=max_results)
            results = resp.get("results", [])
            if results:
                return _format_search_results(results, url_key="url", snippet_key="content")
        except Exception:
            pass  # fall through to DDGS

    try:
        # Package was renamed from duckduckgo_search to ddgs
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found for that query."
        return _format_search_results(results, url_key="href", snippet_key="body")
    except Exception as exc:
        return f"[web_search error] {exc}"


@mcp.tool()
async def fetch_url(url: str) -> str:
    """Fetch and return the text content of a URL.

    Uses crawl4ai for JavaScript-rendered pages (returns clean Markdown).
    Falls back to a plain httpx request with HTML stripping when crawl4ai
    is unavailable or the page fails to crawl.

    Args:
        url: The full URL to fetch (must start with http:// or https://).
    """
    if not url.startswith(("http://", "https://")):
        return "[fetch_url error] URL must start with http:// or https://"

    # httpx first — fast, works for static pages (Wikipedia, docs, etc.)
    httpx_error: str | None = None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; Agent6/1.0; "
                "+https://github.com/perceive-decide-act)"
            )
        }
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=20.0,
        ) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "text/html")
        if "text/html" in content_type:
            extractor = _TextExtractor()
            extractor.feed(resp.text)
            text = extractor.get_text()
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            text = resp.text
        if text.strip():
            return text[:50_000]  # cap at 50K chars to stay within context limits
        httpx_error = "empty response"
    except Exception as exc:
        httpx_error = str(exc)

    # crawl4ai fallback — JS-rendered pages (30s total timeout)
    try:
        async def _crawl(u: str) -> str | None:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=u)
            if result.success and result.markdown:
                t = result.markdown.strip()
                return re.sub(r"\n{3,}", "\n\n", t)
            return None

        text = await asyncio.wait_for(_crawl(url), timeout=30.0)
        if text:
            return text[:50_000]
    except Exception:
        pass

    return f"[fetch_url error] {httpx_error}"


@mcp.tool()
def get_time(tz: str = "UTC") -> str:
    """Return the current date and time in the requested timezone.

    Uses IANA timezone names. Common examples:
      'UTC'            → 2026-05-19T07:02:33+00:00
      'Asia/Tokyo'     → 2026-05-19T16:02:33+09:00
      'America/New_York' → 2026-05-19T03:02:33-04:00
      'Europe/London'  → 2026-05-19T08:02:33+01:00

    Args:
        tz: IANA timezone name (default 'UTC').
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, KeyError):
        try:
            zone = ZoneInfo("UTC")
        except Exception:
            # Last resort: use stdlib UTC (no tzdata needed)
            zone = timezone.utc
    return datetime.now(zone).isoformat()


@mcp.tool()
async def get_weather(city: str, days: int = 3) -> str:
    """Get current and forecast weather for any city (no API key required).

    Uses Open-Meteo geocoding + weather API — free, reliable, no sign-up.
    Always use this tool for weather questions; do NOT use web_search for weather.

    Args:
        city: City name, e.g. 'Tokyo', 'London', 'New York'.
        days: Number of forecast days to return (1–7, default 3).
    """
    days = max(1, min(7, days))
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # Step 1: geocode city → lat/lon
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()
            results = geo_data.get("results")
            if not results:
                return f"[get_weather] City not found: {city!r}"
            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]
            city_label = f"{loc.get('name', city)}, {loc.get('country', '')}"

            # Step 2: fetch forecast
            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": (
                        "temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,weathercode,windspeed_10m_max"
                    ),
                    "current_weather": "true",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            wx.raise_for_status()
            data = wx.json()

        daily = data.get("daily", {})
        dates  = daily.get("time", [])
        t_max  = daily.get("temperature_2m_max", [])
        t_min  = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes  = daily.get("weathercode", [])
        wind   = daily.get("windspeed_10m_max", [])

        cur    = data.get("current_weather", {})
        cur_t  = cur.get("temperature", "?")
        cur_d  = _WMO_CODES.get(int(cur.get("weathercode", 0)), "Unknown")

        lines = [
            f"Weather for {city_label}",
            f"Current: {cur_t}°C, {cur_d}",
            "",
            "Forecast:",
        ]
        for i, date in enumerate(dates):
            code = int(codes[i]) if i < len(codes) else 0
            desc = _WMO_CODES.get(code, f"Code {code}")
            lo   = t_min[i]  if i < len(t_min)  else "?"
            hi   = t_max[i]  if i < len(t_max)  else "?"
            p    = f"{precip[i]:.1f}mm" if i < len(precip) else "?"
            w    = f"{wind[i]:.0f}km/h" if i < len(wind)   else "?"
            lines.append(f"  {date}: {lo}–{hi}°C, {desc}, Rain: {p}, Wind: {w}")

        return "\n".join(lines)

    except Exception as exc:
        return f"[get_weather error] {exc}"


@mcp.tool()
def save_memory(text: str) -> str:
    """Save a fact or note to persistent memory for retrieval in future sessions.

    Use this whenever the user asks you to 'remember' something.
    The stored text will be searchable by keyword in all future runs.

    Args:
        text: The fact or note to save, e.g. "Mom's birthday is May 15, 2026".
    """
    _STOPWORDS = frozenset(
        "a an the and or but in on at to for of with is was are were be been "
        "it its this that these those i my me we our you your he she they their "
        "what when where how why which who do did does have has had will would "
        "could should may might can not no yes get give tell find show make go "
        "take use want need know about just from by up out if so".split()
    )

    try:
        # Load existing items
        items: list[dict] = []
        if _MEMORY_FILE.exists():
            try:
                items = json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                items = []

        # Extract keywords
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 1][:15]

        new_item = {
            "id": uuid.uuid4().hex[:12],
            "kind": "fact",
            "keywords": keywords,
            "descriptor": text[:120],
            "value": {"text": text},
            "artifact_id": None,
            "source": "save_memory_tool",
            "run_id": "manual",
            "goal_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        items.append(new_item)
        _MEMORY_FILE.write_text(
            json.dumps(items, indent=2, default=str),
            encoding="utf-8",
        )
        return f"ok — remembered: {text[:80]}"
    except Exception as exc:
        return f"[save_memory error] {exc}"


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live exchange rates.

    Uses the free Frankfurter API (no API key required).

    Args:
        amount: The numeric amount to convert.
        from_currency: ISO 4217 source currency code (e.g. 'USD').
        to_currency: ISO 4217 target currency code (e.g. 'EUR').
    """
    fc = from_currency.strip().upper()
    tc = to_currency.strip().upper()
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(
                "https://api.frankfurter.app/latest",
                params={"from": fc, "to": tc, "amount": amount},
            )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("rates", {}).get(tc)
        if result is None:
            return f"[currency_convert] Could not convert {fc} → {tc}"
        date = data.get("date", "unknown date")
        return f"{amount} {fc} = {result:.4f} {tc}  (rate as of {date})"
    except Exception as exc:
        return f"[currency_convert error] {exc}"


@mcp.tool()
def read_file(path: str) -> str:
    """Read and return the text content of a file inside the sandbox.

    Args:
        path: Relative path within the sandbox directory.
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"[read_file] File not found: {path}"
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[read_file error] {exc}"


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """List files and directories at a sandbox path.

    Args:
        path: Relative path within the sandbox (default: root of sandbox).
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"[list_dir] Path not found: {path}"
        if not target.is_dir():
            return f"[list_dir] Not a directory: {path}"
        entries = sorted(target.iterdir())
        if not entries:
            return "(empty directory)"
        lines = []
        for entry in entries:
            tag = "DIR " if entry.is_dir() else "FILE"
            size = f"{entry.stat().st_size:,} bytes" if entry.is_file() else ""
            lines.append(f"{tag}  {entry.name}  {size}")
        return "\n".join(lines)
    except Exception as exc:
        return f"[list_dir error] {exc}"


@mcp.tool()
def create_file(path: str, content: str) -> str:
    """Create a new file inside the sandbox with the given content.

    Parent directories are created automatically. Fails if the file exists.

    Args:
        path: Relative path within the sandbox.
        content: Text content to write.
    """
    try:
        target = _safe_path(path)
        if target.exists():
            return f"[create_file] File already exists: {path}. Use update_file to overwrite."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"ok — created {path} ({len(content):,} chars)"
    except Exception as exc:
        return f"[create_file error] {exc}"


@mcp.tool()
def update_file(path: str, content: str) -> str:
    """Overwrite an existing file inside the sandbox with new content.

    Args:
        path: Relative path within the sandbox.
        content: New text content.
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"ok — updated {path} ({len(content):,} chars)"
    except Exception as exc:
        return f"[update_file error] {exc}"


@mcp.tool()
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace a specific text segment in a sandbox file.

    Performs an exact string replacement. Fails if old_text is not found.

    Args:
        path: Relative path within the sandbox.
        old_text: The exact text to find and replace.
        new_text: The replacement text.
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"[edit_file] File not found: {path}"
        original = target.read_text(encoding="utf-8")
        if old_text not in original:
            return f"[edit_file] old_text not found in {path}"
        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return f"ok — edited {path}"
    except Exception as exc:
        return f"[edit_file error] {exc}"


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run()
