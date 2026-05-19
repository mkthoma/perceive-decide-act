"""Action role — pure MCP dispatch, no LLM calls.

Payloads > ARTIFACT_THRESHOLD_BYTES are written to the artifact store;
the caller receives a short descriptor plus the artifact handle.
art: handle guard prevents tool calls with stale artifact ids as paths/urls.

Tool responses are postprocessed from JSON into clean human-readable text
so Decision sees plain strings, not raw JSON wrappers.
"""
from __future__ import annotations

import asyncio
import json

# Hard ceiling on any single MCP tool call.  fetch_url now tries plain HTTP
# first (~3 s); crawl4ai only fires as a fallback, so 30 s is enough for
# either path and frees budget for subsequent Perception/Decision LLM calls.
_TOOL_TIMEOUT_SECS = 30

from mcp import ClientSession

from artifact_store import ARTIFACT_THRESHOLD_BYTES, artifacts
from schemas import ToolCall

_ARTIFACT_HANDLE_ARGS = frozenset({"path", "url", "file_path", "filepath"})


def _has_artifact_handle(tc: ToolCall) -> bool:
    for key, val in tc.arguments.items():
        if key in _ARTIFACT_HANDLE_ARGS and isinstance(val, str) and val.startswith("art:"):
            return True
    return False


def _postprocess(tool_name: str, raw: str) -> str:
    """Convert JSON tool responses into clean human-readable text.

    The MCP server returns typed Python dicts/lists which FastMCP serialises
    as JSON strings.  This layer unwraps them so Decision sees readable text
    and artifacts contain the actual content (not JSON wrappers).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw  # already plain text

    # ── web_search → list[{title, url, snippet}] ────────────────────────── #
    if tool_name == "web_search":
        if isinstance(data, list):
            if not data:
                return "No results found for that query."
            lines: list[str] = []
            for r in data:
                lines.append(f"Title: {r.get('title', '')}")
                lines.append(f"URL:   {r.get('url', '')}")
                lines.append(f"Snippet: {r.get('snippet', '')}")
                lines.append("")
            return "\n".join(lines).strip()
        return raw

    # ── fetch_url → {status, content_type, length_bytes, text} ─────────── #
    if tool_name == "fetch_url":
        if isinstance(data, dict):
            status = int(data.get("status") or 200)
            if status >= 400:
                return f"[fetch_url error] HTTP {status}"
            text = data.get("text", "")
            return text if text.strip() else "[fetch_url] Empty response"
        return raw

    # ── get_time → {iso, human, timezone, offset_hours} ─────────────────── #
    if tool_name == "get_time":
        if isinstance(data, dict):
            return data.get("human") or data.get("iso") or raw
        return raw

    # ── currency_convert → {amount, from, to, rate, converted, date} ────── #
    if tool_name == "currency_convert":
        if isinstance(data, dict) and "converted" in data:
            return (
                f"{data['amount']} {data['from']} = {data['converted']} {data['to']} "
                f"(rate as of {data.get('date', 'unknown')})"
            )
        return raw

    # ── read_file → {path, size_bytes, content, encoding} ───────────────── #
    if tool_name == "read_file":
        if isinstance(data, dict):
            content = data.get("content", "")
            if not content:
                return f"[read_file] Empty file: {data.get('path', '')}"
            return content
        return raw

    # ── list_dir → list[{name, type, size_bytes}] ────────────────────────── #
    if tool_name == "list_dir":
        if isinstance(data, list):
            if not data:
                return "(empty directory)"
            lines = []
            for entry in data:
                tag = "DIR " if entry.get("type") == "dir" else "FILE"
                size = (
                    f"  ({entry.get('size_bytes', 0):,} bytes)"
                    if entry.get("type") == "file"
                    else ""
                )
                lines.append(f"{tag}  {entry.get('name', '')}{size}")
            return "\n".join(lines)
        return raw

    # ── create_file / update_file / edit_file → {ok, path, ...} ─────────── #
    if tool_name in ("create_file", "update_file", "edit_file"):
        if isinstance(data, dict) and data.get("ok"):
            path = data.get("path", "")
            size = data.get("size_bytes", 0)
            if tool_name == "edit_file":
                reps = data.get("replacements", 1)
                return f"ok — edited {path} ({reps} replacement{'s' if reps != 1 else ''})"
            verb = tool_name.split("_")[0]  # create / update
            return f"ok — {verb}d {path} ({size:,} bytes)"
        return raw

    return raw


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
) -> tuple[str, str | None]:
    """Dispatch one MCP tool call; return (descriptor, artifact_id | None)."""
    # Guard: reject artifact handles masquerading as real paths/URLs
    if _has_artifact_handle(tool_call):
        return (
            f"[STOP] Do NOT call '{tool_call.name}' with an art: handle. "
            "The artifact bytes are already in ATTACHED ARTIFACTS above. "
            "Read them and produce your answer now — call NO tool.",
            None,
        )

    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_call.name, arguments=tool_call.arguments),
            timeout=_TOOL_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        return (
            f"[tool_timeout] {tool_call.name} did not respond within "
            f"{_TOOL_TIMEOUT_SECS}s — try a different URL or approach.",
            None,
        )

    # Collapse MCP content blocks into a single text string
    text_parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)
        elif hasattr(block, "data"):
            text_parts.append(f"[binary block: {len(block.data)} bytes]")

    raw_text = "\n".join(text_parts)

    # Unwrap JSON tool responses into plain readable text
    processed_text = _postprocess(tool_call.name, raw_text)
    raw_bytes = processed_text.encode("utf-8")

    # Store as artifact if payload exceeds threshold
    if len(raw_bytes) > ARTIFACT_THRESHOLD_BYTES:
        artifact_id = artifacts.put(
            raw_bytes,
            content_type="text/plain",
            source=tool_call.name,
            descriptor=f"[{tool_call.name}] {processed_text[:80]}",
        )
        preview = processed_text[:200].replace("\n", " ")
        descriptor = (
            f"[artifact {artifact_id}, {len(raw_bytes):,} bytes] "
            f"preview: {preview}…"
        )
        return descriptor, artifact_id

    return processed_text, None
