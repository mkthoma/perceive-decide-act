"""Action role — pure MCP dispatch, no LLM calls.

Payloads > ARTIFACT_THRESHOLD_BYTES are written to the artifact store;
the caller receives a short descriptor plus the artifact handle.
art: handle guard prevents tool calls with stale artifact ids as paths/urls.
"""
from __future__ import annotations

from mcp import ClientSession

from artifact_store import ARTIFACT_THRESHOLD_BYTES, artifacts
from schemas import ToolCall

_ARTIFACT_HANDLE_ARGS = frozenset({"path", "url", "file_path", "filepath"})


def _has_artifact_handle(tc: ToolCall) -> bool:
    for key, val in tc.arguments.items():
        if key in _ARTIFACT_HANDLE_ARGS and isinstance(val, str) and val.startswith("art:"):
            return True
    return False


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

    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)

    # Collapse MCP content blocks into a single text string
    text_parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)
        elif hasattr(block, "data"):
            text_parts.append(f"[binary block: {len(block.data)} bytes]")

    raw_text = "\n".join(text_parts)
    raw_bytes = raw_text.encode("utf-8")

    # Store as artifact if payload exceeds threshold
    if len(raw_bytes) > ARTIFACT_THRESHOLD_BYTES:
        artifact_id = artifacts.put(
            raw_bytes,
            content_type="text/plain",
            source=tool_call.name,
            descriptor=f"[{tool_call.name}] {raw_text[:80]}",
        )
        preview = raw_text[:200].replace("\n", " ")
        descriptor = (
            f"[artifact {artifact_id}, {len(raw_bytes):,} bytes] "
            f"preview: {preview}…"
        )
        return descriptor, artifact_id

    return raw_text, None
