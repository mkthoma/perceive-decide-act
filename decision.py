"""Decision role — picks the next action for one bounded goal.

Returns either a final answer (text) or a single typed ToolCall.
Routes through the gateway with auto_route='decision'; the router pool
selects TINY or LARGE tier based on prompt size.
"""
from __future__ import annotations

import llm_gateway as gw
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

_MAX_ARTIFACT_CHARS = 80_000  # ~20k tokens; truncate larger artifacts

_SYSTEM = """\
You are DECISION, the action selector in an agentic loop.

You receive one GOAL and supporting context. You must return EXACTLY ONE of:
  1. answer   — a direct response you can produce from CONTEXT or ATTACHED ARTIFACTS
  2. tool_call — when you need external data not already present in context

STRICT RULES:
- NEVER return both answer and tool_call in the same response.
- Strings starting with "art:" are internal artifact handles. Do NOT pass them
  as path or url arguments to any tool. The artifact bytes are in ATTACHED ARTIFACTS.
- If HISTORY contains a [STOP] line, the previous tool call was illegal.
  Answer directly from ATTACHED ARTIFACTS — do NOT call any tool.
- For real-time data (current time, live exchange rates, today's weather),
  ALWAYS call the appropriate tool — never answer from memory or assumptions.
- For extraction, list, comparison, recommendation, or synthesis goals: your answer
  must be substantive — at least 3 sentences or a numbered/bulleted list of ≥ 3 items.
- If HISTORY already contains a tool result for this goal, answer from that result
  directly — do not call the same tool again.
- Pick the most specific tool for the task. Prefer fetch_url over web_search when
  you already have a URL."""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "  (none)"
    lines = []
    for item in hits:
        art = f"  [artifact: {item.artifact_id}]" if item.artifact_id else ""
        lines.append(f"  [{item.kind}] {item.descriptor}{art}")
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    entries: list[str] = []
    for h in history[-10:]:
        kind = h.get("kind")
        if kind == "action":
            entries.append(
                f"  iter {h['iter']}: TOOL {h['tool']} → "
                f"{h.get('result_descriptor', '')[:300]}"
            )
        elif kind == "answer":
            entries.append(
                f"  iter {h['iter']}: ANSWER: {h.get('text', '')[:300]}"
            )
    return "\n".join(entries) if entries else "  (empty)"


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return ""
    parts: list[str] = ["\n\nATTACHED ARTIFACTS:"]
    for art_id, raw_bytes in attached:
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = "[binary content]"
        if len(text) > _MAX_ARTIFACT_CHARS:
            text = text[:_MAX_ARTIFACT_CHARS] + f"\n...[truncated — {len(text)} total chars]"
        parts.append(f"\n=== {art_id} ===\n{text}")
    return "".join(parts)


def _build_messages(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
) -> list[dict]:
    user_content = (
        f"GOAL: {goal.text}\n\n"
        f"MEMORY HITS:\n{_format_hits(hits)}\n\n"
        f"HISTORY:\n{_format_history(history)}"
        f"{_format_attached(attached)}"
    )
    return [{"role": "user", "content": user_content}]


async def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    """One LLM call — returns answer text or a single ToolCall."""
    messages = _build_messages(goal, hits, attached, history)

    resp = await gw.chat(
        messages,
        system=_SYSTEM,
        auto_route="decision",
        tools=mcp_tools if mcp_tools else None,
        tool_choice="auto" if mcp_tools else None,
        temperature=0.7,
    )

    # Prefer explicit tool calls; arguments is already a dict in gateway response
    tool_calls = gw.extract_tool_calls(resp)
    if tool_calls:
        tc = tool_calls[0]
        return DecisionOutput(
            tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
        )

    text = gw.extract_text(resp).strip()
    # Some LLMs prefix their response with the option label ("answer\n...").
    # Strip it so it doesn't pollute the final answer shown to the user.
    lower = text.lower()
    if lower.startswith("answer"):
        candidate = text[6:].lstrip(": \n")
        if candidate:
            text = candidate
    return DecisionOutput(answer=text or "Task completed.")
