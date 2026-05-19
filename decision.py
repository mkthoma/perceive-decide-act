"""Decision role — picks the next action for one bounded goal.

Returns either a final answer (text) or a single typed ToolCall.
Routes through the gateway with auto_route='decision'; the router pool
selects TINY or LARGE tier based on prompt size.
"""
from __future__ import annotations

import re

import llm_gateway as gw
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

_MAX_ARTIFACT_CHARS = 80_000  # ~20k tokens; truncate larger artifacts

# Matches vLLM / Groq-style text function call markup that some models emit
# instead of native tool_calls:  <function(name){...}</function>
_FC_RE = re.compile(r"<function\((\w+)\)\s*(\{.*?\})\s*</function>", re.DOTALL)

_SYSTEM = """\
You are DECISION, the action selector in an agentic loop.

You receive one GOAL and supporting context. You must return EXACTLY ONE of:
  1. answer   — a direct response you can produce from CONTEXT or ATTACHED ARTIFACTS
  2. tool_call — when you need external data not already present in context

STRICT RULES:
- NEVER return both answer and tool_call in the same response.
- Before choosing a tool_call, verify the action maps to an available tool.
  If a goal asks you to DO something for which no tool exists (set a calendar
  reminder, send an email, post to social media, create a document, book a
  flight, etc.), answer directly with a clear text description of what should
  be done — do NOT attempt to call a non-existent tool or loop trying.
- MEMORY HITS are part of your context. If a hit's descriptor already contains
  the answer to the current GOAL (e.g. "[fact] Mom's birthday is May 15, 2026"),
  answer directly from it — do NOT say the information is unavailable.
- Strings starting with "art:" are internal artifact handles. Do NOT pass them
  as path or url arguments to any tool. The artifact bytes are in ATTACHED ARTIFACTS.
- If HISTORY contains a [STOP] line, the previous tool call was illegal.
  Answer directly from ATTACHED ARTIFACTS — do NOT call any tool.
- For real-time data (current time, live exchange rates, today's weather),
  ALWAYS call the appropriate tool — never answer from memory or assumptions.
- For WEATHER data, use web_search — e.g. web_search("Tokyo weather Saturday forecast").
  Do NOT use fetch_url for weather; headless rendering of weather sites is too slow.
- get_time uses a 'timezone' parameter (IANA name), e.g.:
    get_time(timezone="Asia/Tokyo")
- For extraction, list, comparison, recommendation, or synthesis goals: your answer
  must be substantive — at least 3 sentences or a numbered/bulleted list of ≥ 3 items.
- If HISTORY already contains a tool result for this goal, answer from that result
  directly — do not call the same tool again.
- If ATTACHED ARTIFACTS do not contain the data needed for this goal, do NOT answer
  saying the data is missing. Call the appropriate tool to fetch it instead.
- If HISTORY shows 3 or more consecutive web_search results with "No results found"
  for the same goal, STOP searching. Answer from your own knowledge or note the
  information is unavailable — never search the same topic a fourth time.
- If HISTORY contains "[SEARCH_EXHAUSTED:" for this goal, answer from your own
  knowledge — do NOT call web_search or fetch_url again.
- Prefer web_search over fetch_url by default. Only call fetch_url when you need
  the FULL rendered content of a specific page AND web_search snippets are not enough.
- If HISTORY contains ANY [tool_timeout] result, do NOT call fetch_url again for
  this goal. Switch to web_search immediately and answer from those results.
- When the user asks to "remember" something, use create_file to save the fact:
    create_file(path="memory/{key}.txt", content="...the fact...")
  Parent directories are created automatically — do NOT call create_file just to
  make a directory; go straight to creating the file.
- To RECALL a previously remembered fact, call read_file on the relevant
  memory/ path, e.g. read_file(path="memory/moms_birthday.txt"). If MEMORY HITS
  include a [tool_outcome] showing a memory/ file was written for a related topic,
  read that file — do NOT answer "I don't know" without checking first."""


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
                f"{h.get('result_descriptor', '')[:800]}"
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
        temperature=1.0,
    )

    # Prefer explicit tool calls; arguments is already a dict in gateway response
    tool_calls = gw.extract_tool_calls(resp)
    if tool_calls:
        tc = tool_calls[0]
        return DecisionOutput(
            tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
        )

    text = gw.extract_text(resp).strip()

    # Some models (vLLM, Groq-style) emit tool calls as text markup instead of
    # native tool_calls.  Detect and parse <function(name){...}</function>.
    fc_match = _FC_RE.search(text)
    if fc_match:
        try:
            import json as _json
            fn_name = fc_match.group(1)
            raw_args = fc_match.group(2)
            args = _json.loads(raw_args)
            # Coerce string integers to int for known numeric parameters
            for key in ("max_results",):
                if key in args and isinstance(args[key], str):
                    try:
                        args[key] = int(args[key])
                    except ValueError:
                        pass
            return DecisionOutput(
                tool_call=ToolCall(name=fn_name, arguments=args)
            )
        except Exception:
            pass  # fall through to text answer

    # Some LLMs prefix their response with the option label ("answer\n...").
    # Strip it so it doesn't pollute the final answer shown to the user.
    lower = text.lower()
    if lower.startswith("answer"):
        candidate = text[6:].lstrip(": \n")
        if candidate:
            text = candidate
    return DecisionOutput(answer=text or "Task completed.")
