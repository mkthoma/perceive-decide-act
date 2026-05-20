"""Decision role — picks the next action for one bounded goal.

Returns either a final answer (text) or a single typed ToolCall.
Routes through the gateway with auto_route='decision'; the router pool
selects TINY or LARGE tier based on prompt size.
"""
from __future__ import annotations

import re

import llm_gateway as gw
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

_MAX_ARTIFACT_CHARS = 50_000  # safety net only — source is already bounded by mcp_server._MAX_FETCH_CHARS

# Matches vLLM / Groq-style text function call markup that some models emit
# instead of native tool_calls:  <function(name){...}</function>
_FC_RE = re.compile(r"<function\((\w+)\)\s*(\{.*?\})\s*</function>", re.DOTALL)

# ------------------------------------------------------------------------------- #
# System prompt — split into preamble + dynamic tool guide + rules               #
# The TOOL SELECTION section is built at call time from the live MCP schema so  #
# adding or renaming a tool in mcp_server.py is automatically reflected here.   #
# ------------------------------------------------------------------------------- #

_SYSTEM_PREAMBLE = """\
You are DECISION, the action selector in an agentic loop.

You receive one GOAL and supporting context. You must return EXACTLY ONE of:
  1. answer   — a direct response you can produce from CONTEXT or ATTACHED ARTIFACTS
  2. tool_call — when you need external data or actions not already present in context

"""

_SYSTEM_RULES = """\
BEHAVIORAL NOTES (apply to whichever tools are available):
- For real-time data (current time, live exchange rates, live weather), call the
  appropriate tool — never answer from memory or training-data assumptions.
- For web search goals: prefer the search tool when snippets contain enough detail;
  use URL-fetch only when you need the full page body after a search gave you a URL.
- For durable memory: use the file-listing tool to discover saved facts, the
  file-read tool to load them, the file-create tool to save new ones (raises if
  exists), and the file-update tool to overwrite an existing file.
  The memory/ directory is always pre-created — write there safely.
  Example paths: "memory/<key>.txt"
- FOR "read N results" GOALS: count URL-fetch calls in HISTORY for this goal.
  Call the URL-fetch tool for the NEXT URL from search results until N calls are
  made or all remaining URLs have timed out. If [tool_timeout] occurs, skip to the
  NEXT URL — do NOT retry the same one. Answer only after all N URLs are attempted.

STRICT RULES:
- NEVER return both answer and tool_call in the same response.
- MEMORY HITS are part of your context. If a hit's descriptor already contains
  the answer to the current GOAL, answer directly from it — do NOT say the
  information is unavailable.
- Strings starting with "art:" are internal artifact handles. Do NOT pass them
  as path or url arguments to any tool. The artifact bytes are in ATTACHED ARTIFACTS.
- If HISTORY contains a [STOP] line, the previous tool call was illegal.
  Answer directly from ATTACHED ARTIFACTS — do NOT call any tool.
- For extraction, list, comparison, recommendation, or synthesis goals: your answer
  must be substantive — at least 3 sentences or a numbered/bulleted list of >= 3 items.
- For recommendation / "which is best" synthesis goals (e.g. "determine which
  activity is most appropriate", "recommend the best option based on X"): your
  answer MUST be fully self-contained and follow this exact structure:
    1. PRESENT ALL OPTIONS: number every option from prior HISTORY answers
       (e.g. if goal-1 found 3 activities, list all 3 numbered).  Do NOT
       drop or skip any option — list them all before making a judgment.
    2. CONTEXT: state the relevant constraint (weather, budget, etc.).
    3. RECOMMENDATION: pick ONE as the single best choice with reasoning.
  The reader has not seen prior sub-goal answers.  Omitting any option from
  step 1 is an error — always enumerate the full set first.
- If HISTORY already contains a tool result for this goal, answer from that result
  directly — do not call the same tool again, UNLESS the goal requires N URL fetches
  (e.g. "read the top 3 results") and fewer than N have been made yet — in that
  case keep fetching the next URL until all N are attempted.
- If ATTACHED ARTIFACTS do not contain the data needed for this goal, do NOT answer
  saying the data is missing. Call the appropriate tool to fetch it instead.
- If HISTORY shows 3 or more consecutive search/fetch results with "No results found"
  for the same goal, stop trying. Answer from your own knowledge or note unavailability.
- If HISTORY contains "[SEARCH_EXHAUSTED:" for this goal, answer from your own
  knowledge — do NOT call any search or fetch tool again.
- If HISTORY contains "[tool_timeout]" for this goal, switch to a different tool
  strategy — do NOT retry the same URL.  EXCEPTION: for "read N results" goals,
  a timeout on one URL means skip to the NEXT URL until all N have been attempted.
- NEVER output "__NO_ANSWER__", "N/A", "NONE", or any single-word placeholder
  as a standalone response.  Always produce either a substantive text answer
  (at least one full sentence) or a single tool_call.
- If ATTACHED ARTIFACTS are present AND HISTORY shows a fetch or search tool
  already returned results for this goal, synthesize your answer directly from
  the artifact content — do not output a placeholder or call a tool again."""


def _build_tool_guide(tools: list[dict]) -> str:
    """Generate the TOOL SELECTION section from the live MCP tool schemas.

    Called once per decision step so the prompt always matches whatever tools
    the MCP server currently exposes — no hardcoding, never stale.
    """
    if not tools:
        return ""

    lines: list[str] = [
        "TOOL SELECTION — reason from the goal and context to choose the right tool:",
        "  Before calling any tool, ask: \"What does this goal need, and which tool",
        "  provides exactly that?\"  The tools below are discovered live from the MCP",
        "  server at runtime — this list is always current.",
        "",
    ]

    for tool in tools:
        name = tool.get("name", "unknown")
        desc = (tool.get("description") or "").strip()
        schema = tool.get("input_schema") or {}
        props: dict = schema.get("properties") or {}
        required: set[str] = set(schema.get("required") or [])

        # Signature: required params positional, optional as param=<type>
        sig_parts: list[str] = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            if pname in required:
                sig_parts.append(pname)
            else:
                sig_parts.append(f"{pname}=<{ptype}>")

        lines.append(f"  {name}({', '.join(sig_parts)})")

        # Tool-level description (from MCP server docstring)
        if desc:
            for dline in desc.splitlines():
                lines.append(f"      {dline.strip()}")

        # Per-parameter descriptions when available
        param_notes: list[str] = []
        for pname, pinfo in props.items():
            pdesc = (pinfo.get("description") or "").strip()
            if pdesc:
                req_label = "required" if pname in required else "optional"
                param_notes.append(f"        {pname} ({req_label}): {pdesc}")
        lines.extend(param_notes)

        lines.append("")

    lines.append(
        "  If NO available tool can satisfy the goal (e.g. set a calendar alert,\n"
        "  send an email, post to social media, book a flight), answer directly\n"
        "  with what the user should do manually — do NOT loop or retry."
    )
    lines.append("")

    return "\n".join(lines)


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
                f"  iter {h['iter']}: ANSWER: {h.get('text', '')[:800]}"
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
    """One LLM call — returns answer text or a single ToolCall.

    The system prompt is assembled at call time: the TOOL SELECTION section is
    generated from the live ``mcp_tools`` schemas so it always reflects whatever
    the MCP server currently exposes — no manual updates needed when tools change.
    """
    messages = _build_messages(goal, hits, attached, history)

    # Build system prompt with live tool catalogue inserted between preamble and rules
    system = _SYSTEM_PREAMBLE + _build_tool_guide(mcp_tools) + _SYSTEM_RULES

    resp = await gw.chat(
        messages,
        system=system,
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
