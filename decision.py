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

# Matches "tool_call:name(..." plain-text tool call format emitted by some models
# when they output the call as a string instead of a native tool_call object.
_TC_TEXT_RE = re.compile(r"tool_call\s*:?\s*(\w+)\(", re.IGNORECASE)

# Matches Llama 3.1/3.2-style function call markup emitted as plain text:
#   <|tool_calls_section_begin|><|tool_call_begin|>functions.<id_or_name>
#   <|tool_call_argument_begin|>{...}<|tool_call_end|>
# The identifier after "functions." is often a tool-use ID (toolu_...) rather
# than the real tool name; we infer the correct name from arg keys below.
_LLAMA_FC_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_call_begin\|>"
    r"functions\.\w+<\|tool_call_argument_begin\|>(\{.*?\})"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)

# ------------------------------------------------------------------------------- #
# System prompt — split into preamble + dynamic tool guide + rules               #
# The TOOL SELECTION section is built at call time from the live MCP schema so  #
# adding or renaming a tool in mcp_server.py is automatically reflected here.   #
# ------------------------------------------------------------------------------- #

_SYSTEM_PREAMBLE = """\
You are DECISION, the action selector in an agentic loop.

You receive one GOAL and supporting context. Before responding, reason through the
following steps in order:

REASONING PROCESS:

  STEP 1 — CLASSIFY THE GOAL TYPE (reasoning type: categorisation)
    What kind of work does this goal require?
    • Acquisition   — fetch / search / retrieve external data NOT yet in context → tool_call
    • Synthesis     — compare / recommend / determine / select from existing context → answer
    • Memory-read   — recall a saved fact → check MEMORY HITS first, then file-read tool
    • Memory-write  — persist a fact or file → file-create or file-update tool_call
    • Real-time     — FETCH current time / live rates / live weather not yet in context
                      → MUST use a tool; never guess
                      EXCEPTION: if the data is already in ATTACHED ARTIFACTS or HISTORY,
                      this is a Synthesis goal — produce an answer, do NOT re-fetch

  STEP 2 — CHECK EXISTING CONTEXT (reasoning type: lookup / evidence scan)
    In this order:
    a. Does MEMORY HITS already contain the answer? → answer directly from it.
    b. Does HISTORY show this goal's tool already returned a result? → synthesize from it.
    c. Are ATTACHED ARTIFACTS present with relevant content? → synthesize from them.
    d. None of the above → a tool_call is needed.

  STEP 3 — SELECT ACTION (exactly one)
    • answer    — when steps 2a / 2b / 2c confirmed sufficient context exists
    • tool_call — when step 2d applies, or when real-time / file I/O is required

  STEP 4 — SELF-CHECK before responding:
    [ ] Am I returning EXACTLY ONE of answer or tool_call (never both)?
    [ ] If answer: is it substantive (≥ 3 sentences or ≥ 3 items for synthesis goals)?
    [ ] If tool_call: is the tool name in the available TOOL SELECTION list?
    [ ] Am I free of art: handles in path / url arguments?
    [ ] For real-time queries: is the data absent from ATTACHED ARTIFACTS and HISTORY?
        (if already present in context, this is Synthesis — produce an answer)
    [ ] For recommendation answers: do I follow OPTIONS → CONTEXT → RECOMMENDATION order?

You must return EXACTLY ONE of:
  1. answer    — a direct response producible from CONTEXT or ATTACHED ARTIFACTS
  2. tool_call — when external data, file access, or live values are needed

"""

_SYSTEM_RULES = """\
BEHAVIORAL NOTES (apply to whichever tools are available):

  Reasoning type: real-time lookup
    For FETCHING current time, live exchange rates, live weather that is NOT yet in
    context → call the appropriate tool. Never answer from training-data.
    If HISTORY or ATTACHED ARTIFACTS already contain fresh data from this run,
    synthesize from it — do NOT re-fetch.

  Reasoning type: web research
    Prefer the search tool when snippets contain enough detail.
    Use URL-fetch only when you need the full page body after a search gave a URL.

  Reasoning type: memory / file I/O
    Use the file-listing tool to discover saved facts, file-read to load them,
    file-create to save new ones (raises if exists), file-update to overwrite.
    The memory/ directory is always pre-created — write there safely.
    Example paths: "memory/<key>.txt"

  Reasoning type: multi-fetch / sequential URL reading
    FOR "read N results" GOALS: count URL-fetch calls in HISTORY for this goal.
    Call the URL-fetch tool for the NEXT URL from search results until N calls are
    made or all remaining URLs have timed out. If [tool_timeout] occurs, skip to the
    NEXT URL — do NOT retry the same one. Answer only after all N URLs are attempted.

STRICT RULES:
- NEVER return both answer and tool_call in the same response.
- Synthesis / recommendation / determination goals (keywords: determine, recommend,
  choose, select, compare, most appropriate, which is best) are NEVER Real-time.
  These goals synthesize from already-fetched data. When ATTACHED ARTIFACTS contain
  the needed information (weather, search results, etc.), produce an answer immediately
  — do NOT call any tool, even if the topic involves weather, time, or live data.
- NEVER emit a tool call as plain text (e.g. "tool_call:name(arg:val)").
  When step 3 selects tool_call, use the native tool_call return format —
  never write it out as a string in an answer field.
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

    # Build system prompt with live tool catalogue inserted between preamble and rules.
    # When mcp_tools is empty (no-tools fallback due to provider rate limits), replace
    # the tool guide with an explicit instruction to answer from context only — this
    # prevents the model from emitting tool call markup or JSON objects as plain text.
    if mcp_tools:
        system = _SYSTEM_PREAMBLE + _build_tool_guide(mcp_tools) + _SYSTEM_RULES
    else:
        _no_tools_note = (
            "IMPORTANT: No tools are available for this request.\n"
            "You MUST return a direct text ANSWER synthesised ONLY from "
            "ATTACHED ARTIFACTS and HISTORY.\n"
            "Do NOT output a tool_call, JSON object, code block, or any markup. "
            "Write plain prose immediately.\n\n"
        )
        system = _no_tools_note + _SYSTEM_PREAMBLE + _SYSTEM_RULES

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

    # Detect "tool_call:name(key:val, ...)" plain-text format.
    # Some models omit native tool_call objects and write the call as a string.
    # Only anchor key detection to positions after ^ or , to avoid treating
    # prose words like "Reminder:" as argument names.
    tc_m = _TC_TEXT_RE.search(text)
    if tc_m:
        fn_name = tc_m.group(1)
        # Find matching closing ')' via balanced-paren scan
        paren_open = tc_m.end() - 1
        depth, close = 0, paren_open
        for _i, _ch in enumerate(text[paren_open:]):
            if _ch == "(":
                depth += 1
            elif _ch == ")":
                depth -= 1
                if depth == 0:
                    close = paren_open + _i
                    break
        raw_args = text[paren_open + 1 : close].strip()
        try:
            import json as _json2
            _parsed = _json2.loads(raw_args) if raw_args else {}
            if not isinstance(_parsed, dict):
                raise ValueError("not a dict")
            args: dict = _parsed
        except Exception:
            # Key:val fallback — only anchors at ^ or after ','
            args = {}
            _anchors = list(re.finditer(r"(?:^|,)\s*(\w+):", raw_args))
            for _idx, _km in enumerate(_anchors):
                _key = _km.group(1)
                _vs = _km.end()
                _ve = _anchors[_idx + 1].start() if _idx + 1 < len(_anchors) else len(raw_args)
                args[_key] = raw_args[_vs:_ve].rstrip(", \n")
        if fn_name and (args or not raw_args):
            return DecisionOutput(
                tool_call=ToolCall(name=fn_name, arguments=args)
            )

    # Detect Llama 3.1/3.2-style tool call markup emitted as plain text.
    # The "function name" field is often a tool-use ID (toolu_...) rather than
    # the real tool name, so we infer it by matching arg keys against the live
    # MCP tool schemas: the tool whose required params are a subset of the
    # provided args is the intended tool.
    llama_m = _LLAMA_FC_RE.search(text)
    if llama_m:
        try:
            import json as _json3
            args = _json3.loads(llama_m.group(1))
            if isinstance(args, dict):
                fn_name: str | None = None
                for _tool in mcp_tools:
                    _schema = (_tool.get("input_schema") or {})
                    _required = set(_schema.get("required") or [])
                    if _required and _required <= set(args.keys()):
                        fn_name = _tool["name"]
                        break
                if fn_name:
                    return DecisionOutput(
                        tool_call=ToolCall(name=fn_name, arguments=args)
                    )
        except Exception:
            pass  # fall through to text answer

    # Detect JSON tool_call objects emitted as plain text, e.g.:
    #   {"type": "tool_call", "name": "web_search", "arguments": {...}}
    #   {"tool_call": {"name": "fetch_url", "arguments": {...}}}
    # Only converts to a real ToolCall when the named tool exists in mcp_tools;
    # if the name is not in mcp_tools (e.g. a hallucinated tool like "weather"),
    # this falls through so agent.py's answer guard can catch and reject it.
    _stripped = text.strip()
    if _stripped.startswith("{") or _stripped.startswith("```"):
        _jt = _stripped
        if _jt.startswith("```"):
            _jt = re.sub(r"```\w*\n?", "", _jt).strip()
        try:
            import json as _json4
            _obj = _json4.loads(_jt)
            if isinstance(_obj, dict):
                _tc_name: str | None = None
                _tc_args: dict = {}
                if _obj.get("type") == "tool_call" and isinstance(_obj.get("name"), str):
                    _tc_name = _obj["name"]
                    _tc_args = _obj.get("arguments") or {}
                elif isinstance(_obj.get("tool_call"), dict):
                    _inner = _obj["tool_call"]
                    _tc_name = _inner.get("name") or ""
                    _tc_args = _inner.get("arguments") or {}
                if _tc_name and isinstance(_tc_args, dict):
                    if any(t["name"] == _tc_name for t in mcp_tools):
                        return DecisionOutput(
                            tool_call=ToolCall(name=_tc_name, arguments=_tc_args)
                        )
        except Exception:
            pass  # fall through; agent.py guard will handle the garbled text

    # Some LLMs prefix their response with the option label ("answer\n...").
    # Strip it so it doesn't pollute the final answer shown to the user.
    lower = text.lower()
    if lower.startswith("answer"):
        candidate = text[6:].lstrip(": \n")
        if candidate:
            text = candidate
    return DecisionOutput(answer=text or "Task completed.")
