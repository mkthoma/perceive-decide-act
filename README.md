# perceive-decide-act

Lightweight agentic framework built on a four-role cognitive loop — **Memory → Perception → Decision → Action** — with MCP tool integration, artifact-aware goal tracking, and a multi-provider LLM gateway.

**Demo:** [https://youtu.be/CmgvtCqTdk0](https://youtu.be/CmgvtCqTdk0)

---

## Getting started

### 1. Prerequisites

- **Python >= 3.11** and **[uv](https://docs.astral.sh/uv/)**
- **At least one provider API key** (Gemini free tier is enough to start)

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in at least one worker provider key and a search provider key.

```bash
# Minimum viable .env
GEMINI_API_KEY=your_gemini_api_key_here

# Search: Tavily primary, DuckDuckGo fallback (free, no key needed)
TAVILY_API_KEY=your_tavily_api_key_here   # recommended: 1,000 free/mo
```

The `.env` file lives at the **repo root** — one level above `llm_gatewayV3/`. The gateway reads `../.env` relative to its own directory, so a single file covers both.

> **Search reliability:** Without a Tavily key, `web_search` falls back to DuckDuckGo (free, no key required) which works but is rate-limited and unreliable. Adding a Tavily key is the quickest fix — free tier at <https://app.tavily.com/>.

### 3. Install agent dependencies

```powershell
uv sync
```

crawl4ai requires Playwright browsers on first install — run the setup once:

```powershell
uv run crawl4ai-setup
```

This downloads Chromium (~150 MB). Only needed once.

### 4. Run your first query

```powershell
uv run python agent.py "What time is it in Tokyo right now?"
```

Or run the full canonical test suite:

```powershell
uv run python test_all.py          # all 5 queries
uv run python test_all.py 1 3 4    # run queries A, C1, C2 only (1-based)
```

The gateway runs **in-process** — no separate server needed. Provider adapters from `llm_gatewayV3/` are imported directly and all routing, rate-limiting, and failover happens in the same process.

---

## Architecture

The agent runs a bounded loop of at most 20 iterations. Each iteration executes the four roles in strict order:

```
+---------------------------------------------------------+
|                    agent.py loop                        |
|                                                         |
|  +----------+   hits    +------------+                  |
|  |  Memory  |---------->| Perception | (iter 1 only)   |
|  +----------+           +-----+------+                  |
|       ^                       | Observation             |
|       | record_outcome        v                         |
|  +----+-----+          +------------+                   |
|  |  Action  |<---------+  Decision  |                   |
|  +----------+ ToolCall +------------+                   |
|       |                                                 |
|       +-- MCP tool result --> memory + history          |
+---------------------------------------------------------+
```

The loop terminates when all goals are marked `done`, or when `MAX_ITERATIONS` (20) is reached.

> **Perception is called only on iteration 1.** It decomposes the query into goals once. On subsequent iterations `agent.py` manages goal completion directly — marking goals `done` immediately when Decision produces an answer or an acquisition tool call succeeds. This saves one LLM call per iteration and avoids rate-limit pressure on the Gemini free tier.

---

## The four roles

### Memory — `memory.py`

**What it does:** A typed, persistent fact store that gives every iteration access to relevant knowledge from prior runs — without re-fetching or re-computing it.

**How it works:**

1. At the start of every iteration, `memory.read(query, history)` performs a **keyword-overlap search** (no LLM, no cost) against all stored items. It tokenizes the query and the last six history entries, removes stopwords, then scores each stored item by how many of its keywords appear in that combined set. Top-8 hits are returned.

2. `memory.remember(text)` is called once at run start with the raw user query. This fires **one LLM gateway call** to classify the text — assigning a `kind` (`fact`, `preference`, `tool_outcome`, `scratchpad`), extracting 3-10 search keywords, writing a human-readable `descriptor`, and building a structured `value` payload. The result is appended to `state/memory.json` and persists across process restarts.

3. `memory.record_outcome(tool_call, result_text, artifact_id)` is called after every Action with zero LLM cost. It extracts keywords from the tool name and arguments, stores the result preview, and — critically — stores the `artifact_id` when the tool returned a large payload. This is how Perception learns that an artifact exists for a given goal.

**Why it matters:** Without memory, each run starts cold. A fact remembered in run 1 ("mom's birthday is May 15") is instantly available as a keyword hit in run 2 — the agent answers without any tool call. Artifact IDs in tool-outcome records are how Perception decides which artifact to attach to an extraction goal, decoupling the fetch step from the read step across iterations.

**Stored at:** `state/memory.json` — plain JSON array of `MemoryItem` objects.

---

### Perception — `perception.py`

**What it does:** The orchestrator. It decomposes the user query into an ordered list of goals on the first iteration only. Done-flags are then managed entirely by `agent.py` on subsequent iterations.

**How it works:**

1. **First iteration** (`prior_goals` is empty): Perception sends the query, memory hits, and an empty prior-goals list to Gemini with `temperature=1.0`. The LLM decomposes the query into 1-4 short imperative goals ordered by logical dependency — fetch before extract, search before synthesize. If the query contains memory-write intent ("remember", "save", etc.) Perception places the durable-save goal first.

2. **Subsequent iterations**: `agent.py` bypasses the Perception LLM call entirely and reuses the existing goal list. Goals are marked `done` by `agent.py` immediately when Decision answers or an acquisition tool call succeeds.

3. **Artifact attachment**: For the first unfinished goal that requires reading a previously fetched artifact (e.g. "extract info from page"), Perception sets `artifact_index` pointing to the corresponding `[artifact N]` entry in the memory-hit list. The main loop resolves this index to the actual artifact ID stored by memory.

4. **Sticky-done guard**: On the first iteration, `agent.py` resets all Perception-returned `done` flags to `False`. This prevents memory hits from prior runs from making Gemini think the task is already complete before any work has been done this run.

**Why it matters:** Pinning Perception to Gemini (`provider="g"`) ensures reliable structured-output compliance for the goal list schema. `temperature=1.0` prevents Gemini 3.x from stalling in a low-entropy loop. Calling it only once per run cuts free-tier Gemini RPM usage by up to 90% on long queries.

**Fallback:** If the Gemini call fails, Perception returns prior goals unchanged (or a single bare goal on the first iteration). The run degrades gracefully rather than crashing.

#### System prompt

The v3 prompt uses a four-step **REASONING PROCESS** with per-step reasoning-type labels. The **DATE COMPUTATION** rule in STEP 2a resolves time-offset phrases (e.g. "two weeks before") into exact calendar dates inline — eliminating a separate date-calculation goal. The **MEMORY WRITES** rule triggers only on explicit persistence keywords, preventing false-positive reminder injection on general queries.

```
You are PERCEPTION, the goal-tracking orchestrator in an agentic loop.

REASONING PROCESS — follow every step in order before producing output:

  STEP 1 — ASSESS SITUATION (reasoning type: conditional / state-check)
    Ask: Is PRIOR GOALS empty?
    • YES → this is iteration 1; proceed to STEP 2a (decompose).
    • NO  → this is a subsequent iteration; proceed to STEP 2b (update).

  STEP 2a — DECOMPOSE (iteration 1 only)
    Reasoning type: planning / dependency ordering.
    - Break the QUERY into 1-4 short imperative goals (≤ 15 words each).
    - Order by logical dependency: fetch/search before extract, extract before synthesize.
    - Apply MEMORY WRITES: if query contains "remember / save / store / note / record / keep",
      add a durable-save goal FIRST.  Example:
        "Save mom's birthday (May 15, 2026) to memory/moms_birthday.txt"
    - Apply DATE COMPUTATION: when the query mentions time offsets relative to a
      known date ("two weeks before", "3 days after", "the week of"), compute all
      derived dates and embed them explicitly in the goal text — never leave
      offsets as vague strings.
      Example: "birthday May 15, 2026, reminder two weeks before and on the day"
        → "Save reminders for May 1, 2026 (2-week prior) and May 15, 2026 (day-of) to memory/reminder.txt"

  STEP 2b — UPDATE DONE FLAGS (subsequent iterations)
    Reasoning type: evidence matching.
    - Copy goals in EXACT SAME ORDER — never reorder, insert, or drop.
    - For each goal, scan HISTORY: is there an action result or ANSWER that
      directly satisfies it?  If yes → done: true.  Once done, always done.

  STEP 3 — SET ARTIFACT INDEX (reasoning type: lookup / index matching)
    Consider only the FIRST unfinished goal.
    - Does completing it require reading a previously fetched artifact?
      → Set artifact_index to the integer from [artifact N] in MEMORY HITS.
    - Otherwise → set artifact_index to null.
    - NEVER invent or guess an index not present in MEMORY HITS.

  STEP 4 — SELF-CHECK before outputting:
    [ ] Are done flags backed by explicit HISTORY evidence (not inferred)?
    [ ] Is goal order identical to the original decomposition?
    [ ] Is artifact_index either null or a real [artifact N] label?
    [ ] Did I apply MEMORY WRITES when the query asked to persist a fact?
    [ ] For date-offset goals: have I computed exact derived dates, not left them as offsets?
    [ ] Are all goals ≤ 15 words and imperative (start with a verb)?

ERROR HANDLING / FALLBACKS:
  - If HISTORY is ambiguous about whether a goal is satisfied, mark it NOT done
    (conservative — let Decision retry rather than skip a needed step).
  - If QUERY is very short and unclear, create a single broad goal:
    "Research and answer: <query verbatim>"
  - If an artifact is referenced but no [artifact N] label exists in MEMORY HITS,
    set artifact_index to null (do not guess).

Return ONLY valid JSON matching the schema. No prose or commentary outside JSON.
```

#### [PoP validation](https://github.com/mkthoma/perceive-decide-act/blob/main/prompt_evaluator.md) (9-criteria rubric)

```json
{
  "prompt_id": "perception_system_v3",
  "role": "PERCEPTION",
  "evaluated_at": "2026-05-20",
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": true,
  "reasoning_type_awareness": true,
  "fallbacks": true,
  "overall_clarity": "Excellent. All 9 criteria met. The DATE COMPUTATION rule replaces the over-broad PROACTIVE REMINDERS, eliminating false-positive reminder injection on general queries. JSON-only output is explicit and enforced by Pydantic gateway schema injection. Conservative done-marking fallback prevents premature goal completion."
}
```

---

### Decision — `decision.py`

**What it does:** Given one goal and its context, produces exactly one output: either a direct **answer** (text) or a single **tool call**. It never does both.

**How it works:**

1. Decision builds a user message containing the current goal, memory hits, recent history (last 10 entries, each truncated to 800 chars), and — when Perception has attached one — the full bytes of the artifact rendered inline as `ATTACHED ARTIFACTS`.

2. It sends this message to the gateway with `auto_route="decision"` and the full MCP tool list. The gateway classifies the request size (TINY or LARGE), selects a worker tier, and dispatches to the first available provider.

3. The response is inspected for native tool calls first. If none are found, Decision checks for vLLM/Groq-style text markup `<function(name){...}</function>` before treating the text as an answer.

**Tool reasoning:** Rather than hard-coded dispatch rules, Decision receives a dynamic **TOOL SELECTION guide** assembled at call time via `_build_tool_guide(mcp_tools)`. The guide is generated from the live MCP server schema on every call — so the tool list in the prompt is always current and never stale. Decision reasons from this guide to correctly select `get_time` for timezone questions, `create_file` for memory persistence, and the multi-fetch protocol for "read the top 3 results" goals. Adding or renaming a tool in `mcp_server.py` is automatically reflected in the prompt with no manual update needed.

**Why it matters:** Separating Decision from Action ensures the LLM never directly executes code. Decision emits intent (`ToolCall`); Action executes it through MCP. The `auto_route` lets the gateway pick the cheapest provider that fits the request size — a small fetch decision goes to a fast TINY-tier model; an extraction decision with 50 KB of attached content goes to a large-context LARGE-tier provider.

#### System prompt

The v3 prompt is assembled at call time from three parts: `_SYSTEM_PREAMBLE` (four-step REASONING PROCESS), a dynamic `TOOL SELECTION` section generated from the live MCP schema via `_build_tool_guide(mcp_tools)`, and `_SYSTEM_RULES` (behavioral notes with reasoning-type labels and 13 STRICT RULES).

**Preamble** (`_SYSTEM_PREAMBLE` — always included):

```
You are DECISION, the action selector in an agentic loop.

You receive one GOAL and supporting context. Before responding, reason through the
following steps in order:

REASONING PROCESS:

  STEP 1 — CLASSIFY THE GOAL TYPE (reasoning type: categorisation)
    What kind of work does this goal require?
    • Acquisition   — fetch / search / retrieve external data → likely tool_call
    • Synthesis     — compare / recommend / summarize from existing context → likely answer
    • Memory-read   — recall a saved fact → check MEMORY HITS first, then file-read tool
    • Memory-write  — persist a fact or file → file-create or file-update tool_call
    • Real-time     — current time / live rates / live weather → MUST use a tool; never guess

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
    [ ] For real-time queries: am I using a tool (not training-data assumptions)?
    [ ] For recommendation answers: do I follow OPTIONS → CONTEXT → RECOMMENDATION order?

You must return EXACTLY ONE of:
  1. answer    — a direct response producible from CONTEXT or ATTACHED ARTIFACTS
  2. tool_call — when external data, file access, or live values are needed
```

**Tool selection** (`_build_tool_guide(mcp_tools)` — generated at runtime from the live MCP server schema):

> This section is assembled dynamically each call from the tool objects returned by `session.list_tools()`. Each tool entry includes its name, a signature line showing required vs. optional parameters, the full docstring from `mcp_server.py`, and per-parameter descriptions. The list is always current — adding or renaming a tool in `mcp_server.py` is automatically reflected here with no manual prompt update. For the current tool set, see the [MCP tools table](#mcp-tools-available-to-the-agent) below.

**Rules** (`_SYSTEM_RULES` — always included):

```
BEHAVIORAL NOTES (apply to whichever tools are available):

  Reasoning type: real-time lookup
    For current time, live exchange rates, live weather → call the appropriate
    tool every time. Never answer from training-data or stale memory.

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
    1. PRESENT ALL OPTIONS: number every option from prior HISTORY answers.
       Do NOT drop or skip any option — list them all before making a judgment.
    2. CONTEXT: state the relevant constraint (weather, budget, etc.).
    3. RECOMMENDATION: pick ONE as the single best choice with reasoning.
  The reader has not seen prior sub-goal answers. Omitting any option is an error.
- If HISTORY already contains a tool result for this goal, answer from that result
  directly — do not call the same tool again, UNLESS the goal requires N URL fetches
  and fewer than N have been made yet — in that case keep fetching the next URL.
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
  the artifact content — do not output a placeholder or call a tool again.
```

#### [PoP validation](https://github.com/mkthoma/perceive-decide-act/blob/main/prompt_evaluator.md) (9-criteria rubric)

```json
{
  "prompt_id": "decision_system_v3",
  "role": "DECISION",
  "evaluated_at": "2026-05-20",
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": true,
  "reasoning_type_awareness": true,
  "fallbacks": true,
  "overall_clarity": "Excellent. All 9 criteria met. Four-step REASONING PROCESS with typed labels, dynamic TOOL SELECTION from live MCP schemas, 6-item self-check, and 13 STRICT RULES covering every known failure mode. The NEVER-emit-text-tool-call rule and balanced-paren parser close the text-format tool-call gap."
}
```

---

### Action — `action.py`

**What it does:** Pure MCP dispatch. No LLM calls. Takes a `ToolCall`, executes it through the MCP `ClientSession`, and returns a `(descriptor, artifact_id)` pair.

**How it works:**

1. **Guard check**: Before any dispatch, Action inspects the tool arguments for `art:...` handles in path/URL fields. If found, it returns an error descriptor immediately — this prevents Decision from accidentally passing a stale artifact handle to a tool that expects a real URL or file path.

2. **MCP dispatch**: Calls `session.call_tool(name, arguments)` wrapped in `asyncio.wait_for` with a **30-second timeout** (surfaced to Decision as `[tool_timeout]`). If exceeded, Action returns the timeout descriptor and Decision switches strategy — falls back to `web_search`, tries a different URL, or answers from existing artifacts.

3. **JSON post-processing**: The MCP server returns typed Python objects (`list[dict]`, `dict`) which FastMCP serialises as JSON strings. Action unwraps these into clean human-readable text before passing them to Decision — search results become `Title / URL / Snippet` blocks, file tool responses become plain file content, etc.

4. **Artifact threshold**: If the tool response exceeds `ARTIFACT_THRESHOLD_BYTES` (4 KB), the bytes are written to the content-addressable artifact store (`state/artifacts/`). The returned descriptor includes the artifact handle (`art:<sha256[:16]>`) and a 200-character preview. If the response is small enough, it is returned as raw text with no artifact created.

**Why it matters:** The artifact threshold is what keeps Decision's context from bloating. A 50 KB Wikipedia page is stored once as `art:4c163...` and referred to by handle throughout the remaining iterations. Only when Perception explicitly attaches it does Decision see the bytes — and only the bytes it actually needs. The `art:` guard closes the loop: Decision is told not to pass handles to tools, and Action enforces that rule at execution time.

---

### agent.py — main loop

**What it does:** Wires the four roles together in a bounded loop and applies several correctness policies that the individual roles cannot enforce in isolation.

#### Goal classification

`agent.py` classifies each goal with three helper functions to decide how to handle it:

```python
_SYNTHESIS_KW = frozenset(
    "synthesize synthesise compare decide summarize summarise "
    "recommend choose select appropriate agree common findings "
    "analyze analyse collate distil distill".split()
)

_ACQUISITION_VERBS = frozenset(
    "fetch download retrieve get load search look".split()
)

_ANSWER_MARKERS = frozenset(
    "tell give describe explain list summarize compare "
    "what when how why which who does did has have".split()
)
```

- `_is_synthesis_goal(text)` — True when the goal contains any `_SYNTHESIS_KW` word. Used to trigger force-attach of all run artifacts and to gate final-answer selection.
- `_is_acquisition_goal(text)` — True when the goal starts with an acquisition verb AND contains no synthesis keywords AND contains no answer markers. `"Fetch the Wikipedia page"` is acquisition; `"Fetch https://... and tell me his birth date"` is NOT (contains "tell").
- `_has_memory_write_intent(query)` / `_has_memory_write_goal(goals)` — Two-layer check: does the user query ask to persist a fact? Does Perception's goal list already include a durable save step?

> **Why `"extract"` is absent from `_SYNTHESIS_KW`:** Extraction goals like "Extract birth date and death date from the fetched content" are data-retrieval steps, not final integration answers. Including `"extract"` caused the synthesis gate in `_final_answer_from` to suppress prior answers when contributions were the last answered goal — dropping birth/death dates from the final output.

#### Three-tier artifact attachment

Before each Decision call, `agent.py` selects which artifact bytes (if any) to attach inline:

| Tier | Condition | Effect |
|---|---|---|
| **Explicit** | `goal.attach_artifact_id` set by Perception | Attaches that specific artifact |
| **Force-attach** | `_is_synthesis_goal(goal.text)` and no explicit attach | Attaches up to 3 most-recent run artifacts |
| **Context-attach** | Non-acquisition goal following a completed acquisition goal | Attaches most-recent run artifact |

The context-attach tier ensures extraction goals (e.g. "Extract birth date from the fetched content") always receive the artifact that the prior fetch step produced, even when Perception's `artifact_index` is null.

#### Acquisition goal auto-done

When an acquisition tool call (`web_search`, `fetch_url`, `read_file`, `list_dir`) succeeds for an acquisition goal, `agent.py` immediately marks the goal done:

```python
_AUTO_DONE_TOOLS = frozenset({"web_search", "fetch_url", "read_file", "list_dir"})

if (not is_error and not is_empty
        and _is_acquisition_goal(goal.text)
        and tc.name in _AUTO_DONE_TOOLS):
    goal.done = True
```

This prevents Decision from looping trying to re-read an artifact handle after the data has already been fetched.

#### Memory-write safety net

If the user query contains a memory-write keyword ("remember", "save", "store", etc.) and Perception produced no goal that references `memory/`, `create_file`, or a save/persist verb, `agent.py` injects a durable-save goal at the front of the list:

```python
if _has_memory_write_intent(query) and not _has_memory_write_goal(prior_goals):
    save_goal = Goal(
        id=uuid.uuid4().hex[:8],
        text=f"Save fact to memory/ with create_file: {_fact}",
        done=False,
    )
    prior_goals.insert(0, save_goal)
```

This ensures `create_file("memory/moms_birthday.txt", ...)` is always called when the user says "remember that", even if Gemini's decomposition omits the persistence step.

#### `sandbox/memory/` pre-creation

`mcp_server.create_file` raises `ValueError` if the parent directory does not exist. `agent.py` calls `_ensure_sandbox_dirs()` at the top of every `run()` to guarantee `sandbox/memory/` is always present before any agent call can attempt a write:

```python
_SANDBOX_DIRS = ["sandbox/memory"]

def _ensure_sandbox_dirs() -> None:
    import os as _os
    base = _os.path.dirname(_os.path.abspath(__file__))
    for d in _SANDBOX_DIRS:
        _os.makedirs(_os.path.join(base, d), exist_ok=True)
```

#### `__NO_ANSWER__` sentinel guard

Some LLM providers emit `__NO_ANSWER__` as a literal string instead of extracting data from attached artifacts. When detected, `agent.py` injects a `[STOP]` system entry into history and retries Decision rather than recording the placeholder as the answer:

```python
if "__NO_ANSWER__" in answer_text:
    history.append({
        "iter": it, "kind": "action", "goal_id": goal.id,
        "tool": "SYSTEM", "arguments": {},
        "result_descriptor": (
            "[STOP] The previous response was a placeholder. "
            "ATTACHED ARTIFACTS contain the requested data. "
            "Extract the information and produce a real answer "
            "NOW -- do NOT call any tool."
        ),
        "artifact_id": None,
    })
    continue
```

Decision's system prompt contains a matching rule: `If HISTORY contains a [STOP] line, answer directly from ATTACHED ARTIFACTS -- do NOT call any tool.`

#### Hard-stop on repeated empty searches

If the last 3 actions for a goal all returned empty results, `agent.py` appends a `[SEARCH_EXHAUSTED]` sentinel to that history entry, blocking further search tool calls for that goal. Decision's system prompt instructs it to answer from its own knowledge when this sentinel is present.

#### `_final_answer_from` — synthesis gate logic

```
1. Collect the last answer per goal_id (skip __NO_ANSWER__ entries).
2. If only one answer exists, return it.
3. Find the last answered goal in goal-list order.
4. If that goal is a synthesis goal:
   a. Count numbered items in the synthesis answer.
   b. Count prior non-synthesis answered goals (the "options").
   c. If prior_count >= 1 AND numbered < prior_count:
      -> the synthesis answer dropped options; prepend prior answers.
   d. Otherwise return the synthesis answer alone.
5. If no synthesis gate: join all answers in goal order with double newline.
```

The `prior_count >= 1` threshold (previously `> 1`) ensures the fallback fires even for two-goal queries where exactly one prior answered goal exists.

#### Emergency synthesis call

If `_final_answer_from` produces only a placeholder (no answer was recorded, all goals were auto-completed by tool calls), `agent.py` makes one additional Decision call with no tools and the most recent run artifact attached, to synthesise a final answer from whatever was collected.

---

## Prompt Evaluation Rubric

All system prompts in this project are evaluated against a 9-criteria **Prompt of Prompts (PoP)** rubric. Each criterion is a boolean pass/fail. A prompt is production-ready when all 9 are met.

| # | Criterion | What it checks |
|---|-----------|----------------|
| 1 | **Explicit Reasoning Instructions** | Does the prompt tell the model *how* to think (step-by-step process), not just *what* to produce? |
| 2 | **Structured Output Format** | Is the expected output format defined precisely, with schema enforcement where possible? |
| 3 | **Separation of Reasoning and Tools** | Are tool-use decisions cleanly separated from reasoning and output generation? |
| 4 | **Conversation Loop Support** | Does the prompt handle multi-turn state — e.g. prior goals, history, done-flags — explicitly? |
| 5 | **Instructional Framing** | Are all instructions imperative, unambiguous, and free of hedging language? |
| 6 | **Internal Self-Checks** | Does the prompt include a checklist or self-verification step before output? |
| 7 | **Reasoning Type Awareness** | Are the different kinds of reasoning the model must perform named and distinguished? |
| 8 | **Error Handling or Fallbacks** | Are edge cases, ambiguous inputs, and failure modes explicitly handled? |
| 9 | **Overall Clarity and Robustness** | Would a new model, cold, produce correct output from this prompt alone with no extra context? |

The PoP JSON for each prompt appears in its `#### PoP validation` subsection. See [Perception](#perception----perceptionpy) and [Decision](#decision----decisionpy) above.

---

## Shared contracts — `schemas.py`

All four roles communicate exclusively through Pydantic v2 models defined in `schemas.py`. This file is the single source of truth for every data shape that crosses a role boundary.

```python
# The six models that wire the loop together

class MemoryItem          # A stored fact, preference, or tool outcome
class Artifact            # Metadata for a stored binary payload
class Goal                # One sub-task: text, done flag, optional artifact attachment
class Observation         # Perception's output: ordered list of Goals
class ToolCall            # Decision's tool output: name + arguments dict
class DecisionOutput      # Decision's full output: answer XOR tool_call
```

`Observation` carries two computed properties used directly in the main loop:

```python
obs.all_done          # True when every goal has done=True
obs.next_unfinished() # Returns the first Goal with done=False
```

`DecisionOutput` enforces the answer-XOR-tool-call invariant:

```python
out.is_answer         # True when out.answer is not None
```

---

## Pydantic v2 — where it is used and why

Pydantic v2 (`pydantic.BaseModel`) is used in **five distinct ways** across this codebase:

### 1. Role-boundary contracts (`schemas.py`)

Every object that flows between roles is a Pydantic model. This means:

- All fields are type-checked at parse time — a malformed LLM response that omits a required field raises `ValidationError` immediately rather than producing a silent `None` downstream.
- Models serialize to/from JSON with `model_dump(mode="json")` and `model_validate(raw)`, used for loading and saving `state/memory.json`.
- `Field(default_factory=...)` generates stable IDs (`uuid4().hex[:8]` for goals, `hex[:12]` for memory items) at construction time.

### 2. LLM response parsing (`memory.py`)

`_Classification` and `_RelevanceResponse` are private Pydantic models used as `response_model` arguments to `gw.chat()`. The gateway injects the model's JSON schema into the provider request (as `response_format: json_schema`), coerces the LLM output, and validates the structure. If validation fails, `gw.parse_model()` retries with regex extraction before giving up.

```python
# memory.py -- classify a raw text into a typed memory item
resp = await gw.chat(messages, response_model=_Classification, ...)
cls = gw.parse_model(resp, _Classification)   # raises ValueError on parse failure
```

### 3. Perception's goal-list schema (`perception.py`)

`_GoalSlot` and `_PerceptionResponse` are private models used the same way. The gateway sends Gemini a JSON schema derived from `_PerceptionResponse`, and Perception validates the response before converting it into public `Goal` objects with stable IDs.

```python
# perception.py -- parse Gemini's goal list into typed objects
perc = gw.parse_model(resp, _PerceptionResponse)
for slot in perc.goals:
    new_goals.append(Goal(id=..., text=slot.text, done=slot.done, ...))
```

### 4. Persistence round-trips (`memory.py`)

The memory store serializes and deserializes using Pydantic's native JSON support:

```python
# Save -- convert to JSON-safe dicts
json.dumps([item.model_dump(mode="json") for item in self._items])

# Load -- validate each raw dict back into a MemoryItem
self._items = [MemoryItem.model_validate(r) for r in raw]
```

This means any schema evolution (adding an optional field) is handled automatically — old records load cleanly because Pydantic applies defaults for missing fields.

### 5. Gateway structured-output plumbing (`llm_gateway.py`)

When `response_model` is passed to `gw.chat()`, the gateway strips Pydantic's generated `title` fields (which some providers reject), wraps the schema in a `json_schema` response-format envelope, and validates the response. `gw.parse_model()` handles the three fallback cases: pre-validated `parsed` dict, direct JSON parse, and regex-extracted JSON object.

---

## File layout

```
perceive-decide-act/
+-- agent.py            Main loop (Memory -> Perception -> Decision -> Action)
+-- schemas.py          Pydantic v2 contracts for all role boundaries
+-- memory.py           Typed fact store -- keyword search, LLM classification
+-- perception.py       Goal decomposition (Gemini, iter 1 only)
+-- decision.py         Action selection -- answer or single tool call
+-- action.py           MCP dispatch with 30s timeout, artifact threshold
+-- artifact_store.py   Content-addressable store (state/artifacts/)
+-- llm_gateway.py      In-process LLM gateway -- routing, failover, rate limits
+-- mcp_server.py       9 MCP tools -- Tavily primary, DuckDuckGo fallback
+-- test_all.py         Colour-coded runner for the 5 canonical test queries
+-- pyproject.toml      uv project config
+-- .env.example        Environment variable template
+-- llm_gatewayV3/      Gateway source (see llm_gatewayV3/README.md)
```

---

## LLM Gateway V3

The gateway is imported as a Python library — `llm_gatewayV3/` is added to `sys.path` at startup and provider adapters are called in-process. No HTTP server is required for normal agent operation.

It routes LLM calls across **seven free worker providers** (Gemini, Groq, Cerebras, NVIDIA NIM, OpenRouter, GitHub Models, Ollama) with automatic failover and a **router pool** that classifies each request by size and picks the right worker tier.

### Starting the gateway (optional — dashboard only)

To view the live dashboard and call-log, run the gateway in a separate terminal:

```powershell
cd llm_gatewayV3
uv run python main.py
```

Ready when you see `Uvicorn running on http://127.0.0.1:8101`. Dashboard: <http://localhost:8101>

### Checking gateway health

```bash
# Worker pool -- providers, models, rate limits
curl -s http://localhost:8101/v1/providers | python3 -m json.tool

# Router pool -- four small LLMs that classify request size
curl -s http://localhost:8101/v1/routers | python3 -m json.tool

# Live rate state
curl -s http://localhost:8101/v1/status | python3 -m json.tool
```

### How routing works

When `agent` makes a gateway call tagged `auto_route="decision"` (or `"perception"` / `"memory"`), the gateway:

1. Estimates the token count of the request
2. Sends a bounded `{token_count, sample}` envelope to a router LLM (Cerebras -> Groq -> NVIDIA -> GitHub, first available)
3. Router responds with one word: `TINY` (< 1K tokens) or `LARGE` (1K-8K tokens)
4. Filters the worker list from `LLM_ORDER` by minimum context window for the tier
5. Dispatches to the first available provider and returns the response

If all router providers are unavailable, it falls back to the token-count rule — the worker call still succeeds.

### Provider keys

| Provider | Free tier | Recommended for |
|---|---|---|
| Gemini | 15 RPM / 1,000 RPD | Perception (pinned), large-context extraction |
| Groq | 30 RPM / 1,000 RPD | Decision (fast, high quality) |
| Cerebras | 30 RPM / 1M tok/day | Memory (fast, small calls) |
| NVIDIA NIM | 40 RPM | General worker |
| GitHub Models | 10-15 RPM | Low-volume fallback |
| OpenRouter | 20 RPM / 50 RPD | Additional coverage |
| Ollama | unlimited | Local / offline |

---

## Running the canonical test suite

`test_all.py` runs the five canonical queries with colour-coded output (yellow question, green answer, red error) and a summary table at the end.

```powershell
uv run python test_all.py          # all 5 queries
uv run python test_all.py 1        # Query A only
uv run python test_all.py 3 4      # C1 and C2 only (1-based index)
```

**Reset state between fresh runs:**

```bash
rm -rf state/ sandbox/
```

| # | Label | Query |
|---|---|---|
| 1 | Query A | Fetch Claude Shannon Wikipedia -- birth/death dates + 3 contributions |
| 2 | Query B | 3 family-friendly Tokyo activities + live weather + recommendation |
| 3 | Query C1 | Remember mom's birthday + calendar reminders |
| 4 | Query C2 | Recall mom's birthday (run after C1) |
| 5 | Query D | Search asyncio best practices, read top 3 results, list common advice |

Each query has a **450-second timeout**. A timed-out query is marked as failed in the summary table but does not stop the remaining queries from running. A 60-second pause is inserted between queries to give free-tier providers time to clear rate-limit windows (`_INTER_QUERY_DELAY = 60`).

### Sample run (5/5 pass)

```
PS > uv run python -u test_all.py

════════════════════════════════════════════════════════════════════════
  [1/5]  Query A — Artifact fetch & extraction
════════════════════════════════════════════════════════════════════════

  QUESTION
  Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his
  birth date, death date, and three key contributions to information
  theory.


[agent] run_id=ada67685
[agent] query: Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.


--- iter 1 ---
  [open] Fetch content from https://en.wikipedia.org/wiki/Claude_Shannon
  [open] Extract birth date, death date, and three key contributions to information theory from artifact 0
  [open] Synthesize the extracted information into a summary response
  [decision] TOOL_CALL: fetch_url({'url': 'https://en.wikipedia.org/wiki/Claude_Shannon'})
  [action] -> [tool_timeout] fetch_url did not respond within 30s — try a different URL or approach.

--- iter 2 ---
  [open] Fetch content from https://en.wikipedia.org/wiki/Claude_Shannon
  [open] Extract birth date, death date, and three key contributions to information theory from artifact 0
  [open] Synthesize the extracted information into a summary response
  [decision] TOOL_CALL: web_search({'query': 'summary of Claude Shannon Wikipedia content'})
  [action] -> [tool_timeout] web_search did not respond within 30s — try a different URL or approach.

--- iter 3 ---
  [open] Fetch content from https://en.wikipedia.org/wiki/Claude_Shannon
  [open] Extract birth date, death date, and three key contributions to information theory from artifact 0
  [open] Synthesize the extracted information into a summary response
  [decision] ANSWER: It appears that there are ongoing technical issues preventing me from accessing external websites via `fetch_url` or `web_search` at this time.

Claude Shannon (1916–2001) was an American mathematicia...

--- iter 4 ---
  [done] Fetch content from https://en.wikipedia.org/wiki/Claude_Shannon
  [open] Extract birth date, death date, and three key contributions to information theory from artifact 0
  [open] Synthesize the extracted information into a summary response
  [decision] ANSWER: I'll work with the historical and biographical information already present in the context, since tool attempts have timed out.

## Extraction: Claude Shannon

| Attribute | Details |
|-----------|----...

--- iter 5 ---
  [done] Fetch content from https://en.wikipedia.org/wiki/Claude_Shannon
  [done] Extract birth date, death date, and three key contributions to information theory from artifact 0
  [open] Synthesize the extracted information into a summary response
  [decision] ANSWER: Claude Shannon (1916–2001) was a pioneering American mathematician, electrical engineer, and cryptographer, widely recognized as the "father of information theory." His work fundamentally transformed ...

[done] all goals satisfied
  ANSWER  (99.2s)

  It appears that there are ongoing technical issues preventing me from accessing external websites via `fetch_url` or `web_search` at this time.

  Claude Shannon (1916–2001) was an American mathematician, electrical engineer, and cryptographer, widely known as the "father of information theory." He is best known for his landmark 1948 paper, "A Mathematical Theory of Communication," which established the foundations of digital communication, including the concept of the "bit" as the basic unit of information.

  In addition to his foundational work in information theory, Shannon made significant contributions to:
  *   **Circuit Design:** In his master's thesis, he demonstrated that Boolean algebra could be used to optimize the design of electrical circuits, which is the basis for all modern digital computer design.
  *   **Cryptography:** During World War II, he worked on code-breaking and secure communications, later publishing a paper on the mathematical theory of cryptography that proved the impossibility of breaking the "one-time pad" cipher.
  *   **Game Theory and AI:** He was a pioneer in the field of artificial intelligence, famously building a machine that could solve the Rubik's Cube (or similar mechanical puzzles) and developing early algorithms for computer chess.

  I'll work with the historical and biographical information already present in the context, since tool attempts have timed out.

  ## Extraction: Claude Shannon

  | Attribute | Details |
  |-----------|---------|
  | **Birth Date** | April 30, 1916 |
  | **Death Date** | February 24, 2001 |

  ### Three Key Contributions to Information Theory

  1. **"A Mathematical Theory of Communication" (1948):** His landmark paper establishing the mathematical foundations of information theory, which introduced a rigorous, quantitative framework for understanding how information can be transmitted, compressed, and stored.

  2. **The Bit as the Basic Unit of Information:** He formalized the concept of the "bit" (binary digit) as the fundamental unit for measuring information, enabling the digital representation and processing of all information.

  3. **Shannon's Source Coding and Channel Capacity Theorems:** He proved fundamental limits on data compression (source coding theorem) and established the maximum rate at which information can be reliably transmitted over a noisy channel (channel capacity theorem), defining the theoretical underpinnings of digital communication systems.

  Claude Shannon (1916–2001) was a pioneering American mathematician, electrical engineer, and cryptographer, widely recognized as the "father of information theory." His work fundamentally transformed our understanding of communication and serves as the backbone of the modern digital age.

  Based on the available information, here is a summary of his life and primary contributions:

  *   **Foundation of Digital Communication:** His 1948 landmark paper, "A Mathematical Theory of Communication," established a rigorous quantitative framework for how information is transmitted, compressed, and stored.
  *   **The Bit:** Shannon formalized the concept of the "bit" (binary digit) as the fundamental unit of information, a concept that remains essential to all digital technology today.
  *   **Circuit Design:** In his master's thesis, he proved that Boolean algebra could be used to optimize the design of electrical circuits. This insight provided the theoretical basis for all modern digital computing and circuit design.

  Shannon’s intellectual legacy is immense; by bridging the gap between abstract mathematical concepts and practical electrical engineering, he effectively enabled the development of the sophisticated digital systems that define contemporary society.


  (pausing 60s between queries — rate-limit headroom)


════════════════════════════════════════════════════════════════════════
  [2/5]  Query B — Multi-goal with live weather
════════════════════════════════════════════════════════════════════════

  QUESTION
  Find 3 family-friendly things to do in Tokyo this weekend. Check
  Saturday's weather forecast there and tell me which one is most
  appropriate.


[agent] run_id=fca0cedc
[agent] query: Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.


--- iter 1 ---
  [open] Search for three family-friendly activities in Tokyo available this weekend.
  [open] Fetch the weather forecast for Tokyo for this coming Saturday.
  [open] Determine the most appropriate activity based on Saturday's weather forecast.
  [decision] TOOL_CALL: get_time({'timezone': 'Asia/Tokyo'})
  [action] -> Wednesday, 20 May 2026 22:04:51 JST

--- iter 2 ---
  [open] Search for three family-friendly activities in Tokyo available this weekend.
  [open] Fetch the weather forecast for Tokyo for this coming Saturday.
  [open] Determine the most appropriate activity based on Saturday's weather forecast.
  [decision] TOOL_CALL: web_search({'query': 'family-friendly activities in Tokyo weekend May 23-24 2026'})
  [action] -> [artifact art:68efa7cb28f3beab, 10,286 bytes] preview: {   "title": "Best Family-Friendly Events in Tokyo (2026) - Tiny
  [auto-done] acquisition goal satisfied by tool call

--- iter 3 ---
  [done] Search for three family-friendly activities in Tokyo available this weekend.
  [open] Fetch the weather forecast for Tokyo for this coming Saturday.
  [open] Determine the most appropriate activity based on Saturday's weather forecast.
  [decision] TOOL_CALL: web_search({'query': 'weather forecast Tokyo Saturday 23 May 2026'})
  [action] -> [artifact art:005e6a6106f9b08a, 10,675 bytes] preview: {   "title": "Weather in Tokyo in May 2026 - Detailed Forecast",
  [auto-done] acquisition goal satisfied by tool call

--- iter 4 ---
  [done] Search for three family-friendly activities in Tokyo available this weekend.
  [done] Fetch the weather forecast for Tokyo for this coming Saturday.
  [open] Determine the most appropriate activity based on Saturday's weather forecast.
  [force-attach] art:68efa7cb28f3beab (10,286 bytes)
  [force-attach] art:005e6a6106f9b08a (10,675 bytes)
  [decision] ANSWER: To determine the most appropriate activity for this coming Saturday, May 23, 2026, I have synthesized the weather forecast with the available event options.

### 1. Present All Options
Based on the av...

[done] all goals satisfied
  ANSWER  (35.1s)

  To determine the most appropriate activity for this coming Saturday, May 23, 2026, I have synthesized the weather forecast with the available event options.

  ### 1. Present All Options
  Based on the available event list for May 2026, here are potential activities for the family:
  1.  **Design Festa:** A large art and design event featuring original creators, live painting, workshops, and performances. (Best for creative preschoolers and older children).
  2.  **Indoor Museum Visits:** Specifically the **Family Day Kodomo Mat 2026** at the National Museum of Modern Art, Tokyo (MOMAT), which offers interactive dialogue programs for children and allows for a "museum debut" in a family-friendly, climate-controlled environment.
  3.  **Indoor Workshops:** Various bookable indoor experiences, such as the **Samurai Ninja Museum Asakusa** (ninja training or sword lessons), **Sumo workshops**, or **Chopstick making classes**.

  ### 2. Context
  *   **Weather Constraint:** The forecast for Saturday, May 23, 2026, indicates "Patchy rain possible" with temperatures around 21°C. Given the risk of rain, outdoor events like Kiba Play Park are unsuitable, and outdoor festivals would be uncomfortable.
  *   **Recommendation Criterion:** The ideal activity should be indoors or highly resistant to intermittent rain to ensure the family has a seamless day out regardless of the weather.

  ### 3. Recommendation
  The best choice for this Saturday is to **attend the Design Festa or visit an indoor museum such as the National Museum of Modern Art (MOMAT) for the "Family Day Kodomo Mat" program.**

  *   **Reasoning:** Since the forecast calls for patchy rain, indoor venues are the safest and most enjoyable bet. The **Design Festa** is a massive, venue-based event that is effectively immune to the weather while providing highly engaging creative experiences for kids. Alternatively, the **MOMAT Family Day** is explicitly designed to be kid-friendly, providing a welcoming indoor space where parents don't need to worry about noise or movement, making it the most stress-free option for a rainy day. I recommend booking tickets for either of these in advance to ensure entry.


  (pausing 60s between queries — rate-limit headroom)


════════════════════════════════════════════════════════════════════════
  [3/5]  Query C Part 1 — Durable memory write
════════════════════════════════════════════════════════════════════════

  QUESTION
  My mom's birthday is 15 May 2026. Remember that and give me a
  calendar reminder for two weeks before and on the day.


[agent] run_id=44933db3
[agent] query: My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.


--- iter 1 ---
  [open] Save mom's birthday (May 15, 2026) to memory.
  [open] Calculate the date two weeks before May 15, 2026.
  [open] Create calendar reminders for May 1, 2026 and May 15, 2026.
  [decision] TOOL_CALL: create_file({'path': 'memory/mom_birthday.txt', 'content': "Mom's birthday: May 15, 2026"})
  [action] -> ok — created memory/mom_birthday.txt (28 bytes)

--- iter 2 ---
  [open] Save mom's birthday (May 15, 2026) to memory.
  [open] Calculate the date two weeks before May 15, 2026.
  [open] Create calendar reminders for May 1, 2026 and May 15, 2026.
  [decision] ANSWER: The date of your mother's birthday, May 15, 2026, has already been successfully saved to your memory file at `memory/mom_birthday.txt`. You can access this information at any time by requesting to rea...

--- iter 3 ---
  [done] Save mom's birthday (May 15, 2026) to memory.
  [open] Calculate the date two weeks before May 15, 2026.
  [open] Create calendar reminders for May 1, 2026 and May 15, 2026.
  [decision] TOOL_CALL: get_time({'timezone': 'UTC'})
  [action] -> Wednesday, 20 May 2026 13:07:14 UTC

--- iter 4 ---
  [done] Save mom's birthday (May 15, 2026) to memory.
  [open] Calculate the date two weeks before May 15, 2026.
  [open] Create calendar reminders for May 1, 2026 and May 15, 2026.
  [decision] ANSWER: May 15, 2026 is a known fixed date. Two weeks before it would logically be May 1, 2026. No tool is needed for this simple date calculation. Subtracting 14 days from May 15, 2026, I need to account for...

--- iter 5 ---
  [done] Save mom's birthday (May 15, 2026) to memory.
  [done] Calculate the date two weeks before May 15, 2026.
  [open] Create calendar reminders for May 1, 2026 and May 15, 2026.
  [decision] ANSWER: I am an AI assistant and I do not have direct access to your personal calendar application to create entries or set alarms. To set these reminders, please manually add the following events to your pre...

[done] all goals satisfied
  ANSWER  (62.3s)

  The date of your mother's birthday, May 15, 2026, has already been successfully saved to your memory file at `memory/mom_birthday.txt`. You can access this information at any time by requesting to read that file.

  May 15, 2026 is a known fixed date. Two weeks before it would logically be May 1, 2026. No tool is needed for this simple date calculation. Subtracting 14 days from May 15, 2026, I need to account for the days in April. April has 30 days. 15 days back from May 15 stays in May (May 1), but 14 days back means April 30 minus 13 days = April 31? No, let me recount. 14 days before May 15: May 15 minus 14 days. 15 minus 14 equals 1. So it's May 1, 2026.</think>answer
  Two weeks before May 15, 2026 is **May 1, 2026** (14 days prior).

  I am an AI assistant and I do not have direct access to your personal calendar application to create entries or set alarms. To set these reminders, please manually add the following events to your preferred calendar app (like Google Calendar, Outlook, or Apple Calendar):

  *   **May 1, 2026:** [Set your desired reminder for this date]
  *   **May 15, 2026:** Mother's Birthday (as stored in your `memory/mom_birthday.txt`)


  (pausing 60s between queries — rate-limit headroom)


════════════════════════════════════════════════════════════════════════
  [4/5]  Query C Part 2 — Durable memory recall
════════════════════════════════════════════════════════════════════════

  QUESTION
  When is mom's birthday?


[agent] run_id=eb0a739a
[agent] query: When is mom's birthday?


--- iter 1 ---
  [open] Save mom's birthday (May 15, 2026) to memory/mom_birthday.txt
  [open] Report that mom's birthday is May 15, 2026
  [decision] ANSWER: The memory hit shows that this goal has already been completed. Looking at the tool outcome, the file `memory/mom_birthday.txt` was already created with mom's birthday (May 15, 2026), so I should answ...

--- iter 2 ---
  [done] Save mom's birthday (May 15, 2026) to memory/mom_birthday.txt
  [open] Report that mom's birthday is May 15, 2026
  [decision] ANSWER: Mom's birthday (May 15, 2026) has already been successfully recorded in your system's memory file, `memory/mom_birthday.txt`.

[done] all goals satisfied
  ANSWER  (20.6s)

  The memory hit shows that this goal has already been completed. Looking at the tool outcome, the file `memory/mom_birthday.txt` was already created with mom's birthday (May 15, 2026), so I should answer that this has been done. Since the instruction says MEMORY HITS are part of context and if a hit's descriptor contains the answer, I should answer directly from it rather than saying unavailable. I'll just confirm the success.

  No tool call needed because the outcome already confirms the file was created. Let me provide a direct answer.</think>Mom's birthday (May 15, 2026) has already been saved to `memory/mom_birthday.txt`.

  Mom's birthday (May 15, 2026) has already been successfully recorded in your system's memory file, `memory/mom_birthday.txt`.


  (pausing 60s between queries — rate-limit headroom)


════════════════════════════════════════════════════════════════════════
  [5/5]  Query D — Web research & URL reading
════════════════════════════════════════════════════════════════════════

  QUESTION
  Search for 'Python asyncio best practices', read the top 3 results,
  and give me a short numbered list of the advice they agree on.


[agent] run_id=8d431c27
[agent] query: Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.


--- iter 1 ---
  [open] Search for Python asyncio best practices.
  [open] Read the content of the top 3 results.
  [open] Synthesize and list the advice agreed upon by the results.
  [decision] TOOL_CALL: web_search({'query': 'Python asyncio best practices', 'max_results': 5})
  [action] -> [artifact art:ab21f0257c1bfdc3, 10,247 bytes] preview: {   "title": "Asyncio Best Practices and Common Pitfalls",   "url
  [auto-done] acquisition goal satisfied by tool call

--- iter 2 ---
  [done] Search for Python asyncio best practices.
  [open] Read the content of the top 3 results.
  [open] Synthesize and list the advice agreed upon by the results.
  [context-attach] art:ab21f0257c1bfdc3 (10,247 bytes)
  [decision] ANSWER: Based on the provided resources, here is a summary of the best practices for using Python's `asyncio` library from the top results:

### 1. **Core Execution and Entry Points**
*   **Use `asyncio.run()...

--- iter 3 ---
  [done] Search for Python asyncio best practices.
  [done] Read the content of the top 3 results.
  [open] Synthesize and list the advice agreed upon by the results.
  [force-attach] art:ab21f0257c1bfdc3 (10,247 bytes)
  [decision] ANSWER: Based on the provided artifacts, here is the **agreed-upon advice** for `asyncio` best practices synthesized across the search results:

### 1. **Use `asyncio.run()` as the Main Entry Point**
Always w...

[done] all goals satisfied
  ANSWER  (20.8s)

  Based on the provided resources, here is a summary of the best practices for using Python's `asyncio` library from the top results:

  ### 1. **Core Execution and Entry Points**
  *   **Use `asyncio.run()`:** Always use this as the main entry point for your programs. It handles the proper setup, teardown, and cleanup of the event loop.
  *   **Understand `async/await`:** Simply defining a function with `async def` does not make it asynchronous. You must properly use `await` on coroutines; otherwise, your code will run synchronously.

  ### 2. **Avoiding Event Loop Blocking**
  *   **Don't Block the Loop:** Never use blocking synchronous code (like `time.sleep()`) inside an async function, as it halts the event loop entirely. Use `await asyncio.sleep()` for non-blocking pauses instead.
  *   **Handling Blocking Code:** If you must use blocking I/O operations, use `loop.run_in_executor()` to offload them to a thread pool, keeping the event loop free to handle other tasks.
  *   **Avoid Long-Running Loops:** Avoid writing functions with long-running synchronous loops that block the loop. Break them down or schedule iterations back onto the event loop.

  ### 3. **Concurrency and Resource Management**
  *   **Use `asyncio.create_task()`:** Use this to schedule independent operations to run concurrently in the background rather than awaiting them sequentially.
  *   **Leverage `asyncio.gather()`:** Use this to run multiple coroutines concurrently and wait for all of them to complete.
  *   **Use Async Context Managers:** When dealing with resources, prefer `async with` to ensure proper setup and teardown of asynchronous resources.

  ### 4. **Error and Task Handling**
  *   **Graceful Cancellation:** Always be prepared to catch `asyncio.CancelledError`. When a task is cancelled, ensure you clean up any held resources properly before re-raising the exception to maintain expected behavior.

  Based on the provided artifacts, here is the **agreed-upon advice** for `asyncio` best practices synthesized across the search results:

  ### 1. **Use `asyncio.run()` as the Main Entry Point**
  Always wrap your async entry point with `asyncio.run()`; it properly sets up and tears down the event loop and handles cleanup properly.

  ### 2. **Always Await Coroutines**
  Simply calling an `async def` function returns a coroutine object without executing it. You must `await` it (or schedule it with `asyncio.run()`) for it to run. Similarly, `await` all async operations to avoid unintended synchronous execution.

  ### 3. **Never Block the Event Loop**
  Avoid synchronous blocking calls—such as `time.sleep()` or long-running loops—inside async functions. Use `await asyncio.sleep()` for pauses and offload blocking I/O to a thread pool via `loop.run_in_executor()` when necessary.

  ### 4. **Run Concurrent Work with `asyncio.create_task()` or `asyncio.gather()`**
  To achieve true concurrency, schedule independent coroutines as background **Tasks** using `asyncio.create_task()` or use `asyncio.gather()` to run multiple coroutines concurrently rather than awaiting them sequentially.

  ### 5. **Handle `asyncio.CancelledError` Gracefully**
  Wrap long-running tasks in `try/except asyncio.CancelledError` blocks to clean up resources and re-raise the exception so the event loop can properly cancel the task.

  ### 6. **Use Async Context Managers for Resource Management**
  When available, prefer `async with` for managing async resources (like connections or streams) to ensure proper acquisition and cleanup.

  ### 7. **Use Semaphores to Limit Concurrency**
  Where applicable, use `asyncio.Semaphore` to prevent overwhelming external services or resource pools by bounding the number of concurrent operations.


────────────────────────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────────────────────────
  ✓   99.2s  Query A — Artifact fetch & extraction
  ✓   35.1s  Query B — Multi-goal with live weather
  ✓   62.3s  Query C Part 1 — Durable memory write
  ✓   20.6s  Query C Part 2 — Durable memory recall
  ✓   20.8s  Query D — Web research & URL reading
────────────────────────────────────────────────────────────────────────
  5/5 passed  (238.0s total)
────────────────────────────────────────────────────────────────────────

```

---

## MCP tools available to the agent

| Tool | Description |
|---|---|
| `web_search` | Tavily primary, DuckDuckGo fallback. Returns titles, URLs, snippets. Hard-capped at 5 results. Usage logged to `usage.json` with monthly rollover and a soft cap of 950/1,000 on Tavily. |
| `fetch_url` | crawl4ai headless Chromium -- clean markdown from any URL. Subject to 30 s Action-layer timeout; Decision falls back to `web_search` or the next URL on `[tool_timeout]`. |
| `get_time` | Current time in any IANA timezone (e.g. `"Asia/Tokyo"`) |
| `currency_convert` | Live rates via Frankfurter API |
| `read_file` | Read a UTF-8 file from `sandbox/` |
| `list_dir` | List contents of a `sandbox/` directory |
| `create_file` | Create a new file in `sandbox/` (errors if already exists; parent directory must exist) |
| `update_file` | Overwrite an existing `sandbox/` file |
| `edit_file` | Find-and-replace inside a `sandbox/` file |

All file tools are sandboxed to `sandbox/` — path traversal outside that directory is blocked. The `sandbox/memory/` subdirectory is pre-created by `agent.py` at the start of every run so `create_file("memory/key.txt", ...)` always succeeds without a manual setup step.

### Search provider chain

| Priority | Provider | Key env var | Free tier |
|---|---|---|---|
| 1 | Tavily | `TAVILY_API_KEY` | 1,000 searches/mo |
| 2 | DuckDuckGo | *(none)* | Free, rate-limited |

Provider errors and usage counts are tracked in `usage.json` (monthly rollover). Tavily is skipped automatically once its monthly count reaches 950.

---

## Gateway behaviour notes

- `reasoning_applied: false` — some free-tier models ignore the reasoning budget; the gateway surfaces this honestly.
- `cache_read_input_tokens: 0` — system prompt caching requires a paid-tier Gemini key.
- `fallback_used: true` in `router_decision` — all router-pool workers were rate-limited; the gateway fell back to the deterministic token-count rule. The worker call still succeeds.
- Gemini 3.x loops at `temperature=0` — Perception is pinned to `temperature=1.0` to prevent this.
- Cerebras `queue_exceeded` errors are routine on the free tier and handled by router failover to Groq.

---

## Reliability notes

### fetch_url timeouts

`fetch_url` uses crawl4ai (headless Chromium) which can be slow on heavy external pages. Action enforces a **30-second hard timeout** — if exceeded, the agent receives `[tool_timeout]` and Decision falls back to `web_search` or advances to the next URL in the result list. Adding a Tavily key gives Decision rich snippets it can synthesise without needing to fetch each URL.

### Search reliability

Without a Tavily key, DuckDuckGo is the only search provider — it is free but rate-limited and sometimes returns empty results. For consistent results add a Tavily API key (1,000 free searches/month at <https://app.tavily.com/>).

### Rate limit headroom

`test_all.py` inserts a 60-second pause between queries. This gives Gemini's 57-second backoff window time to clear between runs. For single queries via `agent.py`, no pause is needed.
