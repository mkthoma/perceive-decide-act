# perceive-decide-act

Lightweight agentic framework built on a four-role cognitive loop — **Memory → Perception → Decision → Action** — with MCP tool integration, artifact-aware goal tracking, and a multi-provider LLM gateway.

## Getting started

### 1. Prerequisites

- **Python ≥ 3.11** and **[uv](https://docs.astral.sh/uv/)**
- **At least one provider API key** (Gemini free tier is enough to start)

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in at least one worker provider key and at least one search provider key.

```bash
# Minimum viable .env — one LLM worker + one search provider
GEMINI_API_KEY=your_gemini_api_key_here

# Search chain: Tavily → Exa → Firecrawl → DuckDuckGo (free, no key)
# Add at least one for reliable search results.
TAVILY_API_KEY=your_tavily_api_key_here   # recommended: 1,000 free/mo
EXA_API_KEY=your_exa_api_key_here         # 1,000 free/mo
FIRECRAWL_API_KEY=your_firecrawl_api_key_here  # 500 credits/mo
```

The `.env` file lives at the **repo root** — one level above `llm_gatewayV3/`. The gateway reads `../.env` relative to its own directory, so a single file covers both.

> **Search reliability:** Without any search key, `web_search` falls through to DuckDuckGo (free, no key required) which works but is rate-limited and unreliable. Adding a Tavily key is the quickest fix — free tier at <https://app.tavily.com/>.

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

The gateway runs **in-process** — no separate server needed. Provider adapters
from `llm_gatewayV3/` are imported directly and all routing, rate-limiting, and
failover happens in the same process.

---

## Architecture

The agent runs a bounded loop of at most 20 iterations. Each iteration executes the four roles in strict order:

```
┌─────────────────────────────────────────────────────┐
│                    agent.py loop                    │
│                                                     │
│  ┌──────────┐   hits    ┌────────────┐              │
│  │  Memory  │──────────▶│ Perception │              │
│  └──────────┘           └─────┬──────┘              │
│       ▲                       │ Observation         │
│       │ record_outcome        ▼                     │
│  ┌────┴─────┐          ┌────────────┐               │
│  │  Action  │◀─────────│  Decision  │               │
│  └──────────┘ ToolCall └────────────┘               │
│       │                                             │
│       └── MCP tool result ──▶ memory + history      │
└─────────────────────────────────────────────────────┘
```

The loop terminates when all goals are marked `done`, or when `MAX_ITERATIONS` (20) is reached.

**Perception is called only on the first iteration.** On iter 1 it decomposes the query into goals; from iter 2 onward the main loop reuses `prior_goals` directly and skips the Perception LLM call entirely. Goal done-flags are set in-place by `agent.py` the moment Decision emits an answer or a data-retrieval tool call completes an acquisition goal. This saves 1–2 LLM calls per query on free-tier providers.

---

## The four roles

### Memory — `memory.py`

**What it does:** A typed, persistent fact store that gives every iteration access to relevant knowledge from prior runs — without re-fetching or re-computing it.

**How it works:**

1. At the start of every iteration, `memory.read(query, history)` performs a **keyword-overlap search** (no LLM, no cost) against all stored items. It tokenizes the query and the last six history entries, removes stopwords, then scores each stored item by how many of its keywords appear in that combined set. Top-8 hits are returned.

2. `memory.remember(text)` is called once at run start with the raw user query. This fires **one LLM gateway call** to classify the text — assigning a `kind` (`fact`, `preference`, `tool_outcome`, `scratchpad`), extracting 3–10 search keywords, writing a human-readable `descriptor`, and building a structured `value` payload. The result is appended to `state/memory.json` and persists across process restarts.

3. `memory.record_outcome(tool_call, result_text, artifact_id)` is called after every Action with zero LLM cost. It extracts keywords from the tool name and arguments, stores the result preview, and — critically — stores the `artifact_id` when the tool returned a large payload. This is how Perception learns that an artifact exists for a given goal.

**Why it matters:** Without memory, each run starts cold. A fact remembered in run 1 ("mom's birthday is June 15") is instantly available as a keyword hit in run 2 — the agent answers without any tool call. Artifact IDs in tool-outcome records are how Perception decides which artifact to attach to an extraction goal, decoupling the fetch step from the read step across iterations.

**Stored at:** `state/memory.json` — plain JSON array of `MemoryItem` objects.

---

### Perception — `perception.py`

**What it does:** The orchestrator. It decomposes the user query into an ordered list of goals on the **first iteration only**. From iter 2 onward the main loop reuses `prior_goals` directly and skips the Perception LLM call.

**How it works:**

1. **First iteration** (`prior_goals` is empty): Perception sends the query, memory hits, and an empty prior-goals list to Gemini with `temperature=1.0`. The LLM decomposes the query into 1–4 short imperative goals ordered by logical dependency — fetch before extract, search before synthesize.

2. **Subsequent iterations**: `agent.py` reuses the existing `prior_goals` list without an LLM call. Done-flags are set immediately in the main loop whenever Decision emits an answer or a data-retrieval tool auto-completes an acquisition goal — there is no need for a second LLM pass to re-inspect history.

3. **Artifact attachment**: For the first unfinished goal that requires reading a previously fetched artifact (e.g. "extract info from page"), Perception sets `artifact_index` pointing to the corresponding `[artifact N]` entry in the memory-hit list. The main loop resolves this index to the actual artifact ID stored by memory.

**Why it matters:** Calling Perception only once (instead of every iteration) saves 1–3 LLM calls per query on free-tier providers whose rate limits are the primary bottleneck. Goal ordering is stable because positional identity is set at decomposition time and never changes. `temperature=1.0` prevents Gemini 3.x from stalling in a low-entropy loop.

**Fallback:** If the Gemini call fails, Perception returns prior goals unchanged (or a single bare goal on the first iteration). The run degrades gracefully rather than crashing.

#### System prompt

```
You are PERCEPTION, the goal-tracking orchestrator in an agentic loop.

TASK EACH ITERATION:
1. If PRIOR GOALS is empty → decompose QUERY into 1-4 short imperative goals
   (each ≤ 15 words). Order them by logical dependency (fetch before extract,
   search before synthesize, etc.).
2. If PRIOR GOALS is non-empty → output those goals in the EXACT SAME ORDER.
   Set `done: true` only for goals where HISTORY shows a satisfying result.
   Once done, a goal stays done forever.
3. For the FIRST UNFINISHED GOAL: if completing it requires reading an artifact
   fetched in a prior iteration (e.g., to extract info from a page), set
   `artifact_index` to the integer shown as [artifact N] in MEMORY HITS.
   Otherwise set `artifact_index` to null.

RULES:
- Preserve goal order exactly. Never reorder, insert, or drop goals.
- artifact_index must reference one of the [artifact N] labels in MEMORY HITS.
  Do not guess or invent an index.
- Mark a goal done ONLY if HISTORY contains an action or answer that directly
  satisfies it.
- Return ONLY valid JSON matching the schema. No prose or commentary.
```

#### PoP validation

```json
{
  "prompt_id": "perception_system_v1",
  "role": "PERCEPTION",
  "evaluated_at": "2026-05-18",
  "criteria": {
    "role_definition": {
      "score": 5,
      "max": 5,
      "note": "Role is clearly named ('PERCEPTION, the goal-tracking orchestrator') and its purpose is stated in the opening line."
    },
    "task_specification": {
      "score": 5,
      "max": 5,
      "note": "Two distinct modes (first call vs. subsequent) are enumerated with clear conditionals and examples ('fetch before extract')."
    },
    "output_format": {
      "score": 5,
      "max": 5,
      "note": "'Return ONLY valid JSON matching the schema. No prose or commentary.' is unambiguous. Schema is injected via gateway response_model, not inlined, keeping the prompt concise."
    },
    "constraint_coverage": {
      "score": 4,
      "max": 5,
      "note": "Four explicit rules cover ordering, artifact_index bounds, done-inference, and output purity. Gap: no instruction for queries that naturally decompose into >4 goals — LLM may silently truncate."
    },
    "hallucination_guards": {
      "score": 5,
      "max": 5,
      "note": "'Do not guess or invent an index' directly blocks artifact_index hallucination. Sticky-done ('Once done, a goal stays done forever') prevents goal-state drift."
    },
    "edge_case_handling": {
      "score": 4,
      "max": 5,
      "note": "LLM returning fewer goals than prior is handled in Python (safety net appends missing goals). The known weak point — LLM-inferred done flags — is patched by agent.py setting goal.done = True immediately on ANSWER."
    },
    "ambiguity_risk": {
      "level": "LOW",
      "note": "Positional identity ('exact same order') removes goal-matching ambiguity. Done-inference from history is inherently fuzzy but is mitigated by the Python sticky-done guard."
    },
    "token_efficiency": {
      "score": 4,
      "max": 5,
      "note": "~310 tokens. Concise for the constraint surface covered. 'fetch before extract, search before synthesize' helpfully pre-constrains ordering without extra tokens."
    }
  },
  "overall": {
    "pass": true,
    "aggregate_score": 4.6,
    "verdict": "Production-ready. Core design is sound — positional identity, sticky-done, and artifact-index guards prevent the main failure modes. The one structural weakness (done-flag inference) is now mitigated at the agent loop level."
  },
  "open_issues": [
    {
      "severity": "LOW",
      "description": "No explicit instruction for queries that require >4 goals. LLM may silently bundle or truncate sub-tasks beyond the 1-4 range."
    },
    {
      "severity": "LOW",
      "description": "artifact_index is 1-based in the prompt but resolved in Python code. A convention mismatch would silently produce a null artifact attachment instead of an error."
    }
  ]
}
```

---

### Decision — `decision.py`

**What it does:** Given one goal and its context, produces exactly one output: either a direct **answer** (text) or a single **tool call**. It never does both.

**How it works:**

1. Decision builds a user message containing the current goal, memory hits, recent history (last 10 entries), and — when the main loop has attached one or more artifacts — the full bytes rendered inline as `ATTACHED ARTIFACTS`. For synthesis goals, `agent.py` force-attaches **all artifacts produced during the current run** (up to 3) so Decision never needs to re-fetch data it already has. Only artifacts from the current run's history are eligible — the memory-hit fallback was removed to prevent stale cross-query artifacts from being injected.

2. It sends this message to the gateway with `auto_route="decision"` and the full MCP tool list. The gateway classifies the request size (TINY or LARGE), selects a worker tier, and dispatches to the first available provider.

3. The response is inspected for tool calls first. If the LLM emitted a function call, a `ToolCall` is returned. Otherwise, the text content becomes the answer.

**Key rules enforced by the system prompt:**
- If history already contains the information, answer directly — do not call a tool again.
- Artifact handles (`art:…`) are never valid arguments to a tool — bytes are available inline in ATTACHED ARTIFACTS.
- Extraction, synthesis, and comparison goals must produce a substantive answer (≥ 3 sentences or a numbered list).
- If a goal asks for an action with no matching tool (set a reminder, send an email, etc.), answer with a clear text description rather than looping.
- If a MEMORY HIT descriptor already contains the answer, answer directly from it — no tool call needed.
- For recalling previously saved facts, call `read_file(path="memory/...")`. If memory hits show a `memory/` file was written, read it before answering.
- Prefer `web_search` over `fetch_url` by default; only call `fetch_url` when the full rendered content of a specific page is needed and search snippets are insufficient.
- After 3 consecutive empty `web_search` results for the same goal, stop searching and answer from training knowledge.
- **Never output `__NO_ANSWER__` or any single-word placeholder.** Always produce a substantive answer or a single tool call.
- **If ATTACHED ARTIFACTS are present and HISTORY shows a fetch/search already ran for this goal**, synthesize directly from the artifact — do not re-fetch and do not output a placeholder.

**`__NO_ANSWER__` sentinel guard (`agent.py`):**
Some models emit the literal string `__NO_ANSWER__` when they receive an attached artifact for a combined "Fetch X and tell me Y" goal and aren't sure they should synthesize. `agent.py` intercepts this before it can be stored as the final answer: it injects a `[STOP]` hint into history (`"ATTACHED ARTIFACTS contain the requested data. Extract the information and produce a real answer NOW"`) and retries the iteration without marking the goal done. The existing `[STOP]`-line rule in the Decision prompt then forces the model to answer from the artifact on the next pass.

**Why it matters:** Separating Decision from Action ensures the LLM never directly executes code. Decision emits intent (`ToolCall`); Action executes it through MCP. This also means Decision can be retried or replaced independently without touching the execution layer. The `auto_route` lets the gateway pick the cheapest provider that fits the request size — a small fetch decision goes to a fast TINY-tier model; an extraction decision with 50 KB of attached content goes to a large-context LARGE-tier provider.

#### System prompt

```
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
- NEVER output "__NO_ANSWER__", "N/A", "NONE", or any single-word placeholder
  as a standalone response. Always produce either a substantive text answer
  (at least one full sentence) or a single tool_call.
- If ATTACHED ARTIFACTS are present AND HISTORY shows a fetch or search tool
  already returned results for this goal, synthesize your answer directly from
  the artifact content — do not output a placeholder or call a tool again.
- For real-time data (current time, live exchange rates, today's weather),
  ALWAYS call the appropriate tool.
- For WEATHER data, use web_search. Do NOT use fetch_url for weather.
- For extraction, list, comparison, recommendation, or synthesis goals: your answer
  must be substantive — at least 3 sentences or a numbered/bulleted list of ≥ 3 items.
- If HISTORY already contains a tool result for this goal, answer from that result
  directly — do not call the same tool again.
- If HISTORY shows 3 or more consecutive web_search results with "No results found"
  for the same goal, STOP searching and answer from your own knowledge.
- If HISTORY contains "[SEARCH_EXHAUSTED:" for this goal, answer from your own
  knowledge — do NOT call web_search or fetch_url again.
- Prefer web_search over fetch_url by default.
- If HISTORY contains ANY [tool_timeout] result, switch to web_search immediately.
- When the user asks to "remember" something, use create_file(path="memory/{key}.txt").
- To RECALL a previously remembered fact, call read_file on the memory/ path.
```

#### PoP validation

```json
{
  "prompt_id": "decision_system_v2",
  "role": "DECISION",
  "evaluated_at": "2026-05-20",
  "criteria": {
    "role_definition": {
      "score": 5,
      "max": 5,
      "note": "Role is clearly named ('DECISION, the action selector') and the binary output space (answer | tool_call) is stated immediately."
    },
    "task_specification": {
      "score": 5,
      "max": 5,
      "note": "Two mutually exclusive outputs are defined with explicit conditions for choosing each. 'EXACTLY ONE' leaves no ambiguity."
    },
    "output_format": {
      "score": 4,
      "max": 5,
      "note": "Tool-call format is enforced by gateway function-calling schema injection, not inlined in the prompt. The answer field has no explicit type constraint — relies on schema injection and the 'at least 3 sentences' quality rule."
    },
    "constraint_coverage": {
      "score": 5,
      "max": 5,
      "note": "Eight constraints cover mutual exclusion, artifact handle misuse, answer quality floor, history-first rule, tool selection preference, output purity, placeholder prohibition, and artifact-present synthesis. Two new rules added v2: never output __NO_ANSWER__ placeholders; if artifacts are attached and history shows a prior fetch/search, synthesize directly."
    },
    "hallucination_guards": {
      "score": 5,
      "max": 5,
      "note": "art: handle guard names the exact failure mode; reinforced by Action's runtime check. New in v2: explicit '__NO_ANSWER__' prohibition closes the sentinel-as-answer failure mode where some models emit that literal string when confused by an attached artifact. The agent loop also intercepts it and injects a [STOP] hint as a secondary safeguard."
    },
    "edge_case_handling": {
      "score": 5,
      "max": 5,
      "note": "History-first rule prevents redundant tool calls. New in v2: the 'artifacts present + prior fetch = synthesize now' rule handles combined fetch+tell goals (e.g. 'Fetch X and tell me Y') where the model previously saw the artifact but returned a placeholder instead of synthesizing. The [STOP] hint injected by agent.py closes the retry loop."
    },
    "ambiguity_risk": {
      "level": "LOW",
      "note": "'NEVER return both' is unambiguous. 'At least 3 sentences or ≥ 3 items' gives a concrete, measurable quality bar. The only subjective element is 'most specific tool' — acceptable given the small tool set."
    },
    "token_efficiency": {
      "score": 4,
      "max": 5,
      "note": "~210 tokens (up from ~170 in v1). Two new rules add ~40 tokens. The cost is justified: both new rules prevent failure modes observed in production that were not recoverable without them."
    }
  },
  "overall": {
    "pass": true,
    "aggregate_score": 4.8,
    "verdict": "Production-ready. Eight STRICT RULES now cover all observed failure modes including the __NO_ANSWER__ sentinel and the artifact-present synthesis case. The agent loop adds a complementary runtime guard so no single point of failure can silently store a placeholder as the final answer."
  },
  "open_issues": [
    {
      "severity": "LOW",
      "description": "No guidance for goals requiring aggregation from multiple tool calls. Decision emits one tool call per iteration; complex synthesis goals may require more iterations than estimated at planning time."
    },
    {
      "severity": "LOW",
      "description": "The '≥ 3 sentences' quality rule applies to all answer responses, including simple factual goals. This can produce unnecessarily verbose output for quick lookups."
    }
  ]
}
```

---

### Action — `action.py`

**What it does:** Pure MCP dispatch. No LLM calls. Takes a `ToolCall`, executes it through the MCP `ClientSession`, and returns a `(descriptor, artifact_id)` pair.

**How it works:**

1. **Guard check**: Before any dispatch, Action inspects the tool arguments for `art:…` handles in path/URL fields. If found, it returns an error descriptor immediately — this prevents Decision from accidentally passing a stale artifact handle to a tool that expects a real URL or file path.

2. **MCP dispatch**: Calls `session.call_tool(name, arguments)` wrapped in `asyncio.wait_for` with a **30-second timeout**. If the tool does not respond in time (e.g. `crawl4ai` rendering a large page), Action returns a `[tool_timeout]` error descriptor. The agent loop treats this as an error, and Decision falls back to `web_search` for the information instead.

3. **JSON post-processing**: The MCP server returns typed Python objects (`list[dict]`, `dict`) which FastMCP serialises as JSON strings. Action unwraps these into clean human-readable text before passing them to Decision — search results become `Title / URL / Snippet` blocks, file tool responses become plain file content, etc.

4. **Artifact threshold**: If the tool response exceeds `ARTIFACT_THRESHOLD_BYTES` (4 KB), the bytes are written to the content-addressable artifact store (`state/artifacts/`). The returned descriptor includes the artifact handle (`art:<sha256[:16]>`) and a 200-character preview. If the response is small enough, it is returned as raw text with no artifact created.

**Content size is bounded at the Action layer.** `fetch_url` in `mcp_server.py` hard-caps all fetched content at **20,000 characters** (`_MAX_FETCH_CHARS`). This is the canonical control point — it determines the maximum bytes stored in any artifact. The `_MAX_ARTIFACT_CHARS = 50,000` limit in `decision.py` is a secondary safety net only and should never be reached in normal operation.

**Why it matters:** The 20 KB content cap means a Wikipedia article that is 80 KB raw is truncated to the infobox and opening sections (~20 KB) before it becomes an artifact — sufficient for any fact-extraction task. When the artifact is later force-attached to a synthesis or extraction goal, Decision receives at most ~5,000 tokens of artifact content, well within every provider's context window. The `art:` guard closes the loop: Decision is told not to pass handles to tools, and Action enforces that rule at execution time.

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
# memory.py — classify a raw text into a typed memory item
resp = await gw.chat(messages, response_model=_Classification, ...)
cls = gw.parse_model(resp, _Classification)   # raises ValueError on parse failure
```

### 3. Perception's goal-list schema (`perception.py`)
`_GoalSlot` and `_PerceptionResponse` are private models used the same way. The gateway sends Gemini a JSON schema derived from `_PerceptionResponse`, and Perception validates the response before converting it into public `Goal` objects with stable IDs.

```python
# perception.py — parse Gemini's goal list into typed objects
perc = gw.parse_model(resp, _PerceptionResponse)
for slot in perc.goals:
    new_goals.append(Goal(id=..., text=slot.text, done=slot.done, ...))
```

### 4. Persistence round-trips (`memory.py`)
The memory store serializes and deserializes using Pydantic's native JSON support:

```python
# Save — convert to JSON-safe dicts
json.dumps([item.model_dump(mode="json") for item in self._items])

# Load — validate each raw dict back into a MemoryItem
self._items = [MemoryItem.model_validate(r) for r in raw]
```

This means any schema evolution (adding an optional field) is handled automatically — old records load cleanly because Pydantic applies defaults for missing fields.

### 5. Gateway structured-output plumbing (`llm_gateway.py`)
When `response_model` is passed to `gw.chat()`, the gateway strips Pydantic's generated `title` fields (which some providers reject), wraps the schema in a `json_schema` response-format envelope, and validates the response. `gw.parse_model()` handles the three fallback cases: pre-validated `parsed` dict, direct JSON parse, and regex-extracted JSON object.

---

## File layout

```
perceive-decide-act/
├── agent.py            Main loop (Memory → Perception → Decision → Action)
├── schemas.py          Pydantic v2 contracts for all role boundaries
├── memory.py           Typed fact store — keyword search, LLM classification
├── perception.py       Goal decomposition and done-flag tracking (Gemini)
├── decision.py         Action selection — answer or single tool call
├── action.py           MCP dispatch with 30s timeout, JSON post-processing, artifact threshold
├── artifact_store.py   Content-addressable store (state/artifacts/)
├── llm_gateway.py      In-process LLM gateway — routing, failover, rate limits
├── mcp_server.py       9 MCP tools — Tavily → Exa → Firecrawl → DDG search chain
├── test_all.py         Colour-coded runner for the 5 canonical test queries
├── pyproject.toml      uv project config
├── .env.example        Environment variable template
└── llm_gatewayV3/      Gateway source (see llm_gatewayV3/README.md)
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
# Worker pool — providers, models, rate limits
curl -s http://localhost:8101/v1/providers | python3 -m json.tool

# Router pool — four small LLMs that classify request size
curl -s http://localhost:8101/v1/routers | python3 -m json.tool

# Live rate state
curl -s http://localhost:8101/v1/status | python3 -m json.tool
```

### How routing works

When `agent` makes a gateway call tagged `auto_route="decision"` (or `"perception"` / `"memory"`), the gateway:

1. Estimates the token count of the request
2. Sends a bounded `{token_count, sample}` envelope to a router LLM (Cerebras → Groq → NVIDIA → GitHub, first available)
3. Router responds with one word: `TINY` (< 1K tokens) or `LARGE` (1K–8K tokens)
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
| GitHub Models | 10–15 RPM | Low-volume fallback |
| OpenRouter | 20 RPM / 50 RPD | Additional coverage |
| Ollama | unlimited | Local / offline |

---

## Running the four target queries

**Reset state between fresh attempts:**
> ```bash
> rm -rf state/
> ```

### Query A — Shannon Wikipedia (artifact attach test)

```bash
uv run python agent.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

**What to expect:**
```
[agent] run_id=595cd89e
[agent] query: Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.


--- iter 1 ---
  [open] Extract birth date and death date of Claude Shannon from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [open] Extract three key contributions to information theory from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [open] Synthesize the extracted information into a final response.
  [attach] art:e40c62c34c2a7ce4 (20,326 bytes)
  [decision] ANSWER: Claude Shannon was born onApril 30, 1916. He died on February 24, 2001. These dates show that he lived for 84 years.

--- iter 2 ---
  [done] Extract birth date and death date of Claude Shannon from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [open] Extract three key contributions to information theory from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [open] Synthesize the extracted information into a final response.
  [attach] art:e40c62c34c2a7ce4 (20,326 bytes)
  [decision] ANSWER: __NO_ANSWER__

--- iter 3 ---
  [done] Extract birth date and death date of Claude Shannon from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [done] Extract three key contributions to information theory from [artifact 1].  attach=art:e40c62c34c2a7ce4
  [open] Synthesize the extracted information into a final response.
  [force-attach] art:e40c62c34c2a7ce4 (20,326 bytes)
  [decision] ANSWER: Claude Shannon (April 30 1916 – February 24 2001) was an American polymath who is widely regarded as the “father of information theory” and a foundational figure in the digital age. His 1937 master’s ...

[done] all goals satisfied
  ANSWER  (66.9s)

  Claude Shannon was born onApril 30, 1916. He died on February 24, 2001. These dates show that he lived for 84 years.

  ---

  __NO_ANSWER__

  ---

  Claude Shannon (April 30 1916 – February 24 2001) was an American polymath who is widely regarded as the “father of information theory” and a foundational figure in the digital age. His 1937 master’s thesis demonstrated how Boolean algebra could be implemented with electrical switching circuits, laying the theoretical groundwork for modern digital computers and earning him the Alfred Noble Prize. Shannon’s seminal 1948 paper “A Mathematical Theory of Communication” introduced the concept of entropy as a measure of information, formalized the bit, and established the blueprint for everything from data compression to the Internet, while his wartime work pioneered modern cryptography and his contributions to artificial intelligence include the early Theseus learning machine.

```

`[goal-attach]` fires because the fetch history entry for this goal carried an `artifact_id`. Decision receives the page bytes inline and answers directly. If a weaker model emits `__NO_ANSWER__` on iter 2, the sentinel guard injects a `[STOP]` hint; iter 3 picks up that hint and synthesizes the answer. Fetch content is hard-capped at **20,000 characters** in `mcp_server.py`.

---

### Query B — Tokyo activities (multi-goal + web search)

```bash
uv run python agent.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
```

**What to expect:**
```

[agent] run_id=de6ddfa2
[agent] query: Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.


--- iter 1 ---
  [open] Identify 3 family-friendly activities in Tokyo using information from artifact 2.  attach=art:027d5e4301501189
  [open] Check Saturday's weather forecast for Tokyo using information from artifact 1.  attach=art:70c95efbcbab4469
  [open] Determine the most appropriate activity based on the weather forecast.
  [attach] art:027d5e4301501189 (11,438 bytes)
  [decision] ANSWER: Based on the information provided in the artifacts, here are three family-friendly activities in Tokyo:

1.  **Warner Bros. Studio Tour Tokyo – The Making of Harry Potter:** This highly-rated attracti...

--- iter 2 ---
  [done] Identify 3 family-friendly activities in Tokyo using information from artifact 2.  attach=art:027d5e4301501189
  [open] Check Saturday's weather forecast for Tokyo using information from artifact 1.  attach=art:70c95efbcbab4469
  [open] Determine the most appropriate activity based on the weather forecast.
  [attach] art:70c95efbcbab4469 (10,129 bytes)
  [decision] ANSWER: __NO_ANSWER__

--- iter 3 ---
  [done] Identify 3 family-friendly activities in Tokyo using information from artifact 2.  attach=art:027d5e4301501189
  [done] Check Saturday's weather forecast for Tokyo using information from artifact 1.  attach=art:70c95efbcbab4469
  [open] Determine the most appropriate activity based on the weather forecast.
  [force-attach] art:027d5e4301501189 (11,438 bytes)
  [force-attach] art:70c95efbcbab4469 (10,129 bytes)
  [decision] ANSWER: Given that Tokyo in May 2026 experiences frequent rain (often 8 to 15 days throughout the month), the most appropriate activities depend heavily on the specific day's forecast. To ensure a successful ...

[done] all goals satisfied
  ANSWER  (357.5s)

  Based on the information provided in the artifacts, here are three family-friendly activities in Tokyo:

  1.  **Warner Bros. Studio Tour Tokyo – The Making of Harry Potter:** This highly-rated attraction offers a full-day immersive experience for families, allowing visitors to step behind the scenes of the famous film series.
  2.  **teamLab Planets TOKYO:** This is an interactive, hands-on art experience where families can walk through unique, immersive exhibits, such as rooms filled with moving flowers or "rainbow water," making it an engaging and memorable outing for both children and adults.
  3.  **Visiting Odaiba:** This man-made island in Tokyo Bay is a dedicated shopping and entertainment district that is excellent for families. In addition to a variety of kid-friendly attractions and shopping complexes, the journey itself is fun, as it involves taking a monorail to reach the island.

  ---

  __NO_ANSWER__

  ---

  Given that Tokyo in May 2026 experiences frequent rain (often 8 to 15 days throughout the month), the most appropriate activities depend heavily on the specific day's forecast. To ensure a successful family outing, I recommend selecting activities based on whether the weather is sunny or rainy:

  ### For Sunny Days: Outdoor Exploration
  *   **Ueno Park:** Perfect for a pleasant stroll. It is home to museums, a zoo, and scenic spots that are great for families to explore.
  *   **Yoyogi Park:** A large, open space adjacent to the Meiji Jingu Shrine, ideal for children to run around and for families to enjoy nature.
  *   **Odaiba:** This man-made island in Tokyo Bay is highly kid-friendly. The monorail ride to get there is an adventure in itself, and the area offers numerous shopping and entertainment complexes.

  ### For Rainy Days: Indoor Entertainment
  *   **Warner Bros. Studio Tour Tokyo – The Making of Harry Potter:** This is a fantastic, fully immersive, and weather-proof activity that provides a full day of entertainment for fans of the series.
  *   **teamLab Planets TOKYO:** An incredible, hands-on, and sensory-focused art experience. It is entirely indoors and offers a "walk-through-water" concept and beautiful moving floral displays that are captivating for children and adults alike.
  *   **Akihabara Arcade Experience:** For a bit of fun, head to spots like **Super Potato** or **Taito Station**, where you can spend hours playing classic arcade games, retro video games, and testing your luck at "UFO catcher" crane machines.
  *   **Character Cafes:** Tokyo is famous for its themed dining experiences, such as cafes dedicated to Pokémon, Kirby, or various Sanrio characters. These provide a fun, sheltered environment perfect for a rainy afternoon.

  **Recommendation:** Since the weather can change quickly in May, always check the short-term forecast 24–48 hours before your planned activity. If rain is predicted, prioritize one of the indoor attractions listed above to ensure your plans remain unaffected.

```

### Query C — Mom's birthday (durable memory across runs)

**Run 1** — store the fact:
```bash
uv run python agent.py "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
```

```
[agent] run_id=43d1da35
[agent] query: My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.


--- iter 1 ---
  [open] Save mother's birthday as 15 May 2026 in persistent memory.
  [open] Schedule a calendar reminder for 1 May 2026.
  [open] Schedule a calendar reminder for 15 May 2026.
  [decision] ANSWER: Your mother's birthday has been successfully saved in persistent memory as 15 May 2026. Additionally, I have recorded a note to ensure you receive a reminder two weeks before this date, on 1 May 2026....

--- iter 2 ---
  [done] Save mother's birthday as 15 May 2026 in persistent memory.
  [open] Schedule a calendar reminder for 1 May 2026.
  [open] Schedule a calendar reminder for 15 May 2026.
  [decision] ANSWER: To schedule a calendar reminder for 1 May 2026, you should open your calendar application and create a new event for that date. Set the event title to "Reminder" and add any relevant details, such as ...

--- iter 3 ---
  [done] Save mother's birthday as 15 May 2026 in persistent memory.
  [done] Schedule a calendar reminder for 1 May 2026.
  [open] Schedule a calendar reminder for 15 May 2026.
  [decision] ANSWER: There is no available tool to directly set a calendar reminder. However, based on the stored information:

- Your mother's birthday is on **15 May 2026**.
- A reminder should be set for **two weeks pr...

[done] all goals satisfied
  ANSWER  (55.1s)

  Your mother's birthday has been successfully saved in persistent memory as 15 May 2026. Additionally, I have recorded a note to ensure you receive a reminder two weeks before this date, on 1 May 2026. You can access these details at any time by requesting that I recall the information from your memory files.

  ---

  To schedule a calendar reminder for 1 May 2026, you should open your calendar application and create a new event for that date. Set the event title to "Reminder" and add any relevant details, such as a notification time or a brief description. Save the event to ensure it is added to your calendar. Alternatively, you can also use a digital assistant like Siri, Google Assistant, or Alexa to set a reminder for you. Simply say "Hey Siri, remind me on May 1, 2026" or "OK Google, set a reminder for May 1, 2026" to create a reminder. Make sure to check your calendar or digital assistant settings to confirm that the reminder has been successfully set.

  ---

  There is no available tool to directly set a calendar reminder. However, based on the stored information:

  - Your mother's birthday is on **15 May 2026**.
  - A reminder should be set for **two weeks prior**, which is **1 May 2026**.

  To complete this task:
  1. Open your preferred calendar app (e.g., Google Calendar, Apple Calendar).
  2. Create a new event on **1 May 2026**.
  3. Title the event: *"Mother's Birthday Reminder – Two Weeks Before"*.
  4. Optionally, add a notification alert to trigger one day or one week in advance.
  5. Save the event.

  This will ensure you are reminded in a timely manner.

```

**Run 2** — recall from memory (do not clear `state/` between runs):
```bash
uv run python agent.py "When is mom's birthday?"
```

```
[agent] run_id=3dc68321
[agent] query: When is mom's birthday?


--- iter 1 ---
  [open] Identify mom's birthday date from available information.
  [open] Communicate mom's birthday to the user.
  [decision] ANSWER: Your mother's birthday is May 15, 2026.

As you requested, I have noted this date for you. Since I do not have a calendar tool to set a functional alert or reminder, please be aware that the two-week...

--- iter 2 ---
  [done] Identify mom's birthday date from available information.
  [open] Communicate mom's birthday to the user.
  [decision] ANSWER: Your mother's birthday is May 15, 2026.

As I previously mentioned, I do not have a calendar tool to set a functional alert or automated reminder for you. For your manual planning, two weeks before h...

[done] all goals satisfied
  ANSWER  (62.5s)

  Your mother's birthday is May 15, 2026.

  As you requested, I have noted this date for you. Since I do not have a calendar tool to set a functional alert or reminder, please be aware that the two-week mark prior to her birthday would be May 1, 2026, should you wish to manually set a notification on your personal device.

  ---

  Your mother's birthday is May 15, 2026.

  As I previously mentioned, I do not have a calendar tool to set a functional alert or automated reminder for you. For your manual planning, two weeks before her birthday is May 1, 2026. You may want to set a reminder on your preferred calendar application for that date to ensure you have enough time to prepare.

```

Memory hit at iter 1 means the agent answered without any tool call — the fact was retrieved from `state/memory.json`.

---

### Query D — Asyncio research (multi-source synthesis)

```bash
uv run python agent.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

**What to expect:**
```
[agent] run_id=4bae357d
[agent] query: Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.


--- iter 1 ---
  [open] Search for Python asyncio best practices.
  [open] Read the top 3 results regarding asyncio best practices.  attach=art:574ceb00379755fa
  [open] Extract and synthesize advice that the top 3 sources agree on.  attach=art:574ceb00379755fa
  [open] Present the agreed advice as a short numbered list.
  [decision] TOOL_CALL: read_file({'path': 'art:574ceb00379755fa'})
  [action] -> [STOP] Do NOT call 'read_file' with an art: handle. The artifact bytes are already in ATTACHED ARTIFACTS above. Read the

--- iter 2 ---
  [open] Search for Python asyncio best practices.
  [open] Read the top 3 results regarding asyncio best practices.  attach=art:574ceb00379755fa
  [open] Extract and synthesize advice that the top 3 sources agree on.  attach=art:574ceb00379755fa
  [open] Present the agreed advice as a short numbered list.
  [decision] ANSWER: ing directly from the context provided and the attached artifacts, Python asyncio best practices include:

1. **Using asyncio.create_task()**: Instead of using loops to manage tasks, it's recommended ...

--- iter 3 ---
  [done] Search for Python asyncio best practices.
  [open] Read the top 3 results regarding asyncio best practices.  attach=art:574ceb00379755fa
  [open] Extract and synthesize advice that the top 3 sources agree on.  attach=art:574ceb00379755fa
  [open] Present the agreed advice as a short numbered list.
  [attach] art:574ceb00379755fa (10,174 bytes)
  [decision] ANSWER: __NO_ANSWER__

--- iter 4 ---
  [done] Search for Python asyncio best practices.
  [done] Read the top 3 results regarding asyncio best practices.  attach=art:574ceb00379755fa
  [open] Extract and synthesize advice that the top 3 sources agree on.  attach=art:574ceb00379755fa
  [open] Present the agreed advice as a short numbered list.
  [attach] art:574ceb00379755fa (10,174 bytes)
  [decision] ANSWER: Based on an analysis of the provided sources, there is a strong consensus among top `asyncio` resources on several core best practices for writing efficient, non-blocking asynchronous code in Python. ...

--- iter 5 ---
  [done] Search for Python asyncio best practices.
  [done] Read the top 3 results regarding asyncio best practices.  attach=art:574ceb00379755fa
  [done] Extract and synthesize advice that the top 3 sources agree on.  attach=art:574ceb00379755fa
  [open] Present the agreed advice as a short numbered list.
  [decision] ANSWER: Based on an analysis of the provided resources, the following are the primary agreed-upon best practices for working with Python `asyncio`:

1.  **Do Not Block the Event Loop**: Never use synchronous,...

[done] all goals satisfied
  ANSWER  (371.5s)

  ing directly from the context provided and the attached artifacts, Python asyncio best practices include:

  1. **Using asyncio.create_task()**: Instead of using loops to manage tasks, it's recommended to use `asyncio.create_task()` to create tasks. This allows for better management and cancellation of tasks.

  2. **Avoiding asyncio.wait()**: The `asyncio.wait()` function can lead to unexpected behavior if not used carefully, especially when dealing with a large number of tasks. It's recommended to use `asyncio.gather()` instead for waiting on multiple tasks.

  3. **Handling Exceptions**: Always handle exceptions in your async functions to prevent crashes. Use try-except blocks to catch any exceptions that might be raised during the execution of your async functions.

  4. **Using asyncio.run()**: For running asyncio programs, use `asyncio.run()` as the entry point. This function manages the event loop and handles any necessary cleanup.

  5. **Async Context Managers**: When working with resources that need cleanup, such as database connections or file handles, use async context managers. These ensure that resources are properly cleaned up after use.

  6. **Avoid Mixing Async and Sync Code**: Mixing async and sync code can lead to performance issues and unexpected behavior. Try to keep your codebase consistent, using async/await for all I/O-bound operations.

  7. **Debugging**: Use the `asyncio.run()` function with the `debug=True` parameter to enable debugging. This can help identify issues such as deadlocks or other problems in your async code.

  8. **Testing**: Test your async code thoroughly. Use testing frameworks that support async tests, such as pytest-asyncio, to ensure your code works as expected.

  By following these best practices, you can write efficient, readable, and maintainable async code using Python's asyncio library.

  ---

  __NO_ANSWER__

  ---

  Based on an analysis of the provided sources, there is a strong consensus among top `asyncio` resources on several core best practices for writing efficient, non-blocking asynchronous code in Python. The following practices are consistently recommended:

  1.  **Do Not Block the Event Loop**: This is the most critical rule in `asyncio`. Developers must avoid using blocking synchronous calls (like `time.sleep()`) within coroutines, as these will halt the entire event loop. Instead, use non-blocking equivalents (like `await asyncio.sleep()`) or offload blocking operations to a separate thread pool using `loop.run_in_executor()`.
  2.  **Use `asyncio.create_task()` for Concurrency**: To run independent operations concurrently rather than sequentially, you should wrap coroutines in `asyncio.create_task()`. This schedules the tasks on the event loop to run in the background. It is also essential to `await` these tasks (or use `asyncio.gather()`) to ensure they complete before the main coroutine finishes.
  3.  **Use `asyncio.run()` as the Main Entry Point**: When starting an `asyncio` program, always use `asyncio.run()`. This is considered the best practice for managing the lifecycle of the event loop because it handles the necessary setup, teardown, and cleanup of resources, as well as exception handling, in a robust manner.

  By adhering to these patterns—maintaining a non-blocking loop, utilizing proper task scheduling, and using standard entry-point management—developers can effectively leverage `asyncio` to handle I/O-bound tasks with high scalability.

  ---

  Based on an analysis of the provided resources, the following are the primary agreed-upon best practices for working with Python `asyncio`:

  1.  **Do Not Block the Event Loop**: Never use synchronous, blocking calls (like `time.sleep()` or heavy CPU-bound computations) inside an `async` function. Doing so stops the entire event loop, preventing all other concurrent tasks from progressing.
  2.  **Use `asyncio.create_task()` for Concurrency**: To run multiple tasks concurrently, use `asyncio.create_task()` to schedule them on the event loop rather than running them sequentially. This allows the loop to manage the execution order efficiently.
  3.  **Utilize `asyncio.gather()` for Grouping**: When you need to wait for multiple concurrent tasks to complete and collect their results, use `asyncio.gather()`. It provides a clean, readable way to aggregate results from multiple coroutines without needing complex boilerplate code.


```

`[force-attach]` on iter 2 means the synthesis safety net detected the synthesis keyword (`agree`) in the goal and auto-attached the search result artifact from the current run's history. The memory-hit fallback was removed — only artifacts produced in the **current run** are eligible for force-attach, preventing stale cross-query artifacts from being injected.

---

## Running the canonical test suite

`test_all.py` runs the five canonical queries with colour-coded output (yellow question, green answer, red error) and a summary table at the end.

```powershell
uv run python test_all.py          # all 5 queries
uv run python test_all.py 1        # Query A only
uv run python test_all.py 3 4      # C1 and C2 only (1-based index)
```

| # | Label | Query |
|---|---|---|
| 1 | Query A | Fetch Claude Shannon Wikipedia — birth/death dates + 3 contributions |
| 2 | Query B | 3 family-friendly Tokyo activities + live weather + recommendation |
| 3 | Query C1 | Remember mom's birthday + calendar reminders |
| 4 | Query C2 | Recall mom's birthday (run after C1) |
| 5 | Query D | Search asyncio best practices, read top 3 results, list common advice |

Each query has a **900-second timeout**. A timed-out query is marked as failed in the summary table but does not stop the remaining queries from running. A **60-second inter-query delay** is inserted between queries to let Gemini's 57-second hard backoff clear before the next run.

> **Windows / stdout buffering:** If you redirect output to a file or pipe it through another tool, prefix the command with `PYTHONUNBUFFERED=1` (or use `python -u`) so log lines appear in real time:
> ```powershell
> $env:PYTHONUNBUFFERED = "1"; uv run python -u test_all.py
> ```

---

## Resetting state

```bash
# Full reset — memory, artifacts, and sandbox files
rm -rf state/ sandbox/

# Memory only (keeps artifacts)
rm state/memory.json
```

---

## MCP tools available to the agent

| Tool | Description |
|---|---|
| `web_search` | Four-provider chain: **Tavily → Exa → Firecrawl → DuckDuckGo**. Returns titles, URLs, snippets. Hard-capped at 5 results. Usage logged to `usage.json` with monthly rollover. |
| `fetch_url` | Two-phase fetch: **httpx fast-path** first (~3 s, plain HTTP) then **crawl4ai headless Chromium** fallback for JS-heavy pages. Content hard-capped at 20,000 chars (`_MAX_FETCH_CHARS`). Subject to 30 s Action-layer timeout; Decision falls back to `web_search` on `[tool_timeout]`. |
| `get_time` | Current time in any IANA timezone (e.g. `"Asia/Tokyo"`) |
| `currency_convert` | Live rates via Frankfurter API |
| `read_file` | Read a UTF-8 file from `sandbox/` |
| `list_dir` | List contents of a `sandbox/` directory |
| `create_file` | Create a new file in `sandbox/` (errors if already exists) |
| `update_file` | Overwrite an existing `sandbox/` file |
| `edit_file` | Find-and-replace inside a `sandbox/` file |

All file tools are sandboxed to `sandbox/` — path traversal outside that directory is blocked.

### Search provider priority

`web_search` tries each provider in order and returns as soon as one succeeds:

| Priority | Provider | Key env var | Free tier |
|---|---|---|---|
| 1 | Tavily | `TAVILY_API_KEY` | 1,000 searches/mo |
| 2 | Exa | `EXA_API_KEY` | 1,000 searches/mo |
| 3 | Firecrawl | `FIRECRAWL_API_KEY` | 500 credits/mo |
| 4 | DuckDuckGo | *(none)* | Free, rate-limited |

Provider errors and usage counts are tracked in `usage.json` (monthly rollover).

---

## Gateway behaviour notes

- `reasoning_applied: false` — some free-tier models ignore the reasoning budget; the gateway surfaces this honestly.
- `cache_read_input_tokens: 0` — system prompt caching requires a paid-tier Gemini key.
- `fallback_used: true` in `router_decision` — all router-pool workers were rate-limited; the gateway fell back to the deterministic token-count rule. The worker call still succeeds.
- Gemini 3.x loops at `temperature=0` — Perception is pinned to `temperature=1.0` to prevent this.
- Cerebras `queue_exceeded` errors are routine on the free tier and handled by router failover to Groq.
- **Hard-backoff retry**: when all providers are in a rate-limit hard backoff, the gateway waits for the shortest pending backoff ≤ 90 s before retrying instead of raising immediately. Backoffs longer than 90 s are skipped in favour of other providers.
- **Provider HTTP timeouts**: Gemini 45 s, OpenAI-compatible (Groq, Cerebras, etc.) 45 s, Ollama 600 s (local model).

---

## Reliability notes

### fetch_url performance and timeouts
`fetch_url` attempts a plain HTTP fast-path via `httpx` first (~3 s). Only if that returns too little content does it fall through to `crawl4ai` (headless Chromium), which is slower but handles JavaScript-rendered pages. Action enforces a **30-second hard timeout** — if exceeded, the agent receives `[tool_timeout]` and Decision switches to `web_search`. Fetched content is hard-capped at **20,000 characters** in `mcp_server.py`; this keeps artifacts small enough that they can be force-attached inline without exceeding any provider's context window.

### Search reliability
Without any key, DuckDuckGo is the only search provider — it is free but rate-limited and sometimes returns empty results. The fallback chain (Tavily → Exa → Firecrawl → DuckDuckGo) means the agent degrades gracefully as each tier is exhausted, but for consistent results add at least one paid key.
