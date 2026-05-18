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

Open `.env` and fill in at least one worker provider key. Gemini is the recommended starting point — the free tier (15 RPM / 1,000 RPD) handles all four example queries with room to spare.

```bash
# Minimum viable .env
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file lives at the **repo root** — one level above `llm_gatewayV3/`. The gateway reads `../.env` relative to its own directory, so a single file covers both.

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

The loop terminates when Perception marks all goals as `done`, or when `MAX_ITERATIONS` is reached.

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

**What it does:** The orchestrator. It decomposes the user query into an ordered list of goals on the first iteration, then updates done-flags on every subsequent iteration by inspecting the history.

**How it works:**

1. **First iteration** (`prior_goals` is empty): Perception sends the query, memory hits, and an empty prior-goals list to Gemini with `temperature=1.0`. The LLM decomposes the query into 1–4 short imperative goals ordered by logical dependency — fetch before extract, search before synthesize.

2. **Subsequent iterations**: Perception sends the same query, the current history, and the prior goals. It outputs the goals in the **exact same order**, setting `done: true` only for goals that have a satisfying action or answer in history. Once `done`, a goal is **sticky** — it can never be flipped back.

3. **Artifact attachment**: For the first unfinished goal that requires reading a previously fetched artifact (e.g. "extract info from page"), Perception sets `artifact_index` pointing to the corresponding `[artifact N]` entry in the memory-hit list. The main loop resolves this index to the actual artifact ID stored by memory.

**Why it matters:** Perception is the only role that can mark work as done. This prevents Decision from re-doing completed steps and ensures goals have stable positions (position = identity, not LLM-generated IDs that can hallucinate). Pinning Perception to Gemini (`provider="g"`) ensures reliable structured-output compliance for the goal list schema. `temperature=1.0` prevents Gemini 3.x from stalling in a low-entropy loop.

**Fallback:** If the Gemini call fails, Perception returns prior goals unchanged (or a single bare goal on the first iteration). The run degrades gracefully rather than crashing.

---

### Decision — `decision.py`

**What it does:** Given one goal and its context, produces exactly one output: either a direct **answer** (text) or a single **tool call**. It never does both.

**How it works:**

1. Decision builds a user message containing the current goal, memory hits, recent history (last 10 entries), and — when Perception has attached one — the full bytes of the artifact rendered inline as `ATTACHED ARTIFACTS`.

2. It sends this message to the gateway with `auto_route="decision"` and the full MCP tool list. The gateway classifies the request size (TINY or LARGE), selects a worker tier, and dispatches to the first available provider.

3. The response is inspected for tool calls first. If the LLM emitted a function call, a `ToolCall` is returned. Otherwise, the text content becomes the answer.

**The strict rule enforced by the system prompt:**
- If history already contains the information, answer directly — do not call a tool again.
- Artifact handles (`art:…`) are never valid arguments to a tool — bytes are available inline in ATTACHED ARTIFACTS.
- Extraction, synthesis, and comparison goals must produce a substantive answer (≥ 3 sentences or a numbered list).

**Why it matters:** Separating Decision from Action ensures the LLM never directly executes code. Decision emits intent (`ToolCall`); Action executes it through MCP. This also means Decision can be retried or replaced independently without touching the execution layer. The `auto_route` lets the gateway pick the cheapest provider that fits the request size — a small fetch decision goes to a fast TINY-tier model; an extraction decision with 50 KB of attached content goes to a large-context LARGE-tier provider.

---

### Action — `action.py`

**What it does:** Pure MCP dispatch. No LLM calls. Takes a `ToolCall`, executes it through the MCP `ClientSession`, and returns a `(descriptor, artifact_id)` pair.

**How it works:**

1. **Guard check**: Before any dispatch, Action inspects the tool arguments for `art:…` handles in path/URL fields. If found, it returns an error descriptor immediately — this prevents Decision from accidentally passing a stale artifact handle to a tool that expects a real URL or file path.

2. **MCP dispatch**: Calls `session.call_tool(name, arguments)` over the stdio MCP transport. The MCP server (`mcp_server.py`) executes the tool and returns content blocks.

3. **Artifact threshold**: If the tool response exceeds `ARTIFACT_THRESHOLD_BYTES` (4 KB), the bytes are written to the content-addressable artifact store (`state/artifacts/`). The returned descriptor includes the artifact handle (`art:<sha256[:16]>`) and a 200-character preview. If the response is small enough, it is returned as raw text with no artifact created.

**Why it matters:** The artifact threshold is what keeps Decision's context from bloating. A 50 KB Wikipedia page is stored once as `art:4c163…` and referred to by handle throughout the remaining iterations. Only when Perception explicitly attaches it does Decision see the bytes — and only the bytes it actually needs, not every tool result from every prior iteration. The `art:` guard closes the loop: Decision is told not to pass handles to tools, and Action enforces that rule at execution time.

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
├── action.py           MCP dispatch with artifact threshold and art: guard
├── artifact_store.py   Content-addressable store (state/artifacts/)
├── llm_gateway.py      In-process LLM gateway — routing, failover, rate limits
├── mcp_server.py       9 MCP tools over stdio transport
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

> **Reset state between fresh attempts:**
> ```bash
> rm -rf state/
> ```

### Query A — Shannon Wikipedia (artifact attach test)

```bash
uv run python agent.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

**What to expect (3 iterations):**
```
[agent] run_id=<id>

--- iter 1 ---
  [open] Fetch the Wikipedia page for Claude Shannon
  [open] Extract birth date, death date, and three contributions
  [decision] TOOL_CALL: fetch_url({'url': 'https://en.wikipedia.org/wiki/Claude_Shannon'})
  [action] -> [artifact art:09ff..., 50,463 bytes] ...

--- iter 2 ---
  [done] Fetch the Wikipedia page for Claude Shannon
  [open] Extract birth date, death date, and three contributions  attach=art:09ff...
  [attach] art:09ff... (50,463 bytes)
  [decision] ANSWER: Claude Shannon (April 30, 1916 - February 24, 2001)...

--- iter 3 ---
  [done] Fetch the Wikipedia page for Claude Shannon
  [done] Extract birth date, death date, and three contributions

[done] all goals satisfied
```

This query tests that Perception detects the fetch goal is done, sets `attach_artifact_id` on the extraction goal, and Decision receives the page bytes directly rather than re-fetching.

---

### Query B — Tokyo activities (multi-goal + web search)

```bash
uv run python agent.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
```

**What to expect (≤ 6 iterations):**
```
--- iter 1 ---
  [open] Find 3 family-friendly things to do in Tokyo
  [open] Check Saturday weather forecast for Tokyo
  [open] Choose the most appropriate activity given the weather
  [decision] TOOL_CALL: web_search(...)

--- iter 2-3 ---
  [done] Find 3 family-friendly activities
  [open] Check Saturday weather ...

--- iter 4-5 ---
  [done] Find + weather
  [open] Choose most appropriate activity ...
  [decision] ANSWER: Given Saturday's forecast of [weather], ...
```

---

### Query C — Mom's birthday (durable memory across runs)

**Run 1** — store the fact:
```bash
uv run python agent.py "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
```

```
[memory.remember] -> classified as fact, keywords: [mom, birthday, may, 2026]
--- iter 1 ---
  [open] Record mom's birthday (15 May 2026) in memory
  [open] Create reminder for 1 May 2026 (two weeks before)
  [open] Create reminder for 15 May 2026
  [decision] TOOL_CALL: create_file(...)
...
FINAL ANSWER: Reminders created. Mom's birthday on 15 May 2026 is recorded.
```

**Run 2** — recall from memory (do not clear `state/` between runs):
```bash
uv run python agent.py "When is mom's birthday?"
```

```
--- iter 1 ---
  [open] Answer when mom's birthday is
  [decision] ANSWER: Mom's birthday is on 15 May 2026.

[done] all goals satisfied
```

Memory hit at iter 1 means the agent answered without any tool call — the fact was retrieved from `state/memory.json`.

---

### Query D — Asyncio research (multi-source synthesis)

```bash
uv run python agent.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

**What to expect (5–7 iterations):**
```
--- iter 1 ---
  [open] Search for Python asyncio best practices
  [open] Fetch the top 3 search results
  [open] Synthesise the common advice into a numbered list
  [decision] TOOL_CALL: web_search(...)

--- iter 2-4 ---
  [done] Search
  [open] Fetch top 3 results (fetches each URL -> artifacts)

--- iter 5 ---
  [done] Search + fetch
  [open] Synthesise common advice
  [force-attach] art:... (... bytes)
  [decision] ANSWER:
    1. Use asyncio.run() as the single program entry point
    2. Prefer asyncio.gather() or TaskGroup for concurrent coroutines
    ...
```

The `[force-attach]` line means the synthesis safety net fired — Perception hadn't set an explicit `attach_artifact_id`, but `agent.py` detected the synthesis keyword in the goal text and auto-attached the most recent artifact.

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
| `web_search` | Tavily (with `TAVILY_API_KEY`) or DuckDuckGo fallback — titles, URLs, snippets |
| `fetch_url` | httpx (static pages) with crawl4ai fallback for JS-rendered content |
| `get_time` | Current UTC time in ISO 8601 |
| `currency_convert` | Live rates via Frankfurter API |
| `read_file` | Read a file from `sandbox/` |
| `list_dir` | List contents of a `sandbox/` directory |
| `create_file` | Create a new file in `sandbox/` |
| `update_file` | Overwrite a `sandbox/` file |
| `edit_file` | Replace a text segment in a `sandbox/` file |

All file tools are sandboxed to `sandbox/` — path traversal outside that directory is blocked.

---

## Gateway behaviour notes

- `reasoning_applied: false` — some free-tier models ignore the reasoning budget; the gateway surfaces this honestly.
- `cache_read_input_tokens: 0` — system prompt caching requires a paid-tier Gemini key.
- `fallback_used: true` in `router_decision` — all router-pool workers were rate-limited; the gateway fell back to the deterministic token-count rule. The worker call still succeeds.
- Gemini 3.x loops at `temperature=0` — Perception is pinned to `temperature=1.0` to prevent this.
- Cerebras `queue_exceeded` errors are routine on the free tier and handled by router failover to Groq.
