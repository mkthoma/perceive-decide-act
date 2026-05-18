# perceive-decide-act

Lightweight agentic framework built on a four-role cognitive loop — **Memory → Perception → Decision → Action** — with MCP tool integration and artifact-aware goal tracking.

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

```
agent loop
  │
  ├─ Memory       Typed, persistent fact store. read() uses keyword overlap
  │               (no LLM). remember() classifies free-form text via one
  │               gateway call. record_outcome() is free.
  │
  ├─ Perception   Orchestrator. Every iteration: decompose query into goals
  │               (first call) or update done-flags from history (subsequent).
  │               Pinned to Gemini via provider="g". temperature=1.0.
  │
  ├─ Decision     Picks one action per iteration: answer OR tool_call.
  │               Routes via auto_route="decision" through the gateway
  │               router pool (TINY / LARGE tier selection).
  │
  └─ Action       Pure MCP dispatch. Payloads > 4 KB are written to the
                  artifact store; Decision sees bytes only when Perception
                  explicitly attaches them to a goal.
```

### Key design properties

| Property | Where enforced |
|---|---|
| Goals have stable identity (position, not LLM-generated IDs) | `perception.py` |
| Artifact attachment decided by Perception, not Decision | `agent.py` loop |
| Artifact handles (`art:…`) are never valid tool arguments | `action.py` guard |
| `done: true` is sticky — never flipped back | `perception.py` |
| Synthesis goals trigger force-attach safety net | `agent.py` loop |
| Memory persists across process restarts | `state/memory.json` |

## File layout

```
perceive-decide-act/
├── agent.py           Main loop (Memory → Perception → Decision → Action)
├── schemas.py          Pydantic v2 contracts for all role boundaries
├── memory.py           Typed fact store with keyword search and LLM classification
├── perception.py       Goal decomposition and done-flag tracking (Gemini)
├── decision.py         Action selection — answer or single tool call
├── action.py           MCP dispatch with artifact threshold and art: guard
├── artifact_store.py   Content-addressable store (state/artifacts/)
├── llm_gateway.py      Async HTTP client for LLM Gateway V3 at :8101
├── mcp_server.py       9 MCP tools over stdio transport
├── pyproject.toml      uv project config
├── .env.example        Environment variable template
└── llm_gatewayV3/      Gateway source (see llm_gatewayV3/README.md)
```

---

## LLM Gateway V3

The gateway is a local FastAPI service that routes LLM calls across **seven free worker providers** (Gemini, Groq, Cerebras, NVIDIA NIM, OpenRouter, GitHub Models, Ollama) with automatic failover and a **router pool** that classifies each request by size and picks the right worker tier.

### Starting the gateway (optional — dashboard only)

The agent uses `llm_gatewayV3/` as a library and does **not** require a running server.
To view the live dashboard and call-log, run the gateway in a separate terminal:

```powershell
cd llm_gatewayV3
uv run python main.py
```

Ready when you see `Uvicorn running on http://0.0.0.0:8101`. Dashboard: <http://localhost:8101>

### Checking gateway health

```bash
# Worker pool — providers, models, rate limits
curl -s http://localhost:8101/v1/providers | python3 -m json.tool

# Router pool — four small LLMs that classify request size
curl -s http://localhost:8101/v1/routers | python3 -m json.tool

# Live rate state
curl -s http://localhost:8101/v1/status | python3 -m json.tool
```

Or open the dashboard: <http://localhost:8101>

### How routing works

When `agent` sends a call tagged `auto_route="decision"` (or `"perception"` / `"memory"`), the gateway:

1. Estimates the token count of the request
2. Sends a bounded `{token_count, sample}` envelope to a router LLM (Cerebras → Groq → NVIDIA → GitHub, first available)
3. Router responds with one word: `TINY` (< 1K tokens) or `LARGE` (1K–8K tokens)
4. Routes to the corresponding worker failover list and dispatches the actual call
5. Returns the worker response enriched with a `router_decision` block

If all four router providers are unavailable, it falls back to the token-count rule — the worker call still succeeds.

### Provider keys

See [llm_gatewayV3/README.md](llm_gatewayV3/README.md) for the full breakdown of worker providers, router providers, rate limits, and configuration options. The short version:

| Provider | Free tier | Recommended for |
|---|---|---|
| Gemini | 15 RPM / 1,000 RPD | Perception (pinned), worker fallback |
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

─── iter 1 ───
  [open] Fetch the Wikipedia page for Claude Shannon
  [open] Extract birth date, death date, and three contributions
  [decision] TOOL_CALL: fetch_url({'url': 'https://en.wikipedia.org/wiki/Claude_Shannon'})
  [action] → [artifact art:09ff..., 263,065 bytes] ...

─── iter 2 ───
  [done] Fetch the Wikipedia page for Claude Shannon
  [open] Extract birth date, death date, and three contributions  attach=art:09ff...
  [attach] art:09ff... (263,065 bytes)
  [decision] ANSWER: Claude Shannon (April 30, 1916 – February 24, 2001)...

─── iter 3 ───
  [done] Fetch the Wikipedia page for Claude Shannon
  [done] Extract birth date, death date, and three contributions

[done] all goals satisfied
```

This query tests that Perception detects the fetch goal is done, sets `attach_artifact_id` on the extraction goal, and Decision receives the page bytes directly rather than re-fetching.

---

### Query B — Tokyo activities (multi-goal + weather constraint)

```bash
uv run python agent.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
```

**What to expect (≤ 6 iterations):**
```
─── iter 1 ───
  [open] Find 3 family-friendly things to do in Tokyo
  [open] Check Saturday weather forecast for Tokyo
  [open] Choose the most appropriate activity given the weather
  [decision] TOOL_CALL: web_search(...)

─── iter 2–3 ───
  [done] Find 3 family-friendly activities
  [open] Check Saturday weather ...

─── iter 4–5 ───
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
[memory.remember] → classified as fact, keywords: [mom, birthday, may, 2026]
─── iter 1 ───
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
─── iter 1 ───
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
─── iter 1 ───
  [open] Search for Python asyncio best practices
  [open] Fetch the top 3 search results
  [open] Synthesise the common advice into a numbered list
  [decision] TOOL_CALL: web_search(...)

─── iter 2–4 ───
  [done] Search
  [open] Fetch top 3 results (fetches each URL → artifacts)

─── iter 5 ───
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
| `fetch_url` | crawl4ai (JS-rendered, returns Markdown) with httpx fallback |
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
