"""Agent6 — four-role cognitive loop.

Roles per iteration (in order):
  Memory     → read(query, history)  → list[MemoryItem]
  Perception → observe(...)          → Observation (goal list with done flags)
  Decision   → next_step(...)        → DecisionOutput (answer | tool_call)
  Action     → execute(session, tc)  → (descriptor, artifact_id?)

Usage:
  uv run python agent.py "Your query here"
  uv run python agent.py   # interactive prompt
"""
from __future__ import annotations

import asyncio
import sys
import uuid

# Windows consoles default to cp1252; reconfigure stdout/stderr to UTF-8 so
# tool results and LLM output with non-ASCII characters don't crash on print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import action
import decision
import perception
from artifact_store import artifacts
from memory import memory
from schemas import Goal, Observation, ToolCall

# --------------------------------------------------------------------------- #
# Configuration                                                                 #
# --------------------------------------------------------------------------- #

MAX_ITERATIONS = 20

_MCP_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "python", "mcp_server.py"],
)

# Keywords that signal a synthesis / extraction goal → force-attach safety net
_SYNTHESIS_KW = frozenset(
    "synthesize synthesise extract compare decide summarize summarise "
    "recommend choose select appropriate agree common findings "
    "analyze analyse collate distil distill".split()
)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _mcp_tools_for_decision(tools: list) -> list[dict]:
    """Convert MCP Tool objects to gateway ToolDef format."""
    result = []
    for t in tools:
        schema = t.inputSchema
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        result.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": schema,
            }
        )
    return result


def _is_synthesis_goal(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & _SYNTHESIS_KW)


def _final_answer_from(history: list[dict], goals: list[Goal]) -> str:
    """Build the final answer from history.

    When multiple goals each produced an answer, join them in goal order so
    every sub-question is represented in the output (e.g. birth date AND
    contributions, not just whichever was answered last).
    """
    # Collect the last answer for each goal_id (later entries overwrite earlier)
    answer_by_goal: dict[str, str] = {}
    for h in history:
        if h.get("kind") == "answer" and h.get("text"):
            gid = h.get("goal_id", "")
            answer_by_goal[gid] = h["text"]

    if not answer_by_goal:
        for h in reversed(history):
            if h.get("kind") == "action":
                return f"Task completed. Last action: {h.get('result_descriptor', '')}"
        return "Task completed with no answer recorded."

    if len(answer_by_goal) == 1:
        return next(iter(answer_by_goal.values()))

    # Multiple goals — return answers in goal order, one paragraph per goal
    ordered: list[str] = []
    seen: set[str] = set()
    for g in goals:
        if g.id in answer_by_goal:
            ordered.append(answer_by_goal[g.id])
            seen.add(g.id)
    # Append any orphaned answers not matched to a current goal
    for gid, ans in answer_by_goal.items():
        if gid not in seen:
            ordered.append(ans)

    return "\n\n---\n\n".join(ordered)


def _print_goals(goals: list[Goal]) -> None:
    for g in goals:
        status = "[done]" if g.done else "[open]"
        attach = f"  attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
        print(f"  {status} {g.text}{attach}")


# --------------------------------------------------------------------------- #
# Main loop                                                                     #
# --------------------------------------------------------------------------- #

async def run(query: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    print(f"\n[agent] run_id={run_id}")
    print(f"[agent] query: {query}\n")

    # Classify the query for durable memory (persists across runs)
    await memory.remember(query, source="user_query", run_id=run_id)

    fatal_error: str | None = None

    async with stdio_client(_MCP_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            mcp_tools = _mcp_tools_for_decision(tools_result.tools)

            for it in range(1, MAX_ITERATIONS + 1):
                # ── Memory ─────────────────────────────────────────────── #
                hits = memory.read(query, history)

                # ── Perception ─────────────────────────────────────────── #
                try:
                    obs = await perception.observe(
                        query, hits, history, prior_goals, run_id
                    )
                except Exception as exc:
                    print(f"\n[agent] ERROR in perception: {exc}")
                    fatal_error = str(exc)
                    break
                prior_goals = obs.goals

                print(f"\n--- iter {it} ---")
                _print_goals(obs.goals)

                if obs.all_done:
                    print("\n[done] all goals satisfied")
                    break

                goal = obs.next_unfinished()
                if goal is None:
                    break

                # ── Artifact attachment ─────────────────────────────────── #
                attached: list[tuple[str, bytes]] = []

                if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                    raw = artifacts.get_bytes(goal.attach_artifact_id)
                    attached.append((goal.attach_artifact_id, raw))
                    print(f"  [attach] {goal.attach_artifact_id} ({len(raw):,} bytes)")

                # Force-attach safety net: synthesis goals with no attachment
                if not attached and _is_synthesis_goal(goal.text):
                    for hit in reversed(hits):
                        if hit.artifact_id and artifacts.exists(hit.artifact_id):
                            raw = artifacts.get_bytes(hit.artifact_id)
                            attached.append((hit.artifact_id, raw))
                            print(
                                f"  [force-attach] {hit.artifact_id} "
                                f"({len(raw):,} bytes)"
                            )
                            break

                # ── Decision ───────────────────────────────────────────── #
                try:
                    out = await decision.next_step(
                        goal, hits, attached, history, mcp_tools
                    )
                except Exception as exc:
                    print(f"\n[agent] ERROR in decision: {exc}")
                    fatal_error = str(exc)
                    break

                if out.is_answer:
                    preview = (out.answer or "")[:200]
                    print(f"  [decision] ANSWER: {preview}{'...' if len(out.answer or '') > 200 else ''}")
                    history.append(
                        {
                            "iter": it,
                            "kind": "answer",
                            "goal_id": goal.id,
                            "text": out.answer,
                        }
                    )
                    # Mark done immediately — don't rely on Perception to infer it
                    # from history. Sticky-done in Perception will preserve this.
                    goal.done = True
                    continue

                # ── Action ─────────────────────────────────────────────── #
                tc = out.tool_call
                assert tc is not None
                print(f"  [decision] TOOL_CALL: {tc.name}({tc.arguments})")

                result_text, art_id = await action.execute(session, tc)
                print(f"  [action] -> {result_text[:120]}")

                memory.record_outcome(
                    tool_call=tc,
                    result_text=result_text,
                    artifact_id=art_id,
                    run_id=run_id,
                    goal_id=goal.id,
                )
                history.append(
                    {
                        "iter": it,
                        "kind": "action",
                        "goal_id": goal.id,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_descriptor": result_text[:300],
                        "artifact_id": art_id,
                    }
                )

            else:
                print(f"\n[agent] reached MAX_ITERATIONS={MAX_ITERATIONS}")

    if fatal_error:
        sys.exit(1)

    return _final_answer_from(history, prior_goals)


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

async def _main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()
    if not query:
        print("No query provided.")
        return
    result = await run(query)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(_main())
