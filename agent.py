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

# First-word verbs that mean "go get the data" — goal is satisfied by a
# successful tool call alone, no textual answer needed.
# Exclude "check" (implies verify/report) and "find" (implies selection).
_ACQUISITION_VERBS = frozenset(
    "fetch download retrieve get load search look".split()
)

# Only these tools can trigger acquisition-goal auto-done.
# Utility tools like get_time, currency_convert should NOT mark a goal done
# just because they returned a value (e.g. get_time for "search weather" goal).
_AUTO_DONE_TOOLS = frozenset({"web_search", "fetch_url", "read_file", "list_dir"})

# Words that signal an answer is expected — goal is NOT purely data retrieval.
# "Fetch https://... and tell me his birth date" needs an ANSWER after the fetch.
_ANSWER_MARKERS = frozenset(
    "tell give describe explain list summarize compare "
    "what when how why which who does did has have".split()
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


def _is_acquisition_goal(text: str) -> bool:
    """True when a goal is satisfied purely by executing the tool call.

    "Fetch the Wikipedia page" → done once fetch_url succeeds.
    "Fetch https://... and tell me his birth date" → NOT acquisition;
      contains answer markers ("tell") so an ANSWER is still needed.
    "Find the best activity based on weather" → NOT acquisition (contains
      synthesis keywords), needs an ANSWER.
    """
    words = text.lower().split()
    return (
        bool(words)
        and words[0] in _ACQUISITION_VERBS
        and not (_SYNTHESIS_KW & set(words))
        and not (_ANSWER_MARKERS & set(words))
    )


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
                # Fast-path exit: if all goals are already marked done from
                # the previous iteration, skip the Perception LLM call and
                # break immediately — saves one gateway round-trip per query.
                if it > 1 and prior_goals and all(g.done for g in prior_goals):
                    print("\n[done] all goals satisfied")
                    break

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

                # Guard: Perception must never mark brand-new goals as done.
                # Memory hits from prior runs can mislead Gemini into thinking
                # the task is already complete before any work this run.
                if it == 1:
                    for g in prior_goals:
                        g.done = False

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

                # Force-attach safety net: synthesis goals with no attachment.
                # Only attach an artifact if its keywords overlap with the goal text
                # so we don't accidentally attach an unrelated cached artifact.
                if not attached and _is_synthesis_goal(goal.text):
                    goal_words = set(goal.text.lower().split())
                    for hit in reversed(hits):
                        if hit.artifact_id and artifacts.exists(hit.artifact_id):
                            artifact_kw = set(getattr(hit, "keywords", []))
                            if not artifact_kw or (goal_words & artifact_kw):
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

                # Auto-complete acquisition goals on successful tool execution.
                # "Fetch the Wikipedia page" is done once fetch_url returns data;
                # no LLM answer is needed. This prevents Decision from looping
                # trying to re-read an artifact handle it can't use as a path.
                # Only fire for data-retrieval tools — utility tools like get_time
                # or currency_convert should not auto-complete a "search" goal
                # just because they returned a value.
                is_error = result_text.startswith("[") and (
                    "error" in result_text[:80].lower()
                    or "timeout" in result_text[:80].lower()
                    or result_text.startswith("[STOP]")
                )
                is_empty = (
                    "no results found" in result_text[:80].lower()
                    or result_text.strip() in ("(empty directory)", "[]", "null", "")
                )
                if (not is_error and not is_empty
                        and _is_acquisition_goal(goal.text)
                        and tc.name in _AUTO_DONE_TOOLS):
                    goal.done = True
                    print(f"  [auto-done] acquisition goal satisfied by tool call")

                # Hard-stop repeated empty searches: if the last 3 actions for
                # this goal all returned "No results found", skip further tool
                # calls and let Decision answer from its knowledge next iter.
                if is_empty:
                    recent_empty = sum(
                        1 for h in history[-6:]
                        if h.get("kind") == "action"
                        and h.get("goal_id") == goal.id
                        and (
                            "no results found" in h.get("result_descriptor", "").lower()
                            or h.get("result_descriptor", "").strip() in ("[]", "null", "")
                        )
                    )
                    if recent_empty >= 2:
                        print(f"  [no-search] 3+ empty results — skipping tool, forcing answer next iter")
                        # Remove tools from mcp_tools for the NEXT decision call
                        # by injecting a sentinel into history
                        history.append(
                            {
                                "iter": it,
                                "kind": "action",
                                "goal_id": goal.id,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                                "result_descriptor": result_text[:300] + " [SEARCH_EXHAUSTED: web_search returned no results after 3 attempts. Do NOT call web_search again. Instead call fetch_url on official documentation or well-known resource URLs for this topic.]",
                                "artifact_id": None,
                            }
                        )
                        memory.record_outcome(
                            tool_call=tc,
                            result_text=result_text,
                            artifact_id=None,
                            run_id=run_id,
                            goal_id=goal.id,
                        )
                        continue

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
                        "result_descriptor": result_text[:800],
                        "artifact_id": art_id,
                    }
                )

            else:
                print(f"\n[agent] reached MAX_ITERATIONS={MAX_ITERATIONS}")

    if fatal_error:
        sys.exit(1)

    answer = _final_answer_from(history, prior_goals)

    # If no real answer was recorded (e.g. all goals auto-completed via tool
    # calls, or Perception prematurely marked extraction goals done before an
    # answer was produced), make one final Decision call to synthesise the
    # answer from whatever is in memory + history.
    _no_real_answer = (
        answer == "Task completed with no answer recorded."
        or answer.startswith("Task completed. Last action:")
    )
    if _no_real_answer and not fatal_error:
        try:
            hits = memory.read(query, history)
            synth_goal = Goal(text=query)
            attached: list[tuple[str, bytes]] = []
            # Prefer the most recent artifact from the run's own history;
            # memory hits may surface stale artifacts from earlier runs.
            for h in reversed(history):
                art_id = h.get("artifact_id")
                if art_id and artifacts.exists(art_id):
                    raw = artifacts.get_bytes(art_id)
                    attached.append((art_id, raw))
                    break
            if not attached:
                for hit in reversed(hits):
                    if hit.artifact_id and artifacts.exists(hit.artifact_id):
                        raw = artifacts.get_bytes(hit.artifact_id)
                        attached.append((hit.artifact_id, raw))
                        break
            async with stdio_client(_MCP_PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    out = await decision.next_step(synth_goal, hits, attached, history, [])
            if out.is_answer and out.answer:
                # Guard: discard system-prompt echoes (some models repeat the
                # system message when confused by an unusual no-tools context).
                _echo_markers = ("You are DECISION", "STRICT RULES:", "You are an")
                if not any(out.answer.startswith(m) for m in _echo_markers):
                    answer = out.answer
        except Exception:
            pass

    return answer


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
