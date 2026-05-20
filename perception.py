"""Perception role — orchestrator that tracks goals across iterations.

Responsibilities each iteration:
1. First call (prior_goals empty): decompose query into 1-4 bounded goals.
2. Subsequent calls: copy goals in order, updating done flags from history.
3. For the first unfinished goal: set artifact_index when bytes are needed.

Pinned to Gemini via provider="g" for reliable structured-output compliance.
Temperature=1.0 prevents Gemini 3.x from looping at low temperature.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel

import llm_gateway as gw
from schemas import Goal, MemoryItem, Observation

# --------------------------------------------------------------------------- #
# Internal schema — what the LLM emits (no goal IDs, positional identity)      #
# --------------------------------------------------------------------------- #

class _GoalSlot(BaseModel):
    text: str
    done: bool = False
    artifact_index: int | None = None


class _PerceptionResponse(BaseModel):
    goals: list[_GoalSlot]


# --------------------------------------------------------------------------- #
# System prompt                                                                 #
# --------------------------------------------------------------------------- #

_SYSTEM = """\
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
- MEMORY WRITES: If the query uses words like "remember", "save", "store",
  "note", "record", or "keep" to ask that a specific fact be retained, you MUST
  include a goal to durably save that fact.  Example goal text:
    "Save mom's birthday (May 15, 2026) to memory/moms_birthday.txt"
  Place this memory-write goal FIRST so the fact is persisted before any
  follow-up actions that depend on it.
- Return ONLY valid JSON matching the schema. No prose or commentary."""


def _build_messages(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    artifact_index_map: dict[int, str],
) -> list[dict]:
    # Format memory hits with [artifact N] labels for artifact-carrying items
    inv_map = {v: k for k, v in artifact_index_map.items()}
    hits_lines: list[str] = []
    for item in hits:
        line = f"  [{item.kind}] {item.descriptor}"
        if item.artifact_id and item.artifact_id in inv_map:
            line += f"  [artifact {inv_map[item.artifact_id]}]"
        hits_lines.append(line)

    hits_text = "\n".join(hits_lines) if hits_lines else "  (none)"

    # Format prior goals
    if prior_goals:
        goals_lines = [
            f"  {i+1}. {'[done]' if g.done else '[open]'} {g.text}"
            for i, g in enumerate(prior_goals)
        ]
        goals_text = "\n".join(goals_lines)
    else:
        goals_text = "  (none — first iteration, decompose the query)"

    # Format history (recent entries only)
    hist_entries: list[str] = []
    for h in history[-12:]:
        kind = h.get("kind")
        if kind == "action":
            hist_entries.append(
                f"  iter {h['iter']}: TOOL {h['tool']}({h.get('arguments', {})}) "
                f"→ {h.get('result_descriptor', '')[:120]}"
            )
        elif kind == "answer":
            hist_entries.append(
                f"  iter {h['iter']}: ANSWER for goal {h.get('goal_id', '?')}: "
                f"{h.get('text', '')[:120]}"
            )
    hist_text = "\n".join(hist_entries) if hist_entries else "  (empty)"

    user_content = (
        f"QUERY:\n  {query}\n\n"
        f"MEMORY HITS:\n{hits_text}\n\n"
        f"HISTORY:\n{hist_text}\n\n"
        f"PRIOR GOALS:\n{goals_text}\n\n"
        "Return the updated goal list as JSON."
    )
    return [{"role": "user", "content": user_content}]


# --------------------------------------------------------------------------- #
# Public interface                                                              #
# --------------------------------------------------------------------------- #

async def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """One LLM call (Gemini) — returns an updated Observation with stable IDs."""
    # Build 1-based index map: index → artifact_id (only for artifact-carrying hits)
    artifact_index_map: dict[int, str] = {}
    idx = 1
    for item in hits:
        if item.artifact_id:
            artifact_index_map[idx] = item.artifact_id
            idx += 1

    messages = _build_messages(query, hits, history, prior_goals, artifact_index_map)

    try:
        resp = await gw.chat(
            messages,
            system=_SYSTEM,
            auto_route="perception",
            provider="g",
            response_model=_PerceptionResponse,
            temperature=1.0,
        )
        perc = gw.parse_model(resp, _PerceptionResponse)
    except Exception as exc:
        # Fallback: return prior goals unchanged (or bare initial goal)
        if prior_goals:
            return Observation(goals=prior_goals)
        return Observation(
            goals=[Goal(id=uuid.uuid4().hex[:8], text=query, done=False)]
        )

    # Convert _PerceptionResponse → Observation with stable IDs
    new_goals: list[Goal] = []
    for i, slot in enumerate(perc.goals):
        # Preserve existing ID at this position
        if i < len(prior_goals):
            goal_id = prior_goals[i].id
            # Sticky-done: once done always done
            done = slot.done or prior_goals[i].done
        else:
            goal_id = uuid.uuid4().hex[:8]
            done = slot.done

        # Map artifact_index → artifact_id
        art_id: str | None = None
        if slot.artifact_index is not None:
            art_id = artifact_index_map.get(slot.artifact_index)

        new_goals.append(
            Goal(
                id=goal_id,
                text=slot.text,
                done=done,
                attach_artifact_id=art_id,
            )
        )

    # Safety net: never lose prior goals if LLM returned fewer
    if len(new_goals) < len(prior_goals):
        for i in range(len(new_goals), len(prior_goals)):
            new_goals.append(prior_goals[i])

    return Observation(goals=new_goals)
