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
    - Apply PROACTIVE REMINDERS: if query contains a near-future time reference
      (weekend / Saturday / Sunday / tonight / tomorrow / next week / specific date)
      AND a recommendation word (which / appropriate / best / suggest / should),
      add a reminder goal LAST.  Example:
        "Save the activity recommendation to memory/reminder.txt"
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
    [ ] Did I apply PROACTIVE REMINDERS for time-bound recommendation queries?
    [ ] For date-offset goals: have I computed exact derived dates, not left them as offsets?
    [ ] Are all goals ≤ 15 words and imperative (start with a verb)?

ERROR HANDLING / FALLBACKS:
  - If HISTORY is ambiguous about whether a goal is satisfied, mark it NOT done
    (conservative — let Decision retry rather than skip a needed step).
  - If QUERY is very short and unclear, create a single broad goal:
    "Research and answer: <query verbatim>"
  - If an artifact is referenced but no [artifact N] label exists in MEMORY HITS,
    set artifact_index to null (do not guess).

Return ONLY valid JSON matching the schema. No prose or commentary outside JSON."""


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
