"""Memory service — typed, persistent, LLM-classified fact store.

Read methods are free (keyword overlap, no LLM).
Write via remember() costs one gateway call for classification.
Write via record_outcome() is free (kind is tool_outcome by construction).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import llm_gateway as gw
from schemas import MemoryItem, ToolCall

_STATE_DIR = Path("state")
_MEMORY_FILE = _STATE_DIR / "memory.json"

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with is was are were be been "
    "it its this that these those i my me we our you your he she they their "
    "what when where how why which who do did does have has had will would "
    "could should may might can not no yes get give tell find show make go "
    "take use want need know about just from by up out if so do than then "
    "here there now".split()
)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


# --------------------------------------------------------------------------- #
# Schema for the LLM classification call                                        #
# --------------------------------------------------------------------------- #

class _RelevanceResponse(BaseModel):
    indices: list[int]


class _Classification(BaseModel):
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict


_CLASSIFY_SYSTEM = """\
You are MEMORY CLASSIFIER. Your job is to read a piece of text and produce a
structured memory record for an agentic system.

REASONING PROCESS — work through each step before writing output:

  STEP 1 — IDENTIFY THE REASONING TYPE
    Ask: what kind of reasoning does this classification require?
    • Lookup/recognition  → matching text to a known category definition
    • Entity extraction   → pulling out a named subject + attribute + value
    • Preference parsing  → detecting subjective like/dislike language
    • Fallback            → text is ambiguous or mixed → use "scratchpad"
    Name the reasoning type in your head before proceeding.

  STEP 2 — CHOOSE THE KIND
    Read the input and decide which category fits best:
    • fact         — verifiable, objective statement about a person, place, or event
    • preference   — user preference, like, dislike, or style choice ("I prefer…", "I like…")
    • tool_outcome — direct result of a tool execution (rarely used; default to fact/scratchpad)
    • scratchpad   — working note, ambiguous, or run-scoped temporary text

  STEP 3 — EXTRACT KEYWORDS (3-10 lowercase tokens)
    Pick the most distinctive nouns, names, dates, and domain terms.
    Avoid stopwords ("the", "a", "is", "was").

  STEP 4 — WRITE DESCRIPTOR AND VALUE
    • descriptor: one line, ≤15 words, human-readable summary.
    • value: use the schema for the chosen kind (see VALUE SCHEMAS below).

  STEP 5 — SELF-CHECK before outputting:
    [ ] Does "kind" exactly match one of the four defined values?
    [ ] Are keywords specific enough to retrieve this record later by keyword search?
    [ ] Is the descriptor ≤ 15 words and free of JSON syntax?
    [ ] Does the value follow the correct schema for the chosen kind?
    [ ] Is the output ONLY a JSON object — no prose before or after?

VALUE SCHEMAS:
  fact:         {"entity": "<subject>",   "attribute": "<aspect>",   "value": "<the fact>"}
  preference:   {"topic":  "<domain>",    "preference": "<what the user prefers>"}
  tool_outcome: {"tool":   "<tool name>", "result": "<outcome summary>"}
  scratchpad:   {"text":   "<original text verbatim>"}

EXAMPLES:
  Input: "Claude Shannon was born on April 30, 1916 in Michigan."
  Reasoning: verifiable fact about a person → kind=fact, entity extraction.
  Output: {"kind":"fact","keywords":["claude","shannon","born","1916","april","michigan"],"descriptor":"Claude Shannon birth date and birthplace","value":{"entity":"Claude Shannon","attribute":"birth_date_and_place","value":"April 30, 1916, Michigan"}}

  Input: "I always prefer dark mode in my code editors."
  Reasoning: subjective user preference → kind=preference, preference parsing.
  Output: {"kind":"preference","keywords":["dark","mode","code","editor","ui"],"descriptor":"User prefers dark mode in code editors","value":{"topic":"code editor UI","preference":"dark mode"}}

  Input: "web_search returned: {title: 'Paris', snippet: 'Capital of France'}"
  Reasoning: direct output of a tool execution → kind=tool_outcome.
  Output: {"kind":"tool_outcome","keywords":["web","search","paris","capital","france"],"descriptor":"web_search result: Paris is capital of France","value":{"tool":"web_search","result":"Paris - Capital of France"}}

  Input: "maybe follow up on the Tokyo itinerary later"
  Reasoning: vague working note, no verifiable entity+attribute+value → kind=scratchpad.
  Output: {"kind":"scratchpad","keywords":["tokyo","itinerary","follow","up"],"descriptor":"Reminder to follow up on Tokyo itinerary","value":{"text":"maybe follow up on the Tokyo itinerary later"}}

ERROR HANDLING:
  - Unsure between "fact" and "scratchpad"? Choose "fact" when the statement contains a verifiable entity+attribute+value triple; otherwise "scratchpad".
  - Unsure between "preference" and "fact"? "Preference" if first-person and subjective; "fact" if third-person and verifiable.
  - Keywords hard to find? Use the most distinctive nouns from the text, even if only 3.
  - If the text is very short (< 5 words), use "scratchpad" and set keywords to individual words.

Return ONLY valid JSON matching the schema. No prose, no explanation, no markdown fences."""


# --------------------------------------------------------------------------- #
# Memory class                                                                  #
# --------------------------------------------------------------------------- #

class Memory:
    def __init__(self) -> None:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._items: list[MemoryItem] = []
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if _MEMORY_FILE.exists():
            raw = json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
            self._items = [MemoryItem.model_validate(r) for r in raw]

    def _save(self) -> None:
        _MEMORY_FILE.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in self._items],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Read — no LLM cost                                                   #
    # ------------------------------------------------------------------ #

    def read(
        self,
        query: str,
        history: list[dict],
        kinds: list[str] | None = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Keyword-overlap search. Fast, no LLM."""
        q_tokens = _tokenize(query)
        hist_text = " ".join(
            str(h.get("result_descriptor", "")) + " " + str(h.get("text", ""))
            for h in history[-6:]
        )
        combined = q_tokens | _tokenize(hist_text)

        candidates = self._items
        if kinds:
            candidates = [i for i in candidates if i.kind in kinds]

        scored: list[tuple[int, MemoryItem]] = []
        for item in candidates:
            score = len(combined & set(item.keywords))
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: (-x[0], -x[1].created_at.timestamp()))
        return [item for _, item in scored[:top_k]]

    def filter(
        self,
        kinds: list[str] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        result = self._items[:]
        if kinds:
            result = [i for i in result if i.kind in kinds]
        if goal_id:
            result = [i for i in result if i.goal_id == goal_id]
        if recent:
            result = result[-recent:]
        return result

    async def relevant(
        self,
        query: str,
        kinds: list[str] | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """LLM-scored relevance. Used only when keyword recall is weak."""
        candidates = self.filter(kinds=kinds)
        if not candidates:
            return []
        items_text = "\n".join(
            f"{i}: [{c.kind}] {c.descriptor}"
            for i, c in enumerate(candidates)
        )
        relevance_system = (
            f"Return JSON {{\"indices\": [<int>, ...]}} listing at most {top_k} "
            "0-based indices from ITEMS that are most relevant to QUERY. "
            "Most relevant first."
        )
        messages = [
            {"role": "user", "content": f"QUERY: {query}\n\nITEMS:\n{items_text}"},
        ]
        try:
            resp = await gw.chat(
                messages,
                system=relevance_system,
                auto_route="memory",
                response_model=_RelevanceResponse,
            )
            rel = gw.parse_model(resp, _RelevanceResponse)
            return [candidates[i] for i in rel.indices if 0 <= i < len(candidates)]
        except Exception:
            return candidates[:top_k]

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    async def remember(
        self,
        raw_text: str,
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> MemoryItem:
        """Classify free-form text via one gateway call, then persist."""
        messages = [{"role": "user", "content": raw_text}]
        try:
            resp = await gw.chat(
                messages,
                system=_CLASSIFY_SYSTEM,
                auto_route="memory",
                response_model=_Classification,
                temperature=1.0,
            )
            cls = gw.parse_model(resp, _Classification)
        except Exception:
            # Fallback: scratchpad with basic keywords
            cls = _Classification(
                kind="scratchpad",
                keywords=list(_tokenize(raw_text))[:10],
                descriptor=raw_text[:100],
                value={"text": raw_text},
            )
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind=cls.kind,
            keywords=cls.keywords,
            descriptor=cls.descriptor,
            value=cls.value,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            created_at=datetime.now(timezone.utc),
        )
        self._items.append(item)
        self._save()
        return item

    def clear(self) -> None:
        """Wipe the in-RAM cache only.

        Disk files (state/memory.json, sandbox/memory/*.txt) are untouched so
        the agent must re-discover facts via read_file / list_dir tool calls.
        Useful in tests to verify the agent reads from disk rather than from
        a warm in-process cache.
        """
        self._items = []

    def record_outcome(
        self,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        """Record tool outcome without an LLM call."""
        kw_src = (
            tool_call.name + " "
            + " ".join(str(v) for v in tool_call.arguments.values())
        )
        keywords = list(_tokenize(kw_src))[:15]
        descriptor = f"[{tool_call.name}] {result_text[:100]}"
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords,
            descriptor=descriptor,
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:500],
            },
            artifact_id=artifact_id,
            source="action",
            run_id=run_id,
            goal_id=goal_id,
            created_at=datetime.now(timezone.utc),
        )
        self._items.append(item)
        self._save()
        return item


# Module-level singleton — persists across calls within a process
memory = Memory()
