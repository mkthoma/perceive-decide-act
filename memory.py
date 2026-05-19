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
Classify the user's text as one of the memory kinds:
  fact        — an objective statement about the world or a person
  preference  — a user preference, like, dislike, or style choice
  tool_outcome — result of a tool execution (rarely used here)
  scratchpad  — working note, temporary, run-scoped

Return JSON matching this schema:
{
  "kind": "fact" | "preference" | "tool_outcome" | "scratchpad",
  "keywords": [list of 3-10 lowercase keyword strings for search],
  "descriptor": "one short human-readable line summarizing the item",
  "value": { structured payload relevant to the kind }
}

For a fact: value = {"entity": ..., "attribute": ..., "value": ...}
For a preference: value = {"topic": ..., "preference": ...}
For scratchpad: value = {"text": <original text>}
Return ONLY valid JSON, no prose."""


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
