"""
test_all.py — Run all four canonical agent queries and show results with
              colour-coded output.

Usage:
    uv run python test_all.py          # run all 4 queries
    uv run python test_all.py 1 3      # run queries 1 and 3 only (1-indexed)
"""
from __future__ import annotations

import asyncio
import sys
import time

# Windows: ensure UTF-8 output so box-drawing chars don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── ANSI colour helpers ──────────────────────────────────────────────────────

# Detect whether the terminal supports colour (skip when piped to a file)
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def cyan(t: str)    -> str: return _c("96", t)
def yellow(t: str)  -> str: return _c("93", t)
def green(t: str)   -> str: return _c("92", t)
def bold(t: str)    -> str: return _c("1",  t)
def dim(t: str)     -> str: return _c("2",  t)
def red(t: str)     -> str: return _c("91", t)


# ── Canonical test queries ───────────────────────────────────────────────────

QUERIES: list[tuple[str, str]] = [
    (
        "Query A — Real-time tool use",
        "What time is it in Tokyo right now?",
    ),
    (
        "Query B — Artifact fetch & extraction",
        (
            "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me "
            "his birth date, death date, and three key contributions to "
            "information theory."
        ),
    ),
    (
        "Query C — Multi-goal planning",
        (
            "Find 3 family-friendly things to do in Tokyo this weekend. "
            "Check Saturday's weather forecast there and tell me which one "
            "is most appropriate."
        ),
    ),
    (
        "Query D — Web research & synthesis",
        (
            "Search for recent asyncio best practices in Python and "
            "summarize the top 3 recommendations."
        ),
    ),
]


# ── Display helpers ──────────────────────────────────────────────────────────

_WIDTH = 72

def _rule(char: str = "─") -> str:
    return char * _WIDTH

def _banner(label: str, n: int, total: int) -> None:
    print()
    print(bold(cyan(_rule("═"))))
    index_tag = dim(f"[{n}/{total}]")
    print(bold(cyan(f"  {index_tag} {label}")))
    print(bold(cyan(_rule("═"))))


def _print_question(query: str) -> None:
    print()
    print(bold(yellow("  QUESTION")))
    # Wrap long queries visually
    words = query.split()
    line, lines = [], []
    for w in words:
        if len(" ".join(line + [w])) > _WIDTH - 4:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        print(yellow(f"  {ln}"))
    print()


def _print_answer(answer: str, elapsed: float) -> None:
    print(bold(green("  ANSWER")))
    print(dim(f"  (completed in {elapsed:.1f}s)"))
    print()
    # Indent each line of the answer
    for ln in answer.splitlines():
        print(green(f"  {ln}"))
    print()


def _print_error(exc: Exception, elapsed: float) -> None:
    print(bold(red("  ERROR")))
    print(dim(f"  (failed after {elapsed:.1f}s)"))
    print()
    print(red(f"  {exc}"))
    print()


def _summary(results: list[tuple[str, bool, float]]) -> None:
    print()
    print(bold(_rule("─")))
    print(bold("  SUMMARY"))
    print(bold(_rule("─")))
    total_time = sum(t for _, _, t in results)
    for label, ok, elapsed in results:
        status = bold(green("  PASS")) if ok else bold(red("  FAIL"))
        print(f"{status}  {dim(f'{elapsed:5.1f}s')}  {label}")
    print(bold(_rule("─")))
    passed = sum(1 for _, ok, _ in results if ok)
    colour = green if passed == len(results) else (yellow if passed > 0 else red)
    print(bold(colour(f"  {passed}/{len(results)} queries passed  ({total_time:.1f}s total)")))
    print(bold(_rule("─")))
    print()


# ── Core runner ──────────────────────────────────────────────────────────────

async def run_query(label: str, query: str) -> tuple[str, float]:
    """Run one query through the agent; return (answer, elapsed_seconds)."""
    # Import here so the MCP server doesn't start until we actually need it
    from agent import run as agent_run
    t0 = time.perf_counter()
    answer = await agent_run(query)
    return answer, time.perf_counter() - t0


async def main(indices: list[int] | None = None) -> None:
    """
    indices — 1-based list of queries to run; None runs all.
    """
    total = len(QUERIES)

    # Build list of (1-based-index, label, query)
    if indices:
        selected = [
            (i, QUERIES[i - 1][0], QUERIES[i - 1][1])
            for i in indices
            if 1 <= i <= total
        ]
    else:
        selected = [(i + 1, label, query) for i, (label, query) in enumerate(QUERIES)]

    results: list[tuple[str, bool, float]] = []

    for orig_idx, label, query in selected:
        _banner(label, orig_idx, total)
        _print_question(query)

        try:
            answer, elapsed = await run_query(label, query)
            _print_answer(answer, elapsed)
            results.append((label, True, elapsed))
        except Exception as exc:
            elapsed = 0.0
            _print_error(exc, elapsed)
            results.append((label, False, elapsed))

    _summary(results)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optional: pass 1-based query numbers as CLI args to run a subset
    # e.g.  uv run python test_all.py 1 3
    indices: list[int] | None = None
    if len(sys.argv) > 1:
        try:
            indices = [int(a) for a in sys.argv[1:]]
        except ValueError:
            print(f"Usage: {sys.argv[0]} [query_numbers...]  (e.g. 1 3)")
            sys.exit(1)

    asyncio.run(main(indices))
