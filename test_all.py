"""
test_all.py — Run the canonical agent test suite and print colour-coded results.

Canonical queries
-----------------
A  Artifact fetch + extraction  (Claude Shannon Wikipedia)
B  Multi-goal + live weather    (Tokyo family activities)
C1 Durable memory write         (mom's birthday remember + reminder)
C2 Durable memory recall        (when is mom's birthday?)
D  Web research + URL reading   (Python asyncio best practices)

Usage:
    uv run python test_all.py            # all 5 queries
    uv run python test_all.py 1 3 4      # run queries A, C1, C2 only (1-based)
"""
from __future__ import annotations

import asyncio
import sys
import time

# Windows: force UTF-8 so box-drawing characters don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── ANSI colour helpers ──────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def cyan(t: str)   -> str: return _c("96", t)
def yellow(t: str) -> str: return _c("93", t)
def green(t: str)  -> str: return _c("92", t)
def red(t: str)    -> str: return _c("91", t)
def bold(t: str)   -> str: return _c("1",  t)
def dim(t: str)    -> str: return _c("2",  t)


# ── Canonical test queries ───────────────────────────────────────────────────

QUERIES: list[tuple[str, str]] = [
    (
        "Query A — Artifact fetch & extraction",
        (
            "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me "
            "his birth date, death date, and three key contributions to "
            "information theory."
        ),
    ),
    (
        "Query B — Multi-goal with live weather",
        (
            "Find 3 family-friendly things to do in Tokyo this weekend. "
            "Check Saturday's weather forecast there and tell me which one "
            "is most appropriate."
        ),
    ),
    (
        "Query C1 — Durable memory write",
        (
            "My mom's birthday is 15 May 2026. Remember that and give me a "
            "calendar reminder for two weeks before and on the day."
        ),
    ),
    (
        "Query C2 — Durable memory recall",
        "When is mom's birthday?",
    ),
    (
        "Query D — Web research & URL reading",
        (
            "Search for 'Python asyncio best practices', read the top 3 "
            "results, and give me a short numbered list of the advice they "
            "agree on."
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
    print(bold(cyan(f"  [{n}/{total}]  {label}")))
    print(bold(cyan(_rule("═"))))


def _print_question(query: str) -> None:
    print()
    print(bold(yellow("  QUESTION")))
    words = query.split()
    line: list[str] = []
    for w in words:
        if len(" ".join(line + [w])) > _WIDTH - 4:
            print(yellow(f"  {' '.join(line)}"))
            line = [w]
        else:
            line.append(w)
    if line:
        print(yellow(f"  {' '.join(line)}"))
    print()


def _print_answer(answer: str, elapsed: float) -> None:
    print(bold(green("  ANSWER")) + dim(f"  ({elapsed:.1f}s)"))
    print()
    for ln in answer.splitlines():
        print(green(f"  {ln}"))
    print()


def _print_error(exc: Exception, elapsed: float) -> None:
    print(bold(red("  ERROR")) + dim(f"  ({elapsed:.1f}s)"))
    print()
    for ln in str(exc).splitlines():
        print(red(f"  {ln}"))
    print()


def _summary(results: list[tuple[str, bool, float]]) -> None:
    print()
    print(bold(_rule()))
    print(bold("  RESULTS"))
    print(bold(_rule()))
    for label, ok, elapsed in results:
        icon   = bold(green("  ✓")) if ok else bold(red("  ✗"))
        timing = dim(f"{elapsed:5.1f}s")
        print(f"{icon}  {timing}  {label}")
    print(bold(_rule()))
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    colour = green if passed == total else (yellow if passed > 0 else red)
    total_t = sum(t for _, _, t in results)
    print(bold(colour(f"  {passed}/{total} passed  ({total_t:.1f}s total)")))
    print(bold(_rule()))
    print()


# ── Runner ───────────────────────────────────────────────────────────────────

_QUERY_TIMEOUT = 180  # seconds per query before we give up


async def _run_one(query: str) -> tuple[str, float]:
    from agent import run as agent_run
    t0 = time.perf_counter()
    answer = await asyncio.wait_for(agent_run(query), timeout=_QUERY_TIMEOUT)
    return answer, time.perf_counter() - t0


async def main(indices: list[int] | None = None) -> None:
    total = len(QUERIES)

    selected = (
        [(i, QUERIES[i - 1][0], QUERIES[i - 1][1]) for i in indices if 1 <= i <= total]
        if indices
        else [(i + 1, label, q) for i, (label, q) in enumerate(QUERIES)]
    )

    results: list[tuple[str, bool, float]] = []

    for n, label, query in selected:
        _banner(label, n, total)
        _print_question(query)
        try:
            answer, elapsed = await _run_one(query)
            _print_answer(answer, elapsed)
            results.append((label, True, elapsed))
        except asyncio.TimeoutError:
            elapsed = _QUERY_TIMEOUT
            _print_error(TimeoutError(f"Query timed out after {_QUERY_TIMEOUT}s"), elapsed)
            results.append((label, False, elapsed))
        except Exception as exc:
            _print_error(exc, 0.0)
            results.append((label, False, 0.0))

    _summary(results)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    indices: list[int] | None = None
    if len(sys.argv) > 1:
        try:
            indices = [int(a) for a in sys.argv[1:]]
        except ValueError:
            print(f"Usage: {sys.argv[0]} [query_numbers...]  (1=A  2=B  3=C1  4=C2  5=D)")
            sys.exit(1)
    asyncio.run(main(indices))
