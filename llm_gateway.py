"""LLM Gateway client — imports llm_gatewayV3/ directly (no HTTP server needed).

Provides the same chat() / extract_text() / extract_tool_calls() / parse_model()
interface as before, but calls provider adapters in-process instead of POSTing
to localhost:8101.  The gateway's Router, RouterPool, and rate-state machinery
are reused exactly — the only thing removed is the FastAPI/HTTP/DB layer.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

# ── add llm_gatewayV3/ to the import path ───────────────────────────────────
_GATEWAY_DIR = Path(__file__).parent / "llm_gatewayV3"
if str(_GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(_GATEWAY_DIR))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import providers as P
from cache import GeminiCache
from router import DEFAULT_ROUTER_ORDER, LIMITS, Router, RouterPool, resolve


# ── Routing config (mirrors main.py) ────────────────────────────────────────

_DEFAULT_ORDER = "ollama,gemini,nvidia,groq,cerebras,openrouter,github"

# Min context window (tokens) required for each tier.
# Candidates are always drawn from router.order (LLM_ORDER) to respect user preference.
_TIER_MIN_CTX = {
    "TINY": 0,
    "LARGE": 8000,
}

_ROUTER_PROMPT = (
    "You are a routing classifier. Given a token_count and a content sample, "
    "output exactly one of: TINY, LARGE, or HUGE.\n\n"
    "Rules:\n"
    "- TINY: token_count below 1000 with simple factual content.\n"
    "- LARGE: token_count between 1000 and 8000, OR token_count below 1000 "
    "but content is dense (code, base64, multilingual, technical).\n"
    "- HUGE: token_count above 8000.\n\n"
    "Output the single word and nothing else."
)


# ── Module-level singletons (lazy-initialized on first chat() call) ──────────

_router: Router | None = None
_router_pool: RouterPool | None = None


def _init() -> tuple[Router, RouterPool]:
    global _router, _router_pool
    if _router is None:
        cache = GeminiCache(ttl_seconds=300)
        order = [x.strip() for x in os.getenv("LLM_ORDER", _DEFAULT_ORDER).split(",") if x.strip()]
        router_order = [x.strip() for x in os.getenv("ROUTER_ORDER", ",".join(DEFAULT_ROUTER_ORDER)).split(",") if x.strip()]
        _router = Router(P.build_providers(cache), order)
        _router_pool = RouterPool(P.build_router_providers(), router_order)
    return _router, _router_pool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.4)


def _tier_from_count(tokens: int) -> str:
    if tokens > 8000:
        return "HUGE"
    if tokens >= 1000:
        return "LARGE"
    return "TINY"


def _parse_tier(text: str) -> str | None:
    up = (text or "").upper()
    for tier in ("HUGE", "LARGE", "TINY"):
        if tier in up:
            return tier
    return None


async def _classify_tier(prompt_text: str, router_pool: RouterPool) -> str:
    estimated = _estimate_tokens(prompt_text)
    if estimated > 8000:
        return "HUGE"

    head, tail = 400, 400
    sample = prompt_text if len(prompt_text) <= head + tail + 10 else prompt_text[:head] + "\n...\n" + prompt_text[-tail:]
    envelope = f"token_count: {estimated}\nsample:\n{sample}"

    for name in router_pool.candidates():
        ok, _ = router_pool.state[name].can_use(LIMITS[name], 400)
        if not ok:
            continue
        provider = router_pool.providers[name]
        router_pool.state[name].record(0)
        try:
            result = await provider.chat(
                messages=[{"role": "user", "content": envelope}],
                system_blocks=_ROUTER_PROMPT,
                max_tokens=8,
                temperature=0,
            )
            tier = _parse_tier(result.get("text", ""))
            if tier == "HUGE" and estimated <= 8000:
                tier = "LARGE"
            if tier:
                return tier
        except Exception:
            continue

    return _tier_from_count(estimated)


def _strip_titles(schema: dict) -> dict:
    cleaned: dict = {}
    for k, v in schema.items():
        if k == "title":
            continue
        if isinstance(v, dict):
            cleaned[k] = _strip_titles(v)
        elif isinstance(v, list):
            cleaned[k] = [_strip_titles(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def _backoff_secs(err: Exception) -> float:
    msg = str(err).lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "queue" in msg:
            return 15
        if "quota" in msg or "rpm" in msg or "per minute" in msg:
            return 60
        if "rpd" in msg or "per day" in msg or "daily" in msg:
            return 3600
        return 30
    if status and 500 <= status < 600:
        return 20
    if status in (401, 403):
        return 600
    return 0


# ── Public API ───────────────────────────────────────────────────────────────

async def chat(
    messages: list[dict],
    *,
    system: str | None = None,
    auto_route: str | None = None,
    provider: str | None = None,
    response_model: Type[BaseModel] | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    cache_system: bool = False,
    reasoning: str | None = None,
) -> dict:
    router, router_pool = _init()

    system_blocks: Any = None
    if system:
        system_blocks = [{"text": system, "cache": True}] if cache_system else system

    # Providers accept a plain dict: {"type","schema","name","strict"}
    # (same shape as ResponseFormat.model_dump(by_alias=True))
    response_format: dict | None = None
    if response_model:
        response_format = {
            "type": "json_schema",
            "schema": _strip_titles(response_model.model_json_schema()),
            "name": "out",
            "strict": True,
        }

    prompt_text = "".join(str(m.get("content", "")) for m in messages)
    est = _estimate_tokens(prompt_text) + 2048

    required_caps: list[str] = []
    if tools:
        required_caps.append("tools")
    if reasoning and reasoning != "off":
        required_caps.append("reasoning")
    if response_format:
        required_caps.append("structured")

    # Determine candidate list
    if auto_route and not provider:
        tier = await _classify_tier(prompt_text, router_pool)
        if tier == "HUGE":
            tier = "LARGE"
        # Respect LLM_ORDER (router.order) and filter by minimum context window.
        min_ctx = _TIER_MIN_CTX.get(tier, 0)
        candidates = [
            p for p in router.order
            if LIMITS.get(p, {}).get("max_ctx", 0) > min_ctx
        ]
    elif provider:
        resolved = resolve(provider) or provider
        candidates = [resolved] if resolved in router.providers else []
        if not candidates:
            raise RuntimeError(f"Unknown provider '{provider}'. Available: {list(router.providers)}")
    else:
        candidates = list(router.order)

    all_attempts: list[dict] = []
    last_err: str | None = None
    initial_candidates = list(candidates)

    # Up to 3 rounds: each round tries all remaining candidates once,
    # then waits for the shortest cooldown (≤10s) before the next round.
    for _round in range(3):
        for _ in range(len(candidates) + 1):
            name, atts = router.pick(est, candidates, required_caps=required_caps)
            all_attempts.extend(atts)
            if name is None:
                break

            prov = router.providers[name]
            router.state[name].record(0)

            try:
                result = await prov.chat(
                    messages,
                    max_tokens=2048,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning=reasoning,
                    response_format=response_format,
                    system_blocks=system_blocks,
                    cache_system=cache_system,
                )
            except P.ProviderError as e:
                last_err = str(e)
                secs = _backoff_secs(e)
                if secs > 0:
                    router.state[name].mark_unavailable(secs, str(e)[:80])
                all_attempts.append({"provider": name, "reason": f"failed: {str(e)[:100]}"})
                candidates = [c for c in candidates if c != name]
                continue
            except Exception as e:
                last_err = str(e)
                all_attempts.append({"provider": name, "reason": f"exception: {str(e)[:100]}"})
                candidates = [c for c in candidates if c != name]
                continue

            tokens = (result.get("input_tokens") or 0) + (result.get("output_tokens") or 0)
            router.state[name].tokens_today += tokens
            router.state[name].tokens_minute.append((time.time(), tokens))

            # Normalize tool_calls — ensure arguments is always a dict
            normalized: list[dict] = []
            for tc in result.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                normalized.append({"id": tc.get("id", ""), "name": tc.get("name", ""), "arguments": args})

            # Best-effort structured output validation
            parsed: dict | None = None
            if response_format and response_model and not normalized:
                try:
                    import jsonschema
                    obj = json.loads(result.get("text", ""))
                    schema = _strip_titles(response_model.model_json_schema())
                    jsonschema.Draft202012Validator(schema).validate(obj)
                    parsed = obj
                except Exception:
                    pass

            return {
                "provider": name,
                "model": result.get("model", name),
                "text": result.get("text", ""),
                "tool_calls": normalized,
                "stop_reason": result.get("stop_reason", "end_turn"),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "parsed": parsed,
            }

        # All candidates tried — find shortest wait across initial set:
        # prefer soft cooldown (≤10 s), fall back to hard backoff (≤90 s).
        # This lets the gateway recover from short rate-limit windows without
        # the caller having to re-issue the request.
        now = time.time()
        min_wait: float | None = None
        min_hard: float | None = None
        for cname in initial_candidates:
            prov_c = router.providers.get(cname)
            if prov_c is None:
                continue
            caps_c = getattr(prov_c, "capabilities", {})
            if required_caps and any(not caps_c.get(c) for c in required_caps):
                continue
            state_c = router.state[cname]
            if state_c.unavailable_until > now:
                hard_wait = state_c.unavailable_until - now
                if hard_wait <= 90 and (min_hard is None or hard_wait < min_hard):
                    min_hard = hard_wait
                continue
            wait_c = LIMITS.get(cname, {}).get("cooldown", 0) - (now - state_c.last_call)
            if 0 < wait_c <= 10 and (min_wait is None or wait_c < min_wait):
                min_wait = wait_c

        # Prefer soft cooldown; fall back to hard backoff if ≤90 s
        effective_wait = min_wait if min_wait is not None else min_hard
        if effective_wait is None:
            break  # all providers in long-term backoff or day quota — give up

        await asyncio.sleep(effective_wait + 0.2)
        # Restore initial candidate list so all providers retry
        candidates = list(initial_candidates)

    raise RuntimeError(
        f"All providers unavailable. Attempts: {all_attempts}. Last error: {last_err}\n"
        "  Add GEMINI_API_KEY or GROQ_API_KEY to .env"
    )


def extract_text(response: dict) -> str:
    return response.get("text", "")


def extract_tool_calls(response: dict) -> list[dict]:
    return response.get("tool_calls") or []


def parse_model(response: dict, model: Type[BaseModel]) -> BaseModel:
    """Parse a gateway response into a Pydantic model.

    Prefers response['parsed'] (pre-validated JSON) when set,
    otherwise falls back to parsing response['text'].
    """
    parsed = response.get("parsed")
    if isinstance(parsed, dict):
        return model.model_validate(parsed)

    text = response.get("text", "").strip()
    if not text:
        raise ValueError(f"Empty response from gateway: {response}")

    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

    try:
        return model.model_validate_json(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return model.model_validate_json(match.group())
        except Exception:
            pass

    raise ValueError(
        f"Could not parse gateway response as {model.__name__}.\n"
        f"Raw text:\n{text[:500]}"
    )
