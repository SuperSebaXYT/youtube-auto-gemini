"""Shared runtime for Animals & Wildlife AI agents.

The agent layer is deliberately batch-oriented: a single Gemini request can
review many topics/claims at once. This keeps the system comfortably below
Gemini free-tier request-per-minute limits and makes the agents easier to
scale later.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from dotenv import load_dotenv

from src.gemini_client import generate

load_dotenv()


class AgentRateLimitError(RuntimeError):
    """Raised after the Gemini gateway exhausts its retry budget."""


MAX_RETRIES = 3
INITIAL_RETRY_SECONDS = 5


def _extract_json(text: str) -> Any:
    """Parse JSON from a clean response or a fenced/extra-text response."""
    if not text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [
        p
        for p in (
            cleaned.find("{"),
            cleaned.find("["),
        )
        if p >= 0
    ]

    if not starts:
        raise ValueError(
            f"Gemini did not return JSON: {cleaned[:600]}"
        )

    start = min(starts)

    end_object = cleaned.rfind("}")
    end_array = cleaned.rfind("]")
    end = max(end_object, end_array)

    if end <= start:
        raise ValueError(
            f"Gemini did not return complete JSON: "
            f"{cleaned[:600]}"
        )

    try:
        return json.loads(
            cleaned[start:end + 1]
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON: {exc}. "
            f"Response: {cleaned[:600]}"
        ) from exc


def _looks_like_transient_error(exc: Exception) -> bool:
    """Recognize temporary provider errors, excluding exhausted daily free quota."""
    message = str(exc).upper()
    if (
        "FREE-MODELS-PER-DAY" in message
        or "OPENROUTER_FREE_TIER_DAILY" in message
    ):
        return False

    return any(
        token in message
        for token in (
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE_LIMIT",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
            "SERVICE_UNAVAILABLE",
            "TEMPORARILY",
            "HIGH DEMAND",
        )
    )


def generate_text(
    prompt: str,
    *,
    max_output_tokens: int = 2500,
    model: str | None = None,
    json_mode: bool = False,
) -> str:
    """Central Gemini gateway with bounded exponential retry for transient errors."""

    for attempt in range(MAX_RETRIES + 1):
        try:
            return generate(
                prompt,
                max_output_tokens=max_output_tokens,
                model=(
                    model
                    or "gemini-3.5-flash-lite"
                ),
                json_mode=json_mode,
            )

        except Exception as exc:
            if (
                not _looks_like_transient_error(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            delay = INITIAL_RETRY_SECONDS * (2 ** attempt)

            print(
                "AI Gateway: temporary Gemini service error "
                f"({str(exc)[:180]}); retrying in {delay}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )

            time.sleep(delay)

    raise AgentRateLimitError(
        "Gemini retry budget exhausted."
    )


def generate_json(
    prompt: str,
    *,
    max_output_tokens: int = 2500,
    model: str | None = None,
) -> Any:
    """Generate and parse JSON through the shared gateway."""

    budget = max_output_tokens
    last_error = None
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            return _extract_json(
                generate_text(
                    prompt,
                    max_output_tokens=budget,
                    model=model,
                    json_mode=True,
                )
            )

        except ValueError as e:
            last_error = e
            if attempt >= max_attempts - 1:
                break

            budget = int(budget * 1.6)

            print(
                "AI Gateway: JSON parse failed; retrying "
                f"with max_output_tokens={budget} "
                f"(attempt {attempt + 2}/{max_attempts})..."
            )

    raise last_error


def run_agent(
    name: str,
    role: str,
    task: str,
    context: str = "",
) -> str:
    """Backward-compatible single-agent text call."""

    prompt = f"""
You are the {name} AI agent for an Animals & Wildlife
YouTube channel.

ROLE:
{role}

TASK:
{task}

CONTEXT:
{context}

Rules:
- Focus on animals and wildlife.
- Be factually accurate.
- Do not invent evidence.
- Do not use fake clickbait.
- Prefer specific, interesting animal facts.
- Keep claims understandable for a general audience.
- Optimize for long-term channel quality.
- Return useful output, not generic advice.
"""

    return generate_text(
        prompt,
        max_output_tokens=2200,
    )


def run_json_batch(
    name: str,
    role: str,
    task: str,
    context: str = "",
    *,
    max_output_tokens: int = 3500,
) -> Any:
    """Run one structured batch agent call."""

    prompt = f"""
You are the {name} AI agent for an Animals & Wildlife
YouTube channel.

ROLE:
{role}

TASK:
{task}

CONTEXT:
{context}

NON-NEGOTIABLE RULES:
- All generated content must remain focused on animals
  and wildlife when the task concerns channel content.
- Factual accuracy beats sensationalism.
- Never invent facts, measurements, citations, or discoveries.
- Treat hypotheses and uncertain claims as hypotheses.
- Do not turn uncertain animal behavior into a guaranteed fact.
- Prefer genuinely interesting accurate topics over dramatic weak ones.
- Avoid space, astronomy, planets, stars, galaxies, NASA,
  and unrelated science unless the task explicitly requires
  comparing animals with those subjects.
- Return ONLY valid JSON.
- No markdown fences.
- No commentary outside JSON.
"""

    return generate_json(
        prompt,
        max_output_tokens=max_output_tokens,
    )
