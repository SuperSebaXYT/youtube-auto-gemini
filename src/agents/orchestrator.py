"""Shared runtime for Animals & Wildlife AI agents.

The agent layer is deliberately batch-oriented: a single Gemini request can
review many topics/claims at once. This keeps the system comfortably below
Gemini free-tier request-per-minute limits and makes the agents easier to
scale later.
"""

from __future__ import annotations

import json
import os
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


def _looks_like_rate_limit(
    exc: Exception,
) -> bool:
    text = str(exc).upper()

    return any(
        token in text
        for token in (
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE_LIMIT",
        )
    )


def generate_text(
    prompt: str,
    *,
    max_output_tokens: int = 2500,
    model: str | None = None,
) -> str:
    """Central Gemini gateway with bounded backoff for 429s."""

    for attempt in range(MAX_RETRIES + 1):
        try:
            return generate(
                prompt,
                max_output_tokens=max_output_tokens,
                model=(
                    model
                    or "gemini-3.5-flash-lite"
                ),
            )

        except Exception as exc:
            if (
                not _looks_like_rate_limit(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            delay = (
                INITIAL_RETRY_SECONDS
                * (2 ** attempt)
            )

            print(
                "AI Gateway: Gemini rate limit "
                f"detected; retrying in {delay}s "
                f"(attempt {attempt + 1}/"
                f"{MAX_RETRIES})..."
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

    for attempt in range(3):
        try:
            return _extract_json(
                generate_text(
                    prompt,
                    max_output_tokens=budget,
                    model=model,
                )
            )

        except ValueError as e:
            last_error = e
            budget = int(budget * 1.6)

            print(
                "AI Gateway: JSON parse failed "
                f"(likely truncated), retrying with "
                f"max_output_tokens={budget} "
                f"(attempt {attempt + 1}/3)..."
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
- Optimize for long-term YouTube channel quality.
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